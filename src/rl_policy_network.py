from enum import IntEnum
from typing import Any, Tuple

import chainer
from chainer import Chain
from chainerrl.distribution import SoftmaxDistribution

from constants import ROW, COLUMN, HORIZONTAL_MASK, VERTICAL_MASK, SQUARE_MASK, Direction, MASK
from model_components import InputLayer, ResNet, OutputLayer


class RLPolicyNetwork(Chain):
    def __init__(self,
                 n_input_channel: int,
                 n_output_channel: int,
                 ) -> None:
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel

        self._reversed_indices = None
        self._bin_mask = None

        # only the weight of links objects in self.init_scope() is updated
        links = {'layer1': InputLayer(n_input_channel, n_output_channel)}
        links.update({
            f'layer{i}': ResNet(n_output_channel, n_output_channel)
            for i in range(2, 10)
        })
        links['layer10'] = OutputLayer(n_output_channel)
        super(RLPolicyNetwork, self).__init__(**links)

    def __call__(self,
                 states: Any,   # (b, n_input_channel, ROW, COLUMN)
                 ) -> Any:
        h = self.layer1(states)  # (b, n_input_channel, ROW, COLUMN)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = self.layer7(h)
        h = self.layer8(h)
        h = self.layer9(h)
        policies = self.layer10(h)  # (b, ROW * COLUMN + 1)
        valid_mask = self.valid_moves(states)
        policies += valid_mask
        # Categoricalは微分不可能なので注意
        action_distribution = SoftmaxDistribution(policies)
        return action_distribution

    def set_cache(self):
        self._reversed_indices = (self.xp.ones((1, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1]
        self._bin_mask = self.xp.left_shift(
            self.xp.ones((1, ROW * COLUMN), dtype='uint64'),
            (self.xp.ones((1, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1])

    def valid_moves(self,
                    states: Any  # (b, n_input_channel, ROW, COLUMN)
                    ) -> Any:
        b = len(states)

        black_masks = states[:, 0, :, :].reshape(b, -1).astype('uint64')                          # (b, 64)
        hex_black_channels = self.xp.left_shift(black_masks, self._reversed_indices).sum(axis=1)  # (b, )
        white_masks = states[:, 1, :, :].reshape(b, -1).astype('uint64')                          # (b, 64)
        hex_white_channels = self.xp.left_shift(white_masks, self._reversed_indices).sum(axis=1)  # (b, )
        is_blacks = states[:, 2, 0, 0].astype(bool)                                               # (b, )

        hex_valid_mask = self.valid_move(hex_black_channels, hex_white_channels, is_blacks)       # (b, )
        # (b, ) -> (b, ROW * COLUMN)
        bin_valid_mask = self.xp.bitwise_and(hex_valid_mask.reshape(-1, 1), self._bin_mask) / self._bin_mask
        bin_pass_mask = self.xp.logical_not(self.xp.any(bin_valid_mask, axis=1, keepdims=1)).astype(float)
        valid_mask = ~(self.xp.concatenate([bin_valid_mask, bin_pass_mask], axis=-1).astype('bool')) * MASK
        return valid_mask

    def valid_move(self,
                   blacks: Any,     # (b, ), numpy.ndarray or cupy.core.core.ndarray
                   whites: Any,     # (b, ), numpy.ndarray or cupy.core.core.ndarray
                   is_blacks: Any,  # (b, ), numpy.ndarray or cupy.core.core.ndarray
                   ) -> Any:
        player, opponent = self.which_turns(blacks, whites, is_blacks)
        blanks = ~(blacks | whites)
        h = opponent & HORIZONTAL_MASK
        v = opponent & VERTICAL_MASK
        s = opponent & SQUARE_MASK
        valid = self._valid_move(player, v, blanks, Direction.LEFT)
        valid |= self._valid_move(player, s, blanks, Direction.UPPERLEFT)
        valid |= self._valid_move(player, h, blanks, Direction.UP)
        valid |= self._valid_move(player, s, blanks, Direction.UPPERRIGHT)
        return valid

    @staticmethod
    def _valid_move(player: Any,           # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    masked_opponent: Any,  # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    blank: Any,            # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    dir_enum: IntEnum
                    ) -> Any:
        direction = int(dir_enum)
        # move to the direction step by step
        one = masked_opponent & (player << direction)
        one |= masked_opponent & (one << direction)
        one |= masked_opponent & (one << direction)
        one |= masked_opponent & (one << direction)
        one |= masked_opponent & (one << direction)
        one |= masked_opponent & (one << direction)
        one_valid_mask = blank & (one << direction)

        the_other = masked_opponent & (player >> direction)
        the_other |= masked_opponent & (the_other >> direction)
        the_other |= masked_opponent & (the_other >> direction)
        the_other |= masked_opponent & (the_other >> direction)
        the_other |= masked_opponent & (the_other >> direction)
        the_other |= masked_opponent & (the_other >> direction)
        the_other_valid_mask = blank & (the_other >> direction)
        return one_valid_mask | the_other_valid_mask

    @staticmethod
    def which_turns(blacks: Any,    # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    whites: Any,    # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    is_blacks: Any  # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    ) -> Tuple[Any, Any]:
        player = blacks * is_blacks + whites * ~is_blacks
        opponent = blacks * ~is_blacks + whites * is_blacks
        return player, opponent

    def predict(self,
                states: Any  # (b, n_input_channel, ROW, COLUMN)
                ) -> chainer.Variable:
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            h = self.layer1(states)
            h = self.layer2(h)
            h = self.layer3(h)
            h = self.layer4(h)
            h = self.layer5(h)
            h = self.layer6(h)
            h = self.layer7(h)
            h = self.layer8(h)
            h = self.layer9(h)
            policies = self.layer10(h)    # (b, ROW * COLUMN + 1)
            valid_mask = self.valid_moves(states)
            policies += valid_mask
            return policies
