from enum import IntEnum
from typing import Any, Tuple

import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain
from chainer.backends import cuda

from constants import HORIZONTAL_MASK, VERTICAL_MASK, SQUARE_MASK, Direction
from model_components import InputLayer, ResNet, OutputLayer


class SLPolicyNetwork(Chain):
    def __init__(self,
                 n_input_channel: int,
                 n_output_channel: int,
                 ) -> None:
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.board_size = 8
        # only the weight of links objects in self.init_scope() is updated
        links = {'layer1': InputLayer(n_input_channel, n_output_channel)}
        links.update({
            f'layer{i}': ResNet(n_output_channel, n_output_channel)
            for i in range(2, 10)
        })
        links['layer10'] = OutputLayer(n_output_channel)
        super(SLPolicyNetwork, self).__init__(**links)

    def __call__(self,
                 states: Any,   # (b, n_input_channel, ROW, COLUMN)
                 actions: Any,  # (b, 1)
                 ) -> Any:
        h = self.layer1(states)     # (b, n_input_channel, ROW, COLUMN)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = self.layer7(h)
        h = self.layer8(h)
        h = self.layer9(h)
        policies = self.layer10(h)  # (b, ROW * COLUMN + 1)
        """ NOTE: in training step, the model does not use valid_mask
        
        valid_mask = self.valid_moves(states)
        policies += valid_mask
        """
        loss = F.softmax_cross_entropy(policies, actions)
        accuracy = F.accuracy(policies, actions)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

    def predict(self,
                states: chainer.Variable  # (b, n_input_channel, ROW, COLUMN)
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
            prediction = self.xp.argmax(policies.data, axis=1)
            return prediction

    def valid_moves(self,
                    states: Any  # (b, n_input_channel, ROW, COLUMN)
                    ) -> Any:
        b = len(states)
        black_channels = self.xp.array([int(''.join(map(str, channels[0].flatten().astype('int'))), 2)
                                        for channels in states], dtype='uint64')                        # (b, 1)
        white_channels = self.xp.array([int(''.join(map(str, channels[1].flatten().astype('int'))), 2)
                                        for channels in states], dtype='uint64')                        # (b, 1)
        is_blacks = self.xp.array([True if channels[2][0][0] == 1. else False for channels in states])  # (b, 1)

        hex_valid_mask = self.valid_move(black_channels, white_channels, is_blacks)
        # (b, 1) -> (b, ROW * COLUMN)
        bin_valid_mask = self.xp.array(
            list(map(lambda x: list(self.xp.binary_repr(cuda.to_cpu(x), width=64)), hex_valid_mask)), 'f')
        pass_mask = self.xp.logical_not(self.xp.any(bin_valid_mask, axis=1, keepdims=-1)).astype(float)
        if self.xp != np:
            cuda.to_gpu(pass_mask)
        valid_mask = self.xp.logical_not(
            self.xp.concatenate([bin_valid_mask, pass_mask], axis=-1).astype('bool')) * -1e6
        return valid_mask

    def valid_move(self,
                   blacks: Any,     # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                   whites: Any,     # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                   is_blacks: Any,  # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                   ) -> Any:
        player, opponent = self.which_turns(blacks, whites, is_blacks)
        blank = self.xp.invert(self.xp.bitwise_or(blacks, whites))
        h = self.xp.bitwise_and(opponent, HORIZONTAL_MASK)
        v = self.xp.bitwise_and(opponent, VERTICAL_MASK)
        s = self.xp.bitwise_and(opponent, SQUARE_MASK)
        valid = self._valid_move(player, v, blank, Direction.LEFT)
        valid = self.xp.bitwise_or(valid, self._valid_move(player, s, blank, Direction.UPPERLEFT))
        valid = self.xp.bitwise_or(valid, self._valid_move(player, h, blank, Direction.UP))
        valid = self.xp.bitwise_or(valid, self._valid_move(player, s, blank, Direction.UPPERRIGHT))
        return valid

    def _valid_move(self,
                    player: Any,           # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    masked_opponent: Any,  # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    blank: Any,            # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    direction: IntEnum
                    ) -> Any:
        temp = int(direction)
        # move to the direction step by step
        one = self.xp.bitwise_and(masked_opponent, self.xp.left_shift(player, temp))
        one = self.xp.bitwise_or(one, self.xp.bitwise_and(masked_opponent, self.xp.left_shift(one, temp)))
        one = self.xp.bitwise_or(one, self.xp.bitwise_and(masked_opponent, self.xp.left_shift(one, temp)))
        one = self.xp.bitwise_or(one, self.xp.bitwise_and(masked_opponent, self.xp.left_shift(one, temp)))
        one = self.xp.bitwise_or(one, self.xp.bitwise_and(masked_opponent, self.xp.left_shift(one, temp)))
        one = self.xp.bitwise_or(one, self.xp.bitwise_and(masked_opponent, self.xp.left_shift(one, temp)))
        one_valid_mask = self.xp.bitwise_and(blank, self.xp.left_shift(one, temp))

        the_other = self.xp.bitwise_and(masked_opponent, self.xp.right_shift(player, temp))
        the_other = self.xp.bitwise_or(
            the_other, self.xp.bitwise_and(masked_opponent, self.xp.right_shift(the_other, temp)))
        the_other = self.xp.bitwise_or(
            the_other, self.xp.bitwise_and(masked_opponent, self.xp.right_shift(the_other, temp)))
        the_other = self.xp.bitwise_or(
            the_other, self.xp.bitwise_and(masked_opponent, self.xp.right_shift(the_other, temp)))
        the_other = self.xp.bitwise_or(
            the_other, self.xp.bitwise_and(masked_opponent, self.xp.right_shift(the_other, temp)))
        the_other = self.xp.bitwise_or(
            the_other, self.xp.bitwise_and(masked_opponent, self.xp.right_shift(the_other, temp)))
        the_other_valid_mask = self.xp.bitwise_and(blank, self.xp.right_shift(the_other, temp))
        return self.xp.bitwise_or(one_valid_mask, the_other_valid_mask)

    def which_turns(self,
                    blacks: Any,    # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    whites: Any,    # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    is_blacks: Any  # (b, 1), numpy.ndarray or cupy.core.core.ndarray
                    ) -> Tuple[Any, Any]:
        player = blacks * is_blacks + whites * self.xp.invert(is_blacks)
        opponent = blacks * self.xp.invert(is_blacks) + whites * is_blacks
        return player, opponent
