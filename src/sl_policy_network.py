from enum import IntEnum
from typing import Tuple

import chainer
import chainer.functions as F
from chainer import Chain

from constants import ROW, COLUMN, HORIZONTAL_MASK, VERTICAL_MASK, SQUARE_MASK, Direction
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
                 states: chainer.Variable,   # (b, n_input_channel, ROW, COLUMN)
                 actions: chainer.Variable,  # (b, 1)
                 ):
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
        valid_mask = self.valid_moves(states)
        policies += valid_mask
        loss = F.softmax_cross_entropy(policies, actions)
        accuracy = F.accuracy(policies, actions)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

    def valid_moves(self,
                    states: chainer.Variable  # (b, n_input_channel, ROW, COLUMN)
                    ) -> chainer.Variable:
        b = len(states)
        black_channels = [int(''.join(map(str, channels[0].flatten().astype('int'))), 2) for channels in states]
        white_channels = [int(''.join(map(str, channels[1].flatten().astype('int'))), 2) for channels in states]
        moves = [True if channels[2][0][0] == 1. else False for channels in states]

        valid_mask = self.xp.zeros((b, ROW * COLUMN + 1), 'f')
        for i in range(b):
            valid = self.valid_move(black_channels[i], white_channels[i], moves[i])
            valid = self.xp.array(list(format(valid, '064b')), 'f')   # (64)
            if any(valid):  # in the case that there is at least one valid move
                valid = self.xp.concatenate((valid, self.xp.array([0.])), axis=0)  # (65)
            else:
                valid = self.xp.concatenate((valid, self.xp.array([1.])), axis=0)  # (65)
            # impose a penalty where the mask value is 0
            valid_mask[i] = ~(valid.astype('bool')) * -1e6
        return valid_mask

    def valid_move(self,
                   black: int,
                   white: int,
                   is_black: bool
                   ) -> int:
        player, opponent = self.which_turn(black, white, is_black)
        blank = ~(black | white)
        h = opponent & HORIZONTAL_MASK
        v = opponent & VERTICAL_MASK
        s = opponent & SQUARE_MASK
        valid = self._valid_move(player, v, blank, Direction.LEFT)
        valid |= self._valid_move(player, s, blank, Direction.UPPERLEFT)
        valid |= self._valid_move(player, h, blank, Direction.UP)
        valid |= self._valid_move(player, s, blank, Direction.UPPERRIGHT)
        return valid

    @staticmethod
    def _valid_move(player: int,
                    masked_opponent: int,
                    blank: int,
                    direction: IntEnum
                    ) -> int:
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
    def which_turn(black: int,
                   white: int,
                   is_black: bool
                   ) -> Tuple[int, int]:
        player = black if is_black else white
        opponent = white if is_black else black
        return player, opponent

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
