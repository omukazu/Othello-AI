from enum import IntEnum
from typing import List, Tuple

import chainer.functions as F
import numpy as np

from constants import ROW, COLUMN, HORIZONTAL_MASK, VERTICAL_MASK, SQUARE_MASK, Direction
from sl_policy_network import SLPolicyNetwork
from rl_policy_network import RLPolicyNetwork


class Env:
    def __init__(self,
                 batch_size: int,
                 xp,
                 player: RLPolicyNetwork,
                 opponent: SLPolicyNetwork
                 ) -> None:
        self.batch_size = batch_size
        self.xp = xp
        self.black = xp.array([0x0000000810000000] * batch_size, dtype='uint64').reshape(-1, 1)  # (b, 1)
        self.white = xp.array([0x0000001008000000] * batch_size, dtype='uint64').reshape(-1, 1)  # (b, 1)
        self.player = player
        self.opponent = opponent

        self.move_channels = xp.ones((batch_size, ROW, COLUMN), dtype='float32')
        self._bin_mask = xp.left_shift(
            xp.ones((batch_size, ROW * COLUMN), dtype='uint64'),
            (xp.ones((batch_size, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1])

    # set up environment and return current observation
    def reset(self
              ):
        self.black = self.xp.array([0x0000000810000000] * self.batch_size, dtype='uint64').reshape(-1, 1)  # (b, 1)
        self.white = self.xp.array([0x0000001008000000] * self.batch_size, dtype='uint64').reshape(-1, 1)  # (b, 1)

    @staticmethod
    def translate(indices
                  ):
        mask = (indices == 64)
        exponents = ((63 - indices) * ~mask + indices * mask).astype('uint64')
        # 2 ** exponents とすると計算の精度の関係でおかしな値になる
        return 1 << exponents

    def step(self,
             action_indices,  # (b, )
             is_black: bool,
             ):
        actions = self.translate(action_indices).reshape(-1, 1)  # (b, )
        reversible_mask = self.reversible(actions, is_black)     # (b, )
        self.black, self.white = \
            self.reverse(actions, is_black, reversible_mask)
        is_black = not is_black

        current_states = self.create_current_states(is_black)    # (b, 3, ROW, COLUMN)
        opponent_action_indices = \
            self.xp.argmax(F.softmax(self.opponent.predict(current_states)).data, axis=1)  # (b)
        opponent_actions = self.translate(opponent_action_indices).reshape(-1, 1).reshape(-1, 1)
        reversible_mask = self.reversible(opponent_actions, is_black)
        self.black, self.white = \
            self.reverse(opponent_actions, is_black, reversible_mask)
        is_black = not is_black

        obs = self.create_current_states(is_black)               # (b, 3, ROW, COLUMN)
        r = 0
        done = True if self.xp.all(actions == 0) and self.xp.all(opponent_actions == 0) else False
        info = None
        return obs, r, done, info

    def create_current_states(self,
                              is_black: bool
                              ):  # (b, 3, ROW, COLUMN)
        black_channels = \
            (self.xp.bitwise_and(self.black, self._bin_mask) / self._bin_mask).reshape(self.batch_size, ROW, COLUMN)
        white_channels = \
            (self.xp.bitwise_and(self.white, self._bin_mask) / self._bin_mask).reshape(self.batch_size, ROW, COLUMN)
        move_channels = self.move_channels if is_black else self.move_channels * 0
        states = self.xp.stack([black_channels, white_channels, move_channels], axis=1).astype('float32')
        return states

    def reversible(self,
                   next_moves: np.array,
                   is_black: bool,
                   ) -> np.array:
        player, opponent = self.which_turn(is_black, self.black, self.white)
        h = ~(player | opponent & HORIZONTAL_MASK)
        v = ~(player | opponent & VERTICAL_MASK)
        s = ~(player | opponent & SQUARE_MASK)
        reversible_mask = self._reversible(next_moves, player, v, Direction.LEFT)
        reversible_mask |= self._reversible(next_moves, player, s, Direction.UPPERLEFT)
        reversible_mask |= self._reversible(next_moves, player, h, Direction.UP)
        reversible_mask |= self._reversible(next_moves, player, s, Direction.UPPERRIGHT)
        return reversible_mask

    def _reversible(self,
                    next_moves: np.array,
                    player: np.array,
                    masked_blank: np.array,
                    dir_enum: IntEnum
                    ) -> np.array:
        direction = int(dir_enum)
        one_reversible_mask = 0
        candidates = ~(player | masked_blank)
        # ~(player | masked_blank) ... candidates of reversible disc
        one_tmp = candidates & (next_moves << direction)
        for i in range(6):
            one_tmp <<= direction
            one_reversible_mask |= (one_tmp >> direction) * ~(one_tmp & player == 0)
            one_tmp *= ((one_tmp & candidates) == one_tmp)
            one_tmp |= one_tmp >> direction

        the_other_reversible_mask = 0
        the_other_tmp = candidates & (next_moves >> direction)
        for i in range(6):
            if self.xp.all(the_other_tmp == 0):
                break
            the_other_tmp >>= direction
            the_other_reversible_mask |= (the_other_tmp << direction) * ~(the_other_tmp & player == 0)
            the_other_tmp *= ((the_other_tmp & candidates) == the_other_tmp)
            the_other_tmp |= the_other_tmp << direction
        return one_reversible_mask | the_other_reversible_mask

    def reverse(self,
                next_moves: List[int],
                is_black: bool,
                reversible_mask: np.array
                ) -> Tuple[np.array, np.array]:
        _mask = reversible_mask ^ next_moves
        if is_black:
            return self.black ^ _mask, self.white ^ reversible_mask
        else:
            return self.black ^ reversible_mask, self.white ^ _mask

    @staticmethod
    def which_turn(is_black: bool,
                   current_black: np.array,
                   current_white: np.array
                   ) -> Tuple[np.array, np.array]:
        player = current_black if is_black else current_white
        opponent = current_white if is_black else current_black
        return player, opponent
