import argparse
from enum import IntEnum
import random
from typing import Any, Tuple

from chainer import serializers
import chainer.functions as F
import numpy as np

from constants import ROW, COLUMN, ROW_TABLE, COLUMN_TABLE, COLUMN_INVERSE_TABLE, TABLE,\
    HORIZONTAL_MASK, VERTICAL_MASK, SQUARE_MASK, BITMASK, Direction
from sl_policy_network import SLPolicyNetwork


class Game:
    def __init__(self,
                 opponent: SLPolicyNetwork
                 ) -> None:
        self.black = np.array([0x0000000810000000], dtype='uint64')
        self.white = np.array([0x0000001008000000], dtype='uint64')
        self.opponent = opponent

        self._reversed_indices = None
        self._bin_mask = None
        self.black_move_channel = np.ones((1, ROW, COLUMN), dtype='float32')
        self.white_move_channel = np.zeros((1, ROW, COLUMN), dtype='float32')
        self._bin_mask = np.left_shift(
            np.ones((1, ROW * COLUMN), dtype='uint64'),
            (np.ones((1, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1])
        self.set_cache()

    def set_cache(self):
        self._reversed_indices = (np.ones((1, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1]
        self._bin_mask = np.left_shift(
            np.ones((1, ROW * COLUMN), dtype='uint64'),
            (np.ones((1, ROW * COLUMN), dtype='uint64').cumsum(axis=1) - 1)[:, ::-1])

    def get_state(self,
                  is_black: bool
                  ):
        black_channel = (np.bitwise_and(self.black, self._bin_mask) / self._bin_mask).astype('float32').reshape(1, ROW, COLUMN)
        white_channel = (np.bitwise_and(self.white, self._bin_mask) / self._bin_mask).astype('float32').reshape(1, ROW, COLUMN)
        move_channel = self.black_move_channel if is_black else self.white_move_channel
        state = np.stack([black_channel, white_channel, move_channel], axis=1)
        return state

    def print_board(self,
                    is_black: bool,
                    memory: int,
                    valid_moves: np.array
                    ) -> None:
        rev = list(format(memory, '064b'))
        val = valid_moves[:-1]
        black_channel = np.bitwise_and(self.black, self._bin_mask) / self._bin_mask
        white_channel = np.bitwise_and(self.white, self._bin_mask) / self._bin_mask
        sb = ''.join(map(str, black_channel.flatten().astype('int')))
        sw = ''.join(map(str, white_channel.flatten().astype('int')))
        board = [0] * 64
        for index, (b, w, r, v) in enumerate(zip(sb, sw, rev, val)):
            if bool(int(b)):
                s = '\033[33m' + ' B' + '\033[0m' if bool(int(r)) else ' B'
            elif bool(int(w)):
                s = '\033[33m' + ' W' + '\033[0m' if bool(int(r)) else ' W'
            else:
                s = COLUMN_INVERSE_TABLE[index % 8]
                s += str(index // 8 + 1)
                if v:
                    s = '\033[31m' + s + '\033[0m'
            board[index] = s
        for row in range(8):
            print(' '.join(map(str, board[row * 8:(row + 1) * 8])))
        valid_indices = np.where(self.valid_move(is_black))[0]
        valid_moves = [COLUMN_INVERSE_TABLE[valid_indice % 8] + str(valid_indice // 8 + 1)
                       if valid_indice != 64 else 'PASS' for valid_indice in valid_indices]
        print(f'valid moves: {" ".join(valid_moves)}')

    def valid_move(self,
                   is_black: Any,  # (b, ), numpy.ndarray or cupy.core.core.ndarray
                   ) -> Any:
        player, opponent = self.which_turn(is_black, self.black, self.white)
        blanks = ~(self.black | self.white)
        h = opponent & HORIZONTAL_MASK
        v = opponent & VERTICAL_MASK
        s = opponent & SQUARE_MASK
        valid = self._valid_move(player, v, blanks, Direction.LEFT)
        valid |= self._valid_move(player, s, blanks, Direction.UPPERLEFT)
        valid |= self._valid_move(player, h, blanks, Direction.UP)
        valid |= self._valid_move(player, s, blanks, Direction.UPPERRIGHT)
        bin_valid_mask = np.bitwise_and(valid, self._bin_mask) / self._bin_mask  # (1, 64)
        bin_pass_mask = np.logical_not(np.any(bin_valid_mask, axis=1, keepdims=True)).astype(float)
        valid = np.concatenate([bin_valid_mask, bin_pass_mask], axis=-1).astype('bool').squeeze(axis=0)
        return valid

    @staticmethod
    def _valid_move(player: Any,           # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    masked_opponent: Any,  # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    blank: Any,            # (b, ), numpy.ndarray or cupy.core.core.ndarray
                    dir_enum: IntEnum
                    ) -> Any:
        direction = dir_enum.value
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

    def reversible(self,
                   next_move: np.array,
                   is_black: bool,
                   ) -> np.array:
        player, opponent = self.which_turn(is_black, self.black, self.white)
        h = ~(player | opponent & HORIZONTAL_MASK)
        v = ~(player | opponent & VERTICAL_MASK)
        s = ~(player | opponent & SQUARE_MASK)
        reversible_mask = self._reversible(next_move, player, v, Direction.LEFT)
        reversible_mask |= self._reversible(next_move, player, s, Direction.UPPERLEFT)
        reversible_mask |= self._reversible(next_move, player, h, Direction.UP)
        reversible_mask |= self._reversible(next_move, player, s, Direction.UPPERRIGHT)
        return reversible_mask

    @staticmethod
    def _reversible(next_move: int,
                    player: np.array,
                    masked_blank: np.array,
                    dir_enum: IntEnum
                    ) -> np.array:
        direction = dir_enum.value
        one_reversible_mask = 0
        # ~(player | masked_blank) ... candidates of reversible disc
        one_temp = ~(player | masked_blank) & (next_move << direction)
        if one_temp:
            for i in range(6):
                one_temp <<= direction
                # can not reverse due to opponent being put between player and blank
                if one_temp & masked_blank:
                    break
                elif one_temp & player:
                    one_reversible_mask |= one_temp >> direction
                    break
                else:
                    one_temp |= one_temp >> direction

        the_other_reversible_mask = 0
        temp = ~(player | masked_blank) & ((next_move >> direction) & BITMASK)
        if temp:
            for i in range(6):
                temp >>= direction
                if temp & masked_blank:
                    break
                elif temp & player:
                    the_other_reversible_mask |= temp << direction
                    break
                else:
                    temp |= temp << direction
        return one_reversible_mask | the_other_reversible_mask

    def reverse(self,
                next_move: int,
                is_black: bool,
                reversible_mask: np.array
                ) -> None:
        _mask = reversible_mask ^ next_move
        if is_black:
            self.black = self.black ^ _mask
            self.white = self.white ^ reversible_mask
        else:
            self.black = self.black ^ reversible_mask
            self.white = self.white ^ _mask

    @staticmethod
    def which_turn(is_black: bool,
                   current_black: np.array,
                   current_white: np.array
                   ) -> Tuple[np.array, np.array]:
        player = current_black if is_black else current_white
        opponent = current_white if is_black else current_black
        return player, opponent


def translate(_input: str  # e.g. 'C4'
              ) -> Any:
    try:
        row = ROW_TABLE[_input[1]]
        column = COLUMN_TABLE[_input[0]]

        translated = ['0'] * 16
        index = TABLE[_input[1]] * 2
        if _input[0] in {'E', 'F', 'G', 'H'}:
            index += 1
        translated[index] = TABLE[_input[0]]
        return int(''.join(map(str, translated)), 16), row * ROW + column
    except KeyError:
        return ('PASS', 64) if _input == 'PASS' else ('Invalid', -1)


def index_to_move(index: int
                  ) -> Any:
    if index < 64:
        row = index // 4
        column = index % 4

        translated = [0] * 16
        translated[row] = 8 >> column
        return int(''.join(map(str, translated)), 16)
    else:
        return 'PASS'


def main():
    parser = argparse.ArgumentParser(description='SLPolicyNetwork')
    parser.add_argument('MODEL', default=None, type=str, help='path to model')
    parser.add_argument('--first', action='store_true', help='switch to first turn')
    args = parser.parse_args()

    is_black = args.first

    n_input_channel = 3
    n_output_channel = 128
    opponent = SLPolicyNetwork(n_input_channel=n_input_channel, n_output_channel=n_output_channel)
    opponent.set_cache()
    serializers.load_npz(args.MODEL, opponent)

    game = Game(opponent)
    memory = 0

    if is_black:
        pass
    else:
        first_move, first_move_index = translate(random.choice(['D3', 'C4', 'F5', 'E6']))
        reversible_mask = game.reversible(first_move, not is_black)
        game.reverse(first_move, not is_black, reversible_mask)
        memory = int(reversible_mask + first_move)

    while True:
        valid_moves = game.valid_move(is_black)
        game.print_board(is_black, memory, valid_moves)
        print('input > ', end='')
        _input = input()
        next_move, next_move_index = translate(_input)
        if next_move == 'PASS':
            state = game.get_state(not is_black)
            next_opponent_index = int(np.argmax(F.softmax(opponent.predict(state)).data))
            next_opponent_move = index_to_move(next_opponent_index)
            print(next_opponent_index)
            if next_opponent_move != 'PASS':
                reversible_mask = game.reversible(next_opponent_move, not is_black)
                game.reverse(next_opponent_move, not is_black, reversible_mask)
                memory = int(reversible_mask+next_opponent_move)
            else:
                print('end')
                break
        elif next_move == 'Invalid':
            print('Invalid input1')
        elif valid_moves[next_move_index]:
            reversible_mask = game.reversible(next_move, is_black)
            game.reverse(next_move, is_black, reversible_mask)

            # opponent turn
            state = game.get_state(not is_black)
            next_opponent_index = int(np.argmax(F.softmax(opponent.predict(state)).data))
            next_opponent_move = index_to_move(next_opponent_index)
            print(next_opponent_index)
            if next_opponent_move != 'PASS':
                reversible_mask = game.reversible(next_opponent_move, not is_black)
                game.reverse(next_opponent_move, not is_black, reversible_mask)
                memory = int(reversible_mask+next_opponent_move)
        else:
            print('Invalid input2')


if __name__ == '__main__':
    main()