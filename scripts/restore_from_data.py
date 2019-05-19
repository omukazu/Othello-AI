import argparse
from enum import IntEnum
from typing import Tuple

from progressbar import ProgressBar

ROW = 8
COLUMN = 8
PASS = 0
TABLE = {'A': '8', 'B': '4', 'C': '2', 'D': '1', 'E': '8', 'F': '4', 'G': '2', 'H': '1',
         '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, }


def translate(move: str  # e.g. 'C3'
              ) -> int or str:
    if move == 'PA':
        return 'PA'
    else:
        translated = ['0'] * 16
        index = TABLE[move[1]] * 2
        if move[0] in {'E', 'F', 'G', 'H'}:
            index += 1
        translated[index] = TABLE[move[0]]
        return int(''.join(translated), 16)


HORIZONTAL_MASK = 0x00ffffffffffff00
VERTICAL_MASK = 0x7e7e7e7e7e7e7e7e
SQUARE_MASK = 0x007e7e7e7e7e7e00


class Direction(IntEnum):
    LEFT = 0x1
    UPPERRIGHT = 0x7
    UP = 0x8
    UPPERLEFT = 0x9


def which_turn(is_black: bool,
               current_black: int,
               current_white: int
               ) -> Tuple[int, int]:
    player = current_black if is_black else current_white
    opponent = current_white if is_black else current_black
    return player, opponent


def valid(is_black: bool,
          current_black: int,
          current_white: int
          ) -> int:
    player, opponent = which_turn(is_black, current_black, current_white)
    current_blank = ~(current_black | current_white)
    h = opponent & HORIZONTAL_MASK
    v = opponent & VERTICAL_MASK
    s = opponent & SQUARE_MASK
    valid_mask = _valid(player, v, current_blank, Direction.LEFT)
    valid_mask |= _valid(player, s, current_blank, Direction.UPPERLEFT)
    valid_mask |= _valid(player, h, current_blank, Direction.UP)
    valid_mask |= _valid(player, s, current_blank, Direction.UPPERRIGHT)
    return valid_mask


def _valid(player: int,
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


def count_flags(bits: int
                ) -> int:
    bits = (bits & 0x5555555555555555) + (bits >> 1 & 0x5555555555555555)
    bits = (bits & 0x3333333333333333) + (bits >> 2 & 0x3333333333333333)
    bits = (bits & 0x0f0f0f0f0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f0f0f0f0f)
    bits = (bits & 0x00ff00ff00ff00ff) + (bits >> 8 & 0x00ff00ff00ff00ff)
    bits = (bits & 0x0000ffff0000ffff) + (bits >> 16 & 0x0000ffff0000ffff)
    return (bits & 0x00000000ffffffff) + (bits >> 32 & 0x00000000ffffffff)


def reversible(next_move: int,
               is_black: bool,
               current_black: int,
               current_white: int
               ) -> int:
    player, opponent = which_turn(is_black, current_black, current_white)
    h = ~(player | opponent & HORIZONTAL_MASK)
    v = ~(player | opponent & VERTICAL_MASK)
    s = ~(player | opponent & SQUARE_MASK)
    reversible_mask = _reversible(next_move, player, v, Direction.LEFT)
    reversible_mask |= _reversible(next_move, player, s, Direction.UPPERLEFT)
    reversible_mask |= _reversible(next_move, player, h, Direction.UP)
    reversible_mask |= _reversible(next_move, player, s, Direction.UPPERRIGHT)
    return reversible_mask


def _reversible(next_move: int,
                player: int,
                masked_blank: int,
                direction: IntEnum
                ) -> int:
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
    temp = ~(player | masked_blank) & (next_move >> direction)
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


def reverse(next_move: int,
            is_black: bool,
            current_black: int,
            current_white: int,
            reversible_mask: int
            ) -> Tuple[int, int]:
    _mask = reversible_mask ^ next_move
    if is_black:
        return current_black ^ _mask, current_white ^ reversible_mask
    else:
        return current_black ^ reversible_mask, current_white ^ _mask


""" flip, mirror, and rotate

refer to https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating
To obtain a rotated or mirrored board, we use flip vertically, horizontally, and diagonally 
"""


def fv(bits: int
       ) -> int:
    bits = (bits << 56) & 0xffffffffffffffff \
           | (bits << 40) & 0x00ff000000000000 \
           | (bits << 24) & 0x0000ff0000000000 \
           | (bits << 8) & 0x000000ff00000000 \
           | (bits >> 8) & 0x00000000ff000000 \
           | (bits >> 24) & 0x0000000000ff0000 \
           | (bits >> 40) & 0x000000000000ff00 \
           | (bits >> 56) & 0x00000000000000ff
    return bits


def fh(bits: int
       ) -> int:
    bits = ((bits >> 1) & 0x5555555555555555) + 2 * (bits & 0x5555555555555555)
    bits = ((bits >> 2) & 0x3333333333333333) + 4 * (bits & 0x3333333333333333)
    bits = ((bits >> 4) & 0x0f0f0f0f0f0f0f0f) + 16 * (bits & 0x0f0f0f0f0f0f0f0f)
    return bits


def fd(bits: int
       ) -> int:
    temp = 0x0f0f0f0f00000000 & (bits ^ (bits << 28))
    bits ^= temp ^ (temp >> 28)
    temp = 0x3333000033330000 & (bits ^ (bits << 14))
    bits ^= temp ^ (temp >> 14)
    temp = 0x5500550055005500 & (bits ^ (bits << 7))
    bits ^= temp ^ (temp >> 7)
    return bits


# for debug
def print_board(current_black: int,
                current_white: int
                ) -> None:
    sb, sw = format(current_black, '064b'), format(current_white, '064b')
    board = [0] * 64
    index = 0
    for b, w in zip(sb, sw):
        board[index] = int(b) * 2 + int(w)
        index += 1

    for i in range(8):
        print(''.join(map(str, board[i * 8:(i + 1) * 8])))
    else:
        print('')


def main():
    parser = argparse.ArgumentParser(description='replay a othello game')
    parser.add_argument('INPUT', help='path to input data')
    parser.add_argument('OUTPUT', help='path to output data of states')
    args = parser.parse_args()

    """　Replay a game based on board and move pairs
    
    To calculate faster, we use bitboard.
    """

    with open(args.INPUT, "r") as inp:
        lines = [line.strip().split() for line in inp]

    invalid_count = 0
    records = []
    bar = ProgressBar(0, len(lines))

    print('*** now restoring ... ***')
    for j, line in enumerate(lines):
        bar.update(j)
        current_black = 0x0000000810000000
        current_white = 0x0000001008000000
        is_black = True
        buffer = [line[0]]  # line[0] == winner
        for i, move in enumerate(line[1:]):
            next_move = translate(move)
            if next_move == 'PA':
                assert valid(is_black, current_black, current_white) == 0
                buffer.append((current_black, current_white, PASS))
                is_black = not is_black
            elif next_move & valid(is_black, current_black, current_white) > 0:
                buffer.append((current_black, current_white, next_move))
                reversible_mask = reversible(next_move, is_black, current_black, current_white)
                current_black, current_white = \
                    reverse(next_move, is_black, current_black, current_white, reversible_mask)
                is_black = not is_black
                """ for debug
                
                print(format(valid('B', initial_black, initial_white), '016x'))
                print_board(current_black, current_white)
                print_board(fv(fh(current_black)), fv(fh(current_white)))
                print_board(fd(current_black), fd(current_white))
                print_board(fv(fh(fd(current_black))), fv(fh(fd(current_white))))
                """
            else:
                invalid_count += 1
                assert invalid_count < 100, 'could not restore too many times'
                break
        else:
            records.append(buffer)

    print('*** begin to write ... ***')
    with open(args.OUTPUT, "w") as out:
        for record in records:
            winner = record[0]
            is_black = True if winner == 'B' else False
            source = record[1::2] if winner == 'B' else record[2::2]
            for b, w, n in source:
                # do not include the action 'PASS' or the case that there's only one valid move
                if count_flags(valid(is_black, b, w)) > 1:
                    out.write(format(b, '064b') + ' ' +
                              format(w, '064b') + ' ' +
                              winner + ' ' +
                              str(format(n, '064b').find('1')) + '\n')
                    out.write(format(fv(fh(b)), '064b') + ' ' +
                              format(fv(fh(w)), '064b') + ' ' +
                              winner + ' ' +
                              str(format(fv(fh(n)), '064b').find('1')) + '\n')
                    out.write(format(fd(b), '064b') + ' ' +
                              format(fd(w), '064b') + ' ' +
                              winner + ' ' +
                              str(format(fd(n), '064b').find('1')) + '\n')
                    out.write(format(fv(fh(fd(b))), '064b') + ' ' +
                              format(fv(fh(fd(w))), '064b') + ' ' +
                              winner + ' ' +
                              str(format(fv(fh(fd(n))), '064b').find('1')) + '\n')


if __name__ == '__main__':
    main()
