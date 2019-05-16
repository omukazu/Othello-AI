import argparse
from enum import IntEnum
from typing import Tuple

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
            ) -> int:
    _mask = reversible_mask ^ next_move
    if is_black:
        return current_black ^ _mask, current_white ^ reversible_mask
    else:
        return current_black ^ reversible_mask, current_white ^ _mask


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
    parser = argparse.ArgumentParser(description='restore a othello game')
    parser.add_argument('INPUT', help='path to input data')
    parser.add_argument('STATE', help='path to output data of states')
    parser.add_argument('ACTION', help='path to output data of actions')
    args = parser.parse_args()

    with open(args.INPUT, "r") as inp:
        lines = [line.strip().split() for line in inp]

    invalid_count = 0
    records = []
    for line in lines:
        current_black = 0x0000000810000000
        current_white = 0x0000001008000000
        is_black = True
        buffer = [line[0]]  # line[0] == winner
        for move in line[1:]:
            next_move = translate(move)
            if next_move == 'PA':
                is_black = not is_black
                buffer.append((current_black, current_white, next_move))
            elif next_move & valid(is_black, current_black, current_white) > 0:
                reversible_mask = reversible(next_move, is_black, current_black, current_white)
                current_black, current_white = \
                    reverse(next_move, is_black, current_black, current_white, reversible_mask)
                is_black = not is_black
                buffer.append((current_black, current_white, next_move))
                """ for debug
                
                print(format(valid('B', initial_black, initial_white), '016x'))
                print_board(current_black, current_white)
                """
            else:
                invalid_count += 1
                assert invalid_count < 100, 'could not restore too many times'
                break
        else:
            records.append(buffer)

    with open(args.STATE, "w") as sta, open(args.ACTION, "w") as act:
        for record in records:
            sta.write(record[0] + ' '.join([str(b) + ' ' + str(w) for b, w, _ in record[1:]]) + '\n')
            act.write(record[0] + ' '.join([str(b) + ' ' + str(w) for b, w, _ in record[1:]]) + '\n')


if __name__ == '__main__':
    main()
