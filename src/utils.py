from typing import List, Tuple

import numpy as np

from constants import ROW, COLUMN


def transform(b: str,  # 64bits
              w: str,
              is_black: str,
              ) -> np.array:
    b_channel = [list(b[i * ROW:i * ROW + COLUMN]) for i in range(ROW)]  # (ROW, COLUMN) = (8, 8)
    w_channel = [list(w[i * ROW:i * ROW + COLUMN]) for i in range(ROW)]
    m_channel = [[1. if is_black is 'B' else 0. for _ in range(COLUMN)] for _ in range(ROW)]
    return np.array([b_channel, w_channel, m_channel], dtype='float32')  # (channels, ROW, COLUMN) = (3, 8, 8)


def load_data(path: str
              ) -> List[Tuple]:
    with open(path, "r") as f:
        data = [line.strip().split() for line in f]
    # (state, action)
    return [(transform(b, w, is_black),
             np.array(64 if n == -1 else int(n), dtype='int32'))
            for b, w, is_black, n in data]


def print_board(states: np.array  # (3, 8, 8)
                ) -> None:
    sb = ''.join(map(str, states[0].flatten().astype('int')))
    sw = ''.join(map(str, states[1].flatten().astype('int')))
    board = [0] * 64
    index = 0
    for b, w in zip(sb, sw):
        board[index] = int(b) * 2 + int(w)
        index += 1

    for i in range(8):
        print(''.join(map(str, board[i * 8:(i + 1) * 8])))
    else:
        print('')
