from enum import IntEnum

ROW = 8
COLUMN = 8

HORIZONTAL_MASK = 0x00ffffffffffff00
VERTICAL_MASK = 0x7e7e7e7e7e7e7e7e
SQUARE_MASK = 0x007e7e7e7e7e7e00
MASK = -1e6
BITMASK = 0xffffffffffffffff

PASS = 64

ROW_TABLE = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, }
COLUMN_TABLE = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
COLUMN_INVERSE_TABLE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
TABLE = {'A': '8', 'B': '4', 'C': '2', 'D': '1', 'E': '8', 'F': '4', 'G': '2', 'H': '1',
         '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, }


class Direction(IntEnum):
    LEFT = 0x1
    UPPERRIGHT = 0x7
    UP = 0x8
    UPPERLEFT = 0x9
