from enum import IntEnum

ROW = 8
COLUMN = 8

HORIZONTAL_MASK = 0x00ffffffffffff00
VERTICAL_MASK = 0x7e7e7e7e7e7e7e7e
SQUARE_MASK = 0x007e7e7e7e7e7e00
MASK = -1e6

COLUMN_INVERSED_TABLE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}


class Direction(IntEnum):
    LEFT = 0x1
    UPPERRIGHT = 0x7
    UP = 0x8
    UPPERLEFT = 0x9
