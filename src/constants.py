from enum import IntEnum

ROW = 8
COLUMN = 8

HORIZONTAL_MASK = 0x00ffffffffffff00
VERTICAL_MASK = 0x7e7e7e7e7e7e7e7e
SQUARE_MASK = 0x007e7e7e7e7e7e00


class Direction(IntEnum):
    LEFT = 0x1
    UPPERRIGHT = 0x7
    UP = 0x8
    UPPERLEFT = 0x9
