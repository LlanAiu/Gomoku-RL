# builtin

# external

# internal
from .constants import BOARD_SIZE

class GameSnapshot:
    def __init__(self):
        self.board: list[list[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]