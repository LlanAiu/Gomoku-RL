# builtin

# external
import numpy as np


# internal
from ..constants import BOARD_SIZE
from ...rl.elements import State


class GameState(State):
    def __init__(self):
        super().__init__()
        self.board: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    def _set_state_representation(self):
        self.representation = np.zeros(BOARD_SIZE * BOARD_SIZE)