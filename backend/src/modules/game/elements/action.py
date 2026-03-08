# builtin

# external
import numpy as np

# internal
from ...rl.elements import Action


class GameAction(Action):
    def __init__(self, player_index: int, move: tuple[int, int]):
        self.player_index: int = player_index
        self.move: np.ndarray = np.array(move, dtype=np.uint8)
        
    def get_player_index(self) -> int:
        return self.player_index
    
    def get_move(self) -> np.ndarray:
        return self.move