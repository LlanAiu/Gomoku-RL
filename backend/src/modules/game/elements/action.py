# builtin

# external

# internal
from ...rl.elements import Action
from ..constants import BOARD_SIZE


class GameAction(Action):
    def __init__(self, player_index: int, move: tuple[int, int]):
        super().__init__()
        self.player_index: int = player_index
        self.move: tuple[int] = move
        
    def get_player_index(self) -> int:
        return self.player_index
    
    def get_move(self) -> tuple[int, int]:
        return self.move
    
    def get_flattened_index(self) -> int:
        row, col = self.move
        return row * BOARD_SIZE + col