# builtin

# external
import numpy as np


# internal
from ..constants import BOARD_SIZE, WIN_COUNT, DIRECTIONS
from ...rl.elements import State
from .action import GameAction


class GameState(State):
    def __init__(self, board: np.ndarray | None = None):
        if board is None:
            self.board: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE))
        else: 
            self.board: np.ndarray = board.copy()
        super().__init__()
        
        self._check_if_won()
        
    def _check_if_won(self):
        self.terminal = False
        self.win_index = -1

        size = self.board.shape[0]
        for row in range(size):
            for col in range(size):
                player = int(self.board[row, col])
                if player == 0:
                    continue

                for d_row, d_col in DIRECTIONS:
                    if not self._is_start_of_run(row, col, d_row, d_col, player):
                        continue

                    count = self._count_direction(row, col, d_row, d_col)
                    if count >= WIN_COUNT:
                        self.terminal = True
                        self.win_index = player
                        return

        if not (self.board == 0).any():
            self.terminal = True
            self.win_index = 0

    def _in_bounds(self, row: int, col: int) -> bool:
        size = self.board.shape[0]
        return 0 <= row < size and 0 <= col < size

    def _count_direction(self, row: int, col: int, d_row: int, d_col: int) -> int:
        player = int(self.board[row, col])
        if player == 0:
            return 0
        count = 1
        current_row = row + d_row
        current_col = col + d_col
        while self._in_bounds(current_row, current_col) and int(self.board[current_row, current_col]) == player:
            count += 1
            current_row += d_row
            current_col += d_col
        return count

    def _is_start_of_run(self, row: int, col: int, d_row: int, d_col: int, player: int) -> bool:
        back_row = row - d_row
        back_col = col - d_col
        return not (self._in_bounds(back_row, back_col) and int(self.board[back_row, back_col]) == player)
    
    def get_representation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().get_representation()
    
    def _set_state_representation(self):
        flattened = self.board.flatten()
        
        player_1_pieces = np.where(flattened == 1, 1.0, 0.0).astype(dtype=np.float32)
        player_2_pieces = np.where(flattened == 2, 1.0, 0.0).astype(dtype=np.float32)
        
        total_cells = float(self.board.size)
        empty_fraction = float(np.sum(flattened == 0)) / total_cells
        empty = np.array([empty_fraction], dtype=np.float32)

        self.representation = (empty, player_1_pieces, player_2_pieces)
        
    def get_board(self) -> np.ndarray:
        return self.board    
        
    def get_win_index(self) -> int:
        return self.win_index
        
    def get_valid_actions(self, player_index: int | None = None) -> list[GameAction]:
        empty_positions = np.argwhere(self.board == 0)
        player_id = 0 if player_index is None else int(player_index)
        return [GameAction(player_id, (int(row), int(col))) for row, col in empty_positions]