# builtin

# external
import numpy as np

# internal
from ...rl.environment import EpisodicRLEnvironment
from ..elements import GameRewardSignal, GameState, GameAction


class GameEnvironment(EpisodicRLEnvironment):
    def __init__(self):
        super().__init__()
          
    def _setup_reward_signal(self):
        self.reward_signal: GameRewardSignal = GameRewardSignal()
    
    def reset(self) -> GameState:
        self.history: list[GameState] = []
        self.current_state: GameState = GameState()
        self.history.append(self.current_state)
        
        return self.current_state
    
    def step(self, action: GameAction) -> tuple[GameState, float]:
        new_board: np.ndarray = self.current_state.get_board().copy()
        
        row, col = action.get_move()

        if not self._move_in_bounds(row, col, new_board):
            print("[WARN][ENV] Selected out-of-bounds move")
            return (self.current_state, 0.0)

        if new_board[row, col] != 0:
            print("[WARN][ENV] Selected invalid move, action masking is likely faulty?")
            return (self.current_state, 0.0)

        new_board[row, col] = action.get_player_index()
        
        new_state = GameState(new_board)
        reward = self.reward_signal.get_reward(self.current_state, new_state, action)
        
        self.history.append(new_state)
        self.current_state = new_state
        
        return (new_state, reward)
    
    def _move_in_bounds(self, row: int, col: int, board: np.ndarray) -> bool:
        rows, cols = board.shape
        
        return 0 <= row < rows and 0 <= col < cols