# builtin

# external
import numpy as np

# internal
from ...rl.agent import ActionValuePolicy
from .action_value_function import GameQFunction
from ..elements import GameState, GameAction
from ..constants import BOARD_SIZE


class GameEpsilonGreedyPolicy(ActionValuePolicy):
    def __init__(
        self, 
        player_index: int, 
        q_function: GameQFunction, 
        epsilon: float = 0.05
    ):
        super().__init__()
        self._player_index: int = player_index
        self._q_function: GameQFunction = q_function
        self._epsilon: float = float(epsilon)
        
    @property
    def q_function(self) -> GameQFunction:
        return self._q_function
    
    def set_player_index(self, player_index: int):
        self._player_index = player_index

    def choose_action(self, state: GameState) -> GameAction:
        valid_actions = state.get_valid_actions(self._player_index)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for selection")

        if np.random.random() < self._epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]

        q_vals = self.q_function.evaluate_all_actions(state)
        
        best_idx = int(np.nanargmax(q_vals))
        row = best_idx // BOARD_SIZE
        col = best_idx % BOARD_SIZE
        
        return GameAction(self._player_index, (int(row), int(col)))

    def update(self, _: np.ndarray):
        return

    def save_parameters(self, _: str):
        return

    def load_parameters(self, _: str):
        return
