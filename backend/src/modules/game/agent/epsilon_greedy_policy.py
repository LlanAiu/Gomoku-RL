# builtin

# external
import numpy as np

# internal
from ...rl.agent import ActionValuePolicy, ActionValueFunction
from ..elements import GameState, GameAction
from ..constants import BOARD_SIZE


class GameEpsilonGreedyPolicy(ActionValuePolicy):
    def __init__(
        self, 
        player_index: int, 
        q_function: ActionValueFunction, 
        epsilon: float = 0.05
    ):
        super().__init__()
        self.player_index: int = player_index
        self.q_function: ActionValueFunction = q_function
        self.epsilon: float = float(epsilon)
        
    @property
    def q_function(self) -> ActionValueFunction:
        return self.q_function

    def choose_action(self, state: GameState) -> GameAction:
        valid_actions = state.get_valid_actions(self.player_index)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for selection")

        if np.random.random() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]

        q_vals = self.q_function.evaluate_all_actions(state)
        
        best_idx = int(np.nanargmax(q_vals))
        row = best_idx // BOARD_SIZE
        col = best_idx % BOARD_SIZE
        
        return GameAction(self.player_index, (int(row), int(col)))

    def update(self, _: np.ndarray):
        return

    def save_parameters(self, _: str):
        return

    def load_parameters(self, _: str):
        return
