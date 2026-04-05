# builtin

# external
import numpy as np

# internal
from ...rl.agent import Policy
from ..elements import GameState, GameAction
from ..constants import BOARD_SIZE, POLICY_OUT_DIM, FEATURE_IN_DIM


class GameEpsilonGreedyPolicy(Policy):
    def __init__(self, player_index: int, q_function, epsilon: float = 0.05):
        self.player_index = player_index
        self.q_function = q_function
        self.epsilon = float(epsilon)

    def choose_action(self, state: GameState) -> GameAction:
        valid_actions = state.get_valid_actions(self.player_index)
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available for selection")

        # epsilon decide
        if np.random.random() < self.epsilon:
            # random valid action
            return valid_actions[np.random.randint(len(valid_actions))]

        # greedy selection using q_function
        q_vals = self.q_function.evaluate_all_actions(state)
        # q_vals uses -inf for invalid actions
        best_idx = int(np.nanargmax(q_vals))
        row = best_idx // BOARD_SIZE
        col = best_idx % BOARD_SIZE
        return GameAction(self.player_index, (int(row), int(col)))

    def update(self, update: np.ndarray):
        # epsilon-greedy has no trainable parameters in this implementation
        return None

    def get_eligibility(self, state: GameState, action: GameAction) -> np.ndarray:
        # No trainable parameters: return zero-shaped gradient matching common policy shapes
        return np.zeros((FEATURE_IN_DIM, POLICY_OUT_DIM), dtype=np.float32)

    def save_parameters(self, path: str) -> None:
        # nothing to save
        return None

    def load_parameters(self, path: str) -> None:
        # nothing to load
        return None
