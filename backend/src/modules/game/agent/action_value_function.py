# builtin

# external
import numpy as np
from pathlib import Path

# internal
from ...rl.agent.action_value_function import ActionValueFunction
from ..elements import GameState, GameAction
from ..constants import FEATURE_IN_DIM, POLICY_OUT_DIM


class GameQFunction(ActionValueFunction):
    def __init__(self, player_index: int):
        super().__init__()
        self._weights = np.random.normal(0.0, 0.05, (FEATURE_IN_DIM, 1)).astype(np.float32)
        self._player_index = player_index
        
    def set_player_index(self, player_index: int):
        self._player_index = player_index

    def evaluate(self, state: GameState, action: GameAction) -> float:
        if state.is_terminal():
            return 0.0

        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        index = action.get_flattened_index()
        if self._player_index == 1:
            player_1_copy[index] = 1.0
        else:
            player_2_copy[index] = 1.0
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        value = self._weights.T @ embedding

        return float(value[0])

    def _get_player_representation(self, empty: np.ndarray, player_1: np.ndarray, player_2: np.ndarray) -> np.ndarray:
        if self._player_index == 1:
            return np.concatenate([empty, player_1, player_2])
        else:
            return np.concatenate([empty, player_2, player_1])
        
    def evaluate_all_actions(self, state: GameState) -> np.ndarray:
        if state.is_terminal():
            return np.full(POLICY_OUT_DIM, -np.inf, dtype=np.float32)

        prefs_masked = np.full(POLICY_OUT_DIM, -np.inf, dtype=np.float32)

        valid_actions = state.get_valid_actions(self._player_index)
        for a in valid_actions:
            index = a.get_flattened_index()
            prefs_masked[index] = self.evaluate(state, a)

        return prefs_masked

    def update(self, update: np.ndarray):
        self._weights += update

    def get_gradient(self, state: GameState, action: GameAction) -> np.ndarray:
        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        index = action.get_flattened_index()
        if self._player_index == 1:
            player_1_copy[index] = 1.0
        else:
            player_2_copy[index] = 1.0
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        
        return embedding.reshape(-1, 1)

    def save_parameters(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "q_weights.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self._weights)

    def load_parameters(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "q_weights.npy"
        if not p.exists():
            raise FileNotFoundError(f"Q weights file not found: {p}")
        self._weights = np.load(p).astype(np.float32)
        print(f"Successfully loaded Q weights from {p}")
