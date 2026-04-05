# builtin

# external
import numpy as np
from pathlib import Path


# internal
from ...rl.agent import ValueFunction
from ..elements import GameState
from ..constants import FEATURE_IN_DIM


class GameValueFunction(ValueFunction):
    def __init__(self, player_index: int):
        super().__init__()
        self._weights = np.random.normal(0.0, 0.01, (FEATURE_IN_DIM, 1)).astype(np.float32)
        self._player_index = player_index
        
    def set_player_index(self, player_index: int):
        self._player_index = player_index
    
    def evaluate_state(self, state: GameState) -> float:
        if state.is_terminal():
            return 0.0
        
        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        predicted_value = self._weights.T @ embedding
        
        return float(predicted_value[0])

    def _get_player_representation(self, empty: np.ndarray, player_1: np.ndarray, player_2: np.ndarray) -> np.ndarray:
        if self._player_index == 1:
            return np.concatenate([empty, player_1, player_2])
        else:
            return np.concatenate([empty, player_2, player_1])
    
    def update(self, update: np.ndarray):
        self._weights += update
        
    def get_gradient(self, state: GameState) -> np.ndarray:
        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        
        return embedding.reshape(-1, 1)

    def save_parameters(self, path: str):
        p = Path(path)
        if p.is_dir():
            p = p / "value_weights.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self._weights)

    def load_parameters(self, path: str):
        p = Path(path)
        if p.is_dir():
            p = p / "value_weights.npy"
        if not p.exists():
            raise FileNotFoundError(f"Value weights file not found: {p}")
        self._weights = np.load(p).astype(np.float32)
        print(f"Successfully loaded value weights from {p}")