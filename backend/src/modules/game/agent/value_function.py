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
        self.weights = np.random.random_sample((FEATURE_IN_DIM, 1))
        self.player_index = player_index
    
    def evaluate_state(self, state: GameState) -> float:
        if state.is_terminal():
            return 0.0
        
        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)
        
        predicted_value = self.weights.T @ embedding
        
        return predicted_value

    def _swap_player_indices(self, state_embedding: np.ndarray) -> np.ndarray:
        embedding_copy: np.ndarray = state_embedding.copy().astype(np.float32)
        player = float(self.player_index)

        player_embedding: np.ndarray = np.where(
            embedding_copy == player, 1.0, 
            np.where(embedding_copy == 0.0, 0.0, -1.0)
        )
        
        return player_embedding
    
    def update(self, update: np.ndarray):
        self.weights += update
        
    def get_gradient(self, state: GameState) -> np.ndarray:
        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)
        
        return embedding.reshape(-1, 1)

    def save_parameters(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "value_weights.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self.weights)

    def load_parameters(self, path: str) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "value_weights.npy"
        if not p.exists():
            raise FileNotFoundError(f"Value weights file not found: {p}")
        self.weights = np.load(p)