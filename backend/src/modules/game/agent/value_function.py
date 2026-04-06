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
        # TODO: Update FEATURE_IN_DIM if needed
        self._weights = np.random.normal(0.0, 0.01, (FEATURE_IN_DIM, 1)).astype(np.float32)
        self._player_index = player_index
        
    def set_player_index(self, player_index: int):
        self._player_index = player_index
    
    # TODO: map state --> state value
    def evaluate_state(self, state: GameState) -> float:
        if state.is_terminal():
            return 0.0
        
        raise NotImplementedError("TODO")
    
    def update(self, update: np.ndarray):
        self._weights += update
    
    # TODO: what is the gradient of this function w.r.t. the weights?  
    def get_gradient(self, state: GameState) -> np.ndarray:
        raise NotImplementedError("TODO")

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