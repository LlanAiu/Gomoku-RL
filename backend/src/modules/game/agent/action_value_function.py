# builtin

# external
import numpy as np
from pathlib import Path

# internal
from ...rl.agent.action_value_function import ActionValueFunction
from ..elements import GameState, GameAction
from ..constants import FEATURE_IN_DIM, POLICY_OUT_DIM, BOARD_SIZE


class GameQFunction(ActionValueFunction):
    def __init__(self, player_index: int):
        super().__init__()
        self._weights = np.random.uniform(0.05, 0.2, (FEATURE_IN_DIM, POLICY_OUT_DIM)).astype(np.float32)
        self._player_index = player_index

    def evaluate(self, state: GameState, action: GameAction) -> float:
        if state.is_terminal():
            return 0.0

        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)

        row, col = action.get_move()
        idx = row * BOARD_SIZE + col
        sim_embedding = embedding.copy()
        sim_embedding[idx] = 1.0

        prefs = self._weights.T @ sim_embedding

        return float(prefs[idx])

    def evaluate_all_actions(self, state: GameState) -> np.ndarray:
        if state.is_terminal():
            return np.full(POLICY_OUT_DIM, -np.inf, dtype=np.float32)

        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)

        prefs_masked = np.full(POLICY_OUT_DIM, -np.inf, dtype=np.float32)

        valid_actions = state.get_valid_actions(self._player_index)
        for a in valid_actions:
            r, c = a.get_move()
            idx = r * BOARD_SIZE + c
            sim_embedding = embedding.copy()
            sim_embedding[idx] = 1.0
            val = float((self._weights.T @ sim_embedding).flatten()[idx])
            prefs_masked[idx] = val

        return prefs_masked

    def update(self, update: np.ndarray):
        self._weights += update

    def get_gradient(self, state: GameState, action: GameAction) -> np.ndarray:
        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)

        row, col = action.get_move()
        a_idx = row * BOARD_SIZE + col
        sim_embedding = embedding.copy()
        sim_embedding[a_idx] = 1.0

        one_hot = np.zeros(POLICY_OUT_DIM, dtype=np.float32)
        one_hot[a_idx] = 1.0

        grad_W = np.outer(sim_embedding, one_hot)

        return grad_W

    def _swap_player_indices(self, state_embedding: np.ndarray) -> np.ndarray:
        embedding_copy: np.ndarray = state_embedding.copy().astype(np.float32)
        player = float(self._player_index)

        player_embedding: np.ndarray = np.where(
            embedding_copy == player, 1.0,
            np.where(embedding_copy == 0.0, 0.0, -1.0)
        )

        return player_embedding

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
