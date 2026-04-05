# builtin

# external
import numpy as np
from pathlib import Path


# internal
from ...rl.agent import ParametrizedPolicy
from ..constants import FEATURE_IN_DIM, POLICY_OUT_DIM, BOARD_SIZE
from ..elements import GameState, GameAction


class GameParametrizedPolicy(ParametrizedPolicy):
    def __init__(self, player_index: int):
        super().__init__()
        self._weights = np.random.normal(0.0, 0.01, (FEATURE_IN_DIM, POLICY_OUT_DIM)).astype(np.float32)
        self._player_index = player_index
        
    def set_player_index(self, player_index: int):
        self._player_index = player_index
        
    def choose_action(self, state: GameState) -> GameAction:
        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)
        preferences = self._weights.T @ embedding
        
        valid_actions = state.get_valid_actions(self._player_index)
        action_mask = self._get_action_mask(valid_actions)
        
        return self._select_softmax(preferences, action_mask)
    
    def _swap_player_indices(self, state_embedding: np.ndarray) -> np.ndarray:
        embedding_copy: np.ndarray = state_embedding.copy().astype(np.float32)
        player = float(self._player_index)

        player_embedding: np.ndarray = np.where(
            embedding_copy == player, 1.0, 
            np.where(embedding_copy == 0.0, 0.0, -1.0)
        )
        
        return player_embedding
        
    def _get_action_mask(self, valid_actions: list[GameAction]) -> np.ndarray:
        mask = np.zeros(POLICY_OUT_DIM, dtype=np.float32)
        
        for action in valid_actions:
            row, col = action.get_move()
            idx = row * BOARD_SIZE + col
            if 0 <= idx < POLICY_OUT_DIM:
                mask[idx] = 1.0

        return mask
    
    def _select_softmax(self, preferences: np.ndarray, action_mask: np.ndarray) -> GameAction:
        masked_prefs = np.where(action_mask.astype(bool), preferences, -np.inf)

        if np.all(np.isneginf(masked_prefs)):
            raise ValueError("No valid actions available for selection")

        max_pref = np.max(masked_prefs[np.isfinite(masked_prefs)])
        exps = np.exp(masked_prefs - max_pref)
        exps[np.isneginf(masked_prefs)] = 0.0

        probs = exps / np.sum(exps)

        idx = int(np.random.choice(len(probs), p=probs))

        row = idx // BOARD_SIZE
        col = idx % BOARD_SIZE

        return GameAction(self._player_index, (int(row), int(col)))
    
    def update(self, update: np.ndarray):
        self._weights += update
        
    def get_eligibility(self, state: GameState, action: GameAction):
        embedding = state.get_representation()
        embedding = self._swap_player_indices(embedding)
        
        preferences = self._weights.T @ embedding
        
        valid_actions = state.get_valid_actions(self._player_index)
        action_mask = self._get_action_mask(valid_actions)
        
        masked_prefs = np.where(action_mask.astype(bool), preferences, -np.inf)

        if np.all(np.isneginf(masked_prefs)):
            raise ValueError("No valid actions available for selection")

        max_pref = np.max(masked_prefs[np.isfinite(masked_prefs)])
        exps = np.exp(masked_prefs - max_pref)
        exps[np.isneginf(masked_prefs)] = 0.0

        probs = exps / np.sum(exps)
        
        one_hot = np.zeros_like(probs)
        row, col = action.get_move()
        a_idx = row * BOARD_SIZE + col
        one_hot[a_idx] = 1.0
        
        grad_W = np.outer(embedding, (one_hot - probs))
        
        return grad_W

    def save_parameters(self, path: str):
        p = Path(path)
        if p.is_dir():
            p = p / "policy_weights.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, self._weights)

    def load_parameters(self, path: str):
        p = Path(path)
        if p.is_dir():
            p = p / "policy_weights.npy"
        if not p.exists():
            raise FileNotFoundError(f"Policy weights file not found: {p}")
        self._weights = np.load(p)
        print(f"Successfully loaded policy weights from {p}")
        