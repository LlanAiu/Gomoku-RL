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
        
        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        preferences = self._weights.T @ embedding
        
        valid_actions = state.get_valid_actions(self._player_index)
        action_mask = self._get_action_mask(valid_actions)
        
        return self._select_softmax(preferences, action_mask)
    
    def _get_player_representation(self, empty: np.ndarray, player_1: np.ndarray, player_2: np.ndarray) -> np.ndarray:
        if self._player_index == 1:
            return np.concatenate([empty, player_1, player_2])
        else:
            return np.concatenate([empty, player_2, player_1])
        
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
        empty, player_1, player_2 = state.get_representation()
        empty_copy, player_1_copy, player_2_copy = empty.copy(), player_1.copy(), player_2.copy()
        
        embedding = self._get_player_representation(empty_copy, player_1_copy, player_2_copy)
        
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
        index = action.get_flattened_index()
        one_hot[index] = 1.0
        
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
        