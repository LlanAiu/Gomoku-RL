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
        
    # TODO: I've left the tail of this function for you
    # feel free to change/remove it as you see fit
    # but they may be helpful in terms of building this out
    def choose_action(self, state: GameState) -> GameAction:
        
        raise NotImplementedError("TODO")
        
        valid_actions = state.get_valid_actions(self._player_index)
        action_mask = self._get_action_mask(valid_actions)
        
        return self._select_softmax(preferences, action_mask)

    # TODO: return a vector marking the valid actions with a 1.0 and invalid ones with 0.0
    def _get_action_mask(self, valid_actions: list[GameAction]) -> np.ndarray:
        raise NotImplementedError("TODO")
    
    # TODO: convert preferences + mask of valid actions --> best action to select
    # BTW -- np.random.choice allows for a sample from a discrete distribution of probabilities
    def _select_softmax(self, preferences: np.ndarray, action_mask: np.ndarray) -> GameAction:
        raise NotImplementedError("TODO")

        idx = int(np.random.choice(len(probs), p=probs))

        row = idx // BOARD_SIZE
        col = idx % BOARD_SIZE

        return GameAction(self._player_index, (int(row), int(col)))
    
    def update(self, update: np.ndarray):
        self._weights += update
        
    # TODO: Remember this is the gradient of ln(policy function)
    # There is more information about this in reference.md that you may find helpful
    def get_eligibility(self, state: GameState, action: GameAction):
        raise NotImplementedError("TODO")

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
        