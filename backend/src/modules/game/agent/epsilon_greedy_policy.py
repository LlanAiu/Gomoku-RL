# builtin

# external
import numpy as np

# internal
from ...rl.agent import ActionValuePolicy
from .action_value_function import GameQFunction
from ..elements import GameState, GameAction
from ..constants import BOARD_SIZE


class GameEpsilonGreedyPolicy(ActionValuePolicy):
    def __init__(
        self, 
        player_index: int, 
        q_function: GameQFunction, 
        epsilon: float = 0.05,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
    ):
        super().__init__()
        self._player_index: int = player_index
        self._q_function: GameQFunction = q_function
        self._epsilon: float = float(epsilon)
        self._epsilon_min: float = float(epsilon_min)
        self._epsilon_decay: float = float(epsilon_decay)
        
    @property
    def q_function(self) -> GameQFunction:
        return self._q_function
    
    def set_player_index(self, player_index: int):
        self._player_index = player_index

    # TODO: How to decide between greedy action vs explore action?
    def choose_action(self, state: GameState) -> GameAction:
        raise NotImplementedError("TODO")

    # TODO: same as above, but only greedy
    def choose_action_inference(self, state: GameState) -> GameAction:
        raise NotImplementedError("TODO")

    # TODO: decay epsilon by the decay rate, but only up until a minimum value
    def after_step(self):
        raise NotImplementedError("TODO")
        return super().after_step()

    def save_parameters(self, _: str):
        return

    def load_parameters(self, _: str):
        return
