# builtin

# external

# internal
from ...rl.agent import Agent
from ...rl.optimization import OptimizationMethod

from .policy import GamePolicy
from .value_function import GameValueFunction

from ..elements import GameState, GameAction


class GameAgent(Agent):
    
    def __init__(self, player_index: int, weights_path: str):
        super().__init__()
        self._policy = GamePolicy(
            player_index=player_index
        )
        self._value_function = GameValueFunction(
            player_index=player_index
        )
    
    @property
    def policy(self) -> GamePolicy:
        return self._policy
    
    @property
    def value_function(self) -> GameValueFunction:
        return self._value_function
    
    @property
    def optimization_method(self) -> OptimizationMethod:
        return None
    
    def decide_train(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)
    
    def decide_inference(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)