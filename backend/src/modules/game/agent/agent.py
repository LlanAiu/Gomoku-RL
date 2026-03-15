# builtin
from pathlib import Path
        
# external

# internal
from ...rl.agent import Agent
from ...rl.optimization import OptimizationMethod, OneStepActorCritic

from .policy import GamePolicy
from .value_function import GameValueFunction

from ..elements import GameState, GameAction


class GameAgent(Agent):
    def __init__(self, player_index: int, weights_path: str):
        super().__init__()
        self.player_index = player_index
        self._policy = GamePolicy(
            player_index=player_index
        )
        self._value_function = GameValueFunction(
            player_index=player_index
        )
        
        self._optimization_method = OneStepActorCritic(
            self._policy,
            self._value_function,
            0.99, 0.02, 0.02
        )
    
        self.load_parameters(weights_path)
    
    @property
    def policy(self) -> GamePolicy:
        return self._policy
    
    @property
    def value_function(self) -> GameValueFunction:
        return self._value_function
    
    @property
    def optimization_method(self) -> OptimizationMethod:
        return self._optimization_method
    
    def get_player_index(self) -> int:
        return self.player_index
    
    def decide_train(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)
    
    def decide_inference(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)
    
    def save_parameters(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)


        self.policy.save_parameters(p)

        if self.value_function is not None:
            self.value_function.save_parameters(p)

    def load_parameters(self, path: str):
        p = Path(path)
        print(f"Loading parameters from: {p}")
        try:
            self.policy.load_parameters(p)
        except FileNotFoundError:
            pass

        if self.value_function is not None:
            try:
                self.value_function.load_parameters(p)
            except FileNotFoundError:
                pass