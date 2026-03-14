# builtin
from pathlib import Path
        
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
        self.player_index = player_index
        self._policy = GamePolicy(
            player_index=player_index
        )
        self._value_function = GameValueFunction(
            player_index=player_index
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
        return None
    
    def get_player_index(self) -> int:
        return self.player_index
    
    def decide_train(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)
    
    def decide_inference(self, state: GameState) -> GameAction:
        return self._policy.choose_action(state)
    
    def save_parameters(self, weights_path: str):
        p = Path(weights_path)
        p.mkdir(parents=True, exist_ok=True)

        policy_path = p / "policy_weights.npy"
        self.policy.save_parameters(policy_path)

        if self.value_function is not None:
            value_path = p / "value_weights.npy"
            self.value_function.save_parameters(value_path)

    def load_parameters(self, weights_path: str):
        p = Path(weights_path)
        policy_path = p / "policy_weights.npy"
        if policy_path.exists():
            self.policy.load_parameters(policy_path)
        else:
            raise FileNotFoundError(f"Policy weights not found: {policy_path}")

        if self.value_function is not None:
            value_path = p / "value_weights.npy"
            if value_path.exists():
                self.value_function.load_parameters(value_path)
            else:
                raise FileNotFoundError(f"Value weights not found: {value_path}")