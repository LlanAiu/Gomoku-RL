# builtin
from pathlib import Path
from typing import Literal
        
# external

# internal
from ...rl.agent import Agent
from ...rl.optimization import OptimizationMethod, OneStepActorCritic, OneStepTDActionValue

from .epsilon_greedy_policy import GameEpsilonGreedyPolicy
from .parametrized_policy import GameParametrizedPolicy
from .value_function import GameValueFunction
from .action_value_function import GameQFunction

from ..elements import GameState, GameAction


class GameAgent(Agent):
    def __init__(
        self, 
        player_index: int, 
        weights_path: str, 
        mode: Literal["policy", "action_value"]
    ):
        super().__init__()
        self._player_index: int = player_index
        self._mode: Literal["policy", "action_value"] = mode

        if mode == "action_value":
            self._q_function = GameQFunction(player_index=player_index)
            self._policy = GameEpsilonGreedyPolicy(
                player_index=player_index, 
                q_function=self._q_function, 
                epsilon=0.05
            )
            self._optimization_method = OneStepTDActionValue(
                policy=self._policy, 
                q_function=self._q_function, 
                discount=1.0, 
                step_size=0.005
            )
            self._value_function = None
        else:
            self._policy = GameParametrizedPolicy(player_index=player_index)
            self._value_function = GameValueFunction(player_index=player_index)
            self._optimization_method = OneStepActorCritic(
                policy=self._policy, 
                value_function=self._value_function, 
                discount=1.0, 
                policy_step_size=0.02, 
                value_step_size=0.02
            )
            self._q_function = None

        self.load_parameters(weights_path)
    
    @property
    def policy(self) -> GameParametrizedPolicy | GameEpsilonGreedyPolicy:
        return self._policy
    
    @property
    def value_function(self) -> GameValueFunction | None:
        return self._value_function

    @property
    def q_function(self) -> GameQFunction | None:
        return self._q_function
    
    @property
    def optimization_method(self) -> OptimizationMethod:
        return self._optimization_method
    
    def set_player_index(self, player_index: int):
        self._player_index = player_index
        self._policy.set_player_index(player_index)
        
        if self._value_function is not None:
            self._value_function.set_player_index(player_index)
            
        if self._q_function is not None:
            self._q_function.set_player_index(player_index)
    
    def get_player_index(self) -> int:
        return self._player_index
    
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

        if self.q_function is not None:
            self.q_function.save_parameters(p)

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

        if self.q_function is not None:
            try:
                self.q_function.load_parameters(p)
            except FileNotFoundError:
                pass