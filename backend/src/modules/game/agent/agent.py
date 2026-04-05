# builtin
from pathlib import Path
        
# external

# internal
from ...rl.agent import Agent
from ...rl.optimization import OptimizationMethod, OneStepActorCritic, OneStepTDActionValue

from .policy import GamePolicy
from .value_function import GameValueFunction
from .game_q_function import GameQFunction

from ..elements import GameState, GameAction


class GameAgent(Agent):
    def __init__(self, player_index: int, weights_path: str, mode: str = "policy"):
        """mode: 'policy' (policy-gradient/actor-critic) or 'action_value' (Q-learning/SARSA)

        When 'action_value' the agent will create a `GameQFunction` and use
        `OneStepTDActionValue` as the optimization method. Policy remains a
        `GamePolicy` (used for behavior/exploration).
        """
        super().__init__()
        self.player_index = player_index
        self.mode = mode

        # construct behavior policy depending on mode
        if mode == "action_value":
            # create q-function first, then an epsilon-greedy behavior policy
            self._q_function = GameQFunction(player_index=player_index)
            from .game_epsilon_policy import GameEpsilonGreedyPolicy

            self._policy = GameEpsilonGreedyPolicy(
                player_index=player_index, q_function=self._q_function, epsilon=0.1
            )
            self._value_function = None
            # use off-policy Q-learning by default
            self._optimization_method = OneStepTDActionValue(
                self._policy, self._q_function, 0.99, 0.02, off_policy=True
            )
        else:
            # default: policy-gradient / actor-critic
            self._policy = GamePolicy(player_index=player_index)
            self._value_function = GameValueFunction(player_index=player_index)
            self._q_function = None
            self._optimization_method = OneStepActorCritic(
                self._policy, self._value_function, 0.99, 0.02, 0.02
            )

        self.load_parameters(weights_path)
    
    @property
    def policy(self) -> GamePolicy:
        return self._policy
    
    @property
    def value_function(self) -> GameValueFunction:
        return self._value_function

    @property
    def q_function(self):
        return getattr(self, "_q_function", None)
    
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