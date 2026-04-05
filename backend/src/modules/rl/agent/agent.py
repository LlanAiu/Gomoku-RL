# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action
from .policies.policy import Policy
from .value_function import ValueFunction
from .action_value_function import ActionValueFunction
from ..optimization import OptimizationMethod


class Agent(ABC):
    
    @property
    @abstractmethod
    def policy(self) -> Policy:
        pass
    
    @property
    @abstractmethod
    def value_function(self) -> ValueFunction | None:
        pass

    @property
    @abstractmethod
    def q_function(self) -> ActionValueFunction | None:
        pass
    
    @property
    @abstractmethod
    def optimization_method(self) -> OptimizationMethod:
        pass
    
    @abstractmethod
    def decide_train(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def decide_inference(self, state: State) -> Action:
        pass
    
    @abstractmethod 
    def save_parameters(self, path: str):
        pass
    
    @abstractmethod
    def load_parameters(self, path: str):
        pass
    
    def reset(self):
        self.optimization_method.reset()
    
    def improve(self, old_state: State, action: Action, new_state: State, reward: float) -> dict:
        return self.optimization_method.improve(old_state, action, new_state, reward)