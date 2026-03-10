# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action
from .policy import Policy
from .value_function import ValueFunction
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
    def optimization_method(self) -> OptimizationMethod:
        pass
    
    @abstractmethod
    def decide_train(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def decide_inference(self, state: State) -> Action:
        pass
    
    def improve(self, old_state: State, new_state: State, reward: float):
        self.optimization_method.improve_policy(old_state, new_state, reward)