# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action
from .policy import Policy
from .value_function import ValueFunction
from .q_function import QFunction
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
    def q_function(self) -> QFunction | None:
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
    def save_parameters(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load_parameters(self, path: str) -> None:
        pass
    
    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        # forward to optimization method and return any diagnostics for logging
        return self.optimization_method.improve(old_state, action, new_state, reward)