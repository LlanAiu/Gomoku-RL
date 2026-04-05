# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action
from ..agent import Policy
from ..agent import ValueFunction


class OptimizationMethod(ABC):
    
    @property
    @abstractmethod
    def policy(self) -> Policy:
        pass
    
    @property
    @abstractmethod
    def value_function(self) -> ValueFunction | None:
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        pass