# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State
from .policy import Policy
from .value_function import ValueFunction


class OptimizationMethod(ABC):
    
    @property
    @abstractmethod
    def policy() -> Policy:
        pass
    
    @property
    @abstractmethod
    def value_function() -> ValueFunction | None:
        pass
    
    @abstractmethod
    def improve_policy(self, old_state: State, new_state: State, reward: float):
        pass