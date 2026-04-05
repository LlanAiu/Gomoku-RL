# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action
from ..agent import Policy


class OptimizationMethod(ABC):
    def __init__(self, discount: float):
        self.discount: float = discount
    
    @property
    def discount(self) -> float:
        return self.discount
    
    @property
    @abstractmethod
    def policy(self) -> Policy:
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        pass