# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State


class ValueFunction(ABC):
    
    @abstractmethod
    def evaluate_state(self, state: State) -> float:
        pass
    
    @abstractmethod
    def save_parameters(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load_parameters(self, path: str) -> None:
        pass