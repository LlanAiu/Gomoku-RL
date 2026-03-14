# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State


class ValueFunction(ABC):
    
    @abstractmethod
    def evaluate_state(self, state: State) -> float:
        pass