# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State


class ValueFunction(ABC):
    
    @abstractmethod
    def evaluate_state(state: State) -> float:
        pass