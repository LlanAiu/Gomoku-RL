# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action

class Policy(ABC):
    
    @abstractmethod
    def choose_action(self, state: State) -> Action:
        pass