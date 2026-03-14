# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action

class Policy(ABC):
    
    @abstractmethod
    def choose_action(self, state: State) -> Action:
        pass

    @abstractmethod 
    def save_parameters(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load_parameters(self, path: str) -> None:
        pass