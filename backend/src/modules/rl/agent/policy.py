# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action

class Policy(ABC):
    
    @abstractmethod
    def evaluate_actions(state: State) -> list[tuple[Action, float]]:
        pass
    
    
    @abstractmethod
    def choose_action(state: State) -> Action:
        pass
    
    
    # TODO: figure out a GPI framework
    @abstractmethod
    def improve_policy():
        pass