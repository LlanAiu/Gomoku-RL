# builtin
from abc import ABC, abstractmethod
from typing import Any

# external

# internal
from .action import Action

class State(ABC):
    def __init__(self):
        self.representation: Any = None
        self.terminal: bool = False
        self._set_state_representation()
    
    def is_terminal(self) -> bool:
        return self.terminal
    
    def get_representation(self) -> Any:
        return self.representation
    
    @abstractmethod
    def _set_state_representation(self):
        pass    
    
    @abstractmethod
    def get_valid_actions(self) -> list[Action]:
        pass
    