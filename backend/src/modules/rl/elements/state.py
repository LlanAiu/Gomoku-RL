# builtin
from abc import ABC, abstractmethod

# external
import numpy as np

# internal
from .action import Action

class State(ABC):
    def __init__(self):
        self.representation = np.empty(0)
        self.terminal: bool = False
        self._set_state_representation()
    
    def is_terminal(self) -> bool:
        return self.terminal
    
    def get_representation(self) -> np.ndarray:
        return self.representation
    
    @abstractmethod
    def _set_state_representation(self):
        pass    
    
    @abstractmethod
    def get_valid_actions(self) -> list[Action]:
        pass
    