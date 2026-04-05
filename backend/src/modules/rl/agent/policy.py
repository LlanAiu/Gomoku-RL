# builtin
from abc import ABC, abstractmethod

# external
import numpy as np

# internal
from ..elements import State, Action

class Policy(ABC):
    
    @abstractmethod
    def choose_action(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def update(self, update: np.ndarray):
        pass
    
    @abstractmethod
    def get_eligibility(self, state: State, action: Action) -> np.ndarray:
        pass
 
    @abstractmethod 
    def save_parameters(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load_parameters(self, path: str) -> None:
        pass