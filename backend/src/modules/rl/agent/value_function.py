# builtin
from abc import ABC, abstractmethod

# external
import numpy as np

# internal
from ..elements import State


class ValueFunction(ABC):
    
    @abstractmethod
    def evaluate_state(self, state: State) -> float:
        pass
    
    @abstractmethod
    def update(self, update: np.ndarray):
        pass
    
    @abstractmethod
    def get_gradient(self, state: State) -> np.ndarray:
        pass
    
    @abstractmethod
    def save_parameters(self, path: str):
        pass
    
    @abstractmethod
    def load_parameters(self, path: str):
        pass