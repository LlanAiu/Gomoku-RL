# builtin
from abc import ABC, abstractmethod

# external
import numpy as np

# internal
from ..elements import State, Action


class QFunction(ABC):
    @abstractmethod
    def evaluate(self, state: State, action: Action) -> float:
        pass

    @abstractmethod
    def evaluate_all_actions(self, state: State) -> np.ndarray:
        """Return an array of action-values for all output actions.

        Implementations may place -inf for invalid actions so that callers
        can safely take max over the returned array.
        """
        pass

    @abstractmethod
    def update(self, update: np.ndarray):
        pass

    @abstractmethod
    def get_gradient(self, state: State, action: Action) -> np.ndarray:
        pass

    @abstractmethod
    def save_parameters(self, path: str) -> None:
        pass

    @abstractmethod
    def load_parameters(self, path: str) -> None:
        pass
