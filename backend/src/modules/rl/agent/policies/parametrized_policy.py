# builtin
from abc import abstractmethod

# external
import numpy as np

# internal
from ...elements import State, Action
from .policy import Policy


class ParametrizedPolicy(Policy):
    
    @abstractmethod
    def get_eligibility(self, state: State, action: Action) -> np.ndarray:
        pass