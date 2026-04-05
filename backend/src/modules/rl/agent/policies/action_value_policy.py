# builtin
from abc import abstractmethod

# external

# internal
from ..action_value_function import ActionValueFunction
from .policy import Policy

class ActionValuePolicy(Policy):
    
    @property
    @abstractmethod
    def q_function(self) -> ActionValueFunction:
        pass