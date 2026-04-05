# builtin
from abc import abstractmethod

# external

# internal
from ..action_value_function import ActionValueFunction
from ...elements import State, Action
from .policy import Policy

class ActionValuePolicy(Policy):
    
    @property
    @abstractmethod
    def q_function(self) -> ActionValueFunction:
        pass
    
    @abstractmethod
    def choose_action_inference(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def after_step(self):
        pass