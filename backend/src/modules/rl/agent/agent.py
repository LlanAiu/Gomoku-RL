# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action


class Agent(ABC):
    
    @property
    @abstractmethod
    def policy(self):
        pass
    
    @property
    @abstractmethod
    def value_function(self):
        pass
    
    @property
    @abstractmethod
    def optimization_method(self):
        pass
    
    @abstractmethod
    def decide_train(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def decide_inference(self, state: State) -> Action:
        pass
    
    @abstractmethod
    def improve(self, old_state: State, new_state: State, reward: float):
        pass