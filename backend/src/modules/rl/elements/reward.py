# builtin
from abc import ABC, abstractmethod

# external

# internal
from .state import State
from .action import Action


class RewardSignal(ABC):
    
    @abstractmethod
    def get_reward(state: State, action: Action) -> float:
        pass