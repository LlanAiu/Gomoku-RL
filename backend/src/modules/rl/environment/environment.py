# builtin
from abc import ABC, abstractmethod

# external

# internal
from ..elements import State, Action


class EpisodicRLEnvironment(ABC):
    def __init__(self):
        self.history: list[State] = []
        self.current_state: State | None = None
        
        self._setup_reward_signal()
        self.reset()
    
    def is_episode_over(self) -> bool:
        if self.current_state is None: 
            return False
        
        return self.current_state.is_terminal()
    
    @abstractmethod
    def _setup_reward_signal(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action: Action) -> tuple[State, float]:
        pass