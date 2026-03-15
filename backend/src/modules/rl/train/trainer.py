# builtin
from abc import ABC, abstractmethod

# external
from tqdm import tqdm

# internal
from ..environment import EpisodicRLEnvironment
from ..agent import Agent


class EpisodicTrainer(ABC):
    def __init__(self):
        self.environment: EpisodicRLEnvironment | None = None
        self.agent: Agent | None = None
        
        self._setup_environment()
        self._setup_agent()
    
    @abstractmethod
    def _setup_environment(self):
        pass
    
    @abstractmethod
    def _setup_agent(self):
        pass
    
    def run_train_episode(self):
        if self.environment is None or self.agent is None:
            raise RuntimeError("Cannot train when environment/agent/method is not set!")

        state = self.environment.reset()
        self.agent.optimization_method.reset()
        
        while not state.is_terminal():
            action = self.agent.decide_train(state)
        
            new_state, reward = self.environment.step(action)

            self.agent.improve(state, action, new_state, reward)
        
            state = new_state
    
    def train_multiple(self, num_episodes: int):
        for _ in tqdm(range(num_episodes)):
            self.run_train_episode()