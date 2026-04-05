# builtin
from abc import ABC, abstractmethod
import os

# external
from tqdm import tqdm

# internal
from ...log import Logger
from ..environment import EpisodicRLEnvironment
from ..agent import Agent


class EpisodicTrainer(ABC):
    def __init__(self, save_path: str):
        self._environment: EpisodicRLEnvironment | None = None
        self._agent: Agent | None = None
        
        self._save_path = save_path
        self._logger = Logger.get_instance(save_dir=os.path.join(self._save_path, "logs"))
        
        self._setup_environment()
        self._setup_agent()
    
    @abstractmethod
    def _setup_environment(self):
        pass
    
    @abstractmethod
    def _setup_agent(self):
        pass
    
    @abstractmethod
    def _before_episode(self):
        pass
    
    @abstractmethod
    def _after_step(self):
        pass
    
    @abstractmethod
    def _after_episode(self):
        pass
    
    def run_train_episode(self):
        if self._environment is None or self._agent is None:
            raise RuntimeError("Cannot train when environment/agent/method is not set!")
        
        state = self._environment.reset()
        self._agent.reset()
        step_count = 0
        
        self._before_episode()
        
        while not state.is_terminal():
            action = self._agent.decide_train(state)
        
            new_state, reward = self._environment.step(action)

            diagnostics = self._agent.improve(state, action, new_state, reward)
            self._logger.log_dict(diagnostics, episode=self.train_episode, timestep=step_count)
        
            state = new_state
            step_count += 1
            
            self._after_step()
        
        self._logger.log_scalar("episode_length", step_count, episode=self.train_episode)
        
        self._after_episode()
    
    def train_multiple(self, num_episodes: int):
        self.train_episode = 1
        
        for _ in tqdm(range(num_episodes)):
            self.run_train_episode()
            self.train_episode += 1
            
        self.save_results()
        
    def save_results(self):
        self._agent.save_parameters(self._save_path)
        
        self._logger.save_csv()
        self._logger.plot(rolling_window=5)