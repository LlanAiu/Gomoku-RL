# builtin
from abc import ABC, abstractmethod
import os

# external
from tqdm import tqdm

# internal
from ...log import Logger
from ..environment import EpisodicRLEnvironment
from ..elements import State, Action
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
    
    def run_train_episode(self):
        if self._environment is None or self._agent is None:
            raise RuntimeError("Cannot train when environment/agent/method is not set!")
        
        state = self._environment.reset()
        self._agent.reset()
        step_count = 0
        episode_reward = 0
        
        self._before_episode()
        
        while not state.is_terminal():
            action = self._agent.decide_train(state)
        
            new_state, reward = self._environment.step(action)

            self._update_train(
                step=step_count,
                state=state,
                action=action,
                new_state=new_state,
                reward=reward
            )
        
            state = new_state
            step_count += 1
            episode_reward += reward
            
            self._after_step()
        
        self._logger.log_scalar("episode_reward", episode_reward, episode=self.train_episode)
        self._logger.log_scalar("episode_length", step_count, episode=self.train_episode)
        
        self._after_episode()
    
    def _before_episode(self):
        return
    
    def _update_train(
        self, 
        step: int, 
        state: State, 
        action: Action, 
        new_state: State, 
        reward: float
    ):
        diagnostics = self._agent.improve(state, action, new_state, reward)
        self._logger.log_dict(diagnostics, episode=self.train_episode, timestep=step)

    def _after_step(self):
        return
    
    def _after_episode(self):
        return
    
    def train_multiple(self, num_episodes: int):
        self.train_episode = 1
        
        for _ in tqdm(range(num_episodes)):
            self.run_train_episode()
            self.train_episode += 1
            
        self.save_results()
        
    def save_results(self):
        self._agent.save_parameters(f"{self._save_path}/parameters")
        
        self._logger.save_csv()
        self._logger.plot(rolling_window=5)