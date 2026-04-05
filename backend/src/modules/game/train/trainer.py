# builtin
import os
from typing import Literal

# external

# internal
from ...rl.train import EpisodicTrainer
from ..environment import GameEnvironment
from ..agent import GameAgent
from ...log.logger import Logger

class GameTrainer(EpisodicTrainer):
    def __init__(self, save_path: str, mode: Literal["policy", "action_value"]):
        self.mode: Literal["policy", "action_value"] = mode
        super().__init__(save_path)
        
        self.logger = Logger.get_instance(save_dir=os.path.join(self.save_path, "logs"))

    def _setup_environment(self):
        self.environment = GameEnvironment()

    def _setup_agent(self):
        self.agent: GameAgent = GameAgent(1, "test_train/parameters", self.mode)
        
    def run_train_episode(self):
        if self.environment is None or self.agent is None:
            raise RuntimeError("Cannot train when environment/agent/method is not set!")

        current_player = 1
        state = self.environment.reset()
        self.agent.reset()
        self.agent.set_player_index(current_player)
        
        agent_rewards = {1: 0.0, 2: 0.0}
        step_count = 0
        
        while not state.is_terminal():
            action = self.agent.decide_train(state)
        
            new_state, reward = self.environment.step(action)

            diagnostics = self.agent.improve(state, action, new_state, reward)

            try:
                pid = self.agent.get_player_index()
                self.logger.log_scalar(f"player_{pid}/step_reward", float(reward),
                                       episode=self.train_episode, timestep=step_count)
            except Exception:
                pass

            pid = self.agent.get_player_index()
            agent_rewards[pid] += float(reward)
            
            if diagnostics and isinstance(diagnostics, dict):
                try:
                    self.logger.log_dict(diagnostics, episode=self.train_episode, timestep=step_count)
                except Exception:
                    pass

            step_count += 1
            state = new_state
            
            current_player = 2 if (current_player == 1) else 1

        try:
            self.logger.log_scalar("episode_length", step_count, episode=self.train_episode)
            for pid, total in agent_rewards.items():
                self.logger.log_scalar(f"player_{pid}/episode_total_reward", total, episode=self.train_episode)
        except Exception:
            pass
    
    def save_results(self):
        self.agent.save_parameters(f"{self.save_path}/parameters")
        try:
            self.logger.save_csv()
            self.logger.plot(rolling_window=5)
        except Exception:
            pass