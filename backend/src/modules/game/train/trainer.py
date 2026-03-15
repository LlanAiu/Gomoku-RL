# builtin
import os

# external

# internal
from ...rl.train import EpisodicTrainer
from ..environment import GameEnvironment
from ..agent import GameAgent
from ...log.logger import Logger

class GameTrainer(EpisodicTrainer):
    def __init__(self, save_path: str):
        super().__init__(save_path)
        # create logger for this trainer (logs saved under save_path/logs)
        self.logger = Logger.get_instance(save_dir=os.path.join(self.save_path, "logs"))

    def _setup_environment(self):
        self.environment = GameEnvironment()

    def _setup_agent(self):
        self.agent_1 = GameAgent(1, "test_train/player_one")
        self.agent_2 = GameAgent(2, "test_train/player_two")
        
    def run_train_episode(self):
        if self.environment is None or self.agent_1 is None or self.agent_2 is None:
            raise RuntimeError("Cannot train when environment/agent/method is not set!")

        current_agent = self.agent_1
        state = self.environment.reset()
        self.agent_1.optimization_method.reset()
        self.agent_2.optimization_method.reset()
        # per-player cumulative rewards for this episode
        agent_rewards = {self.agent_1.get_player_index(): 0.0,
                 self.agent_2.get_player_index(): 0.0}
        step_count = 0
        
        while not state.is_terminal():
            action = current_agent.decide_train(state)
        
            new_state, reward = self.environment.step(action)

            # perform optimization step and capture diagnostics
            diagnostics = current_agent.improve(state, action, new_state, reward)

            try:
                # log per-agent step reward (split by player)
                pid = current_agent.get_player_index()
                self.logger.log_scalar(f"player_{pid}/step_reward", float(reward),
                                       episode=self.train_episode, timestep=step_count)
            except Exception:
                pass

            # accumulate per-agent reward
            pid = current_agent.get_player_index()
            agent_rewards[pid] += float(reward)

            # log diagnostics returned by optimizer, namespaced by player
            if diagnostics and isinstance(diagnostics, dict):
                try:
                    namespaced = {f"player_{pid}/{k}": float(v) for k, v in diagnostics.items()}
                    self.logger.log_dict(namespaced, episode=self.train_episode, timestep=step_count)
                except Exception:
                    pass

            step_count += 1
        
            state = new_state
            
            current_agent = self.agent_1 if current_agent.get_player_index() == 2 else self.agent_2

        try:
            # log overall episode length once
            self.logger.log_scalar("episode_length", step_count, episode=self.train_episode)
            # log per-player episode totals
            for pid, total in agent_rewards.items():
                self.logger.log_scalar(f"player_{pid}/episode_total_reward", total, episode=self.train_episode)
        except Exception:
            pass

        # capture diagnostics from the optimization step if provided and log per-agent
        # Note: diagnostics are captured during improve and should be returned by Agent.improve
    
    def save_results(self):
        self.agent_1.save_parameters(f"{self.save_path}/player_one")
        self.agent_2.save_parameters(f"{self.save_path}/player_two")
        # persist logs and plots
        try:
            self.logger.save_csv()
            # create plots (non-blocking if matplotlib missing)
            self.logger.plot(rolling_window=5)
        except Exception:
            pass