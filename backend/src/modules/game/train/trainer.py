# builtin

# external

# internal
from ...rl.train import EpisodicTrainer
from ..environment import GameEnvironment
from ..agent import GameAgent

class GameTrainer(EpisodicTrainer):
    def __init__(self, save_path: str):
        super().__init__(save_path)

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
        
        while not state.is_terminal():
            action = current_agent.decide_train(state)
        
            new_state, reward = self.environment.step(action)

            current_agent.improve(state, action, new_state, reward)
        
            state = new_state
            
            current_agent = self.agent_1 if current_agent.get_player_index() == 2 else self.agent_2
    
    def save_results(self):
        self.agent_1.save_parameters(f"{self.save_path}/player_one")
        self.agent_2.save_parameters(f"{self.save_path}/player_two")