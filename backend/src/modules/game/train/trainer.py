# builtin
from typing import Literal

# external

# internal
from ...rl.train import EpisodicTrainer
from ..environment import GameEnvironment
from ..agent import GameAgent


class GameTrainer(EpisodicTrainer):
    def __init__(self, save_path: str, mode: Literal["policy", "action_value"]):
        self._mode: Literal["policy", "action_value"] = mode
        self._current_player = 1
        super().__init__(save_path)

    def _setup_environment(self):
        self._environment = GameEnvironment()

    def _setup_agent(self):
        self._agent: GameAgent = GameAgent(
            player_index=self._current_player, 
            weights_path=f"{self._save_path}/parameters", 
            mode=self._mode
        )
        
    def _before_episode(self):
        self._current_player = 1
        self._agent.set_player_index(self._current_player)
    
    def _after_step(self):
        self._current_player = 2 if (self._current_player == 1) else 1
        self._agent.set_player_index(self._current_player)
    
    def _after_episode(self):
        return