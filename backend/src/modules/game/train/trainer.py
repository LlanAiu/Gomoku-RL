# builtin
from typing import Literal

# external

# internal
from ...rl.train import EpisodicTrainer
from ..elements import GameState, GameAction
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
        self._current_player: int = 1
        self._agent.set_player_index(self._current_player)
        self._previous_record: tuple[int, GameState, GameAction, float] | None = None
        
    def _update_train(
        self, 
        step: int, 
        state: GameState, 
        action: GameAction, 
        new_state: GameState, 
        reward: float
    ):
        if self._previous_record is not None:
            prev_player, prev_state, prev_action, prev_reward = self._previous_record

            self._agent.set_player_index(prev_player)
            diagnostics = self._agent.improve(prev_state, prev_action, new_state, prev_reward)
            self._logger.log_dict(diagnostics, episode=self.train_episode, timestep=step)

            self._agent.set_player_index(action.get_player_index())
            self._previous_record = None

        if new_state.is_terminal():
            self._agent.set_player_index(action.get_player_index())
            diagnostics = self._agent.improve(state, action, new_state, reward)
            self._logger.log_dict(diagnostics, episode=self.train_episode, timestep=step)
        else:
            self._previous_record = (action.get_player_index(), state, action, reward)
    
    def _after_step(self):
        self._current_player = 2 if (self._current_player == 1) else 1
        self._agent.set_player_index(self._current_player)