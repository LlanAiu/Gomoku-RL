# builtin

# external

# internal
from ...rl.elements import RewardSignal
from .action import GameAction
from .state import GameState
from ..constants import WIN_REWARD, DRAW_REWARD, LOSS_REWARD


class GameRewardSignal(RewardSignal):
    def __init__(self):
        super().__init__()
    
    def get_reward(self, _: GameState, new_state: GameState, action: GameAction) -> float:
        if new_state.is_terminal():
            win_index = new_state.get_win_index()
            if win_index > 0:
                if win_index == action.get_player_index():
                    return WIN_REWARD
                else:
                    return LOSS_REWARD
            
            return DRAW_REWARD
        
        return 0.0
