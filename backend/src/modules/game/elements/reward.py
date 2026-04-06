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
    
    def get_reward(self, old_state: GameState, new_state: GameState, action: GameAction) -> float:
        # TODO: Write this!
        # Put hard-coded constants in the constants.py file
        
        raise NotImplementedError("TODO")
