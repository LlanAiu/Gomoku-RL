# builtin

# external

# internal
from ...rl.elements import RewardSignal
from .action import GameAction
from .state import GameState


class GameRewardSignal(RewardSignal):
    def __init__(self):
        super().__init__()
    
    def get_reward(self, old_state: GameState, new_state: GameState, action: GameAction) -> float:
        # TODO: Write this! i.e. How much is a win worth? A loss? An intermediate move?
        # Put hard-coded constants in the constants.py file
        
        raise NotImplementedError("TODO")
