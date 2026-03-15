# builtin
from pathlib import Path

# external
import numpy as np
from fastapi import APIRouter

# internal
from .io import AiMoveInput, AiMoveOutput
from src.modules.game.agent import GameAgent
from src.modules.game.elements import GameState, GameAction


ai_move_router = APIRouter(prefix="/ai-move")

_agent = GameAgent(2, "./src/scripts/test_train/player_two")


@ai_move_router.post("/")
async def get_ai_move(input: AiMoveInput) -> AiMoveOutput:
    
    board: np.ndarray = np.array(input.state.board)
    state: GameState = GameState(board)
    
    move: GameAction = _agent.decide_inference(state)
    return AiMoveOutput(move=move.get_move())
