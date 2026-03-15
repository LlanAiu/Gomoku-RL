# builtin
from typing import Tuple

# external
from pydantic import BaseModel

# internal
from src.modules import GameSnapshot


class AiMoveInput(BaseModel):
	state: GameSnapshot


class AiMoveOutput(BaseModel):
	move: Tuple[int, int]

