# builtin
from typing import Tuple

# external
from pydantic import BaseModel

# internal
from src.modules import GameSnapshot


class RandomMoveInput(BaseModel):
    state: GameSnapshot

class RandomMoveOutput(BaseModel):
    move: Tuple[int, int]