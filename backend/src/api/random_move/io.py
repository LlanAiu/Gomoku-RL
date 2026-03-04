# builtin

# external
from pydantic import BaseModel
from typing import List, Tuple


class GameState(BaseModel):
    board: List[List[int]]


class RandomMoveInput(BaseModel):
    state: GameState


class RandomMoveOutput(BaseModel):
    move: Tuple[int, int]