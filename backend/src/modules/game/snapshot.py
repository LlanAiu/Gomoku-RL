# builtin
from typing import List

# external
from pydantic import BaseModel

# internal


class GameSnapshot(BaseModel):
    board: List[List[int]]