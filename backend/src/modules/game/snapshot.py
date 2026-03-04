# builtin
from typing import List, Optional

# external
from pydantic import BaseModel

# internal


class GameSnapshot(BaseModel):
    board: List[List[int]]
    finished: Optional[bool] = False
    win_index: Optional[int] = -1