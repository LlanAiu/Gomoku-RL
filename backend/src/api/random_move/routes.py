# builtin

# external
from fastapi import APIRouter
import random
from typing import List

# internal
from .io import RandomMoveInput, RandomMoveOutput


random_move_router = APIRouter(prefix="/random-move")


@random_move_router.post("/")
async def get_random_move(input: RandomMoveInput) -> RandomMoveOutput:
    board: List[List[int]] = input.state.board
    empty_positions: list[tuple[int, int]] = []
    for r, row in enumerate(board):
        for c, v in enumerate(row):
            if v == 0:
                empty_positions.append((r, c))

    if not empty_positions:
        return RandomMoveOutput(move=(-1, -1))

    move = random.choice(empty_positions)
    return RandomMoveOutput(move=move)