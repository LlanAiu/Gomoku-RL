# builtin

# external
from fastapi import APIRouter

# internal
from .io import RandomMoveInput, RandomMoveOutput


random_move_router = APIRouter(prefix="random-move")

@random_move_router.post("/")
async def get_random_move(input: RandomMoveInput) -> RandomMoveOutput:
    return RandomMoveOutput(move=(0, 0))