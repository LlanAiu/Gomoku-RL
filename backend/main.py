# builtin
from contextlib import asynccontextmanager

# external
from fastapi import FastAPI

# internal
from src.globals import Environment
from src.api import random_move_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    environment: Environment = Environment()
    app.state.environment = environment

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root() -> dict[str, str]:
    return {"Hello": "World"}

app.include_router(router=random_move_router)