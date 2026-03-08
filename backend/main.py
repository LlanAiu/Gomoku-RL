# builtin
from contextlib import asynccontextmanager

# external
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# internal
from src.globals import EnvironmentVariables
from src.api import random_move_router


env_vars: EnvironmentVariables = EnvironmentVariables()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.env_vars = env_vars
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[env_vars.ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root() -> dict[str, str]:
    return {"Hello": "World"}

app.include_router(router=random_move_router)