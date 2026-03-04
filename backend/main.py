# builtin
from contextlib import asynccontextmanager

# external
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# internal
from src.globals import Environment
from src.api import random_move_router


environment: Environment = Environment()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.environment = environment
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[environment.ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root() -> dict[str, str]:
    return {"Hello": "World"}

app.include_router(router=random_move_router)