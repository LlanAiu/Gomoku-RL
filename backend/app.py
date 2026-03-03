from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!"}


class EchoRequest(BaseModel):
    text: str


@app.post("/echo")
def echo(req: EchoRequest):
    return {"echo": req.text}
