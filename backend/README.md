# Backend

Highly recommend [`uv`](https://docs.astral.sh/uv/getting-started/installation/) -- definitely the best python PM I've seen thus far: 

### Setup
- Install python + deps: `uv sync`
- Make an `.env.development` at the `backend` root with the following fields:
```
ENV_TYPE="development"
ALLOWED_ORIGIN="http://localhost:5173"
```
- Run dev server: `uv run fastapi dev`