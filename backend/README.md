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

### Organization

This is pretty much your default FastAPI project structure, with some extra additions:

```
backend/
| - src/
  | - api/
    | - ai_move/
      | - routes.py
      | - io.py
    | - ... (more routes)
  | - globals/
    | - env_vars.py (environment variables)
  | - modules/
    | - game/
      | - ... (all Gomoku specific implementations)
    | - rl/
      | - ... (abstract RL mechanics)
    | - log/
      | - ... (Over-engineered Chat-written logger :yikes:)
  | - scripts/
    | - train_gomoku.py (the RL training script)
| - tests/
  | - modules/
    | - test_game_state.py
    | - ... (many more AI-generated tests that are pretty garbage -- but feel free to swap them out with your own better ones)
| - main.py
| - .env.development
| - pyproject.toml
| - ...
```

In terms of what you actually will be working with, this is almost exclusively in the `modules/rl` and `modules/game` directories (the exception: you'll fiddle with the hard-coded model loading in `api/ai_move/routes.py`)

You'll notice that `scripts/` is not typically part of a FastAPI project -- and you're right: these aren't run as part of the backend server, they're standalone routines that you can run via: `uv run python src/scripts/train_gomoku.py` (CWD @ `backend`)

That's all in terms of setup, I've also included a `reference.md` file in this directory for a more detailed overview of the RL build.

### Things You'll Do
- Finish the game environment
- Write the game reward signal
- Write a representation for the game state
- Complete the policy gradient method
    - parametrized policy + value function
- OR complete the action value method
    - epsilon-greedy policy + action value function
- Finish the game trainer
- Train a model (and fiddle with hyperparameters, potentially)
- Load trained model into the `ai_move` route