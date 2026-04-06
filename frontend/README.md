# Frontend

Install [`node.js`](https://nodejs.org/en/download) and [`npm`](https://nodejs.org/en/download) (their installs are bundled) (or [`pnpm`](https://pnpm.io/installation)) beforehand:

Note: if you use `yarn`, you're on your own (pls don't).

### Setup
- Install deps: `npm install` (or `pnpm install`)
- Make an `.env.development` at the `frontend` root with the following fields:
```
VITE_BACKEND_URL="http://localhost:8000"
```
- Run dev server: `npm run dev` (or `pnpm dev`)

### Organization

This is genuinely just a normal Vite + React + React Router frontend. It's also very bare and lightweight -- there's not much you need to do here.

```
frontend/
| - src/
  | - components/
    | - game/
      | - ... (React component for the Gomoku page)
  | - lib/
    | - game/
      | - ... (Logic for how Gomoku works + backend interactions)
  | - App.tsx
  | - main.tsx
  | - ...
| - index.html
| - .env.development
| - package.json
| - ...
```

NOTE: for now, the backend request sent in `lib/game/actions.ts` goes to the `/random-move` endpoint -- play a game (and try to lose) to see what that's like.

When your RL work is done, change this to be `/ai-move` to actually use the model you've trained.