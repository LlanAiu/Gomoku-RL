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

