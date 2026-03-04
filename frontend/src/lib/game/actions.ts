// builtin

// external

// internal
import type { GameState, Move } from './types';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000';

export async function getRandomMove(state: GameState): Promise<Move> {
    const payload = { state };

    const res = await fetch(`${BACKEND_URL}/random-move/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
    }

    const data = await res.json();
    return data.move as Move;
}