// builtin

// external
import React, { useState } from 'react';

// internal

import { type Board, type Move, type GameState, BOARD_SIZE } from '../../lib/game/types';
import { getRandomMove } from '../../lib/game/actions';

function emptyBoard(): Board {
    return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));
}

export default function GamePage() {
    const [board, setBoard] = useState<Board>(emptyBoard());
    const [thinking, setThinking] = useState(false);

    const handleReset = () => {
        setBoard(emptyBoard());
        setThinking(false);
    };

    const handleClick = async (r: number, c: number) => {
        if (thinking) return;
        if (board[r][c] !== 0) return;

        const newBoard = board.map((row) => row.slice());
        newBoard[r][c] = 1;
        setBoard(newBoard);

        setThinking(true);
        try {
            const state: GameState = { board: newBoard };
            const move: Move = await getRandomMove(state);
            if (move[0] >= 0) {
                const [mr, mc] = move;
                const next = newBoard.map((row) => row.slice());
                if (next[mr][mc] === 0) {
                    next[mr][mc] = 2;
                    setBoard(next);
                }
            }
        } catch (err) {
            console.error('Failed to get move', err);
        } finally {
            setThinking(false);
        }
    };

    return (
        <div style={{ padding: 12 }}>
            <div style={{ marginBottom: 8 }}>
                <button type="button" onClick={handleReset}>Reset</button>
                <span style={{ marginLeft: 12 }}>{thinking ? 'Computer thinking...' : ''}</span>
            </div>

            <div style={{ display: 'inline-block', border: '1px solid #444' }}>
                {board.map((row, r) => (
                    <div key={`row-${r}`} style={{ display: 'flex' }}>
                        {row.map((cell, c) => (
                            <button
                                key={`cell-${c}`}
                                type='button'
                                onClick={() => handleClick(r, c)}
                                style={{
                                    width: 28,
                                    height: 28,
                                    border: '1px solid #999',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    background: '#f7f7f7',
                                    cursor: cell === 0 && !thinking ? 'pointer' : 'default',
                                    fontSize: 14,
                                }}
                            >
                                {cell === 1 ? '●' : cell === 2 ? '○' : ''}
                            </button>
                        ))}
                    </div>
                ))}
            </div>
        </div>
    );
}