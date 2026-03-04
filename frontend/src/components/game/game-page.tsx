// builtin

// external
import { useState } from 'react';
import './game-page.css';

// internal
import { type Board, type Move, type GameState, BOARD_SIZE, type Coordinate } from '../../lib/game/types';
import { getRandomMove } from '../../lib/game/actions';


function emptyBoard(): Board {
    return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));
}

export default function GamePage() {
    const [board, setBoard] = useState<Board>(emptyBoard());
    const [thinking, setThinking] = useState(false);

    const SPACING = 36;
    const MARGIN = 20;
    const GRID_SIZE = BOARD_SIZE;
    const SVG_SIZE = MARGIN * 2 + SPACING * (GRID_SIZE - 1);
    const PIECE_RADIUS = Math.floor(SPACING * 0.4);

    function handleReset() {
        setBoard(emptyBoard());
        setThinking(false);
    };

    async function handleClick(row: number, column: number) {
        if (thinking) return;
        if (board[row][column] !== 0) return;

        const newBoard = board.map((row) => row.slice());
        newBoard[row][column] = 1;
        setBoard(newBoard);

        setThinking(true);
        try {
            const state: GameState = { board: newBoard };
            const move: Move = await getRandomMove(state);
            if (move[0] >= 0) {
                const [moveRow, moveColumn] = move;
                const next = newBoard.map((row) => row.slice());
                if (next[moveRow][moveColumn] === 0) {
                    next[moveRow][moveColumn] = 2;
                    setBoard(next);
                }
            }
        } catch (err) {
            console.error('Failed to get move', err);
        } finally {
            setThinking(false);
        }
    };

    function xy(row: number, column: number): Coordinate {
        return {
            x: MARGIN + column * SPACING,
            y: MARGIN + row * SPACING,
        };
    };

    return (
        <div className="game-root">
            <div className="game-header">
                <button type="button" onClick={handleReset} className="game-reset">Reset</button>
                <span className="game-status">{thinking ? 'Computer thinking...' : ''}</span>
            </div>

            <div className="game-board-wrapper">
                <svg className="game-svg" width={SVG_SIZE} height={SVG_SIZE}>
                    {[...Array(GRID_SIZE)].map((_, i) => {
                        const pos = MARGIN + i * SPACING;
                        return (
                            <g key={`line-${i}`}>
                                <line x1={MARGIN} y1={pos} x2={MARGIN + SPACING * (GRID_SIZE - 1)} y2={pos} stroke="#444" strokeWidth={1} />
                                <line x1={pos} y1={MARGIN} x2={pos} y2={MARGIN + SPACING * (GRID_SIZE - 1)} stroke="#444" strokeWidth={1} />
                            </g>
                        );
                    })}

                    {board.map((row, r) => (
                        row.map((cell, c) => {
                            const { x, y } = xy(r, c);
                            const hitSize = SPACING;
                            return (
                                <g key={`pt-${r}-${c}`}>
                                    <rect
                                        x={x - hitSize / 2}
                                        y={y - hitSize / 2}
                                        width={hitSize}
                                        height={hitSize}
                                        fill="transparent"
                                        onClick={() => handleClick(r, c)}
                                        className={board[r][c] === 0 && !thinking ? 'game-hit' : 'game-hit game-hit-disabled'}
                                    />

                                    {cell === 1 && (
                                        <circle cx={x} cy={y} r={PIECE_RADIUS} fill="#111" />
                                    )}
                                    {cell === 2 && (
                                        <circle cx={x} cy={y} r={PIECE_RADIUS} fill="#fff" stroke="#111" strokeWidth={1} />
                                    )}
                                </g>
                            );
                        })
                    ))}
                </svg>
            </div>
        </div>
    );
}