/** biome-ignore-all lint/a11y/noStaticElementInteractions: Game UI elements */
/** biome-ignore-all lint/suspicious/noArrayIndexKey: Non-index keys unnecessary for board elements */
// builtin

// external
import { useState } from 'react';

// internal
import { type Board, type Move, type GameState, BOARD_SIZE, type Coordinate } from '../../lib/game/types';
import { getRandomMove } from '../../lib/game/actions';
import { evaluateBoard } from '../../lib/game/logic';
import './game-page.css';


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
        const currentEval = evaluateBoard(board);
        if (currentEval.finished) return;
        if (board[row][column] !== 0) return;

        const newBoard = board.map((r) => r.slice());
        newBoard[row][column] = 1;
        setBoard(newBoard);

        const postEval = evaluateBoard(newBoard);
        if (postEval.finished) {
            return;
        }

        setThinking(true);
        try {
            const state: GameState = { board: newBoard, finished: postEval.finished, win_index: postEval.win_index };
            const move: Move = await getRandomMove(state);
            if (move[0] >= 0) {
                const [moveRow, moveColumn] = move;
                const next = newBoard.map((r) => r.slice());
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

    const evalState = evaluateBoard(board);

    const statusText = evalState.finished
        ? (evalState.win_index === 1 ? 'Player wins' : evalState.win_index === 2 ? 'Computer wins' : 'Draw')
        : (thinking ? 'Computer thinking...' : '');

    return (
        <div className="game-root">
            <div className="game-header">
                <button type="button" onClick={handleReset} className="game-reset">Reset</button>
                <span className="game-status">{statusText}</span>
            </div>

            <div className="game-board-wrapper">
                <svg className="game-svg" width={SVG_SIZE} height={SVG_SIZE}>
                    <title>Game Board</title>
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