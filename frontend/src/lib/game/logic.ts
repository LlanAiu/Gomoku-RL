// builtin

// external

// internal
import { WinIndex, type Board, type WinnerResult } from './types';


const WIN_COUNT: number = 5;
const DIRECTIONS: Array<[number, number]> = [
    [0, 1],
    [1, 0],
    [1, 1],
    [-1, 1],
];

function inBounds(board: Board, row: number, column: number) {
    return row >= 0 && column >= 0 && row < board.length && column < board.length;
}

function countDirection(
    board: Board,
    row: number,
    col: number,
    deltaRow: number,
    deltaCol: number
) {
    const player = board[row][col];
    if (!player) return 0;
    let count = 1;
    let currentRow = row + deltaRow;
    let currentCol = col + deltaCol;
    while (inBounds(board, currentRow, currentCol) && board[currentRow][currentCol] === player) {
        count += 1;
        currentRow += deltaRow;
        currentCol += deltaCol;
    }
    return count;
}

export function evaluateBoard(board: Board): WinnerResult {
    const boardSize = board.length;

    for (let row = 0; row < boardSize; row++) {
        for (let col = 0; col < boardSize; col++) {
            const player = board[row][col];
            if (!player) continue;

            for (const [deltaRow, deltaCol] of DIRECTIONS) {
                const backRow = row - deltaRow;
                const backCol = col - deltaCol;
                if (inBounds(board, backRow, backCol) && board[backRow][backCol] === player) {
                    continue;
                }

                const count = countDirection(board, row, col, deltaRow, deltaCol);
                if (count >= WIN_COUNT) {
                    return { finished: true, win_index: player };
                }
            }
        }
    }

    let hasEmpty = false;
    for (let row = 0; row < boardSize; row++) {
        for (let col = 0; col < boardSize; col++) {
            if (board[row][col] === 0) {
                hasEmpty = true;
                break;
            }
        }
        if (hasEmpty) break;
    }

    if (!hasEmpty) {
        return { finished: true, win_index: WinIndex.DRAW };
    }

    return { finished: false, win_index: WinIndex.NONE };
}

export default evaluateBoard;
