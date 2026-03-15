// builtin

// external

// internal


export type Board = number[][];
export type Move = [number, number];

export const WinIndex = {
    NONE: -1,
    DRAW: 0,
    PLAYER_1: 1,
    PLAYER_2: 2,
} as const;

export interface GameState {
    board: Board;
    finished: boolean;
    win_index: number;
}

export interface WinnerResult {
    finished: boolean;
    win_index: number;
};

export interface Coordinate {
    x: number;
    y: number;
}

export const BOARD_SIZE = 9;
