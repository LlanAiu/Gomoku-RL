# builtin

# external

# internal

BOARD_SIZE = 9

WIN_COUNT = 5
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (-1, 1)]

WIN_REWARD = 5.0
LOSS_REWARD = -5.0
DRAW_REWARD = 0.0

WIN_INDICES = {
    "PLAYER_1": 1,
    "PLAYER_2": 2,
    "DRAW": 0,
    "NONE": -1
}

FEATURE_IN_DIM = 2 * BOARD_SIZE * BOARD_SIZE + 1
POLICY_OUT_DIM = BOARD_SIZE * BOARD_SIZE