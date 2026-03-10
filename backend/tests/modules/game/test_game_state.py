import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

from modules.game.elements.state import GameState
from modules.game.elements.action import GameAction
from modules.game.environment.environment import GameEnvironment


def make_empty_board(size):
    return np.zeros((size, size), dtype=np.uint8)


def test_horizontal_win():
    size = 15
    board = make_empty_board(size)
    row = 7
    
    for c in range(3, 8):
        board[row, c] = 1

    state = GameState(board)
    assert state.terminal is True
    assert int(state.win_index) == 1


def test_vertical_and_diagonal_win():
    size = 15

    board = make_empty_board(size)
    col = 5
    for r in range(4, 9):
        board[r, col] = 2
    state = GameState(board)
    assert state.terminal is True
    assert int(state.win_index) == 2

    board = make_empty_board(size)
    start_r, start_c = 2, 2
    for i in range(5):
        board[start_r + i, start_c + i] = 1
    state = GameState(board)
    assert state.terminal is True
    assert int(state.win_index) == 1


def test_no_win_not_terminal():
    size = 15
    board = make_empty_board(size)
    board[0, 0] = 1
    board[1, 1] = 2
    state = GameState(board)
    assert state.terminal is False
    assert int(state.win_index) == -1


def test_environment_step_and_bounds():
    env = GameEnvironment()
    current = env.reset()

    action = GameAction(1, (0, 0))
    new_state, reward = env.step(action)
    assert new_state.board[0, 0] == 1
    assert env.current_state is new_state

    action_oob = GameAction(1, (-1, 0))
    returned_state, reward = env.step(action_oob)
    assert returned_state is env.current_state
    assert reward == 0.0
