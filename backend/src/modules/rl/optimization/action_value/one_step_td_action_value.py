# builtin
from __future__ import annotations

# external

# internal
from ...elements import State, Action
from ...agent import Policy, ActionValueFunction
from .action_value_method import ActionValueMethod


class OneStepTDActionValue(ActionValueMethod):
    def __init__(
        self,
        policy: Policy,
        q_function: ActionValueFunction,
        discount: float,
        step_size: float,
    ):
        super().__init__(discount, step_size)

        self._policy = policy
        self._q_function = q_function

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def q_function(self) -> ActionValueFunction:
        return self._q_function

    def reset(self):
        return

    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        q_previous = float(self._q_function.evaluate(old_state, action))

        if new_state.is_terminal():
            target = reward
        else:
            next_action = self._policy.choose_action(new_state)
            q_next = float(self._q_function.evaluate(new_state, next_action))
            target = reward + self._discount * q_next

        delta = target - q_previous

        grad = self._q_function.get_gradient(old_state, action)
        q_update = self._step_size * delta * grad
        self._q_function.update(q_update)

        try:
            import numpy as _np

            metrics = {
                "delta": float(delta),
                "q_update_norm": float(_np.linalg.norm(q_update)),
            }
        except Exception:
            metrics = {"delta": float(delta)}

        return metrics
