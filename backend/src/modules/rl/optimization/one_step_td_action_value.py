# builtin
from __future__ import annotations

# external

# internal
from ..elements import State, Action
from ..agent import Policy
from .optimization_method import OptimizationMethod


class OneStepTDActionValue(OptimizationMethod):
    """One-step TD(0) update for action-value function approximators.

    This optimizer supports two modes:
    - on_policy (SARSA): uses the provided `policy` to sample the next action
      and bootstraps from Q(new_state, next_action).
    - off_policy (Q-learning): bootstraps from max_a' Q(new_state, a').

    The `q_function` passed in is expected to implement the following methods:
    - `evaluate(state, action) -> float`
    - `get_gradient(state, action) -> np.ndarray` (shape matching parameters)
    - `update(update: np.ndarray)` to apply parameter updates
    - optional: `save_parameters` / `load_parameters` if persistence is needed
    """

    def __init__(
        self,
        policy: Policy | None,
        q_function,
        discount: float,
        step_size: float,
        off_policy: bool = False,
    ):
        super().__init__()

        self._policy = policy
        self._q_function = q_function

        self.discount = discount
        self.step_size = step_size
        self.off_policy = off_policy

    @property
    def policy(self) -> Policy | None:
        return self._policy

    @property
    def value_function(self):
        # action-value case does not expose a state value function
        return None

    def reset(self):
        return None

    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        # evaluate current estimate
        q_old = float(self._q_function.evaluate(old_state, action))

        # build bootstrap target
        if new_state.is_terminal():
            target = reward
        else:
            if self.off_policy:
                # Q-learning style: use max_a' Q(s', a')
                try:
                    next_values = self._q_function.evaluate_all_actions(new_state)
                    max_next = float(max(next_values))
                except Exception:
                    # fallback: if evaluate_all_actions isn't available, try sampling via policy
                    if self._policy is None:
                        raise RuntimeError("off_policy=True requires q_function.evaluate_all_actions or a policy")
                    next_action = self._policy.choose_action(new_state)
                    max_next = float(self._q_function.evaluate(new_state, next_action))

                target = reward + self.discount * max_next
            else:
                # SARSA-style: sample next action from policy
                if self._policy is None:
                    raise RuntimeError("on-policy TD requires a policy to sample the next action")
                next_action = self._policy.choose_action(new_state)
                q_next = float(self._q_function.evaluate(new_state, next_action))
                target = reward + self.discount * q_next

        delta = target - q_old

        # compute parameter update and apply
        grad = self._q_function.get_gradient(old_state, action)
        q_update = self.step_size * delta * grad
        self._q_function.update(q_update)

        # diagnostics
        try:
            import numpy as _np

            metrics = {
                "delta": float(delta),
                "q_update_norm": float(_np.linalg.norm(q_update)),
            }
        except Exception:
            metrics = {"delta": float(delta)}

        return metrics
