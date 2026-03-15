# builtin

# external

# internal
from ..elements import State, Action
from ..agent import Policy, ValueFunction
from .optimization_method import OptimizationMethod


class OneStepActorCritic(OptimizationMethod):
    
    def __init__(self, policy: Policy, value_function: ValueFunction, discount: float, policy_step_size: float, value_step_size: float):
        super().__init__()
        
        self._policy = policy
        self._value_function = value_function
        
        self.discount = discount
        self.policy_step_size = policy_step_size
        self.value_step_size = value_step_size
        
    @property
    def policy(self) -> Policy:
        return self._policy
    
    @property
    def value_function(self) -> ValueFunction:
        return self._value_function
    
    def reset(self):
        self.policy_discount = 1.0
    
    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        delta = reward + self.discount * self._value_function.evaluate_state(new_state) - self._value_function.evaluate_state(old_state)

        value_update = self.value_step_size * delta * self._value_function.get_gradient(old_state)
        # apply value update
        self._value_function.update(value_update)

        policy_update = self.policy_step_size * delta * self.policy_discount * self._policy.get_eligibility(old_state, action)
        # apply policy update
        self._policy.update(policy_update)

        # decay for eligibility traces
        self.policy_discount *= self.discount

        # return diagnostics for logging: delta and norms of updates
        try:
            import numpy as _np
            metrics = {
                "delta": float(delta),
                "value_update_norm": float(_np.linalg.norm(value_update)),
                "policy_update_norm": float(_np.linalg.norm(policy_update)),
            }
        except Exception:
            metrics = {"delta": float(delta)}

        return metrics