# builtin

# external

# internal
from ...elements import State, Action
from ...agent import Policy, ValueFunction
from .policy_gradient_method import PolicyGradientMethod


class OneStepActorCritic(PolicyGradientMethod):
    
    def __init__(
        self, 
        policy: Policy, 
        value_function: ValueFunction, 
        discount: float, 
        policy_step_size: float, 
        value_step_size: float
    ):
        super().__init__(discount, policy_step_size)
        
        self._policy = policy
        self._value_function = value_function
        
        self.value_step_size = value_step_size
        
    @property
    def policy(self) -> Policy:
        return self._policy
    
    @property
    def value_function(self) -> ValueFunction:
        return self._value_function
    
    def reset(self):
        self._policy_discount = 1.0
    
    def improve(self, old_state: State, action: Action, new_state: State, reward: float):
        value_previous = self._value_function.evaluate_state(old_state)
        if new_state.is_terminal():
            target = reward
        else:
            target = reward + self._discount * self._value_function.evaluate_state(new_state)
        
        delta = target - value_previous

        value_update = self.value_step_size * delta * self._value_function.get_gradient(old_state)
        self._value_function.update(value_update)

        policy_update = self.policy_step_size * delta * self.policy_discount * self._policy.get_eligibility(old_state, action)
        self._policy.update(policy_update)

        self._policy_discount *= self._discount

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