# builtin
from abc import abstractmethod
# external

# internal
from ...agent import ValueFunction
from ..optimization_method import OptimizationMethod


class PolicyGradientMethod(OptimizationMethod):
    
    def __init__(self, discount: float, policy_step_size: float):
        super().__init__(discount)
        self._policy_discount: float = discount
        self._policy_step_size: float = policy_step_size
    
    @property
    def policy_step_size(self) -> float:
        return self._policy_step_size
    
    @property
    def policy_discount(self) -> float:
        return self._policy_discount
    
    @property
    @abstractmethod
    def value_function(self) -> ValueFunction | None:
        pass