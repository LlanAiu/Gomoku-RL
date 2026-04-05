# builtin
from abc import abstractmethod

# external

# internal
from ...agent import ActionValueFunction
from ..optimization_method import OptimizationMethod

class ActionValueMethod(OptimizationMethod):
    def __init__(self, discount: float, step_size: float):
        super().__init__(discount)
        self._step_size: float = step_size
        
    @property
    def step_size(self) -> float:
        return self._step_size
    
    @property
    @abstractmethod
    def q_function(self) -> ActionValueFunction:
        pass