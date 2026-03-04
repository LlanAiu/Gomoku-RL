# builtin

# external
import numpy as np

# internal
from .computational_node import ComputationalNode


class ParameterNode(ComputationalNode):
    
    def __init__(self, shape: list[int], learning_rate: float):
        super().__init__()
        self.value = np.zeros(shape=shape, dtype=np.float32)
        self.should_update = True
        self.learning_rate = learning_rate
    
    def process_forward(self, _: np.ndarray) -> np.ndarray:
        return self.value
    
    def update_backward(self, gradient: np.ndarray) -> np.ndarray:
        self.values -= self.learning_rate * gradient
        return gradient