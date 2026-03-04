# builtin
from __future__ import annotations
from abc import ABC, abstractmethod

# external
import numpy as np

# internal


class ComputationalNode(ABC):
    
    def __init__(self, learning_rate: float):
        self.input_nodes: list[ComputationalNode] = []
        self.output_nodes: list[ComputationalNode] = []
        self.learning_rate: float = learning_rate
        self.should_update: bool = True
    
    def add_input_node(self, node: ComputationalNode):
        self.input_nodes.append(node)
        node._add_output_node(self)
    
    def _add_output_node(self, node: ComputationalNode):
        self.output_nodes.append(node)
    
    @abstractmethod
    def process_forward(self, input: np.ndarray) -> np.ndarray:
        ...
    
    @abstractmethod
    def update_backward(self, gradient: np.ndarray) -> np.ndarray:
        ...