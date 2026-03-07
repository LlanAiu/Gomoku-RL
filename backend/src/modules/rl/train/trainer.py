# builtin
from abc import ABC, abstractmethod

# external

# internal

def Trainer(ABC):
    def __init__(self):
        self._setup_environment()
        self._setup_agent()
    
    @abstractmethod
    def _setup_environment():
        pass
    
    @abstractmethod
    def _setup_agent():
        pass
    
    
    