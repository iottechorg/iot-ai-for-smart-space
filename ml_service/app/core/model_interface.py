# ml_service/app/core/model_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ModelInterface(ABC):
    """Interface for ML models"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get model version"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction"""
        pass
    
    @abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> bool:
        """Train the model"""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model"""
        pass
    
    @abstractmethod
    async def save(self, path: str) -> bool:
        """Save the model"""
        pass
    
    @abstractmethod
    async def load(self, path: str) -> bool:
        """Load the model"""
        pass