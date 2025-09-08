
# gateway/app/core/message_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import json
import struct
import logging

logger = logging.getLogger(__name__)

class MessageProcessor(ABC):
    """
    Abstract base class for message processing
    Handles data format conversion between raw bytes and structured data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def decode(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Decode raw bytes into structured data"""
        pass
    
    @abstractmethod
    async def encode(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Encode structured data into raw bytes"""
        pass
    
    @classmethod
    @abstractmethod
    def get_processor_info(cls) -> Dict[str, Any]:
        """Get information about this processor type"""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for this processor type"""
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate structured data format"""
        return isinstance(data, dict)

