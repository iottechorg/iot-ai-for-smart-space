# gateway/app/core/device_connection.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class DeviceConnection(ABC):
    """
    Abstract base class for all device connections
    Follows Interface Segregation and Open/Closed principles
    """
    
    def __init__(self, connection_id: str, config: Dict[str, Any]):
        self.connection_id = connection_id
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.retry_count = 0
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5.0)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the device"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the device"""
        pass
    
    @abstractmethod
    async def read_raw(self) -> Optional[bytes]:
        """Read raw data from the device"""
        pass
    
    @abstractmethod
    async def write_raw(self, data: bytes) -> bool:
        """Write raw data to the device"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connection is healthy"""
        pass
    
    @classmethod
    @abstractmethod
    def get_connection_info(cls) -> Dict[str, Any]:
        """Get information about this connection type"""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for this connection type"""
        pass
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        return self.last_error
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect"""
        if self.retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded for {self.connection_id}")
            return False
        
        self.status = ConnectionStatus.RECONNECTING
        self.retry_count += 1
        
        logger.info(f"Reconnecting {self.connection_id} (attempt {self.retry_count}/{self.max_retries})")
        
        await asyncio.sleep(self.retry_delay)
        
        if await self.connect():
            self.retry_count = 0
            return True
        
        return False

