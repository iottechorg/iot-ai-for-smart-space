# gateway/app/core/device_interface.py (Updated with Write Support)
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class DeviceInterface(ABC):
    """
    Interface for all IoT devices
    Follows Interface Segregation Principle
    Supports both read (sensor) and write (actuator) operations
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """Get device unique identifier"""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get device type"""
        pass
    
    @abstractmethod
    async def read_data(self) -> Dict[str, Any]:
        """Read data from the device"""
        pass
    
    @abstractmethod
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure device parameters"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check device health"""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for this device type"""
        pass
    
    @classmethod
    @abstractmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get device information (name, description, version, etc.)"""
        pass
    
    def supports_write(self) -> bool:
        """Check if device supports write operations (actuator capability)"""
        return hasattr(self, 'write_data') and callable(getattr(self, 'write_data'))
    
    def get_writable_properties(self) -> List[str]:
        """Get list of properties that can be written to this device"""
        if not self.supports_write():
            return []
        
        device_info = self.get_device_info()
        return device_info.get('writable_properties', [])
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write data/commands to the device (for actuators)
        This method should be implemented by devices that support write operations
        
        Args:
            command: Dictionary containing the command data
            
        Returns:
            Dictionary with execution result
        """
        return {
            "success": False,
            "error": "Device does not support write operations",
            "device_id": self.get_id()
        }
    
    def get_command_schema(self) -> Dict[str, Any]:
        """
        Get the schema for valid commands this device accepts
        This helps with command validation
        """
        if not self.supports_write():
            return {}
        
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


