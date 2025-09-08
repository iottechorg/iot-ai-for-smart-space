

# gateway/app/core/physical_device.py
from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime
from .device_interface import DeviceInterface
from .device_connection import DeviceConnection
from .message_processor import MessageProcessor

logger = logging.getLogger(__name__)

class PhysicalDevice(DeviceInterface):
    """
    Base class for physical devices that combine connection + processor + DeviceInterface
    Follows Composition pattern and SOLID principles
    """
    
    def __init__(self, device_id: str, connection: DeviceConnection, 
                 processor: MessageProcessor, config: Dict[str, Any]):
        self.device_id = device_id
        self.connection = connection
        self.processor = processor
        self.config = config
        self.location = config.get('location', {"lat": 0.0, "lon": 0.0})
        self.is_active = True
        self.last_reading = None
        self.last_command = None
        self.connection_retries = 0
        self.max_connection_retries = config.get('max_connection_retries', 3)
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return self.config.get('device_type', 'physical_device')
    
    async def read_data(self) -> Dict[str, Any]:
        """Read data from physical device"""
        if not self.is_active:
            return {}
        
        try:
            # Ensure connection is established
            if not await self._ensure_connected():
                return self._create_error_reading("Connection failed")
            
            # Read raw data from device
            raw_data = await self.connection.read_raw()
            if raw_data is None:
                return self._create_error_reading("No data received")
            
            # Process raw data
            processed_data = await self.processor.decode(raw_data)
            if processed_data is None:
                return self._create_error_reading("Data processing failed")
            
            # Create standardized reading
            reading = {
                "device_id": self.device_id,
                "device_type": self.get_type(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "active",
                "connection_status": self.connection.get_status().value,
                "location": self.location,
                "data": processed_data
            }
            
            self.last_reading = reading
            self.connection_retries = 0  # Reset on successful read
            
            logger.debug(f"Successfully read data from {self.device_id}")
            return reading
            
        except Exception as e:
            logger.error(f"Error reading from device {self.device_id}: {e}")
            return self._create_error_reading(str(e))
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Write command to physical device"""
        try:
            # Ensure connection is established
            if not await self._ensure_connected():
                return {
                    "success": False,
                    "error": "Connection failed",
                    "device_id": self.device_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Encode command
            encoded_command = await self.processor.encode(command)
            if encoded_command is None:
                return {
                    "success": False,
                    "error": "Command encoding failed",
                    "device_id": self.device_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Send command to device
            success = await self.connection.write_raw(encoded_command)
            
            result = {
                "success": success,
                "device_id": self.device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "command": command
            }
            
            if success:
                self.last_command = command
                logger.info(f"Successfully sent command to {self.device_id}")
            else:
                result["error"] = "Failed to send command to device"
                logger.error(f"Failed to send command to {self.device_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error writing to device {self.device_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "device_id": self.device_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _ensure_connected(self) -> bool:
        """Ensure device connection is established"""
        from .device_connection import ConnectionStatus
        
        if self.connection.get_status() == ConnectionStatus.CONNECTED:
            return True
        
        if self.connection_retries >= self.max_connection_retries:
            logger.error(f"Max connection retries exceeded for {self.device_id}")
            return False
        
        logger.info(f"Attempting to connect to device {self.device_id}")
        self.connection_retries += 1
        
        return await self.connection.connect()
    
    def _create_error_reading(self, error_message: str) -> Dict[str, Any]:
        """Create error reading when device communication fails"""
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "error",
            "error": error_message,
            "connection_status": self.connection.get_status().value,
            "location": self.location
        }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure device parameters"""
        try:
            # Update local config
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
            
            # If device supports configuration commands, send them
            if 'configuration' in config:
                config_command = config['configuration']
                result = await self.write_data(config_command)
                return result.get('success', False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring device {self.device_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check device health"""
        try:
            # Check connection health
            if not await self.connection.health_check():
                return False
            
            # Try to read data as health check
            reading = await self.read_data()
            return reading.get('status') != 'error'
            
        except Exception as e:
            logger.error(f"Health check failed for device {self.device_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from physical device"""
        await self.connection.disconnect()
        logger.info(f"Disconnected from device {self.device_id}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "connection_type": self.connection.__class__.__name__,
            "connection_status": self.connection.get_status().value,
            "last_error": self.connection.get_last_error(),
            "processor_type": self.processor.__class__.__name__
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for physical devices"""
        return {
            "sample_interval": 30,
            "location": {"lat": 0.0, "lon": 0.0},
            "max_connection_retries": 3,
            "connection_timeout": 10.0,
            "read_timeout": 5.0
        }
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get device information"""
        return {
            "type": "physical_device",
            "name": "Physical Device Base",
            "description": "Base class for physical IoT devices",
            "version": "1.0",
            "capabilities": ["read", "write", "configure", "health_check"]
        }


