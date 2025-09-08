# gateway/app/core/connection_factory.py
from typing import Dict, Any, Type, Optional, List
import logging
from .device_connection import DeviceConnection
from .message_processor import MessageProcessor

logger = logging.getLogger(__name__)

class ConnectionFactory:
    """
    Factory for creating device connections and message processors
    Follows Factory pattern and Open/Closed principle
    """
    
    def __init__(self):
        self.connection_types: Dict[str, Type[DeviceConnection]] = {}
        self.processor_types: Dict[str, Type[MessageProcessor]] = {}
        self._register_built_in_types()
    
    def _register_built_in_types(self):
        """Register built-in connection and processor types"""
        try:
            from connections.serial_connection import SerialConnection
            self.register_connection_type("serial", SerialConnection)
        except ImportError:
            logger.warning("Serial connection not available")
        
        try:
            from connections.wifi_connection import WiFiConnection
            self.register_connection_type("wifi", WiFiConnection)
        except ImportError:
            logger.warning("WiFi connection not available")
        
        try:
            from connections.i2c_connection import I2CConnection
            self.register_connection_type("i2c", I2CConnection)
        except ImportError:
            logger.warning("I2C connection not available")
        
        try:
            from processors.json_processor import JSONProcessor
            self.register_processor_type("json", JSONProcessor)
        except ImportError:
            logger.warning("JSON processor not available")
        
        try:
            from processors.modbus_processor import ModbusProcessor
            self.register_processor_type("modbus", ModbusProcessor)
        except ImportError:
            logger.warning("Modbus processor not available")
    
    def register_connection_type(self, name: str, connection_class: Type[DeviceConnection]):
        """Register a new connection type"""
        self.connection_types[name] = connection_class
        logger.info(f"Registered connection type: {name}")
    
    def register_processor_type(self, name: str, processor_class: Type[MessageProcessor]):
        """Register a new processor type"""
        self.processor_types[name] = processor_class
        logger.info(f"Registered processor type: {name}")
    
    def create_connection(self, connection_type: str, connection_id: str, 
                         config: Dict[str, Any]) -> Optional[DeviceConnection]:
        """Create a device connection instance"""
        if connection_type not in self.connection_types:
            logger.error(f"Unknown connection type: {connection_type}")
            return None
        
        try:
            connection_class = self.connection_types[connection_type]
            connection = connection_class(connection_id, config)
            logger.info(f"Created {connection_type} connection: {connection_id}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create {connection_type} connection: {e}")
            return None
    
    def create_processor(self, processor_type: str, config: Dict[str, Any]) -> Optional[MessageProcessor]:
        """Create a message processor instance"""
        if processor_type not in self.processor_types:
            logger.error(f"Unknown processor type: {processor_type}")
            return None
        
        try:
            processor_class = self.processor_types[processor_type]
            processor = processor_class(config)
            logger.info(f"Created {processor_type} processor")
            return processor
            
        except Exception as e:
            logger.error(f"Failed to create {processor_type} processor: {e}")
            return None
    
    def get_available_connections(self) -> List[str]:
        """Get list of available connection types"""
        return list(self.connection_types.keys())
    
    def get_available_processors(self) -> List[str]:
        """Get list of available processor types"""
        return list(self.processor_types.keys())
    
    def get_connection_info(self, connection_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a connection type"""
        if connection_type in self.connection_types:
            return self.connection_types[connection_type].get_connection_info()
        return None
    
    def get_processor_info(self, processor_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a processor type"""
        if processor_type in self.processor_types:
            return self.processor_types[processor_type].get_processor_info()
        return None