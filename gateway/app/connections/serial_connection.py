
# gateway/app/connections/serial_connection.py
import serial_asyncio
import serial
from typing import Dict, Any, Optional
import asyncio
import logging
from core.device_connection import DeviceConnection, ConnectionStatus

logger = logging.getLogger(__name__)

class SerialConnection(DeviceConnection):
    """
    Serial/UART connection implementation
    Supports RS232, RS485, USB Serial devices
    """
    
    def __init__(self, connection_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, config)
        self.serial_connection: Optional[serial_asyncio.SerialTransport] = None
        self.protocol: Optional[asyncio.Protocol] = None
        self.read_buffer = bytearray()
        self.write_lock = asyncio.Lock()
        
        # Serial-specific configuration
        self.port = config['port']  # e.g., '/dev/ttyUSB0', 'COM3'
        self.baudrate = config.get('baudrate', 9600)
        self.bytesize = config.get('bytesize', 8)
        self.parity = config.get('parity', 'N')
        self.stopbits = config.get('stopbits', 1)
        self.timeout = config.get('timeout', 1.0)
        self.read_timeout = config.get('read_timeout', 5.0)
    
    async def connect(self) -> bool:
        """Establish serial connection"""
        try:
            self.status = ConnectionStatus.CONNECTING
            logger.info(f"Connecting to serial device on {self.port} at {self.baudrate} baud")
            
            # Create serial connection with asyncio
            transport, protocol = await serial_asyncio.create_serial_connection(
                loop=asyncio.get_event_loop(),
                protocol_factory=lambda: SerialProtocol(self),
                url=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            
            self.serial_connection = transport
            self.protocol = protocol
            self.status = ConnectionStatus.CONNECTED
            self.last_error = None
            
            logger.info(f"Serial connection established: {self.connection_id}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to serial device {self.port}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close serial connection"""
        try:
            if self.serial_connection:
                self.serial_connection.close()
                self.serial_connection = None
                self.protocol = None
            
            self.status = ConnectionStatus.DISCONNECTED
            logger.info(f"Serial connection closed: {self.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing serial connection {self.connection_id}: {e}")
            return False
    
    async def read_raw(self) -> Optional[bytes]:
        """Read raw data from serial port"""
        if self.status != ConnectionStatus.CONNECTED:
            return None
        
        try:
            # Wait for data with timeout
            if await asyncio.wait_for(self._wait_for_data(), timeout=self.read_timeout):
                data = bytes(self.read_buffer)
                self.read_buffer.clear()
                return data
            return None
            
        except asyncio.TimeoutError:
            logger.debug(f"Read timeout on serial connection {self.connection_id}")
            return None
        except Exception as e:
            logger.error(f"Error reading from serial connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return None
    
    async def write_raw(self, data: bytes) -> bool:
        """Write raw data to serial port"""
        if self.status != ConnectionStatus.CONNECTED or not self.serial_connection:
            return False
        
        try:
            async with self.write_lock:
                self.serial_connection.write(data)
                await asyncio.sleep(0.01)  # Small delay for write completion
                return True
                
        except Exception as e:
            logger.error(f"Error writing to serial connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return False
    
    async def health_check(self) -> bool:
        """Check if serial connection is healthy"""
        if self.status != ConnectionStatus.CONNECTED or not self.serial_connection:
            return False
        
        try:
            # Try to write a simple health check command (if supported)
            # This could be device-specific
            return True
        except Exception:
            return False
    
    async def _wait_for_data(self) -> bool:
        """Wait for data to arrive in buffer"""
        while len(self.read_buffer) == 0:
            await asyncio.sleep(0.1)
        return True
    
    @classmethod
    def get_connection_info(cls) -> Dict[str, Any]:
        """Get serial connection information"""
        return {
            "type": "serial",
            "name": "Serial/UART Connection",
            "description": "RS232/RS485/USB Serial communication",
            "supported_interfaces": ["RS232", "RS485", "USB-Serial", "UART"],
            "typical_devices": ["Arduino", "ESP32", "Modbus RTU", "Custom Sensors"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default serial configuration"""
        return {
            "port": "/dev/ttyUSB0",  # Platform-specific default
            "baudrate": 9600,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1.0,
            "read_timeout": 5.0,
            "max_retries": 3,
            "retry_delay": 5.0
        }


class SerialProtocol(asyncio.Protocol):
    """Protocol handler for serial communication"""
    
    def __init__(self, connection: SerialConnection):
        self.connection = connection
    
    def connection_made(self, transport):
        """Called when connection is established"""
        logger.debug(f"Serial protocol connected: {self.connection.connection_id}")
    
    def data_received(self, data):
        """Called when data is received"""
        self.connection.read_buffer.extend(data)
    
    def connection_lost(self, exc):
        """Called when connection is lost"""
        if exc:
            logger.error(f"Serial connection lost: {exc}")
            self.connection.status = ConnectionStatus.ERROR
            self.connection.last_error = str(exc)
        else:
            self.connection.status = ConnectionStatus.DISCONNECTED

