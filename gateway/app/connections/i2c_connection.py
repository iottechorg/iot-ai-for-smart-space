# gateway/app/connections/i2c_connection.py
try:
    import board
    import busio
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    board = None
    busio = None

import asyncio
from typing import Dict, Any, Optional, List
import logging
from core.device_connection import DeviceConnection, ConnectionStatus

logger = logging.getLogger(__name__)

class I2CConnection(DeviceConnection):
    """
    I2C connection implementation
    Requires CircuitPython libraries (board, busio)
    """
    
    def __init__(self, connection_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, config)
        self.i2c_bus = None
        self.device_address = config['address']  # 7-bit I2C address
        self.bus_number = config.get('bus', 1)
        self.frequency = config.get('frequency', 100000)  # 100kHz default
        
        if not I2C_AVAILABLE:
            logger.warning("I2C libraries not available. Install CircuitPython libraries for I2C support.")
    
    async def connect(self) -> bool:
        """Establish I2C connection"""
        if not I2C_AVAILABLE:
            self.status = ConnectionStatus.ERROR
            self.last_error = "I2C libraries not available"
            return False
        
        try:
            self.status = ConnectionStatus.CONNECTING
            logger.info(f"Connecting to I2C device at address 0x{self.device_address:02X}")
            
            # Initialize I2C bus
            self.i2c_bus = busio.I2C(board.SCL, board.SDA, frequency=self.frequency)
            
            # Scan for device
            if await self._scan_device():
                self.status = ConnectionStatus.CONNECTED
                self.last_error = None
                logger.info(f"I2C connection established: {self.connection_id}")
                return True
            else:
                raise Exception(f"Device not found at address 0x{self.device_address:02X}")
                
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to I2C device: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close I2C connection"""
        try:
            if self.i2c_bus:
                self.i2c_bus.deinit()
                self.i2c_bus = None
            
            self.status = ConnectionStatus.DISCONNECTED
            logger.info(f"I2C connection closed: {self.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing I2C connection {self.connection_id}: {e}")
            return False
    
    async def read_raw(self) -> Optional[bytes]:
        """Read raw data from I2C device"""
        if self.status != ConnectionStatus.CONNECTED or not self.i2c_bus:
            return None
        
        try:
            read_length = self.config.get('read_length', 16)
            buffer = bytearray(read_length)
            
            # Perform I2C read in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._i2c_read, buffer
            )
            
            return bytes(buffer)
            
        except Exception as e:
            logger.error(f"Error reading from I2C device {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return None
    
    def _i2c_read(self, buffer: bytearray):
        """Blocking I2C read operation"""
        while not self.i2c_bus.try_lock():
            pass
        try:
            self.i2c_bus.readfrom_into(self.device_address, buffer)
        finally:
            self.i2c_bus.unlock()
    
    async def write_raw(self, data: bytes) -> bool:
        """Write raw data to I2C device"""
        if self.status != ConnectionStatus.CONNECTED or not self.i2c_bus:
            return False
        
        try:
            # Perform I2C write in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._i2c_write, data
            )
            return True
            
        except Exception as e:
            logger.error(f"Error writing to I2C device {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return False
    
    def _i2c_write(self, data: bytes):
        """Blocking I2C write operation"""
        while not self.i2c_bus.try_lock():
            pass
        try:
            self.i2c_bus.writeto(self.device_address, data)
        finally:
            self.i2c_bus.unlock()
    
    async def read_register(self, register: int, length: int = 1) -> Optional[bytes]:
        """Read from specific register"""
        if self.status != ConnectionStatus.CONNECTED or not self.i2c_bus:
            return None
        
        try:
            buffer = bytearray(length)
            await asyncio.get_event_loop().run_in_executor(
                None, self._i2c_read_register, register, buffer
            )
            return bytes(buffer)
            
        except Exception as e:
            logger.error(f"Error reading register 0x{register:02X}: {e}")
            return None
    
    def _i2c_read_register(self, register: int, buffer: bytearray):
        """Blocking I2C register read operation"""
        while not self.i2c_bus.try_lock():
            pass
        try:
            self.i2c_bus.writeto_then_readfrom(
                self.device_address, 
                bytes([register]), 
                buffer
            )
        finally:
            self.i2c_bus.unlock()
    
    async def write_register(self, register: int, data: bytes) -> bool:
        """Write to specific register"""
        if self.status != ConnectionStatus.CONNECTED or not self.i2c_bus:
            return False
        
        try:
            write_data = bytes([register]) + data
            await asyncio.get_event_loop().run_in_executor(
                None, self._i2c_write, write_data
            )
            return True
            
        except Exception as e:
            logger.error(f"Error writing register 0x{register:02X}: {e}")
            return False
    
    async def _scan_device(self) -> bool:
        """Scan for device on I2C bus"""
        try:
            scan_result = await asyncio.get_event_loop().run_in_executor(
                None, self._i2c_scan
            )
            return self.device_address in scan_result
        except Exception:
            return False
    
    def _i2c_scan(self) -> List[int]:
        """Blocking I2C bus scan"""
        while not self.i2c_bus.try_lock():
            pass
        try:
            return self.i2c_bus.scan()
        finally:
            self.i2c_bus.unlock()
    
    async def health_check(self) -> bool:
        """Check if I2C device is responsive"""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            # Try to read a single byte
            test_data = await self.read_raw()
            return test_data is not None
        except Exception:
            return False
    
    @classmethod
    def get_connection_info(cls) -> Dict[str, Any]:
        """Get I2C connection information"""
        return {
            "type": "i2c",
            "name": "I2C Bus Connection",
            "description": "Inter-Integrated Circuit communication",
            "supported_speeds": ["100kHz", "400kHz", "1MHz"],
            "typical_devices": ["Sensors", "ADCs", "RTCs", "EEPROMs", "Display Controllers"],
            "requires": "CircuitPython libraries (board, busio)"
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default I2C configuration"""
        return {
            "address": 0x48,  # Common sensor address
            "bus": 1,
            "frequency": 100000,  # 100kHz
            "read_length": 16,
            "max_retries": 3,
            "retry_delay": 1.0
        }

