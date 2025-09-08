
# gateway/app/connections/wifi_connection.py
import aiohttp
import asyncio
import socket
from typing import Dict, Any, Optional
import logging
from core.device_connection import DeviceConnection, ConnectionStatus

logger = logging.getLogger(__name__)

class WiFiConnection(DeviceConnection):
    """
    WiFi/Ethernet TCP/UDP connection implementation
    Supports HTTP/HTTPS, TCP sockets, UDP sockets
    """
    
    def __init__(self, connection_id: str, config: Dict[str, Any]):
        super().__init__(connection_id, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.socket: Optional[socket.socket] = None
        
        # WiFi/Network specific configuration
        self.host = config['host']
        self.port = config.get('port', 80)
        self.protocol = config.get('protocol', 'http')  # http, https, tcp, udp
        self.timeout = config.get('timeout', 10.0)
        self.keep_alive = config.get('keep_alive', True)
    
    async def connect(self) -> bool:
        """Establish network connection"""
        try:
            self.status = ConnectionStatus.CONNECTING
            logger.info(f"Connecting to {self.protocol}://{self.host}:{self.port}")
            
            if self.protocol in ['http', 'https']:
                await self._connect_http()
            elif self.protocol == 'tcp':
                await self._connect_tcp()
            elif self.protocol == 'udp':
                await self._connect_udp()
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")
            
            self.status = ConnectionStatus.CONNECTED
            self.last_error = None
            logger.info(f"Network connection established: {self.connection_id}")
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            return False
    
    async def _connect_http(self):
        """Establish HTTP/HTTPS connection"""
        connector = aiohttp.TCPConnector(
            keepalive_timeout=30 if self.keep_alive else 0,
            limit=10
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    async def _connect_tcp(self):
        """Establish TCP socket connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        await asyncio.get_event_loop().run_in_executor(
            None, self.socket.connect, (self.host, self.port)
        )
    
    async def _connect_udp(self):
        """Establish UDP socket connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.timeout)
        # UDP doesn't require explicit connection
    
    async def disconnect(self) -> bool:
        """Close network connection"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.socket:
                self.socket.close()
                self.socket = None
            
            self.status = ConnectionStatus.DISCONNECTED
            logger.info(f"Network connection closed: {self.connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing network connection {self.connection_id}: {e}")
            return False
    
    async def read_raw(self) -> Optional[bytes]:
        """Read raw data from network connection"""
        if self.status != ConnectionStatus.CONNECTED:
            return None
        
        try:
            if self.protocol in ['http', 'https']:
                return await self._read_http()
            elif self.protocol in ['tcp', 'udp']:
                return await self._read_socket()
            return None
            
        except Exception as e:
            logger.error(f"Error reading from network connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return None
    
    async def _read_http(self) -> Optional[bytes]:
        """Read data via HTTP GET request"""
        if not self.session:
            return None
        
        endpoint = self.config.get('read_endpoint', '/data')
        url = f"{self.protocol}://{self.host}:{self.port}{endpoint}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                logger.warning(f"HTTP read failed with status {response.status}")
                return None
    
    async def _read_socket(self) -> Optional[bytes]:
        """Read data from TCP/UDP socket"""
        if not self.socket:
            return None
        
        buffer_size = self.config.get('buffer_size', 1024)
        
        if self.protocol == 'tcp':
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.socket.recv, buffer_size
            )
        else:  # UDP
            data, addr = await asyncio.get_event_loop().run_in_executor(
                None, self.socket.recvfrom, buffer_size
            )
        
        return data if data else None
    
    async def write_raw(self, data: bytes) -> bool:
        """Write raw data to network connection"""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            if self.protocol in ['http', 'https']:
                return await self._write_http(data)
            elif self.protocol in ['tcp', 'udp']:
                return await self._write_socket(data)
            return False
            
        except Exception as e:
            logger.error(f"Error writing to network connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return False
    
    async def _write_http(self, data: bytes) -> bool:
        """Write data via HTTP POST request"""
        if not self.session:
            return False
        
        endpoint = self.config.get('write_endpoint', '/command')
        url = f"{self.protocol}://{self.host}:{self.port}{endpoint}"
        
        async with self.session.post(url, data=data) as response:
            return response.status in [200, 201, 202]
    
    async def _write_socket(self, data: bytes) -> bool:
        """Write data to TCP/UDP socket"""
        if not self.socket:
            return False
        
        if self.protocol == 'tcp':
            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.send, data
            )
        else:  # UDP
            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.sendto, data, (self.host, self.port)
            )
        
        return True
    
    async def health_check(self) -> bool:
        """Check if network connection is healthy"""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            if self.protocol in ['http', 'https'] and self.session:
                # Try a simple HEAD request
                health_endpoint = self.config.get('health_endpoint', '/health')
                url = f"{self.protocol}://{self.host}:{self.port}{health_endpoint}"
                async with self.session.head(url) as response:
                    return response.status < 400
            elif self.socket:
                # For TCP/UDP, check if socket is still valid
                return True
            return False
            
        except Exception:
            return False
    
    @classmethod
    def get_connection_info(cls) -> Dict[str, Any]:
        """Get WiFi connection information"""
        return {
            "type": "wifi",
            "name": "WiFi/Ethernet Connection",
            "description": "TCP/UDP/HTTP network communication",
            "supported_protocols": ["HTTP", "HTTPS", "TCP", "UDP"],
            "typical_devices": ["ESP32", "Raspberry Pi", "Arduino WiFi", "Industrial Gateways"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default WiFi configuration"""
        return {
            "host": "192.168.1.100",
            "port": 80,
            "protocol": "http",
            "timeout": 10.0,
            "keep_alive": True,
            "buffer_size": 1024,
            "read_endpoint": "/data",
            "write_endpoint": "/command",
            "health_endpoint": "/health",
            "max_retries": 3,
            "retry_delay": 5.0
        }


