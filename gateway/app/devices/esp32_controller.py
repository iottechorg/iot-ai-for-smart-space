

# gateway/app/devices/esp32_controller.py
from typing import Dict, Any, List
import logging
from datetime import datetime
from core.physical_device import PhysicalDevice
from core.connection_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class ESP32Controller(PhysicalDevice):
    """
    ESP32-based IoT controller with WiFi connectivity
    Supports multiple sensors and actuators
    """
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        config = self.get_default_config()
        if location:
            config['location'] = location
        
        # Create WiFi connection and JSON processor
        factory = ConnectionFactory()
        connection = factory.create_connection(
            "wifi",
            device_id,
            config['connection']
        )
        processor = factory.create_processor(
            "json",
            config['processor']
        )
        
        super().__init__(device_id, connection, processor, config)
        
        # ESP32-specific state
        self.last_rssi = -50
        self.wifi_connected = True
        self.uptime = 0
    
    def get_type(self) -> str:
        return "esp32_controller"
    
    def supports_write(self) -> bool:
        return True
    
    def get_writable_properties(self) -> List[str]:
        return ["relay_states", "pwm_outputs", "wifi_config", "deep_sleep"]
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "esp32_controller",
            "name": "ESP32 IoT Controller",
            "description": "Versatile WiFi-enabled microcontroller with sensors and actuators",
            "version": "3.0",
            "manufacturer": "Espressif Systems",
            "capabilities": ["wifi", "sensors", "actuators", "deep_sleep", "ota_updates"],
            "writable_properties": ["relay_states", "pwm_outputs", "wifi_config", "deep_sleep"],
            "readable_properties": ["sensors", "relay_states", "wifi_status", "system_info"],
            "connection_type": "wifi",
            "protocol": "http_json"
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 20,
            "location_base": {"lat": 40.7484, "lon": -73.9857},
            "location_offset": 0.003,
            "default_instances": 3,
            "device_type": "esp32_controller",
            "connection": {
                "host": "192.168.1.100",  # Will be configured per instance
                "port": 80,
                "protocol": "http",
                "timeout": 10.0,
                "read_endpoint": "/api/status",
                "write_endpoint": "/api/command",
                "health_endpoint": "/api/health"
            },
            "processor": {
                "encoding": "utf-8",
                "validate_schema": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sensors": {"type": "object"},
                        "actuators": {"type": "object"},
                        "system": {"type": "object"}
                    },
                    "required": ["sensors", "system"]
                }
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Read ESP32 status via HTTP"""
        reading = await super().read_data()
        
        if reading.get('status') == 'active' and 'data' in reading:
            data = reading['data']
            
            # Add ESP32-specific processing
            if 'system' in data:
                system = data['system']
                
                # Track WiFi status
                if 'wifi' in system:
                    wifi_info = system['wifi']
                    self.last_rssi = wifi_info.get('rssi', self.last_rssi)
                    self.wifi_connected = wifi_info.get('connected', True)
                    
                    # Add WiFi quality assessment
                    reading['data']['wifi_quality'] = self._assess_wifi_quality(self.last_rssi)
                
                # Track uptime
                self.uptime = system.get('uptime', 0)
                reading['data']['uptime_hours'] = round(self.uptime / 3600, 1)
                
                # Add memory status
                if 'free_heap' in system and 'total_heap' in system:
                    free_heap = system['free_heap']
                    total_heap = system['total_heap']
                    memory_usage = ((total_heap - free_heap) / total_heap) * 100
                    reading['data']['memory_usage_percent'] = round(memory_usage, 1)
            
            # Process sensor data
            if 'sensors' in data:
                sensors = data['sensors']
                
                # Add derived sensor values
                if 'temperature' in sensors and 'humidity' in sensors:
                    # Calculate dew point
                    temp = sensors['temperature']
                    humidity = sensors['humidity']
                    dew_point = self._calculate_dew_point(temp, humidity)
                    reading['data']['sensors']['dew_point'] = dew_point
        
        return reading
    
    def _assess_wifi_quality(self, rssi: int) -> str:
        """Assess WiFi signal quality based on RSSI"""
        if rssi >= -30:
            return "excellent"
        elif rssi >= -50:
            return "very_good"
        elif rssi >= -60:
            return "good"
        elif rssi >= -70:
            return "fair"
        else:
            return "poor"
    
    def _calculate_dew_point(self, temperature: float, humidity: float) -> float:
        """Calculate dew point from temperature and humidity"""
        # Magnus formula approximation
        a = 17.27
        b = 237.7
        
        alpha = ((a * temperature) / (b + temperature)) + math.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return round(dew_point, 1)
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ESP32 commands"""
        # Validate relay commands
        if "relay_states" in command:
            relays = command["relay_states"]
            if isinstance(relays, dict):
                for relay_id, state in relays.items():
                    if not isinstance(state, bool):
                        return {
                            "success": False,
                            "error": f"Relay {relay_id} state must be boolean",
                            "device_id": self.device_id
                        }
        
        # Validate PWM commands
        if "pwm_outputs" in command:
            pwm_outputs = command["pwm_outputs"]
            if isinstance(pwm_outputs, dict):
                for pin, value in pwm_outputs.items():
                    if not isinstance(value, (int, float)) or not 0 <= value <= 255:
                        return {
                            "success": False,
                            "error": f"PWM value for pin {pin} must be 0-255",
                            "device_id": self.device_id
                        }
        
        # Handle deep sleep command
        if "deep_sleep" in command:
            sleep_time = command["deep_sleep"]
            if isinstance(sleep_time, (int, float)) and sleep_time > 0:
                # ESP32 will go to sleep and disconnect
                result = await super().write_data(command)
                if result.get('success'):
                    # Mark connection as expected to disconnect
                    logger.info(f"ESP32 {self.device_id} entering deep sleep for {sleep_time} seconds")
                return result
        
        return await super().write_data(command)
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "relay_states": {
                    "type": "object",
                    "patternProperties": {
                        "^relay_[0-9]+$": {"type": "boolean"}
                    },
                    "description": "Set relay states (relay_1: true/false)"
                },
                "pwm_outputs": {
                    "type": "object",
                    "patternProperties": {
                        "^pin_[0-9]+$": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 255
                        }
                    },
                    "description": "Set PWM output values (pin_13: 0-255)"
                },
                "wifi_config": {
                    "type": "object",
                    "properties": {
                        "ssid": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "description": "Update WiFi configuration"
                },
                "deep_sleep": {
                    "type": "number",
                    "minimum": 1,
                    "description": "Enter deep sleep for specified seconds"
                }
            }
        }



