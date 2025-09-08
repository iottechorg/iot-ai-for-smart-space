# gateway/app/devices/smart_light.py
# Example actuator device with read/write capabilities

import asyncio
import random
from datetime import datetime
from typing import Dict, Any, List

from core.device_interface import DeviceInterface

class SmartLight(DeviceInterface):
    """Smart light with both sensor and actuator capabilities"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
        
        # Device state
        self.is_on = False
        self.brightness = 0  # 0-100
        self.color = {"r": 255, "g": 255, "b": 255}  # RGB
        self.last_command_time = None
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "smart_light"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "smart_light",
            "name": "Smart LED Light",
            "description": "Controllable LED light with brightness and color control",
            "version": "2.0",
            "manufacturer": "LightTech Solutions",
            "capabilities": ["illumination", "color_control", "dimming", "energy_monitoring"],
            "writable_properties": ["power", "brightness", "color", "schedule"],
            "readable_properties": ["power", "brightness", "color", "energy_consumption", "status"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 60,
            "location_base": {"lat": 40.7590, "lon": -73.9845},
            "location_offset": 0.003,
            "default_instances": 5,
            "max_brightness": 100,
            "power_consumption_per_brightness": 0.8,  # watts per brightness unit
            "default_color": {"r": 255, "g": 255, "b": 255}
        }
    
    def get_writable_properties(self) -> List[str]:
        return ["power", "brightness", "color", "schedule"]
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "power": {
                    "type": "boolean",
                    "description": "Turn light on/off"
                },
                "brightness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Set brightness level (0-100)"
                },
                "color": {
                    "type": "object",
                    "properties": {
                        "r": {"type": "integer", "minimum": 0, "maximum": 255},
                        "g": {"type": "integer", "minimum": 0, "maximum": 255},
                        "b": {"type": "integer", "minimum": 0, "maximum": 255}
                    },
                    "required": ["r", "g", "b"],
                    "description": "Set RGB color"
                },
                "schedule": {
                    "type": "object",
                    "properties": {
                        "on_time": {"type": "string", "format": "time"},
                        "off_time": {"type": "string", "format": "time"}
                    },
                    "description": "Set automatic on/off schedule"
                }
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Read current light status and energy consumption"""
        if not self.is_active:
            return {}
        
        # Calculate energy consumption
        power_consumption = 0
        if self.is_on:
            power_consumption = self.brightness * self.config.get("power_consumption_per_brightness", 0.8)
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "power": self.is_on,
            "brightness": self.brightness,
            "color": self.color,
            "power_consumption": round(power_consumption, 2),
            "last_command_time": self.last_command_time.isoformat() + "Z" if self.last_command_time else None,
            "location": self.location
        }
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute commands to control the light"""
        try:
            executed_commands = []
            
            # Handle power command
            if "power" in command:
                power_state = bool(command["power"])
                self.is_on = power_state
                if not power_state:
                    self.brightness = 0
                executed_commands.append(f"power: {'on' if power_state else 'off'}")
            
            # Handle brightness command
            if "brightness" in command:
                brightness = int(command["brightness"])
                if 0 <= brightness <= 100:
                    self.brightness = brightness
                    if brightness > 0:
                        self.is_on = True
                    executed_commands.append(f"brightness: {brightness}")
                else:
                    return {
                        "success": False,
                        "error": "Brightness must be between 0 and 100",
                        "device_id": self.device_id
                    }
            
            # Handle color command
            if "color" in command:
                color = command["color"]
                if isinstance(color, dict) and all(k in color for k in ["r", "g", "b"]):
                    if all(0 <= color[k] <= 255 for k in ["r", "g", "b"]):
                        self.color = {
                            "r": int(color["r"]),
                            "g": int(color["g"]),
                            "b": int(color["b"])
                        }
                        executed_commands.append(f"color: RGB({color['r']}, {color['g']}, {color['b']})")
                    else:
                        return {
                            "success": False,
                            "error": "RGB values must be between 0 and 255",
                            "device_id": self.device_id
                        }
                else:
                    return {
                        "success": False,
                        "error": "Color must be an object with r, g, b properties",
                        "device_id": self.device_id
                    }
            
            # Handle schedule command (simplified)
            if "schedule" in command:
                schedule = command["schedule"]
                # Here you would implement scheduling logic
                executed_commands.append(f"schedule: {schedule}")
            
            self.last_command_time = datetime.utcnow()
            
            return {
                "success": True,
                "executed_commands": executed_commands,
                "current_state": {
                    "power": self.is_on,
                    "brightness": self.brightness,
                    "color": self.color
                },
                "device_id": self.device_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "device_id": self.device_id
            }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure light parameters"""
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        return True
    
    async def health_check(self) -> bool:
        """Check light health"""
        return random.random() < 0.95


# gateway/app/devices/air_quality_sensor.py
# Just save this file and the gateway will automatically discover and start it!


# USAGE EXAMPLES:

# 1. To add a new device type:
#    - Create a new .py file in the devices/ folder
#    - Implement DeviceInterface with the required methods
#    - Define get_device_info() and get_default_config()
#    - Save the file - it will be automatically discovered!

# 2. The system will:
#    - Automatically detect the new device file
#    - Import and validate the device class
#    - Generate configuration based on get_default_config()
#    - Create instances according to default_instances
#    - Start monitoring the new devices
#    - Update the configuration file

# 3. Hot-plugging example:
#    While the gateway is running, create this simple device:

"""
# gateway/app/devices/temperature_sensor.py

from datetime import datetime
from typing import Dict, Any
import random
from core.device_interface import DeviceInterface

class TemperatureSensor(DeviceInterface):
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "temperature_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "temperature_sensor",
            "name": "Temperature Monitor",
            "description": "Basic temperature monitoring",
            "version": "1.0",
            "manufacturer": "TempTech",
            "capabilities": ["temperature"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 30,
            "location_base": {"lat": 40.7580, "lon": -73.9855},
            "location_offset": 0.01,
            "default_instances": 1,
            "unit": "celsius"
        }
    
    async def read_data(self) -> Dict[str, Any]:
        if not self.is_active:
            return {}
        
        temperature = random.normalvariate(20, 5)
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active",
            "temperature": round(temperature, 1),
            "unit": self.config.get("unit", "celsius"),
            "location": self.location
        }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        return True
    
    async def health_check(self) -> bool:
        return True
"""

# Save this file and within 2 seconds, the gateway will:
# 1. Detect the new file
# 2. Import and validate the TemperatureSensor class
# 3. Create instances according to default_instances
# 4. Register them with the gateway
# 5. Start monitoring automatically
# 6. Update the configuration file

# The system is now truly plug-and-play with zero hardcoded device types!