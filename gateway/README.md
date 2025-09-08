

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