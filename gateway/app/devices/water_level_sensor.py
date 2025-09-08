# gateway/app/devices/water_level_sensor.py (Updated)
import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from core.device_interface import DeviceInterface

class WaterLevelSensor(DeviceInterface):
    """Water level sensor implementation with self-contained configuration"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "water_level_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "water_level_sensor",
            "name": "Water Level Monitor",
            "description": "Monitors water level in reservoirs and drainage systems",
            "version": "1.0",
            "manufacturer": "AquaTech Solutions",
            "capabilities": ["water_level", "flow_rate", "quality_index"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 45,
            "location_base": {"lat": 40.7500, "lon": -73.9850},
            "location_offset": 0.005,
            "default_instances": 2,
            "max_level": 100,
            "min_level": 0,
            "alert_threshold": 80
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Simulate water level data reading"""
        if not self.is_active:
            return {}
        
        # Simulate water level with some variation
        base_level = random.normalvariate(50, 15)
        water_level = max(0, min(100, base_level))
        
        flow_rate = random.normalvariate(25, 8)
        quality_index = random.randint(70, 95)
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "water_level": round(water_level, 2),
            "flow_rate": round(flow_rate, 2),
            "quality_index": quality_index,
            "location": self.location
        }
    
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Configure sensor parameters"""
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        return True
    
    async def health_check(self) -> bool:
        """Check sensor health"""
        return random.random() < 0.95
