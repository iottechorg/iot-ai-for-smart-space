import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from core.device_interface import DeviceInterface

class AirQualitySensor(DeviceInterface):
    """Air quality sensor - automatically discovered and configured"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "air_quality_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Device information - completely self-contained"""
        return {
            "type": "air_quality_sensor",
            "name": "Air Quality Monitor",
            "description": "Monitors PM2.5, PM10, CO2, and other air quality metrics",
            "version": "2.1",
            "manufacturer": "EcoSense Technologies",
            "capabilities": ["pm25", "pm10", "co2", "humidity", "temperature"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Default configuration - no hardcoded values in discovery system"""
        return {
            "sample_interval": 60,  # Air quality changes slowly
            "location_base": {"lat": 40.7589, "lon": -73.9851},
            "location_offset": 0.02,
            "default_instances": 2,
            "pm25_threshold": 35,
            "pm10_threshold": 50,
            "co2_threshold": 1000,
            "temperature_unit": "celsius"
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Simulate air quality data reading"""
        if not self.is_active:
            return {}
        
        # Simulate air quality metrics
        pm25 = max(0, random.normalvariate(25, 10))
        pm10 = max(0, random.normalvariate(35, 15))
        co2 = max(300, random.normalvariate(450, 100))
        humidity = max(0, min(100, random.normalvariate(65, 15)))
        temperature = random.normalvariate(22, 8)
        
        # Calculate air quality index (simplified)
        aqi = max(pm25 * 2, pm10 * 1.5, (co2 - 300) / 10)
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "co2": round(co2, 1),
            "humidity": round(humidity, 1),
            "temperature": round(temperature, 1),
            "aqi": round(aqi, 1),
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
        return random.random() < 0.92  # Slightly lower reliability