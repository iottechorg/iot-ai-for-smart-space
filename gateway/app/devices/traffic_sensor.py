# gateway/app/devices/traffic_sensor.py (Updated to implement new interface)
import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from core.device_interface import DeviceInterface

class TrafficSensor(DeviceInterface):
    """Traffic sensor implementation with self-contained configuration"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "traffic_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "traffic_sensor",
            "name": "Traffic Flow Sensor",
            "description": "Monitors vehicle count, speed, and lane occupancy",
            "version": "1.0",
            "manufacturer": "SmartCity Systems",
            "capabilities": ["vehicle_count", "speed_monitoring", "lane_occupancy"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 30,
            "location_base": {"lat": 40.7128, "lon": -74.0060},
            "location_offset": 0.01,
            "default_instances": 3,
            "vehicle_detection_threshold": 0.75,
            "lanes": 4,
            "speed_limit": 60
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Simulate traffic data reading"""
        if not self.is_active:
            return {}
        
        # Simulate vehicle count (higher during peak hours)
        current_hour = datetime.now().hour
        peak_factor = 1.0
        if 7 <= current_hour <= 9 or 16 <= current_hour <= 18:  # Peak hours
            peak_factor = 2.0
        
        vehicle_count = int(random.normalvariate(20 * peak_factor, 5))
        avg_speed = max(5, random.normalvariate(60 - (vehicle_count / 2), 10))
        lane_occupancy = min(100, max(0, vehicle_count * 5))
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "vehicle_count": vehicle_count,
            "average_speed": round(avg_speed, 1),
            "lane_occupancy": round(lane_occupancy, 1),
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