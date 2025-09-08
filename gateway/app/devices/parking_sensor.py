# gateway/app/devices/parking_sensor.py
# Yet another example - demonstrates how easy it is to add new devices

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from core.device_interface import DeviceInterface

class ParkingSensor(DeviceInterface):
    """Smart parking sensor - automatically discovered and managed"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
        self.occupied = random.choice([True, False])
        self.last_change = datetime.utcnow()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "parking_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Device information"""
        return {
            "type": "parking_sensor",
            "name": "Smart Parking Space Monitor",
            "description": "Monitors parking space occupancy and duration",
            "version": "2.0",
            "manufacturer": "ParkSmart Technologies",
            "capabilities": ["occupancy_detection", "duration_tracking", "vehicle_classification"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "sample_interval": 15,  # Parking changes quickly
            "location_base": {"lat": 40.7614, "lon": -73.9776},
            "location_offset": 0.001,  # Close together for parking spots
            "default_instances": 10,  # Many parking spaces
            "detection_sensitivity": 0.85,
            "max_parking_duration": 7200,  # 2 hours in seconds
            "spot_type": "street"  # street, garage, lot
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Simulate parking sensor data"""
        if not self.is_active:
            return {}
        
        # Randomly change occupancy status
        if random.random() < 0.1:  # 10% chance of status change
            self.occupied = not self.occupied
            self.last_change = datetime.utcnow()
        
        # Calculate duration
        duration = (datetime.utcnow() - self.last_change).total_seconds()
        
        # Simulate vehicle classification if occupied
        vehicle_type = None
        if self.occupied:
            vehicle_type = random.choice(["car", "truck", "motorcycle", "van"])
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "occupied": self.occupied,
            "duration": round(duration, 1),
            "vehicle_type": vehicle_type,
            "spot_type": self.config.get("spot_type", "street"),
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
        return random.random() < 0.98

