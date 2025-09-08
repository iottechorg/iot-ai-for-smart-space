# gateway/app/devices/noise_sensor.py
# Another example device - just save this file and it will be auto-discovered!

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from core.device_interface import DeviceInterface

class NoiseSensor(DeviceInterface):
    """Noise level sensor - demonstrates true plug-and-play capability"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "noise_sensor"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Device information"""
        return {
            "type": "noise_sensor",
            "name": "Environmental Noise Monitor",
            "description": "Monitors ambient noise levels and sound frequency analysis",
            "version": "1.3",
            "manufacturer": "SoundTech Solutions",
            "capabilities": ["decibel_level", "frequency_analysis", "peak_detection"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "sample_interval": 20,
            "location_base": {"lat": 40.7505, "lon": -73.9934},
            "location_offset": 0.008,
            "default_instances": 4,
            "noise_threshold": 70,  # dB
            "frequency_range": [20, 20000],  # Hz
            "peak_detection_sensitivity": 0.8
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Simulate noise level data reading"""
        if not self.is_active:
            return {}
        
        # Simulate noise levels (higher during day, lower at night)
        current_hour = datetime.now().hour
        base_noise = 45 if 22 <= current_hour or current_hour <= 6 else 55
        
        # Add random variations
        noise_level = max(30, base_noise + random.normalvariate(0, 8))
        
        # Simulate frequency analysis
        low_freq = random.normalvariate(40, 10)   # 20-250 Hz
        mid_freq = random.normalvariate(45, 12)   # 250-4000 Hz
        high_freq = random.normalvariate(35, 8)   # 4000+ Hz
        
        # Detect peaks
        peak_detected = noise_level > self.config.get("noise_threshold", 70)
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "noise_level": round(noise_level, 1),
            "low_frequency": round(low_freq, 1),
            "mid_frequency": round(mid_freq, 1),
            "high_frequency": round(high_freq, 1),
            "peak_detected": peak_detected,
            "time_of_day": current_hour,
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
        return random.random() < 0.97


