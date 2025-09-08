# gateway/app/core/device_registry.py
from typing import Dict, List, Optional
from .device_interface import DeviceInterface

class DeviceRegistry:
    """
    Registry for IoT devices
    Single Responsibility: Manages device registration and retrieval
    """
    
    def __init__(self):
        self.devices: Dict[str, DeviceInterface] = {}
    
    def register_device(self, device: DeviceInterface):
        """Register a new device"""
        self.devices[device.get_id()] = device
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device"""
        if device_id in self.devices:
            del self.devices[device_id]
            return True
        return False
    
    def get_device(self, device_id: str) -> Optional[DeviceInterface]:
        """Get a device by ID"""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> List[DeviceInterface]:
        """Get all registered devices"""
        return list(self.devices.values())
    
    def get_all_device_ids(self) -> List[str]:
        """Get all registered device IDs"""
        return list(self.devices.keys())
    
    def get_devices_by_type(self, device_type: str) -> List[DeviceInterface]:
        """Get all devices of a specific type"""
        return [d for d in self.devices.values() if d.get_type() == device_type]