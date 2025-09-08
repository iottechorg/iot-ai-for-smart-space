# gateway/app/devices/smart_irrigation.py
# Example of an irrigation controller

import asyncio
import random
from datetime import datetime, time
from typing import Dict, Any, List

from core.device_interface import DeviceInterface

class SmartIrrigation(DeviceInterface):
    """Smart irrigation controller for garden/landscape management"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
        
        # Device state
        self.zones = {
            "zone_1": {"active": False, "duration": 0, "start_time": None},
            "zone_2": {"active": False, "duration": 0, "start_time": None},
            "zone_3": {"active": False, "duration": 0, "start_time": None},
            "zone_4": {"active": False, "duration": 0, "start_time": None}
        }
        self.soil_moisture = {f"zone_{i}": random.randint(30, 70) for i in range(1, 5)}
        self.water_pressure = 45.0  # PSI
        self.total_water_used = 0.0  # liters
        self.last_command_time = None
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "smart_irrigation"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "smart_irrigation",
            "name": "Smart Irrigation Controller",
            "description": "Multi-zone irrigation system with soil moisture monitoring",
            "version": "2.5",
            "manufacturer": "AquaGarden Tech",
            "capabilities": ["zone_control", "moisture_monitoring", "scheduling", "water_conservation"],
            "writable_properties": ["zone_control", "schedule", "duration", "manual_override"],
            "readable_properties": ["zone_status", "soil_moisture", "water_pressure", "water_usage"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 120,  # Slower sampling for irrigation
            "location_base": {"lat": 40.7505, "lon": -73.9934},
            "location_offset": 0.01,
            "default_instances": 2,
            "max_duration": 3600,  # 1 hour max
            "min_soil_moisture": 30,
            "water_flow_rate": 15  # liters per minute per zone
        }
    
    def get_writable_properties(self) -> List[str]:
        return ["zone_control", "schedule", "duration", "manual_override"]
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "zone_control": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string", "enum": ["zone_1", "zone_2", "zone_3", "zone_4", "all"]},
                        "action": {"type": "string", "enum": ["start", "stop"]},
                        "duration": {"type": "integer", "minimum": 60, "maximum": 3600}
                    },
                    "required": ["zone", "action"],
                    "description": "Control irrigation zones"
                },
                "schedule": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string", "enum": ["zone_1", "zone_2", "zone_3", "zone_4"]},
                        "days": {"type": "array", "items": {"type": "string"}},
                        "start_time": {"type": "string", "format": "time"},
                        "duration": {"type": "integer", "minimum": 60, "maximum": 3600}
                    },
                    "description": "Set irrigation schedule"
                },
                "manual_override": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "reason": {"type": "string"}
                    },
                    "description": "Enable/disable manual override mode"
                }
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Read irrigation system status"""
        if not self.is_active:
            return {}
        
        # Update zone status and soil moisture
        current_time = datetime.utcnow()
        water_used_this_cycle = 0
        
        for zone_id, zone_data in self.zones.items():
            if zone_data["active"] and zone_data["start_time"]:
                # Check if duration has elapsed
                elapsed = (current_time - zone_data["start_time"]).total_seconds()
                if elapsed >= zone_data["duration"]:
                    zone_data["active"] = False
                    zone_data["start_time"] = None
                    zone_data["duration"] = 0
                else:
                    # Calculate water usage
                    flow_rate = self.config.get("water_flow_rate", 15)  # L/min
                    water_used_this_cycle += (elapsed / 60) * flow_rate
                    
                    # Increase soil moisture while watering
                    if zone_id in self.soil_moisture:
                        self.soil_moisture[zone_id] = min(100, self.soil_moisture[zone_id] + 2)
            else:
                # Gradually decrease soil moisture when not watering
                if zone_id in self.soil_moisture:
                    self.soil_moisture[zone_id] = max(0, self.soil_moisture[zone_id] - random.uniform(0.1, 0.5))
        
        self.total_water_used += water_used_this_cycle
        
        # Simulate water pressure variations
        self.water_pressure += random.uniform(-2, 2)
        self.water_pressure = max(30, min(60, self.water_pressure))
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "zones": self.zones,
            "soil_moisture": {k: round(v, 1) for k, v in self.soil_moisture.items()},
            "water_pressure": round(self.water_pressure, 1),
            "total_water_used": round(self.total_water_used, 2),
            "active_zones": [zone for zone, data in self.zones.items() if data["active"]],
            "last_command_time": self.last_command_time.isoformat() + "Z" if self.last_command_time else None,
            "location": self.location
        }
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute irrigation control commands"""
        try:
            executed_commands = []
            
            # Handle zone control
            if "zone_control" in command:
                zone_cmd = command["zone_control"]
                zone = zone_cmd.get("zone")
                action = zone_cmd.get("action")
                duration = zone_cmd.get("duration", 600)  # Default 10 minutes
                
                if zone == "all":
                    zones_to_control = list(self.zones.keys())
                elif zone in self.zones:
                    zones_to_control = [zone]
                else:
                    return {
                        "success": False,
                        "error": f"Invalid zone: {zone}",
                        "device_id": self.device_id
                    }
                
                if action == "start":
                    for z in zones_to_control:
                        self.zones[z]["active"] = True
                        self.zones[z]["duration"] = duration
                        self.zones[z]["start_time"] = datetime.utcnow()
                    executed_commands.append(f"started {len(zones_to_control)} zone(s) for {duration}s")
                    
                elif action == "stop":
                    for z in zones_to_control:
                        self.zones[z]["active"] = False
                        self.zones[z]["start_time"] = None
                        self.zones[z]["duration"] = 0
                    executed_commands.append(f"stopped {len(zones_to_control)} zone(s)")
                    
                else:
                    return {
                        "success": False,
                        "error": f"Invalid action: {action}. Use 'start' or 'stop'",
                        "device_id": self.device_id
                    }
            
            # Handle schedule (simplified implementation)
            if "schedule" in command:
                schedule = command["schedule"]
                zone = schedule.get("zone")
                if zone and zone in self.zones:
                    executed_commands.append(f"schedule set for {zone}: {schedule}")
                else:
                    return {
                        "success": False,
                        "error": f"Invalid zone for schedule: {zone}",
                        "device_id": self.device_id
                    }
            
            # Handle manual override
            if "manual_override" in command:
                override = command["manual_override"]
                enabled = override.get("enabled", False)
                executed_commands.append(f"manual override: {'enabled' if enabled else 'disabled'}")
            
            self.last_command_time = datetime.utcnow()
            
            return {
                "success": True,
                "executed_commands": executed_commands,
                "current_state": {
                    "active_zones": [zone for zone, data in self.zones.items() if data["active"]],
                    "soil_moisture": self.soil_moisture,
                    "water_pressure": round(self.water_pressure, 1)
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
        """Configure irrigation parameters"""
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        return True
    
    async def health_check(self) -> bool:
        """Check irrigation system health"""
        return random.random() < 0.96