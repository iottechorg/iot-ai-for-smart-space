import asyncio
import random
from datetime import datetime
from typing import Dict, Any, List

from core.device_interface import DeviceInterface

class SmartThermostat(DeviceInterface):
    """Smart thermostat with temperature control capabilities"""
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        self.device_id = device_id
        self.location = location or {"lat": 0.0, "lon": 0.0}
        self.is_active = True
        self.config = self.get_default_config()
        
        # Device state
        self.current_temperature = 22.0  # Celsius
        self.target_temperature = 22.0
        self.mode = "auto"  # auto, heat, cool, off
        self.fan_speed = "auto"  # auto, low, medium, high
        self.is_heating = False
        self.is_cooling = False
        self.last_command_time = None
    
    def get_id(self) -> str:
        return self.device_id
    
    def get_type(self) -> str:
        return "smart_thermostat"
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "smart_thermostat",
            "name": "Smart Climate Controller",
            "description": "Intelligent thermostat with precise climate control",
            "version": "3.1",
            "manufacturer": "ClimaTech Systems",
            "capabilities": ["temperature_control", "humidity_monitoring", "scheduling", "energy_optimization"],
            "writable_properties": ["target_temperature", "mode", "fan_speed", "schedule"],
            "readable_properties": ["current_temperature", "target_temperature", "humidity", "mode", "status"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 30,
            "location_base": {"lat": 40.7580, "lon": -73.9855},
            "location_offset": 0.005,
            "default_instances": 3,
            "min_temperature": 10,
            "max_temperature": 35,
            "default_target": 22,
            "temperature_tolerance": 0.5
        }
    
    def get_writable_properties(self) -> List[str]:
        return ["target_temperature", "mode", "fan_speed", "schedule"]
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_temperature": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 35,
                    "description": "Set target temperature in Celsius"
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "heat", "cool", "off"],
                    "description": "Set thermostat mode"
                },
                "fan_speed": {
                    "type": "string",
                    "enum": ["auto", "low", "medium", "high"],
                    "description": "Set fan speed"
                },
                "schedule": {
                    "type": "object",
                    "properties": {
                        "weekday_temp": {"type": "number"},
                        "weekend_temp": {"type": "number"},
                        "night_temp": {"type": "number"}
                    },
                    "description": "Set temperature schedule"
                }
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Read current thermostat status"""
        if not self.is_active:
            return {}
        
        # Simulate temperature changes based on heating/cooling
        if self.is_heating:
            self.current_temperature += random.uniform(0.1, 0.3)
        elif self.is_cooling:
            self.current_temperature -= random.uniform(0.1, 0.3)
        else:
            # Natural temperature drift
            self.current_temperature += random.uniform(-0.1, 0.1)
        
        # Update heating/cooling status based on target
        tolerance = self.config.get("temperature_tolerance", 0.5)
        if self.mode == "auto":
            if self.current_temperature < self.target_temperature - tolerance:
                self.is_heating = True
                self.is_cooling = False
            elif self.current_temperature > self.target_temperature + tolerance:
                self.is_heating = False
                self.is_cooling = True
            else:
                self.is_heating = False
                self.is_cooling = False
        elif self.mode == "heat":
            self.is_heating = self.current_temperature < self.target_temperature
            self.is_cooling = False
        elif self.mode == "cool":
            self.is_heating = False
            self.is_cooling = self.current_temperature > self.target_temperature
        else:  # mode == "off"
            self.is_heating = False
            self.is_cooling = False
        
        # Simulate humidity
        humidity = random.normalvariate(45, 10)
        humidity = max(20, min(80, humidity))
        
        return {
            "device_id": self.device_id,
            "device_type": self.get_type(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "active" if self.is_active else "inactive",
            "current_temperature": round(self.current_temperature, 1),
            "target_temperature": self.target_temperature,
            "humidity": round(humidity, 1),
            "mode": self.mode,
            "fan_speed": self.fan_speed,
            "is_heating": self.is_heating,
            "is_cooling": self.is_cooling,
            "last_command_time": self.last_command_time.isoformat() + "Z" if self.last_command_time else None,
            "location": self.location
        }
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute thermostat control commands"""
        try:
            executed_commands = []
            
            # Handle target temperature
            if "target_temperature" in command:
                temp = float(command["target_temperature"])
                min_temp = self.config.get("min_temperature", 10)
                max_temp = self.config.get("max_temperature", 35)
                
                if min_temp <= temp <= max_temp:
                    self.target_temperature = temp
                    executed_commands.append(f"target_temperature: {temp}°C")
                else:
                    return {
                        "success": False,
                        "error": f"Temperature must be between {min_temp}°C and {max_temp}°C",
                        "device_id": self.device_id
                    }
            
            # Handle mode change
            if "mode" in command:
                mode = command["mode"].lower()
                valid_modes = ["auto", "heat", "cool", "off"]
                
                if mode in valid_modes:
                    self.mode = mode
                    executed_commands.append(f"mode: {mode}")
                else:
                    return {
                        "success": False,
                        "error": f"Mode must be one of: {valid_modes}",
                        "device_id": self.device_id
                    }
            
            # Handle fan speed
            if "fan_speed" in command:
                fan_speed = command["fan_speed"].lower()
                valid_speeds = ["auto", "low", "medium", "high"]
                
                if fan_speed in valid_speeds:
                    self.fan_speed = fan_speed
                    executed_commands.append(f"fan_speed: {fan_speed}")
                else:
                    return {
                        "success": False,
                        "error": f"Fan speed must be one of: {valid_speeds}",
                        "device_id": self.device_id
                    }
            
            # Handle schedule
            if "schedule" in command:
                schedule = command["schedule"]
                # Here you would implement scheduling logic
                executed_commands.append(f"schedule: {schedule}")
            
            self.last_command_time = datetime.utcnow()
            
            return {
                "success": True,
                "executed_commands": executed_commands,
                "current_state": {
                    "current_temperature": round(self.current_temperature, 1),
                    "target_temperature": self.target_temperature,
                    "mode": self.mode,
                    "fan_speed": self.fan_speed
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
        """Configure thermostat parameters"""
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        return True
    
    async def health_check(self) -> bool:
        """Check thermostat health"""
        return random.random() < 0.98