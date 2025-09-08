# gateway/app/devices/arduino_sensor.py
from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime
from core.physical_device import PhysicalDevice
from core.connection_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class ArduinoSensor(PhysicalDevice):
    """
    Arduino-based environmental sensor
    Connects via Serial/USB and uses JSON protocol
    """
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        # Get default config first
        config = self.get_default_config()
        if location:
            config['location'] = location
        
        # Create connection and processor
        factory = ConnectionFactory()
        connection = factory.create_connection(
            "serial", 
            device_id, 
            config['connection']
        )
        processor = factory.create_processor(
            "json",
            config['processor']
        )
        
        # Initialize physical device
        super().__init__(device_id, connection, processor, config)
    
    def get_type(self) -> str:
        return "arduino_sensor"
    
    def supports_write(self) -> bool:
        return True
    
    def get_writable_properties(self) -> List[str]:
        return ["led_state", "sampling_rate", "calibration"]
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "arduino_sensor",
            "name": "Arduino Environmental Sensor",
            "description": "Multi-sensor Arduino device for environmental monitoring",
            "version": "2.1",
            "manufacturer": "DIY Electronics",
            "capabilities": ["temperature", "humidity", "light", "led_control"],
            "writable_properties": ["led_state", "sampling_rate", "calibration"],
            "readable_properties": ["temperature", "humidity", "light_level", "battery_voltage"],
            "connection_type": "serial",
            "protocol": "json"
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 15,
            "location_base": {"lat": 40.7589, "lon": -73.9851},
            "location_offset": 0.002,
            "default_instances": 2,
            "device_type": "arduino_sensor",
            "connection": {
                "port": "/dev/ttyUSB0",  # Will be configured per instance
                "baudrate": 9600,
                "timeout": 2.0,
                "read_timeout": 5.0
            },
            "processor": {
                "encoding": "utf-8",
                "validate_schema": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "humidity": {"type": "number"},
                        "light_level": {"type": "number"},
                        "battery_voltage": {"type": "number"}
                    },
                    "required": ["temperature", "humidity"]
                }
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Override to add Arduino-specific processing"""
        reading = await super().read_data()
        
        if reading.get('status') == 'active' and 'data' in reading:
            # Add computed values
            data = reading['data']
            
            # Calculate heat index if we have temp and humidity
            if 'temperature' in data and 'humidity' in data:
                heat_index = self._calculate_heat_index(
                    data['temperature'], 
                    data['humidity']
                )
                reading['data']['heat_index'] = heat_index
            
            # Add battery status
            if 'battery_voltage' in data:
                battery_pct = self._calculate_battery_percentage(data['battery_voltage'])
                reading['data']['battery_percentage'] = battery_pct
                reading['data']['battery_status'] = self._get_battery_status(battery_pct)
        
        return reading
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index from temperature and humidity"""
        # Convert to Fahrenheit for calculation
        temp_f = (temp_c * 9/5) + 32
        
        # Simplified heat index calculation
        if temp_f < 80:
            return temp_c
        
        hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity 
              - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2 
              - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity 
              + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2)
        
        # Convert back to Celsius
        return round((hi - 32) * 5/9, 1)
    
    def _calculate_battery_percentage(self, voltage: float) -> int:
        """Calculate battery percentage from voltage"""
        # Typical Li-ion battery: 3.0V (empty) to 4.2V (full)
        min_voltage = 3.0
        max_voltage = 4.2
        
        percentage = ((voltage - min_voltage) / (max_voltage - min_voltage)) * 100
        return max(0, min(100, int(percentage)))
    
    def _get_battery_status(self, percentage: int) -> str:
        """Get battery status string"""
        if percentage > 75:
            return "excellent"
        elif percentage > 50:
            return "good"
        elif percentage > 25:
            return "fair"
        elif percentage > 10:
            return "low"
        else:
            return "critical"
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Arduino-specific commands"""
        # Validate commands before sending
        if "led_state" in command:
            if command["led_state"] not in [True, False, "on", "off"]:
                return {
                    "success": False,
                    "error": "led_state must be true/false or 'on'/'off'",
                    "device_id": self.device_id
                }
        
        if "sampling_rate" in command:
            rate = command["sampling_rate"]
            if not isinstance(rate, (int, float)) or rate < 1 or rate > 3600:
                return {
                    "success": False,
                    "error": "sampling_rate must be between 1 and 3600 seconds",
                    "device_id": self.device_id
                }
        
        # Send command to Arduino
        return await super().write_data(command)
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "led_state": {
                    "type": ["boolean", "string"],
                    "enum": [True, False, "on", "off"],
                    "description": "Turn LED on or off"
                },
                "sampling_rate": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 3600,
                    "description": "Set sampling rate in seconds"
                },
                "calibration": {
                    "type": "object",
                    "properties": {
                        "temperature_offset": {"type": "number"},
                        "humidity_offset": {"type": "number"}
                    },
                    "description": "Calibration offsets for sensors"
                }
            }
        }


# gateway/app/devices/modbus_energy_meter.py
from typing import Dict, Any, List
import logging
from datetime import datetime
from core.physical_device import PhysicalDevice
from core.connection_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class ModbusEnergyMeter(PhysicalDevice):
    """
    Industrial energy meter with Modbus RTU communication
    Reads power consumption, voltage, current, etc.
    """
    
    def __init__(self, device_id: str, location: Dict[str, float] = None):
        config = self.get_default_config()
        if location:
            config['location'] = location
        
        # Create connection and processor
        factory = ConnectionFactory()
        connection = factory.create_connection(
            "serial",  # Modbus RTU over RS485/Serial
            device_id,
            config['connection']
        )
        processor = factory.create_processor(
            "modbus",
            config['processor']
        )
        
        super().__init__(device_id, connection, processor, config)
        
        # Modbus register mapping
        self.registers = {
            'voltage_l1': 0x0000,
            'voltage_l2': 0x0002,
            'voltage_l3': 0x0004,
            'current_l1': 0x0006,
            'current_l2': 0x0008,
            'current_l3': 0x000A,
            'power_total': 0x000C,
            'energy_total': 0x0010,
            'frequency': 0x0014,
            'power_factor': 0x0016
        }
    
    def get_type(self) -> str:
        return "modbus_energy_meter"
    
    def supports_write(self) -> bool:
        return True  # Some meters support configuration
    
    def get_writable_properties(self) -> List[str]:
        return ["reset_energy", "set_demand_period", "calibration"]
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        return {
            "type": "modbus_energy_meter",
            "name": "Industrial Energy Meter",
            "description": "Three-phase power and energy monitoring device",
            "version": "1.5",
            "manufacturer": "PowerTech Industries",
            "capabilities": ["power_monitoring", "energy_measurement", "power_quality"],
            "writable_properties": ["reset_energy", "set_demand_period", "calibration"],
            "readable_properties": ["voltage", "current", "power", "energy", "frequency", "power_factor"],
            "connection_type": "serial",
            "protocol": "modbus_rtu"
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "sample_interval": 60,  # Energy meters typically read less frequently
            "location_base": {"lat": 40.7505, "lon": -73.9934},
            "location_offset": 0.01,
            "default_instances": 1,
            "device_type": "modbus_energy_meter",
            "connection": {
                "port": "/dev/ttyUSB1",
                "baudrate": 9600,
                "bytesize": 8,
                "parity": "N",
                "stopbits": 1,
                "timeout": 2.0,
                "read_timeout": 5.0
            },
            "processor": {
                "slave_id": 1,
                "protocol": "rtu",
                "byte_order": "big",
                "word_order": "big"
            }
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Read multiple Modbus registers and format as energy data"""
        try:
            # Read all registers in one or multiple requests
            energy_data = {}
            
            # Read voltage registers (3 phases)
            voltage_command = {
                "function_code": 4,  # Read Input Registers
                "address": self.registers['voltage_l1'],
                "quantity": 6  # 3 voltages, 2 registers each (32-bit float)
            }
            
            # Send command using parent's write_data (which encodes and sends)
            # Then manually read response - this is a simplified example
            # In practice, you'd want a more sophisticated Modbus client
            
            # For demonstration, we'll simulate reading the registers
            energy_data = {
                "voltage_l1": 230.5,
                "voltage_l2": 231.2,
                "voltage_l3": 229.8,
                "current_l1": 15.3,
                "current_l2": 16.1,
                "current_l3": 14.9,
                "power_total": 11.2,  # kW
                "energy_total": 1234.5,  # kWh
                "frequency": 50.02,  # Hz
                "power_factor": 0.95
            }
            
            # Calculate additional metrics
            energy_data.update({
                "voltage_average": round((energy_data["voltage_l1"] + 
                                       energy_data["voltage_l2"] + 
                                       energy_data["voltage_l3"]) / 3, 1),
                "current_average": round((energy_data["current_l1"] + 
                                        energy_data["current_l2"] + 
                                        energy_data["current_l3"]) / 3, 1),
                "power_per_phase": round(energy_data["power_total"] / 3, 2),
                "load_balance": self._calculate_load_balance([
                    energy_data["current_l1"],
                    energy_data["current_l2"], 
                    energy_data["current_l3"]
                ])
            })
            
            return {
                "device_id": self.device_id,
                "device_type": self.get_type(),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "active",
                "connection_status": self.connection.get_status().value,
                "location": self.location,
                "data": energy_data
            }
            
        except Exception as e:
            logger.error(f"Error reading energy meter {self.device_id}: {e}")
            return self._create_error_reading(str(e))
    
    def _calculate_load_balance(self, currents: List[float]) -> float:
        """Calculate load balance percentage (0-100, where 100 is perfectly balanced)"""
        if not currents or len(currents) != 3:
            return 0.0
        
        avg_current = sum(currents) / len(currents)
        if avg_current == 0:
            return 100.0
        
        max_deviation = max(abs(current - avg_current) for current in currents)
        balance = max(0, 100 - (max_deviation / avg_current * 100))
        
        return round(balance, 1)
    
    async def write_data(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle energy meter commands"""
        if "reset_energy" in command and command["reset_energy"]:
            # Send Modbus command to reset energy counter
            reset_command = {
                "function_code": 6,  # Write Single Register
                "address": 0x8000,  # Reset register (device-specific)
                "value": 1
            }
            return await super().write_data(reset_command)
        
        if "set_demand_period" in command:
            period = command["set_demand_period"]
            if 1 <= period <= 60:  # 1-60 minutes
                demand_command = {
                    "function_code": 6,
                    "address": 0x8001,  # Demand period register
                    "value": period
                }
                return await super().write_data(demand_command)
            else:
                return {
                    "success": False,
                    "error": "Demand period must be between 1 and 60 minutes",
                    "device_id": self.device_id
                }
        
        return await super().write_data(command)
    
    def get_command_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reset_energy": {
                    "type": "boolean",
                    "description": "Reset energy accumulator to zero"
                },
                "set_demand_period": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "description": "Set demand measurement period in minutes"
                },
                "calibration": {
                    "type": "object",
                    "properties": {
                        "voltage_scale": {"type": "number"},
                        "current_scale": {"type": "number"}
                    },
                    "description": "Calibration scaling factors"
                }
            }
        }


# Example device configuration file: gateway/app/device_config_physical.json
# {
#   "devices": {
#     "arduino_sensor": {
#       "enabled": true,
#       "instance_count": 2,
#       "is_physical": true,
#       "connection_required": true,
#       "auto_discovery": true,
#       "info": {
#         "type": "arduino_sensor",
#         "name": "Arduino Environmental Sensor",
#         "connection_type": "serial",
#         "protocol": "json"
#       },
#       "config": {
#         "sample_interval": 15,
#         "location_base": {"lat": 40.7589, "lon": -73.9851},
#         "location_offset": 0.002,
#         "connection": {
#           "port": "/dev/ttyUSB0",
#           "baudrate": 9600,
#           "timeout": 2.0
#         },
#         "processor": {
#           "encoding": "utf-8",
#           "validate_schema": true
#         }
#       }
#     },
#     "modbus_energy_meter": {
#       "enabled": true,
#       "instance_count": 1,
#       "is_physical": true,
#       "connection_required": true,
#       "auto_discovery": true,
#       "info": {
#         "type": "modbus_energy_meter",
#         "name": "Industrial Energy Meter",
#         "connection_type": "serial",
#         "protocol": "modbus_rtu"
#       },
#       "config": {
#         "sample_interval": 60,
#         "location_base": {"lat": 40.7505, "lon": -73.9934},
#         "connection": {
#           "port": "/dev/ttyUSB1",
#           "baudrate": 9600,
#           "parity": "N"
#         },
#         "processor": {
#           "slave_id": 1,
#           "protocol": "rtu"
#         }
#       }
#     },
#     "esp32_controller": {
#       "enabled": true,
#       "instance_count": 3,
#       "is_physical": true,
#       "connection_required": true,
#       "auto_discovery": true,
#       "info": {
#         "type": "esp32_controller",
#         "name": "ESP32 IoT Controller",
#         "connection_type": "wifi",
#         "protocol": "http_json"
#       },
#       "config": {
#         "sample_interval": 20,
#         "location_base": {"lat": 40.7484, "lon": -73.9857},
#         "location_offset": 0.003,
#         "connection": {
#           "host": "192.168.1.100",
#           "port": 80,
#           "protocol": "http"
#         },
#         "processor": {
#           "encoding": "utf-8",
#           "validate_schema": true
#         }
#       }
#     },
#     "smart_thermostat": {
#       "enabled": true,
#       "instance_count": 3,
#       "is_physical": false,
#       "info": {
#         "type": "smart_thermostat",
#         "name": "Smart Climate Controller"
#       },
#       "config": {
#         "sample_interval": 30,
#         "location_base": {"lat": 40.7580, "lon": -73.9855},
#         "location_offset": 0.005
#       }
#     }
#   },
#   "global_settings": {
#     "monitoring_interval": 30,
#     "default_instance_count": 1,
#     "enable_hot_plugging": true,
#     "auto_reconnect": true
#   },
#   "connection_settings": {
#     "serial_scan_ports": ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"],
#     "wifi_network_scan": true,
#     "wifi_default_network": "192.168.1.0/24",
#     "i2c_bus_scan": true,
#     "i2c_default_bus": 1,
#     "connection_timeout": 10.0,
#     "max_retries": 3
#   }
# }