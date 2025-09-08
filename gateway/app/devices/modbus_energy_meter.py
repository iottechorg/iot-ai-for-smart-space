
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