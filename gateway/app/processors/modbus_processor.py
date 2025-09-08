

# gateway/app/processors/modbus_processor.py
import struct
from typing import Dict, Any, Optional, List, Union
import logging
from core.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

class ModbusProcessor(MessageProcessor):
    """
    Modbus RTU/TCP message processor
    Handles Modbus protocol encoding/decoding
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.slave_id = config.get('slave_id', 1)
        self.byte_order = config.get('byte_order', 'big')  # 'big' or 'little'
        self.word_order = config.get('word_order', 'big')  # for 32-bit values
        self.protocol = config.get('protocol', 'rtu')  # 'rtu' or 'tcp'
    
    async def decode(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Decode Modbus message into structured data"""
        try:
            if len(raw_data) < 5:  # Minimum Modbus message length
                logger.warning("Modbus message too short")
                return None
            
            if self.protocol == 'rtu':
                return await self._decode_rtu(raw_data)
            else:  # TCP
                return await self._decode_tcp(raw_data)
                
        except Exception as e:
            logger.error(f"Failed to decode Modbus message: {e}")
            return None
    
    async def _decode_rtu(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Decode Modbus RTU message"""
        if len(data) < 5:
            return None
        
        # RTU format: [slave_id][function_code][data...][crc16]
        slave_id = data[0]
        function_code = data[1]
        
        # Verify CRC (last 2 bytes)
        if not self._verify_crc(data[:-2], data[-2:]):
            logger.warning("Modbus RTU CRC check failed")
            return None
        
        # Extract data portion
        data_bytes = data[2:-2]
        
        return {
            "protocol": "modbus_rtu",
            "slave_id": slave_id,
            "function_code": function_code,
            "data": self._parse_modbus_data(function_code, data_bytes),
            "raw_data": data.hex()
        }
    
    async def _decode_tcp(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Decode Modbus TCP message"""
        if len(data) < 8:  # MBAP header + PDU
            return None
        
        # TCP format: [transaction_id][protocol_id][length][unit_id][function_code][data...]
        transaction_id = struct.unpack('>H', data[0:2])[0]
        protocol_id = struct.unpack('>H', data[2:4])[0]
        length = struct.unpack('>H', data[4:6])[0]
        unit_id = data[6]
        function_code = data[7]
        
        if protocol_id != 0:  # Must be 0 for Modbus
            logger.warning("Invalid Modbus TCP protocol ID")
            return None
        
        # Extract data portion
        data_bytes = data[8:]
        
        return {
            "protocol": "modbus_tcp",
            "transaction_id": transaction_id,
            "unit_id": unit_id,
            "function_code": function_code,
            "data": self._parse_modbus_data(function_code, data_bytes),
            "raw_data": data.hex()
        }
    
    def _parse_modbus_data(self, function_code: int, data_bytes: bytes) -> Dict[str, Any]:
        """Parse Modbus data based on function code"""
        try:
            if function_code in [1, 2]:  # Read Coils/Discrete Inputs
                return self._parse_bits(data_bytes)
            elif function_code in [3, 4]:  # Read Holding/Input Registers
                return self._parse_registers(data_bytes)
            elif function_code == 5:  # Write Single Coil
                return self._parse_single_coil(data_bytes)
            elif function_code == 6:  # Write Single Register
                return self._parse_single_register(data_bytes)
            elif function_code == 15:  # Write Multiple Coils
                return self._parse_multiple_coils(data_bytes)
            elif function_code == 16:  # Write Multiple Registers
                return self._parse_multiple_registers(data_bytes)
            else:
                return {"raw_bytes": data_bytes.hex()}
                
        except Exception as e:
            logger.error(f"Error parsing Modbus data: {e}")
            return {"raw_bytes": data_bytes.hex(), "error": str(e)}
    
    def _parse_bits(self, data: bytes) -> Dict[str, Any]:
        """Parse coil/discrete input data"""
        if len(data) < 1:
            return {}
        
        byte_count = data[0]
        bit_data = data[1:1+byte_count]
        
        bits = []
        for byte_val in bit_data:
            for i in range(8):
                bits.append(bool(byte_val & (1 << i)))
        
        return {
            "type": "bits",
            "count": byte_count * 8,
            "values": bits[:byte_count * 8]
        }
    
    def _parse_registers(self, data: bytes) -> Dict[str, Any]:
        """Parse holding/input register data"""
        if len(data) < 1:
            return {}
        
        byte_count = data[0]
        register_data = data[1:1+byte_count]
        
        # Parse as 16-bit registers
        registers = []
        for i in range(0, len(register_data), 2):
            if i + 1 < len(register_data):
                if self.byte_order == 'big':
                    reg_val = struct.unpack('>H', register_data[i:i+2])[0]
                else:
                    reg_val = struct.unpack('<H', register_data[i:i+2])[0]
                registers.append(reg_val)
        
        return {
            "type": "registers",
            "count": len(registers),
            "values": registers
        }
    
    def _parse_single_coil(self, data: bytes) -> Dict[str, Any]:
        """Parse single coil write response"""
        if len(data) >= 4:
            address = struct.unpack('>H', data[0:2])[0]
            value = struct.unpack('>H', data[2:4])[0]
            return {
                "type": "single_coil",
                "address": address,
                "value": bool(value == 0xFF00)
            }
        return {}
    
    def _parse_single_register(self, data: bytes) -> Dict[str, Any]:
        """Parse single register write response"""
        if len(data) >= 4:
            address = struct.unpack('>H', data[0:2])[0]
            value = struct.unpack('>H', data[2:4])[0]
            return {
                "type": "single_register",
                "address": address,
                "value": value
            }
        return {}
    
    def _parse_multiple_coils(self, data: bytes) -> Dict[str, Any]:
        """Parse multiple coils write response"""
        if len(data) >= 4:
            address = struct.unpack('>H', data[0:2])[0]
            quantity = struct.unpack('>H', data[2:4])[0]
            return {
                "type": "multiple_coils",
                "start_address": address,
                "quantity": quantity
            }
        return {}
    
    def _parse_multiple_registers(self, data: bytes) -> Dict[str, Any]:
        """Parse multiple registers write response"""
        if len(data) >= 4:
            address = struct.unpack('>H', data[0:2])[0]
            quantity = struct.unpack('>H', data[2:4])[0]
            return {
                "type": "multiple_registers",
                "start_address": address,
                "quantity": quantity
            }
        return {}
    
    async def encode(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Encode structured data into Modbus message"""
        try:
            if self.protocol == 'rtu':
                return await self._encode_rtu(data)
            else:  # TCP
                return await self._encode_tcp(data)
                
        except Exception as e:
            logger.error(f"Failed to encode Modbus message: {e}")
            return None
    
    async def _encode_rtu(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Encode Modbus RTU message"""
        slave_id = data.get('slave_id', self.slave_id)
        function_code = data.get('function_code', 3)
        
        # Build PDU (Protocol Data Unit)
        pdu = self._build_pdu(function_code, data)
        if pdu is None:
            return None
        
        # Build complete RTU frame
        frame = bytes([slave_id]) + pdu
        
        # Add CRC
        crc = self._calculate_crc(frame)
        frame += struct.pack('<H', crc)  # CRC is little-endian
        
        return frame
    
    async def _encode_tcp(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Encode Modbus TCP message"""
        transaction_id = data.get('transaction_id', 1)
        unit_id = data.get('unit_id', self.slave_id)
        function_code = data.get('function_code', 3)
        
        # Build PDU
        pdu = self._build_pdu(function_code, data)
        if pdu is None:
            return None
        
        # Build MBAP header
        protocol_id = 0
        length = len(pdu) + 1  # PDU + unit_id
        
        mbap = struct.pack('>HHHB', transaction_id, protocol_id, length, unit_id)
        
        return mbap + pdu
    
    def _build_pdu(self, function_code: int, data: Dict[str, Any]) -> Optional[bytes]:
        """Build Protocol Data Unit based on function code"""
        try:
            if function_code in [1, 2, 3, 4]:  # Read functions
                address = data.get('address', 0)
                quantity = data.get('quantity', 1)
                return struct.pack('>BHH', function_code, address, quantity)
            
            elif function_code == 5:  # Write Single Coil
                address = data.get('address', 0)
                value = 0xFF00 if data.get('value', False) else 0x0000
                return struct.pack('>BHH', function_code, address, value)
            
            elif function_code == 6:  # Write Single Register
                address = data.get('address', 0)
                value = data.get('value', 0)
                return struct.pack('>BHH', function_code, address, value)
            
            elif function_code == 15:  # Write Multiple Coils
                return self._build_write_multiple_coils_pdu(data)
            
            elif function_code == 16:  # Write Multiple Registers
                return self._build_write_multiple_registers_pdu(data)
            
            else:
                logger.warning(f"Unsupported function code: {function_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error building PDU: {e}")
            return None
    
    def _build_write_multiple_coils_pdu(self, data: Dict[str, Any]) -> bytes:
        """Build PDU for writing multiple coils"""
        function_code = 15
        address = data.get('address', 0)
        values = data.get('values', [])
        quantity = len(values)
        
        # Pack bits into bytes
        byte_count = (quantity + 7) // 8
        packed_bytes = bytearray(byte_count)
        
        for i, value in enumerate(values):
            if value:
                byte_index = i // 8
                bit_index = i % 8
                packed_bytes[byte_index] |= (1 << bit_index)
        
        pdu = struct.pack('>BHHB', function_code, address, quantity, byte_count)
        pdu += bytes(packed_bytes)
        
        return pdu
    
    def _build_write_multiple_registers_pdu(self, data: Dict[str, Any]) -> bytes:
        """Build PDU for writing multiple registers"""
        function_code = 16
        address = data.get('address', 0)
        values = data.get('values', [])
        quantity = len(values)
        byte_count = quantity * 2
        
        pdu = struct.pack('>BHHB', function_code, address, quantity, byte_count)
        
        for value in values:
            if self.byte_order == 'big':
                pdu += struct.pack('>H', value)
            else:
                pdu += struct.pack('<H', value)
        
        return pdu
    
    def _calculate_crc(self, data: bytes) -> int:
        """Calculate Modbus RTU CRC16"""
        crc = 0xFFFF
        
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        
        return crc
    
    def _verify_crc(self, data: bytes, received_crc: bytes) -> bool:
        """Verify Modbus RTU CRC"""
        calculated_crc = self._calculate_crc(data)
        received_crc_val = struct.unpack('<H', received_crc)[0]
        return calculated_crc == received_crc_val
    
    @classmethod
    def get_processor_info(cls) -> Dict[str, Any]:
        """Get Modbus processor information"""
        return {
            "type": "modbus",
            "name": "Modbus Protocol Processor",
            "description": "Processes Modbus RTU/TCP messages",
            "supported_protocols": ["Modbus RTU", "Modbus TCP"],
            "supported_functions": [1, 2, 3, 4, 5, 6, 15, 16],
            "features": ["CRC validation", "multiple data types", "endianness control"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default Modbus processor configuration"""
        return {
            "slave_id": 1,
            "protocol": "rtu",
            "byte_order": "big",
            "word_order": "big",
            "timeout": 1.0,
            "retries": 3
        }
