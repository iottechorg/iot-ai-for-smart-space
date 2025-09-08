
# gateway/app/processors/json_processor.py
import json
from typing import Dict, Any, Optional
import logging
from core.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

class JSONProcessor(MessageProcessor):
    """
    JSON message processor
    Handles JSON encoding/decoding with validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.encoding = config.get('encoding', 'utf-8')
        self.validate_schema = config.get('validate_schema', True)
        self.schema = config.get('schema', {})
    
    async def decode(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Decode JSON bytes into structured data"""
        try:
            # Convert bytes to string
            json_string = raw_data.decode(self.encoding)
            
            # Parse JSON
            data = json.loads(json_string)
            
            # Validate schema if enabled
            if self.validate_schema and self.schema:
                if not self._validate_schema(data):
                    logger.warning("JSON data failed schema validation")
                    return None
            
            logger.debug(f"Decoded JSON data: {data}")
            return data
            
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode bytes to string: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error decoding JSON: {e}")
            return None
    
    async def encode(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Encode structured data into JSON bytes"""
        try:
            # Validate data
            if not self.validate_data(data):
                logger.error("Invalid data format for JSON encoding")
                return None
            
            # Validate schema if enabled
            if self.validate_schema and self.schema:
                if not self._validate_schema(data):
                    logger.warning("Data failed schema validation")
                    return None
            
            # Convert to JSON string
            json_string = json.dumps(data, separators=(',', ':'))
            
            # Encode to bytes
            json_bytes = json_string.encode(self.encoding)
            
            logger.debug(f"Encoded JSON data: {len(json_bytes)} bytes")
            return json_bytes
            
        except Exception as e:
            logger.error(f"Failed to encode JSON: {e}")
            return None
    
    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Basic schema validation"""
        if not self.schema:
            return True
        
        try:
            # Simple validation - check required fields
            required_fields = self.schema.get('required', [])
            for field in required_fields:
                if field not in data:
                    return False
            
            # Check field types
            properties = self.schema.get('properties', {})
            for field, field_schema in properties.items():
                if field in data:
                    expected_type = field_schema.get('type')
                    if expected_type and not self._check_type(data[field], expected_type):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True
    
    @classmethod
    def get_processor_info(cls) -> Dict[str, Any]:
        """Get JSON processor information"""
        return {
            "type": "json",
            "name": "JSON Message Processor",
            "description": "Processes JSON formatted messages",
            "supported_encodings": ["utf-8", "ascii", "latin-1"],
            "features": ["schema_validation", "pretty_printing", "error_handling"]
        }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default JSON processor configuration"""
        return {
            "encoding": "utf-8",
            "validate_schema": True,
            "schema": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "device_id": {"type": "string"},
                    "data": {"type": "object"}
                },
                "required": ["device_id"]
            }
        }
