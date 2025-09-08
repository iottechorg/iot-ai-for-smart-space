# app/core/target_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging
import json
import re
import math
import ast
import operator
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TargetGeneratorInterface(ABC):
    """Interface for all target generators"""
    
    @abstractmethod
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target value from metrics"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate generator configuration"""
        pass

class RuleBasedTargetGenerator(TargetGeneratorInterface):
    """Rule-based target generation using conditions and operators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.rules = config.get('rules', [])
        self.default_target = config.get('default_target', None)
        self.operators = {
            'gt': operator.gt,
            'gte': operator.ge,
            'lt': operator.lt,
            'lte': operator.le,
            'eq': operator.eq,
            'ne': operator.ne,
            'in': lambda x, y: x in y,
            'not_in': lambda x, y: x not in y,
            'between': lambda x, y: y[0] <= x <= y[1],
            'contains': lambda x, y: y in str(x),
            'regex': lambda x, y: bool(re.search(y, str(x)))
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate rule-based configuration"""
        try:
            rules = config.get('rules', [])
            if not isinstance(rules, list):
                return False
            
            for rule in rules:
                if not isinstance(rule, dict):
                    return False
                if 'condition' not in rule or 'target' not in rule:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Rule validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using rule evaluation"""
        try:
            for rule in self.rules:
                condition = rule.get('condition')
                target = rule.get('target')
                
                # Handle default rule
                if condition == "default":
                    return target
                
                # Evaluate condition
                if await self._evaluate_condition(condition, metrics, context):
                    return self._process_target(target, metrics, context)
            
            # Return default if no rule matched
            return self.default_target
            
        except Exception as e:
            logger.error(f"Error in rule-based target generation: {e}")
            return self.default_target
    
    async def _evaluate_condition(self, condition: Dict[str, Any], metrics: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate a single condition"""
        try:
            # Handle logical operators
            if 'and' in condition:
                return all(await self._evaluate_condition(c, metrics, context) for c in condition['and'])
            
            if 'or' in condition:
                return any(await self._evaluate_condition(c, metrics, context) for c in condition['or'])
            
            if 'not' in condition:
                return not await self._evaluate_condition(condition['not'], metrics, context)
            
            # Handle field conditions
            for field, criteria in condition.items():
                if field in ['and', 'or', 'not']:
                    continue
                
                # Get value from metrics or context
                value = self._get_value(field, metrics, context)
                if value is None:
                    return False
                
                # Evaluate criteria
                if not self._evaluate_criteria(value, criteria):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _get_value(self, field: str, metrics: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Get value from metrics or context with dot notation support"""
        # Check metrics first
        if '.' in field:
            # Support nested access like 'sensor.temperature'
            parts = field.split('.')
            value = metrics
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
        else:
            value = metrics.get(field)
        
        # Check context if not found in metrics
        if value is None and context:
            value = context.get(field)
        
        return value
    
    def _evaluate_criteria(self, value: Any, criteria: Union[Dict[str, Any], Any]) -> bool:
        """Evaluate criteria against value"""
        if not isinstance(criteria, dict):
            # Simple equality check
            return value == criteria
        
        for op, threshold in criteria.items():
            if op not in self.operators:
                logger.warning(f"Unknown operator: {op}")
                continue
            
            try:
                if not self.operators[op](value, threshold):
                    return False
            except Exception as e:
                logger.error(f"Error applying operator {op}: {e}")
                return False
        
        return True
    
    def _process_target(self, target: Any, metrics: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Process target value, supporting templates and calculations"""
        if isinstance(target, str):
            # Template substitution
            if '{' in target and '}' in target:
                try:
                    return target.format(**metrics, **(context or {}))
                except Exception as e:
                    logger.warning(f"Template substitution failed: {e}")
                    return target
        
        return target

class FormulaTargetGenerator(TargetGeneratorInterface):
    """Formula-based target generation using mathematical expressions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.expression = config.get('expression', '')
        self.return_type = config.get('return_type', 'auto')  # auto, int, float, str, bool
        
        # Safe functions and constants
        self.safe_functions = {
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'len': len, 'int': int, 'float': float,
            'str': str, 'bool': bool,
            'sqrt': math.sqrt, 'pow': math.pow, 'log': math.log,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pi': math.pi, 'e': math.e
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate formula configuration"""
        try:
            expression = config.get('expression', '')
            if not expression:
                return False
            
            # Try to parse the expression
            ast.parse(expression, mode='exec')
            return True
            
        except Exception as e:
            logger.error(f"Formula validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using formula evaluation"""
        try:
            # Create safe execution context
            safe_dict = {
                '__builtins__': {},
                **self.safe_functions,
                **metrics,
                **(context or {})
            }
            
            # Execute the formula
            exec(self.expression, safe_dict)
            result = safe_dict.get('result')
            
            # Apply return type conversion
            return self._convert_result(result)
            
        except Exception as e:
            logger.error(f"Error in formula target generation: {e}")
            return None
    
    def _convert_result(self, result: Any) -> Any:
        """Convert result to specified type"""
        if self.return_type == 'auto':
            return result
        
        try:
            if self.return_type == 'int':
                return int(result)
            elif self.return_type == 'float':
                return float(result)
            elif self.return_type == 'str':
                return str(result)
            elif self.return_type == 'bool':
                return bool(result)
        except Exception as e:
            logger.warning(f"Type conversion failed: {e}")
        
        return result

class SQLQueryTargetGenerator(TargetGeneratorInterface):
    """SQL-based target generation (requires database connection)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.query_template = config.get('query', '')
        self.db_client = None  # Will be injected
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes default
        self.cache = {}
    
    def set_db_client(self, db_client):
        """Inject database client"""
        self.db_client = db_client
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate SQL configuration"""
        try:
            query = config.get('query', '')
            if not query.strip():
                return False
            
            # Basic SQL validation (can be enhanced)
            forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER']
            query_upper = query.upper()
            for keyword in forbidden_keywords:
                if keyword in query_upper:
                    logger.error(f"Forbidden SQL keyword: {keyword}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"SQL validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using SQL query"""
        try:
            if not self.db_client:
                logger.error("Database client not available")
                return None
            
            # Create cache key
            cache_key = hash(json.dumps(metrics, sort_keys=True))
            
            # Check cache
            if cache_key in self.cache:
                cached_result, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_ttl:
                    return cached_result
            
            # Format query with metrics
            formatted_query = self.query_template.format(**metrics, **(context or {}))
            
            # Execute query
            result = self.db_client.query(formatted_query)
            
            # Extract result (assuming single value)
            if result.result_rows:
                target_value = result.result_rows[0][0]
                
                # Cache result
                self.cache[cache_key] = (target_value, datetime.now())
                
                return target_value
            
            return None
            
        except Exception as e:
            logger.error(f"Error in SQL target generation: {e}")
            return None

class LookupTableTargetGenerator(TargetGeneratorInterface):
    """Lookup table-based target generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.lookup_table = config.get('lookup_table', {})
        self.key_fields = config.get('key_fields', [])
        self.default_target = config.get('default_target', None)
        self.interpolation = config.get('interpolation', False)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate lookup table configuration"""
        try:
            lookup_table = config.get('lookup_table', {})
            key_fields = config.get('key_fields', [])
            
            if not isinstance(lookup_table, dict) or not isinstance(key_fields, list):
                return False
            
            if not key_fields:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Lookup table validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using lookup table"""
        try:
            # Build lookup key
            key_values = []
            for field in self.key_fields:
                value = metrics.get(field)
                if value is None and context:
                    value = context.get(field)
                if value is None:
                    return self.default_target
                key_values.append(str(value))
            
            lookup_key = ':'.join(key_values)
            
            # Direct lookup
            if lookup_key in self.lookup_table:
                return self.lookup_table[lookup_key]
            
            # Interpolation for numeric keys (if enabled)
            if self.interpolation and len(self.key_fields) == 1:
                return self._interpolate_value(float(key_values[0]))
            
            return self.default_target
            
        except Exception as e:
            logger.error(f"Error in lookup table target generation: {e}")
            return self.default_target
    
    def _interpolate_value(self, key: float) -> Any:
        """Linear interpolation for single numeric key"""
        try:
            # Convert lookup table keys to float and sort
            numeric_keys = []
            for k in self.lookup_table.keys():
                try:
                    numeric_keys.append((float(k), self.lookup_table[k]))
                except ValueError:
                    continue
            
            if len(numeric_keys) < 2:
                return self.default_target
            
            numeric_keys.sort(key=lambda x: x[0])
            
            # Find interpolation bounds
            if key <= numeric_keys[0][0]:
                return numeric_keys[0][1]
            if key >= numeric_keys[-1][0]:
                return numeric_keys[-1][1]
            
            # Linear interpolation
            for i in range(len(numeric_keys) - 1):
                x1, y1 = numeric_keys[i]
                x2, y2 = numeric_keys[i + 1]
                
                if x1 <= key <= x2:
                    if isinstance(y1, (int, float)) and isinstance(y2, (int, float)):
                        # Numeric interpolation
                        return y1 + (y2 - y1) * (key - x1) / (x2 - x1)
                    else:
                        # Non-numeric: return closest
                        return y1 if abs(key - x1) < abs(key - x2) else y2
            
            return self.default_target
            
        except Exception as e:
            logger.error(f"Interpolation error: {e}")
            return self.default_target

class MLModelTargetGenerator(TargetGeneratorInterface):
    """Target generation using a pre-trained ML model"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get('model_name', '')
        self.feature_mapping = config.get('feature_mapping', {})
        self.model = None  # Will be loaded
        self.scaler = None
    
    def set_model(self, model, scaler=None):
        """Inject trained model"""
        self.model = model
        self.scaler = scaler
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate ML model configuration"""
        try:
            model_name = config.get('model_name', '')
            feature_mapping = config.get('feature_mapping', {})
            
            if not model_name or not isinstance(feature_mapping, dict):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"ML model validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using ML model prediction"""
        try:
            if not self.model:
                logger.error("ML model not loaded")
                return None
            
            # Map features
            features = []
            for model_feature, metric_field in self.feature_mapping.items():
                value = metrics.get(metric_field)
                if value is None and context:
                    value = context.get(metric_field)
                if value is None:
                    logger.warning(f"Missing feature: {metric_field}")
                    return None
                features.append(float(value))
            
            # Prepare features
            import numpy as np
            features_array = np.array([features])
            
            # Apply scaling if available
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in ML model target generation: {e}")
            return None

class CompositeTargetGenerator(TargetGeneratorInterface):
    """Composite generator that combines multiple generators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.generators = []
        self.combination_method = config.get('combination_method', 'first_valid')  # first_valid, majority, weighted_avg
        self.weights = config.get('weights', [])
        
        # Initialize sub-generators
        for gen_config in config.get('generators', []):
            generator = TargetGeneratorFactory.create_generator(gen_config)
            self.generators.append(generator)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate composite configuration"""
        try:
            generators = config.get('generators', [])
            if not isinstance(generators, list) or len(generators) < 2:
                return False
            
            for gen_config in generators:
                generator = TargetGeneratorFactory.create_generator(gen_config)
                if not generator.validate_config(gen_config):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Composite validation error: {e}")
            return False
    
    async def generate_target(self, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Generate target using composite method"""
        try:
            results = []
            
            # Collect results from all generators
            for generator in self.generators:
                try:
                    result = await generator.generate_target(metrics, context)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Sub-generator failed: {e}")
                    results.append(None)
            
            # Combine results
            if self.combination_method == 'first_valid':
                for result in results:
                    if result is not None:
                        return result
                return None
            
            elif self.combination_method == 'majority':
                # Return most common non-None result
                valid_results = [r for r in results if r is not None]
                if not valid_results:
                    return None
                
                from collections import Counter
                counter = Counter(valid_results)
                return counter.most_common(1)[0][0]
            
            elif self.combination_method == 'weighted_avg':
                # Weighted average for numeric results
                valid_results = [(r, w) for r, w in zip(results, self.weights) if r is not None and isinstance(r, (int, float))]
                if not valid_results:
                    return None
                
                total_weight = sum(w for _, w in valid_results)
                weighted_sum = sum(r * w for r, w in valid_results)
                return weighted_sum / total_weight
            
            return None
            
        except Exception as e:
            logger.error(f"Error in composite target generation: {e}")
            return None

class TargetGeneratorFactory:
    """Factory for creating target generators"""
    
    GENERATOR_TYPES = {
        'rule_based': RuleBasedTargetGenerator,
        'formula': FormulaTargetGenerator,
        'sql_query': SQLQueryTargetGenerator,
        'lookup_table': LookupTableTargetGenerator,
        'ml_model': MLModelTargetGenerator,
        'composite': CompositeTargetGenerator
    }
    
    @staticmethod
    def create_generator(config: Dict[str, Any]) -> TargetGeneratorInterface:
        """Create target generator from configuration"""
        generator_type = config.get('type', 'rule_based')
        
        if generator_type not in TargetGeneratorFactory.GENERATOR_TYPES:
            raise ValueError(f"Unsupported target generator type: {generator_type}")
        
        generator_class = TargetGeneratorFactory.GENERATOR_TYPES[generator_type]
        generator = generator_class(config)
        
        # Validate configuration
        if not generator.validate_config(config):
            raise ValueError(f"Invalid configuration for {generator_type} generator")
        
        return generator
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported generator types"""
        return list(TargetGeneratorFactory.GENERATOR_TYPES.keys())

# Usage example configurations:

# Rule-based configuration example
RULE_BASED_CONFIG = {
    "type": "rule_based",
    "default_target": "unknown",
    "rules": [
        {
            "condition": {
                "and": [
                    {"vehicle_count": {"gt": 50}},
                    {"average_speed": {"lt": 30}}
                ]
            },
            "target": "high"
        },
        {
            "condition": {
                "or": [
                    {"vehicle_count": {"between": [25, 50]}},
                    {"average_speed": {"between": [30, 50]}}
                ]
            },
            "target": "medium"
        },
        {
            "condition": "default",
            "target": "low"
        }
    ]
}

# Formula-based configuration example
FORMULA_CONFIG = {
    "type": "formula",
    "return_type": "float",
    "expression": """
# Calculate congestion index
congestion_factor = vehicle_count / max(average_speed, 1)
time_factor = 1.2 if 7 <= hour <= 9 or 17 <= hour <= 19 else 1.0
result = congestion_factor * time_factor
"""
}

# Lookup table configuration example
LOOKUP_TABLE_CONFIG = {
    "type": "lookup_table",
    "key_fields": ["vehicle_count", "weather"],
    "interpolation": False,
    "default_target": "medium",
    "lookup_table": {
        "0:sunny": "low",
        "25:sunny": "medium", 
        "50:sunny": "high",
        "0:rainy": "medium",
        "25:rainy": "high",
        "50:rainy": "critical"
    }
}

# Composite configuration example
COMPOSITE_CONFIG = {
    "type": "composite",
    "combination_method": "weighted_avg",
    "weights": [0.6, 0.4],
    "generators": [
        RULE_BASED_CONFIG,
        FORMULA_CONFIG
    ]
}