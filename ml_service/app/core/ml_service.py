# app/core/ml_service.py - PURE ML LOGIC (FIXED TRAINING)

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time
from .model_registry import ModelRegistry
from .model_trainer import ModelTrainer
from .storage_manager import StorageManager
from .target_generator import TargetGeneratorFactory

logger = logging.getLogger(__name__)

class MLService:
    """
    Pure ML service without HTTP concerns - FIXED training integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.storage_manager = StorageManager()
        self.model_registry = ModelRegistry(self.storage_manager)
        self.model_trainer = ModelTrainer(self.storage_manager, self.model_registry)
        
        # Target generators per model - FIXED: Initialize properly
        self.target_generators = {}
        
        # Database client (will be injected or initialized)
        self.db_client = None
        
        # Model cache for fast predictions
        self.model_cache = {}
        self.last_cache_update = {}
        
        # Event handlers
        self.event_handlers = []
        self.is_initialized = False

        # Cache management settings
        self.max_cached_models = config.get('max_cached_models', 3)  # Keep only latest 3
        self.cache_cleanup_enabled = config.get('cache_cleanup_enabled', True)
    
    
    async def initialize(self) -> bool:
        """Initialize the ML service with FIXED target generation"""
        try:
            logger.info("Initializing ML Service...")
            
            # Initialize storage
            await self.storage_manager.initialize()
            logger.info("Storage initialized")
            
            # Initialize database connection
            await self._initialize_database()
            logger.info("Database initialized")
            
            # Load model configurations
            await self._load_model_configs()
            logger.info("Model configurations loaded")
            
            # FIXED: Initialize target generators AFTER loading configs
            await self._initialize_target_generators()
            logger.info(f"Target generators initialized: {len(self.target_generators)}")
            
            # Load trained models into cache
            await self._load_models_to_cache()
            logger.info("Models loaded to cache")
            
            self.is_initialized = True
            # Start background training if enabled
            if self.config.get('enable_background_training', True):
                self.start_background_training()
            
            logger.info("ML Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML service: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_database(self):
        """Initialize database connection"""
        try:
            import clickhouse_connect
            
            self.db_client = clickhouse_connect.get_client(
                host=self.config.get('clickhouse_host', 'clickhouse'),
                port=self.config.get('clickhouse_port', 8123),
                username=self.config.get('clickhouse_user', 'default'),
                password=self.config.get('clickhouse_password', ''),
                database=self.config.get('clickhouse_db', 'smartcity'),
                connect_timeout=10
            )
            
            # Test connection
            result = self.db_client.query("SELECT 1")
            logger.info("Database connection successful")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Continue without database - some functionality will be limited
            self.db_client = None
    
    async def _initialize_target_generators(self):
        """FIXED: Initialize target generators for each model"""
        logger.info("Initializing target generators...")
        
        for model_name, config in self.model_registry.model_configs.items():
            try:
                target_config = config.get('training', {}).get('target_generation')
                if target_config:
                    generator = TargetGeneratorFactory.create_generator(target_config)
                    
                    # Inject database client if needed
                    if hasattr(generator, 'set_db_client') and self.db_client:
                        generator.set_db_client(self.db_client)
                    
                    self.target_generators[model_name] = generator
                    logger.info(f"✅ Target generator for {model_name}: {target_config.get('type')}")
                else:
                    logger.warning(f"⚠️  No target generation config for {model_name}")
            
            except Exception as e:
                logger.error(f"❌ Error initializing target generator for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Initialized {len(self.target_generators)} target generators")
    
    async def predict(
        self, 
        device_type: str, 
        device_id: str, 
        data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a prediction using the appropriate model"""
        try:
            if not self.is_initialized:
                raise RuntimeError("ML Service not initialized")
            
            # Determine model to use
            if not model_name:
                models = self._get_models_for_device_type(device_type)
                if not models:
                    raise ValueError(f"No models available for device type: {device_type}")
                model_name = models[0]
            
            # Get model from cache
            model = self.model_cache.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found in cache")
            
            # Add device context to data
            prediction_data = {
                **data,
                'device_type': device_type,
                'device_id': device_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Make prediction
            result = await model.predict(prediction_data)
            
            # Add metadata
            result.update({
                'device_type': device_type,
                'device_id': device_id,
                'model_name': model_name,
                'service_version': '2.0.0'
            })
            
            # Emit prediction event
            await self._emit_event('prediction_made', {
                'device_type': device_type,
                'device_id': device_id,
                'model_name': model_name,
                'prediction': result,
                'input_data': data
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {device_type}/{device_id}: {e}")
            await self._emit_event('prediction_error', {
                'device_type': device_type,
                'device_id': device_id,
                'error': str(e),
                'input_data': data
            })
            raise
    
    async def train_model(
        self, 
        model_name: str, 
        training_data: Optional[Dict[str, Any]] = None,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """FIXED: Train a model with proper target generation and storage"""
        try:
            if not self.is_initialized:
                raise RuntimeError("ML Service not initialized")
            
            logger.info(f"🚂 Starting training for model: {model_name}")
            
            # Check if model exists
            config = self.model_registry.get_model_config(model_name)
            if not config:
                raise ValueError(f"Model configuration for {model_name} not found")
            
            # Generate training data if not provided
            if training_data is None:
                logger.info(f"📊 Generating training data for {model_name}")
                training_data = await self._generate_training_data_with_targets(model_name)
            
            if not training_data['features']:
                raise ValueError(f"No training data available for {model_name}")
            
            logger.info(f"📈 Training data: {len(training_data['features'])} samples")
            
            # Validate training data
            validation_result = await self.model_trainer.validate_training_data(model_name, training_data)
            if not validation_result.get('valid', False):
                raise ValueError(f"Training data validation failed: {validation_result.get('error')}")
            
            # Emit training start event
            await self._emit_event('training_started', {
                'model_name': model_name,
                'samples': len(training_data.get('features', [])),
                'force_retrain': force_retrain
            })
            
            # Train the model
            success = await self.model_trainer.train_model(model_name, training_data)
            
            if success:
                # Get trained model
                model = await self.model_registry.get_model(model_name)
                if model:
                    # FIXED: Explicitly save to storage
                    await self._save_model_to_storage(model_name, model)
                    
                    # Update cache
                    self.model_cache[model_name] = model
                    self.last_cache_update[model_name] = datetime.utcnow().isoformat()

                    # AUTO-CLEANUP: Clean cache after successful training
                    await self._auto_cleanup_after_training(model_name)
                
                logger.info(f"✅ Training completed successfully for {model_name}")
                
                # Emit training success event
                await self._emit_event('training_completed', {
                    'model_name': model_name,
                    'success': True,
                    'version': model.get_version() if model else None
                })
                
                return {
                    'success': True,
                    'model_name': model_name,
                    'version': model.get_version() if model else None,
                    'training_samples': len(training_data.get('features', [])),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"❌ Training failed for {model_name}")
                
                # Emit training failure event
                await self._emit_event('training_failed', {
                    'model_name': model_name,
                    'error': 'Training process failed'
                })
                
                return {
                    'success': False,
                    'model_name': model_name,
                    'error': 'Training process failed',
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Training error for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Emit training error event
            await self._emit_event('training_error', {
                'model_name': model_name,
                'error': str(e)
            })
            
            raise
    
    async def _generate_training_data_with_targets(self, model_name: str) -> Dict[str, Any]:
        """FIXED: Generate training data using configured target generators"""
        try:
            config = self.model_registry.get_model_config(model_name)
            if not config:
                logger.error(f"No config found for model: {model_name}")
                return {"features": [], "targets": []}
            
            # Get training parameters
            device_types = config.get('model', {}).get('device_types', [])
            features = config.get('training', {}).get('features', [])
            
            if not device_types or not features:
                logger.error(f"Invalid config for model: {model_name}")
                return {"features": [], "targets": []}
            
            # Get target generator
            target_generator = self.target_generators.get(model_name)
            if not target_generator:
                logger.warning(f"No target generator found for {model_name}, using fallback")
                return await self._generate_fallback_training_data(model_name, device_types, features)
            
            logger.info(f"📊 Fetching training data for {model_name}")
            logger.info(f"   Device types: {device_types}")
            logger.info(f"   Features: {features}")
            logger.info(f"   Target generator: {type(target_generator).__name__}")
            
            # Fetch raw data from database
            raw_data = await self._fetch_raw_training_data(device_types, features)
            
            if not raw_data:
                logger.warning(f"No raw training data found for model: {model_name}")
                return {"features": [], "targets": []}
            
            logger.info(f"📈 Found {len(raw_data)} raw data points")
            
            # Generate targets using the configured generator
            feature_data = []
            target_data = []
            
            for i, data_point in enumerate(raw_data):
                try:
                    # Extract features
                    feature_dict = {}
                    for feature in features:
                        feature_dict[feature] = data_point.get(feature, 0.0)
                    
                    # Generate target using the configured generator
                    target = await target_generator.generate_target(feature_dict)
                    
                    if target is not None:
                        feature_data.append(feature_dict)
                        target_data.append(target)
                    
                    # Log progress every 100 samples
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Processed {i + 1}/{len(raw_data)} samples")
                
                except Exception as e:
                    logger.warning(f"Error generating target for sample {i}: {e}")
                    continue
            
            logger.info(f"✅ Generated {len(feature_data)} training samples for {model_name}")
            
            return {
                "features": feature_data,
                "targets": target_data
            }
        
        except Exception as e:
            logger.error(f"Error generating training data for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"features": [], "targets": []}
    
    async def _fetch_raw_training_data(self, device_types: List[str], features: List[str]) -> List[Dict[str, Any]]:
        """FIXED: Fetch raw training data from database"""
        try:
            if not self.db_client:
                logger.error("Database client not available")
                return []
            
            # Build query
            device_types_str = "', '".join(device_types)
            features_str = "', '".join(features)
            
            query = f"""
            SELECT 
                device_type,
                device_id,
                metric_name,
                metric_value,
                timestamp
            FROM device_data 
            WHERE device_type IN ('{device_types_str}')
                AND metric_name IN ('{features_str}')
                AND timestamp >= now() - INTERVAL 7 DAY
            ORDER BY device_type, device_id, timestamp
            LIMIT 5000
            """
            
            logger.debug(f"Executing query: {query}")
            result = self.db_client.query(query)
            
            # Process results into feature dictionaries
            data_points = {}
            for row in result.result_rows:
                device_type, device_id, metric_name, metric_value, timestamp = row
                key = (device_type, device_id, str(timestamp))
                
                if key not in data_points:
                    data_points[key] = {
                        'device_type': device_type,
                        'device_id': device_id,
                        'timestamp': timestamp
                    }
                
                data_points[key][metric_name] = metric_value
            
            # Convert to list and filter complete records
            raw_data = []
            for data_point in data_points.values():
                # Check if all required features are present
                if all(feature in data_point for feature in features):
                    raw_data.append(data_point)
            
            logger.info(f"📊 Found {len(raw_data)} complete data points from database")
            return raw_data
        
        except Exception as e:
            logger.error(f"Error fetching raw training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _generate_fallback_training_data(self, model_name: str, device_types: List[str], features: List[str]) -> Dict[str, Any]:
        """FIXED: Fallback training data generation"""
        logger.warning(f"Using fallback training data generation for {model_name}")
        
        try:
            # Fetch raw data
            raw_data = await self._fetch_raw_training_data(device_types, features)
            
            if not raw_data:
                return {"features": [], "targets": []}
            
            # Use old hard-coded target generation
            feature_data = []
            target_data = []
            
            for data_point in raw_data:
                try:
                    device_type = data_point.get('device_type', '')
                    target = self._generate_target_fallback(device_type, data_point)
                    
                    if target is not None:
                        feature_dict = {}
                        for feature in features:
                            feature_dict[feature] = data_point.get(feature, 0.0)
                        
                        feature_data.append(feature_dict)
                        target_data.append(target)
                
                except Exception as e:
                    logger.warning(f"Error in fallback target generation: {e}")
                    continue
            
            logger.info(f"Generated {len(feature_data)} samples using fallback method")
            
            return {
                "features": feature_data,
                "targets": target_data
            }
        
        except Exception as e:
            logger.error(f"Error in fallback training data generation: {e}")
            return {"features": [], "targets": []}
    
    def _generate_target_fallback(self, device_type: str, metrics: Dict[str, Any]) -> Any:
        """Fallback target generation (old hard-coded method)"""
        try:
            if device_type == "traffic_sensor":
                vehicle_count = metrics.get('vehicle_count', 0)
                avg_speed = metrics.get('average_speed', 50)
                
                if vehicle_count > 50 and avg_speed < 30:
                    return "high"
                elif vehicle_count > 25 and avg_speed < 50:
                    return "medium"
                else:
                    return "low"
            
            elif device_type == "water_level_sensor":
                water_level = metrics.get('water_level', 0)
                flow_rate = metrics.get('flow_rate', 0)
                
                risk = (water_level / 5.0) * 0.6 + (flow_rate / 10.0) * 0.4
                return min(100, max(0, risk * 100))
            
            return 0
        
        except Exception as e:
            logger.error(f"Error in fallback target generation: {e}")
            return None
    
    async def _save_model_to_storage(self, model_name: str, model) -> bool:
        """FIXED: Ensure model is saved to storage"""
        try:
            # Save to latest
            model_path = f"models/{model_name}/latest"
            save_success = await model.save(model_path)
            
            if save_success:
                logger.info(f"💾 Model {model_name} saved to storage: {model_path}")
                
                # Also save versioned copy
                versioned_path = f"models/{model_name}/{model.get_version()}"
                await model.save(versioned_path)
                logger.info(f"💾 Model {model_name} versioned copy saved: {versioned_path}")
                
                # Verify save by checking storage
                try:
                    files = await self.storage_manager.list_files(f"models/{model_name}/")
                    logger.info(f"✅ Verified: {len(files)} files saved for {model_name}")
                    for file in files[:5]:  # Show first 5 files
                        logger.info(f"   📄 {file}")
                except Exception as e:
                    logger.warning(f"Could not verify saved files: {e}")
                
                return True
            else:
                logger.error(f"❌ Failed to save model {model_name} to storage")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model {model_name} to storage: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about models"""
        try:
            if model_name:
                model = self.model_cache.get(model_name)
                if not model:
                    raise ValueError(f"Model {model_name} not found")
                
                config = self.model_registry.get_model_config(model_name)
                return {
                    'name': model_name,
                    'version': model.get_version(),
                    'is_trained': getattr(model, 'is_trained', False),
                    'last_updated': self.last_cache_update.get(model_name),
                    'config': config,
                    'device_types': config.get('model', {}).get('device_types', []),
                    'has_target_generator': model_name in self.target_generators
                }
            else:
                models_info = []
                for name, model in self.model_cache.items():
                    config = self.model_registry.get_model_config(name)
                    models_info.append({
                        'name': name,
                        'version': model.get_version(),
                        'is_trained': getattr(model, 'is_trained', False),
                        'last_updated': self.last_cache_update.get(name),
                        'device_types': config.get('model', {}).get('device_types', []) if config else [],
                        'has_target_generator': name in self.target_generators
                    })
                
                return {
                    'models': models_info,
                    'total_models': len(models_info),
                    'storage_healthy': await self._check_storage_health(),
                    'target_generators': len(self.target_generators)
                }
        
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise
    
    async def get_training_data_preview(self, model_name: str) -> Dict[str, Any]:
        """DEBUG: Get preview of training data generation"""
        try:
            training_data = await self._generate_training_data_with_targets(model_name)
            
            return {
                'model_name': model_name,
                'features_count': len(training_data.get('features', [])),
                'targets_count': len(training_data.get('targets', [])),
                'sample_features': training_data.get('features', [])[:3],
                'sample_targets': training_data.get('targets', [])[:3],
                'has_target_generator': model_name in self.target_generators,
                'target_generator_type': type(self.target_generators.get(model_name)).__name__ if model_name in self.target_generators else None
            }
        
        except Exception as e:
            logger.error(f"Error getting training data preview: {e}")
            return {'error': str(e)}
    
    async def get_storage_status(self) -> Dict[str, Any]:
        """Get storage status and statistics"""
        try:
            files = await self.storage_manager.list_files()
            model_files = [f for f in files if f.startswith("models/")]
            
            models_in_storage = {}
            for file_path in model_files:
                parts = file_path.split('/')
                if len(parts) >= 3:
                    model_name = parts[1]
                    if model_name not in models_in_storage:
                        models_in_storage[model_name] = []
                    models_in_storage[model_name].append(file_path)
            
            return {
                'healthy': await self._check_storage_health(),
                'total_files': len(files),
                'model_files': len(model_files),
                'models_in_storage': len(models_in_storage),
                'model_details': models_in_storage
            }
        except Exception as e:
            logger.error(f"Error getting storage status: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def save_model(self, model_name: str) -> bool:
        """Save model to storage"""
        try:
            if model_name not in self.model_cache:
                raise ValueError(f"Model {model_name} not found in cache")
            
            model = self.model_cache[model_name]
            success = await self._save_model_to_storage(model_name, model)
            
            if success:
                await self._emit_event('model_saved', {
                    'model_name': model_name,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return success
        
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    async def load_model(self, model_name: str) -> bool:
        """Load model from storage"""
        try:
            config = self.model_registry.get_model_config(model_name)
            if not config:
                raise ValueError(f"Model config {model_name} not found")
            
            # Create model instance
            model_type = config.get('model', {}).get('type')
            if model_type == 'sklearn':
                from ..models.sklearn_model import SklearnModel
                model = SklearnModel(model_name, config, self.storage_manager)
            elif model_type == 'tensorflow':
                from ..models.tensorflow_model import TensorflowModel
                model = TensorflowModel(model_name, config, self.storage_manager)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load from storage
            model_path = f"models/{model_name}/latest"
            success = await model.load(model_path)
            
            if success:
                # Update cache
                self.model_cache[model_name] = model
                self.last_cache_update[model_name] = datetime.utcnow().isoformat()
                await self.model_registry.register_model(model)
                
                await self._emit_event('model_loaded', {
                    'model_name': model_name,
                    'version': model.get_version(),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return success
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    # Event handling
    def add_event_handler(self, handler):
        """Add event handler for ML service events"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler):
        """Remove event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered handlers"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'ml_service'
        }
        
        for handler in self.event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    # Helper methods
    async def _load_model_configs(self):
        """Load model configurations from files"""
        import os
        import yaml
        
        config_path = self.config.get('model_config_path', '/app/config')
        if not os.path.exists(config_path):
            logger.warning(f"Config path not found: {config_path}")
            return
        
        for file in os.listdir(config_path):
            if file.endswith('.yaml') or file.endswith('.yml'):
                try:
                    with open(os.path.join(config_path, file), 'r') as f:
                        config = yaml.safe_load(f)
                        model_name = config.get('model', {}).get('name')
                        if model_name:
                            self.model_registry.register_model_config(model_name, config)
                            logger.info(f"Loaded model config: {model_name}")
                        else:
                            logger.warning(f"No model name found in config file: {file}")
                except Exception as e:
                    logger.error(f"Error loading config file {file}: {e}")
    
    async def _load_models_to_cache(self):
        """Load trained models into memory cache"""
        try:
            loaded_count = await self.model_registry.load_models()
            
            # Cache models for fast access
            for model_name in self.model_registry.get_all_model_configs():
                model = await self.model_registry.get_model(model_name)
                if model:
                    self.model_cache[model_name] = model
                    self.last_cache_update[model_name] = datetime.utcnow().isoformat()
                    logger.info(f"Cached model: {model_name}")
            
            logger.info(f"Loaded {loaded_count} trained models, cached {len(self.model_cache)} total models")
            
        except Exception as e:
            logger.error(f"Error loading models to cache: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_models_for_device_type(self, device_type: str) -> List[str]:
        """Get model names that can process data from device type"""
        models = []
        for name, config in self.model_registry.model_configs.items():
            device_types = config.get('model', {}).get('device_types', [])
            if device_type in device_types and name in self.model_cache:
                models.append(name)
        return models
    
    async def _check_storage_health(self) -> bool:
        """Check if storage is healthy"""
        try:
            await self.storage_manager.list_files("models/")
            return True
        except Exception:
            return False
        
    def start_background_training(self):
        """Start background training thread"""
        def background_training_loop():
            logger.info("🕐 Started background training thread")
            while self.is_initialized:
                try:
                    training_interval_hours = self.config.get('training_interval_hours', 24)
                    logger.info(f"⏰ Next training cycle in {training_interval_hours} hours")
                    
                    # Wait for the specified interval (in smaller chunks for clean shutdown)
                    wait_time = training_interval_hours * 3600  # Convert to seconds
                    while wait_time > 0 and self.is_initialized:
                        time.sleep(min(60, wait_time))  # Sleep in 1-minute chunks
                        wait_time -= 60
                    
                    if not self.is_initialized:
                        break
                    
                    # Train all models periodically
                    logger.info("🚂 Starting background training cycle")
                    for model_name in self.model_registry.get_all_model_configs():
                        if not self.is_initialized:
                            break
                        
                        try:
                            logger.info(f"🔄 Background training for model: {model_name}")
                            
                            # Use async training in background thread
                            result = asyncio.run(self.train_model(model_name, force_retrain=False))
                            
                            if result.get('success', False):
                                logger.info(f"✅ Background training successful: {model_name}")
                            else:
                                logger.warning(f"⚠️ Background training failed: {model_name}")
                            
                            # Wait between models to avoid overwhelming the system
                            time.sleep(30)
                        
                        except Exception as e:
                            logger.error(f"❌ Background training error for {model_name}: {e}")
                            continue
                    
                    logger.info("🏁 Background training cycle completed")
                
                except Exception as e:
                    logger.error(f"Error in background training loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
            
            logger.info("🛑 Background training thread stopped")
        
        # Start the background thread
        import threading
        self.background_training_thread = threading.Thread(
            target=background_training_loop, 
            daemon=True,
            name="BackgroundTraining"
        )
        self.background_training_thread.start()
        logger.info(f"🚀 Background training started (interval: {self.config.get('training_interval_hours', 24)}h)")

    async def cleanup_model_cache(self, keep_latest: int = None) -> Dict[str, Any]:
            """
            Clean up model cache, keeping only the latest N models
            
            Args:
                keep_latest: Number of latest models to keep (default from config)
                
            Returns:
                Cleanup statistics
            """
            try:
                if keep_latest is None:
                    keep_latest = self.max_cached_models
                
                if not self.cache_cleanup_enabled:
                    logger.info("🚫 Cache cleanup disabled")
                    return {"cleanup_enabled": False}
                
                logger.info(f"🧹 Starting cache cleanup (keeping latest {keep_latest} models)")
                
                # Get current cache state
                before_count = len(self.model_cache)
                before_models = list(self.model_cache.keys())
                
                if before_count <= keep_latest:
                    logger.info(f"✅ Cache already optimal ({before_count} ≤ {keep_latest})")
                    return {
                        "cleanup_needed": False,
                        "models_before": before_count,
                        "models_after": before_count,
                        "models_removed": [],
                        "models_kept": before_models
                    }
                
                # Sort models by last update time (newest first)
                model_timestamps = []
                for model_name, last_update in self.last_cache_update.items():
                    if model_name in self.model_cache:
                        timestamp = datetime.fromisoformat(last_update) if isinstance(last_update, str) else datetime.now()
                        model_timestamps.append((model_name, timestamp))
                
                # Sort by timestamp (newest first)
                model_timestamps.sort(key=lambda x: x[1], reverse=True)
                
                # Determine which models to keep and remove
                models_to_keep = [name for name, _ in model_timestamps[:keep_latest]]
                models_to_remove = [name for name, _ in model_timestamps[keep_latest:]]
                
                # Remove old models from cache
                removed_models = []
                for model_name in models_to_remove:
                    try:
                        # Remove from model cache
                        if model_name in self.model_cache:
                            del self.model_cache[model_name]
                            logger.info(f"🗑️  Removed {model_name} from cache")
                        
                        # Remove from timestamp tracking
                        if model_name in self.last_cache_update:
                            del self.last_cache_update[model_name]
                        
                        removed_models.append(model_name)
                        
                        # Emit event
                        await self._emit_event('model_cache_removed', {
                            'model_name': model_name,
                            'reason': 'cache_cleanup'
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Error removing {model_name} from cache: {e}")
                
                after_count = len(self.model_cache)
                
                logger.info(f"✅ Cache cleanup completed:")
                logger.info(f"   Before: {before_count} models")
                logger.info(f"   After: {after_count} models")
                logger.info(f"   Removed: {len(removed_models)} models")
                logger.info(f"   Kept: {models_to_keep}")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                return {
                    "cleanup_needed": True,
                    "models_before": before_count,
                    "models_after": after_count,
                    "models_removed": removed_models,
                    "models_kept": models_to_keep,
                    "memory_freed": True
                }
                
            except Exception as e:
                logger.error(f"❌ Error during cache cleanup: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"error": str(e)}
        
    async def _auto_cleanup_after_training(self, model_name: str):
        """
        Automatically clean cache after training if enabled
        """
        try:
            if self.cache_cleanup_enabled and len(self.model_cache) > self.max_cached_models:
                logger.info(f"🧹 Auto-cleanup triggered after training {model_name}")
                cleanup_result = await self.cleanup_model_cache()
                
                if cleanup_result.get("cleanup_needed"):
                    logger.info(f"✅ Auto-cleanup: removed {len(cleanup_result.get('models_removed', []))} models")
                
        except Exception as e:
            logger.warning(f"⚠️  Auto-cleanup failed: {e}")

    async def get_cache_status(self) -> Dict[str, Any]:
            """Get detailed cache status information"""
            try:
                cache_info = {}
                total_memory_estimate = 0
                
                for model_name, model in self.model_cache.items():
                    last_update = self.last_cache_update.get(model_name, "unknown")
                    version = model.get_version() if hasattr(model, 'get_version') else "unknown"
                    is_trained = getattr(model, 'is_trained', False)
                    
                    # Rough memory estimate (very approximate)
                    memory_estimate = 0
                    if hasattr(model, 'model') and model.model:
                        try:
                            import sys
                            memory_estimate = sys.getsizeof(model.model) / 1024 / 1024  # MB
                        except:
                            memory_estimate = 10  # Default estimate
                    
                    total_memory_estimate += memory_estimate
                    
                    cache_info[model_name] = {
                        "version": version,
                        "is_trained": is_trained,
                        "last_updated": last_update,
                        "memory_estimate_mb": round(memory_estimate, 2)
                    }
                
                return {
                    "total_cached_models": len(self.model_cache),
                    "max_cached_models": self.max_cached_models,
                    "cleanup_enabled": self.cache_cleanup_enabled,
                    "total_memory_estimate_mb": round(total_memory_estimate, 2),
                    "models": cache_info,
                    "cleanup_needed": len(self.model_cache) > self.max_cached_models
                }
                
            except Exception as e:
                logger.error(f"Error getting cache status: {e}")
                return {"error": str(e)}
        
    async def force_cache_cleanup(self, keep_latest: int = 1) -> Dict[str, Any]:
        """
        Force immediate cache cleanup (useful for memory pressure)
        
        Args:
            keep_latest: Number of models to keep (default 1 for memory optimization)
        """
        logger.info(f"🚨 Force cache cleanup requested (keep {keep_latest})")
        return await self.cleanup_model_cache(keep_latest)