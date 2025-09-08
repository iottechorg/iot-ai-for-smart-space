# ml_service/app/core/model_registry.py - FIXED VERSION
from typing import Dict, Any, List, Optional
import logging
from .model_interface import ModelInterface
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for ML models with proper storage integration"""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        self.models: Dict[str, ModelInterface] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
    
    async def register_model(self, model: ModelInterface) -> bool:
        """Register a model in the registry"""
        try:
            model_name = model.get_name()
            self.models[model_name] = model
            logger.info(f"Registered model: {model_name} v{model.get_version()}")
            return True
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    def register_model_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Register a model configuration"""
        self.model_configs[model_name] = config
        logger.info(f"Registered config for model: {model_name}")
        return True
    
    async def get_model(self, model_name: str) -> Optional[ModelInterface]:
        """Get a model by name"""
        return self.models.get(model_name)
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a model configuration"""
        return self.model_configs.get(model_name)
    
    async def load_models(self) -> int:
        """Load all registered models from storage - FIXED WITH PROPER STORAGE"""
        loaded_count = 0
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Attempting to load model: {model_name}")
                
                # Determine model type and create appropriate model instance
                model_type = config.get('model', {}).get('type')
                
                if model_type == 'sklearn':
                    try:
                        from ..models.sklearn_model import SklearnModel
                        model = SklearnModel(model_name, config, self.storage_manager)
                    except ImportError as e:
                        logger.error(f"Cannot import SklearnModel: {e}")
                        logger.info(f"Skipping model {model_name} (sklearn not available)")
                        continue
                elif model_type == 'tensorflow':
                    try:
                        from ..models.tensorflow_model import TensorflowModel
                        model = TensorflowModel(model_name, config, self.storage_manager)
                    except ImportError as e:
                        logger.error(f"Cannot import TensorflowModel: {e}")
                        logger.info(f"Skipping model {model_name} (tensorflow not available)")
                        continue
                else:
                    logger.warning(f"Unsupported model type: {model_type}")
                    continue
                
                # Check if model exists in storage
                model_path = f"models/{model_name}/latest"
                if await self.storage_manager.file_exists(f"{model_path}/model.pkl"):
                    logger.info(f"Found saved model for {model_name}, loading...")
                    # Load the model
                    if await model.load(model_path):
                        await self.register_model(model)
                        loaded_count += 1
                        logger.info(f"Successfully loaded model {model_name} from storage")
                    else:
                        logger.error(f"Failed to load model {model_name} from storage")
                        # Still register the model instance for future training
                        await self.register_model(model)
                else:
                    logger.info(f"No saved model found for {model_name}, registering new instance")
                    # Register the model instance for future training
                    await self.register_model(model)
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Loaded {loaded_count} models from storage, registered {len(self.models)} total models")
        return loaded_count
    
    def get_all_model_configs(self) -> List[str]:
        """Get the names of all registered model configurations"""
        return list(self.model_configs.keys())

    async def get_models_for_device_type(self, device_type: str) -> List[Dict[str, Any]]:
        """Get models that can process data from a specific device type"""
        matching_models = []
        
        for model_name, config in self.model_configs.items():
            device_types = config.get('model', {}).get('device_types', [])
            
            if device_type in device_types:
                # Check if model is loaded
                if model_name in self.models:
                    model = self.models[model_name]
                    matching_models.append({
                        'name': model_name,
                        'version': model.get_version(),
                        'type': config.get('model', {}).get('type'),
                        'status': 'loaded' if getattr(model, 'is_trained', False) else 'not_trained'
                    })
                else:
                    # Model config exists but not loaded yet
                    matching_models.append({
                        'name': model_name,
                        'version': '0.1.0',
                        'type': config.get('model', {}).get('type'),
                        'status': 'config_only'
                    })
        
        return matching_models

    async def save_model(self, model_name: str) -> bool:
        """Save a specific model to storage"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found in registry")
                return False
            
            model = self.models[model_name]
            model_path = f"models/{model_name}/latest"
            
            success = await model.save(model_path)
            if success:
                logger.info(f"Model {model_name} saved to storage")
                
                # Also save with version
                versioned_path = f"models/{model_name}/{model.get_version()}"
                await model.save(versioned_path)
                logger.info(f"Model {model_name} also saved with version {model.get_version()}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False

    async def list_stored_models(self) -> List[Dict[str, Any]]:
        """List all models stored in storage"""
        try:
            stored_models = []
            
            # List all files with 'models/' prefix
            files = await self.storage_manager.list_files("models/")
            
            # Group by model name
            model_names = set()
            for file_path in files:
                parts = file_path.split('/')
                if len(parts) >= 2:
                    model_names.add(parts[1])
            
            for model_name in model_names:
                # Check for latest version
                latest_path = f"models/{model_name}/latest"
                if await self.storage_manager.file_exists(f"{latest_path}/metadata.json"):
                    try:
                        # Get metadata
                        metadata = await self.storage_manager.get_file_metadata(f"{latest_path}/metadata.json")
                        
                        stored_models.append({
                            'name': model_name,
                            'path': latest_path,
                            'last_modified': metadata.get('last_modified') if metadata else None,
                            'in_registry': model_name in self.models
                        })
                    except Exception as e:
                        logger.warning(f"Could not get metadata for {model_name}: {e}")
                        stored_models.append({
                            'name': model_name,
                            'path': latest_path,
                            'last_modified': None,
                            'in_registry': model_name in self.models
                        })
            
            return stored_models
            
        except Exception as e:
            logger.error(f"Error listing stored models: {e}")
            return []


