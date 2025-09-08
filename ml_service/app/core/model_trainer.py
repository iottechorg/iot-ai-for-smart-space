# ml_service/app/core/model_trainer.py - FIXED VERSION
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .model_registry import ModelRegistry
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages model training jobs with proper storage integration"""
    
    def __init__(self, storage_manager: StorageManager, model_registry: ModelRegistry):
        self.storage_manager = storage_manager
        self.model_registry = model_registry
        self.active_jobs = {}
    
    async def train_model(
        self, 
        model_name: str, 
        training_data: Dict[str, Any],
        test_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Train a model with the provided data"""
        try:
            # Get model configuration
            config = self.model_registry.get_model_config(model_name)
            if not config:
                logger.error(f"Model configuration for {model_name} not found")
                return False
            
            # Get model instance or create new one
            model = await self.model_registry.get_model(model_name)
            
            if not model:
                logger.error(f"Model {model_name} not found in registry")
                return False
            
            # Start training
            job_id = f"train_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.active_jobs[job_id] = {
                "model_name": model_name,
                "status": "running",
                "started_at": datetime.now().isoformat()
            }
            
            logger.info(f"Starting training job {job_id} for model {model_name}")
            logger.info(f"Training data: {len(training_data.get('features', []))} samples")
            
            # Train the model
            success = await model.train(training_data)
            
            if success:
                # Register the updated model
                await self.model_registry.register_model(model)
                
                # Save to storage
                save_success = await self.model_registry.save_model(model_name)
                if not save_success:
                    logger.warning(f"Model {model_name} trained successfully but failed to save to storage")
                
                # Evaluate if test data provided
                if test_data:
                    try:
                        metrics = await model.evaluate(test_data)
                        logger.info(f"Model {model_name} evaluation: {metrics}")
                        self.active_jobs[job_id]["metrics"] = metrics
                    except Exception as e:
                        logger.warning(f"Evaluation failed for {model_name}: {e}")
                
                self.active_jobs[job_id]["status"] = "completed"
                self.active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                logger.info(f"Training job {job_id} completed successfully")
                return True
            else:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                self.active_jobs[job_id]["error"] = "Training failed"
                
                logger.error(f"Training job {job_id} failed")
                return False
        
        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Update job status
            job_id = f"train_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                self.active_jobs[job_id]["error"] = str(e)
            
            return False
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        return self.active_jobs.get(job_id, {"status": "not_found"})
    
    def get_all_training_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all training jobs"""
        return self.active_jobs
    
    async def validate_training_data(self, model_name: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training data for a model"""
        try:
            config = self.model_registry.get_model_config(model_name)
            if not config:
                return {"valid": False, "error": "Model configuration not found"}
            
            features = training_data.get('features', [])
            targets = training_data.get('targets', [])
            
            if not features:
                return {"valid": False, "error": "No features provided"}
            
            if not targets:
                return {"valid": False, "error": "No targets provided"}
            
            if len(features) != len(targets):
                return {"valid": False, "error": "Features and targets length mismatch"}
            
            # Check required features
            required_features = config.get('training', {}).get('features', [])
            if not required_features:
                return {"valid": False, "error": "No required features configured"}
            
            # Check if first sample has all required features
            if features and isinstance(features[0], dict):
                sample_features = set(features[0].keys())
                missing_features = set(required_features) - sample_features
                if missing_features:
                    return {
                        "valid": False, 
                        "error": f"Missing required features: {list(missing_features)}"
                    }
            
            return {
                "valid": True,
                "samples": len(features),
                "features": len(required_features),
                "model_type": config.get('inference', {}).get('output_format', 'classification')
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}