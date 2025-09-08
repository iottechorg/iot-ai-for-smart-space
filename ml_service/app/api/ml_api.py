# app/api/ml_api.py - Enhanced API with training debug endpoints

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from typing import Optional

from .schemas import (
    PredictionRequest, PredictionResponse, TrainingRequest, TrainingResponse,
    ModelsListResponse, StorageStatusResponse, HealthResponse
)
from ..core.ml_service import MLService

logger = logging.getLogger(__name__)

class MLServiceAPI:
    """Enhanced HTTP API wrapper for ML service with training debug"""
    
    def __init__(self, ml_service: MLService):
        self.ml_service = ml_service
        self.app = FastAPI(
            title="Smart City ML Service API",
            description="Machine Learning service for IoT device predictions with enhanced training",
            version="2.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._setup_event_handlers()
    
    def _setup_routes(self):
        """Setup all API routes with enhanced training support"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Enhanced health check with training status"""
            try:
                model_info = await self.ml_service.get_model_info()
                storage_status = await self.ml_service.get_storage_status()
                
                return HealthResponse(
                    status="healthy" if self.ml_service.is_initialized else "initializing",
                    timestamp=datetime.utcnow().isoformat(),
                    ml_service_initialized=self.ml_service.is_initialized,
                    models_loaded=model_info.get('total_models', 0),
                    storage_healthy=storage_status.get('healthy', False),
                    version="2.1.0"
                )
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.utcnow().isoformat(),
                    ml_service_initialized=False,
                    models_loaded=0,
                    storage_healthy=False,
                    version="2.1.0"
                )
        
        @self.app.get("/models", response_model=ModelsListResponse)
        async def list_models():
            """List all available models with training info"""
            try:
                model_info = await self.ml_service.get_model_info()
                return ModelsListResponse(**model_info)
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_name}")
        async def get_model_info(model_name: str):
            """Get detailed information about a specific model"""
            try:
                model_info = await self.ml_service.get_model_info(model_name)
                return model_info
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make a prediction"""
            try:
                result = await self.ml_service.predict(
                    request.device_type,
                    request.device_id,
                    request.data,
                    request.model_name
                )
                return PredictionResponse(**result)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/train/{model_name}", response_model=TrainingResponse)
        async def train_model(
            model_name: str, 
            request: Optional[TrainingRequest] = None,
            background_tasks: BackgroundTasks = None
        ):
            """Enhanced training endpoint with better error reporting"""
            try:
                if request is None:
                    request = TrainingRequest(model_name=model_name)
                
                # Check if model exists
                try:
                    await self.ml_service.get_model_info(model_name)
                except ValueError:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                # For immediate training (synchronous)
                result = await self.ml_service.train_model(
                    model_name,
                    request.training_data,
                    request.force_retrain
                )
                return TrainingResponse(**result)
                    
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Training error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_name}/save")
        async def save_model(model_name: str):
            """Save model to storage"""
            try:
                success = await self.ml_service.save_model(model_name)
                if success:
                    return {"message": f"Model {model_name} saved successfully"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to save model {model_name}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load model from storage"""
            try:
                success = await self.ml_service.load_model(model_name)
                if success:
                    return {"message": f"Model {model_name} loaded successfully"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/storage/status", response_model=StorageStatusResponse)
        async def storage_status():
            """Get storage status"""
            try:
                status = await self.ml_service.get_storage_status()
                return StorageStatusResponse(**status)
            except Exception as e:
                logger.error(f"Error getting storage status: {e}")
                return StorageStatusResponse(
                    healthy=False,
                    error=str(e)
                )
        
        @self.app.get("/cache/status")
        async def cache_status():
            """Get model cache status and memory usage"""
            try:
                status = await self.ml_service.get_cache_status()
                return status
            except Exception as e:
                logger.error(f"Error getting cache status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cache/cleanup")
        async def cleanup_cache(keep_latest: int = 3):
            """Manually trigger cache cleanup"""
            try:
                if keep_latest < 1 or keep_latest > 10:
                    raise HTTPException(status_code=400, detail="keep_latest must be between 1 and 10")
                
                result = await self.ml_service.cleanup_model_cache(keep_latest)
                return result
            except Exception as e:
                logger.error(f"Error cleaning cache: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cache/force-cleanup")
        async def force_cleanup_cache():
            """Force aggressive cache cleanup (keep only 1 model)"""
            try:
                result = await self.ml_service.force_cache_cleanup(keep_latest=1)
                return result
            except Exception as e:
                logger.error(f"Error force cleaning cache: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/cache/memory")
        async def cache_memory_info():
            """Get memory usage information"""
            try:
                import psutil
                import os
                
                # Process memory info
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                
                # Cache status
                cache_status = await self.ml_service.get_cache_status()
                
                return {
                    "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                    "process_memory_percent": round(process.memory_percent(), 2),
                    "cache_memory_estimate_mb": cache_status.get("total_memory_estimate_mb", 0),
                    "cached_models": cache_status.get("total_cached_models", 0),
                    "cleanup_recommended": cache_status.get("cleanup_needed", False)
                }
            except ImportError:
                return {
                    "error": "psutil not available for memory monitoring",
                    "cache_status": await self.ml_service.get_cache_status()
                }
            except Exception as e:
                logger.error(f"Error getting memory info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # DEBUG ENDPOINTS for training troubleshooting
        @self.app.get("/debug/target-generators")
        async def debug_target_generators():
            """DEBUG: Check target generators status"""
            try:
                generators_info = {}
                for model_name in self.ml_service.target_generators:
                    generator = self.ml_service.target_generators[model_name]
                    config = self.ml_service.model_registry.get_model_config(model_name)
                    
                    generators_info[model_name] = {
                        "type": type(generator).__name__,
                        "config": config.get('training', {}).get('target_generation', {}) if config else {}
                    }
                
                return {
                    "target_generators": generators_info,
                    "total_generators": len(self.ml_service.target_generators),
                    "model_configs": len(self.ml_service.model_registry.model_configs)
                }
            except Exception as e:
                logger.error(f"Error getting target generators info: {e}")
                return {"error": str(e)}
        
        @self.app.get("/debug/training-data/{model_name}")
        async def debug_training_data(model_name: str):
            """DEBUG: Preview training data generation for a model"""
            try:
                preview = await self.ml_service.get_training_data_preview(model_name)
                return preview
            except Exception as e:
                logger.error(f"Error getting training data preview: {e}")
                return {"error": str(e)}
        
        @self.app.post("/debug/test-storage")
        async def debug_test_storage():
            """DEBUG: Test storage connectivity and operations"""
            try:
                import tempfile
                import os
                
                # Test basic storage operations
                test_key = "debug/test_file.txt"
                test_content = f"Storage test at {datetime.utcnow().isoformat()}"
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(test_content)
                    test_file = f.name
                
                try:
                    # Upload
                    await self.ml_service.storage_manager.upload_file(test_file, test_key)
                    
                    # Check exists
                    exists = await self.ml_service.storage_manager.file_exists(test_key)
                    
                    # List files
                    files = await self.ml_service.storage_manager.list_files("debug/")
                    
                    # Download to verify
                    download_file = tempfile.mktemp()
                    await self.ml_service.storage_manager.download_file(test_key, download_file)
                    
                    with open(download_file, 'r') as f:
                        downloaded_content = f.read()
                    
                    # Clean up
                    await self.ml_service.storage_manager.delete_file(test_key)
                    os.unlink(test_file)
                    os.unlink(download_file)
                    
                    return {
                        "upload_success": True,
                        "file_exists": exists,
                        "files_listed": len(files),
                        "content_match": downloaded_content == test_content,
                        "storage_type": type(self.ml_service.storage_manager.storage).__name__,
                        "test_content": test_content[:50] + "..." if len(test_content) > 50 else test_content
                    }
                
                except Exception as e:
                    # Clean up on error
                    try:
                        os.unlink(test_file)
                    except:
                        pass
                    raise e
                    
            except Exception as e:
                logger.error(f"Storage test error: {e}")
                return {"error": str(e)}
        
        @self.app.post("/debug/force-train/{model_name}")
        async def debug_force_train(model_name: str):
            """DEBUG: Force training with detailed logging"""
            try:
                logger.info(f"🐛 DEBUG: Force training {model_name}")
                
                # Check target generator
                has_generator = model_name in self.ml_service.target_generators
                generator_type = type(self.ml_service.target_generators[model_name]).__name__ if has_generator else None
                
                logger.info(f"🐛 Target generator: {has_generator} ({generator_type})")
                
                # Check storage health
                storage_healthy = await self.ml_service._check_storage_health()
                logger.info(f"🐛 Storage healthy: {storage_healthy}")
                
                # Get training data preview first
                preview = await self.ml_service.get_training_data_preview(model_name)
                logger.info(f"🐛 Training data preview: {preview.get('features_count', 0)} features, {preview.get('targets_count', 0)} targets")
                
                if preview.get('features_count', 0) == 0:
                    return {
                        "error": "No training data available",
                        "debug_info": {
                            "has_target_generator": has_generator,
                            "generator_type": generator_type,
                            "storage_healthy": storage_healthy,
                            "preview": preview
                        }
                    }
                
                # Proceed with training
                result = await self.ml_service.train_model(model_name, force_retrain=True)
                
                return {
                    "training_result": result,
                    "debug_info": {
                        "has_target_generator": has_generator,
                        "generator_type": generator_type,
                        "storage_healthy": storage_healthy,
                        "preview": preview
                    }
                }
                
            except Exception as e:
                logger.error(f"🐛 DEBUG training error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {"error": str(e)}
        
        @self.app.get("/debug/logs/{model_name}")
        async def debug_get_recent_logs(model_name: str):
            """DEBUG: Get recent logs related to a model (placeholder)"""
            # This would require log aggregation - for now return placeholder
            return {
                "message": "Log endpoint placeholder",
                "model_name": model_name,
                "note": "Implement log aggregation for detailed debugging"
            }
        
        @self.app.get("/version")
        async def get_version():
            """Get API version information"""
            return {
                "api_version": "2.1.0",
                "service": "ml_service",
                "timestamp": datetime.utcnow().isoformat(),
                "features": [
                    "enhanced_training",
                    "target_generation",
                    "debug_endpoints",
                    "storage_integration"
                ]
            }
    
    def _setup_event_handlers(self):
        """Setup event handlers for ML service events"""
        async def log_event_handler(event):
            """Enhanced event logging"""
            event_type = event.get('type')
            data = event.get('data', {})
            
            if event_type in ['training_started', 'training_completed', 'training_failed']:
                logger.info(f"🚂 Training Event: {event_type} - {data.get('model_name')} - {data}")
            elif event_type in ['prediction_made', 'prediction_error']:
                logger.debug(f"🔮 Prediction Event: {event_type} - {data.get('model_name')} - {data.get('device_type')}")
            elif event_type in ['model_saved', 'model_loaded']:
                logger.info(f"💾 Storage Event: {event_type} - {data.get('model_name')}")
            else:
                logger.info(f"📝 ML Event: {event_type} - {data}")
        
        # Add event handler to ML service
        self.ml_service.add_event_handler(log_event_handler)

