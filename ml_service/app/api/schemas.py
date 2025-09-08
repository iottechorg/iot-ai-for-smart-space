
# app/api/schemas.py - Enhanced schemas with training support

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    device_type: str = Field(..., description="Type of device")
    device_id: str = Field(..., description="Device identifier")
    data: Dict[str, Any] = Field(..., description="Sensor data")
    model_name: Optional[str] = Field(None, description="Specific model to use")

class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Prediction result")
    confidence: float = Field(..., description="Prediction confidence")
    model_name: str = Field(..., description="Model used")
    model_version: str = Field(..., description="Model version")
    device_type: str = Field(..., description="Device type")
    device_id: str = Field(..., description="Device ID")
    timestamp: str = Field(..., description="Prediction timestamp")
    service_version: str = Field(..., description="Service version")

class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Model to train")
    force_retrain: bool = Field(False, description="Force retraining")
    training_data: Optional[Dict[str, Any]] = Field(None, description="Custom training data")

class TrainingResponse(BaseModel):
    success: bool = Field(..., description="Training success")
    model_name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    training_samples: Optional[int] = Field(None, description="Training samples")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: str = Field(..., description="Completion timestamp")

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    is_trained: bool = Field(..., description="Training status")
    last_updated: Optional[str] = Field(None, description="Last update")
    device_types: List[str] = Field(..., description="Supported device types")
    has_target_generator: bool = Field(..., description="Has target generator")

class ModelsListResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="Models list")
    total_models: int = Field(..., description="Total models")
    storage_healthy: bool = Field(..., description="Storage health")
    target_generators: int = Field(..., description="Target generators count")

class StorageStatusResponse(BaseModel):
    healthy: bool = Field(..., description="Storage health")
    total_files: Optional[int] = Field(None, description="Total files")
    model_files: Optional[int] = Field(None, description="Model files")
    models_in_storage: Optional[int] = Field(None, description="Models in storage")
    model_details: Optional[Dict[str, List[str]]] = Field(None, description="Model details")
    error: Optional[str] = Field(None, description="Error message")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp")
    ml_service_initialized: bool = Field(..., description="ML service status")
    models_loaded: int = Field(..., description="Models loaded")
    storage_healthy: bool = Field(..., description="Storage health")
    version: str = Field(..., description="Service version")