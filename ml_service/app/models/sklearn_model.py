# ml_service/app/models/sklearn_model.py - FIXED VERSION WITH PROPER STORAGE
import logging
import pickle
import os
import json
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

from ..core.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class SklearnModel(ModelInterface):
    """Scikit-learn model implementation with proper storage integration"""
    
    def __init__(self, name: str, config: Dict[str, Any], storage_manager=None):
        self.name = name
        self.config = config
        self.storage_manager = storage_manager
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.version = "0.1.0"
        self.is_trained = False
        
        # Get model algorithm from config
        self.algorithm = config.get('model', {}).get('algorithm', 'random_forest')
        self.model_type = config.get('inference', {}).get('output_format', 'classification')
        
    def get_name(self) -> str:
        return self.name
    
    def get_version(self) -> str:
        return self.version
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction"""
        try:
            if not self.is_trained:
                logger.warning(f"Model {self.name} not trained, using fallback prediction")
                return await self._fallback_prediction(input_data)
            
            # Extract features
            features = self._extract_features(input_data)
            if features is None:
                logger.error("Failed to extract features from input data")
                return {"prediction": None, "confidence": 0.0}
            
            # Prepare features
            features_array = np.array([features])
            
            # Scale features if scaler exists
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            # Make prediction
            if self.model_type == 'classification':
                prediction = self.model.predict(features_array)[0]
                
                # Get confidence if available
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features_array)[0]
                    confidence = float(np.max(probabilities))
                else:
                    confidence = 0.8  # Default confidence
                
                # Decode label if label encoder exists
                if self.label_encoder:
                    prediction = self.label_encoder.inverse_transform([prediction])[0]
                
            else:  # regression
                prediction = float(self.model.predict(features_array)[0])
                confidence = 0.8  # Default confidence for regression
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_version": self.version
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return await self._fallback_prediction(input_data)
    
    async def _fallback_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction when model is not trained"""
        try:
            device_type = input_data.get('device_type', '')
            
            if 'traffic' in device_type.lower():
                # Traffic prediction fallback
                vehicle_count = input_data.get('vehicle_count', 0)
                avg_speed = input_data.get('average_speed', 50)
                
                if vehicle_count > 50 and avg_speed < 30:
                    prediction = "high"
                elif vehicle_count > 25 and avg_speed < 50:
                    prediction = "medium"
                else:
                    prediction = "low"
                
                return {"prediction": prediction, "confidence": 0.6}
                
            elif 'water' in device_type.lower():
                # Water level prediction fallback
                water_level = input_data.get('water_level', 0)
                flow_rate = input_data.get('flow_rate', 0)
                
                risk = (water_level / 5.0) * 0.6 + (flow_rate / 10.0) * 0.4
                risk_percentage = min(100, max(0, risk * 100))
                
                return {"prediction": risk_percentage, "confidence": 0.6}
            
            return {"prediction": "unknown", "confidence": 0.5}
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {"prediction": "error", "confidence": 0.0}
    
    def _extract_features(self, input_data: Dict[str, Any]) -> Optional[list]:
        """Extract features from input data"""
        try:
            features_config = self.config.get('training', {}).get('features', [])
            features = []
            
            for feature_name in features_config:
                if feature_name in input_data:
                    try:
                        feature_value = float(input_data[feature_name])
                        features.append(feature_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert feature {feature_name} to float")
                        features.append(0.0)  # Default value
                else:
                    logger.warning(f"Feature {feature_name} not found in input data")
                    features.append(0.0)  # Default value
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    async def train(self, training_data: Dict[str, Any]) -> bool:
        """Train the model"""
        try:
            features = training_data.get('features', [])
            targets = training_data.get('targets', [])
            
            if not features or not targets:
                logger.error("No training data provided")
                return False
            
            if len(features) != len(targets):
                logger.error("Features and targets length mismatch")
                return False
            
            logger.info(f"Training {self.name} with {len(features)} samples")
            
            # Convert to numpy arrays
            X = np.array([list(f.values()) if isinstance(f, dict) else f for f in features])
            y = np.array(targets)
            
            # Handle categorical targets for classification
            if self.model_type == 'classification':
                if y.dtype == 'object' or isinstance(y[0], str):
                    self.label_encoder = LabelEncoder()
                    y = self.label_encoder.fit_transform(y)
            
            # Split data
            validation_split = self.config.get('training', {}).get('validation_split', 0.2)
            if len(X) > 10:  # Only split if we have enough data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create model based on algorithm and type
            if self.model_type == 'classification':
                if self.algorithm == 'random_forest':
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif self.algorithm == 'logistic_regression':
                    self.model = LogisticRegression(random_state=42)
                else:
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # regression
                if self.algorithm == 'random_forest':
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif self.algorithm == 'linear_regression':
                    self.model = LinearRegression()
                else:
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            
            if self.model_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Model {self.name} training completed. Accuracy: {accuracy:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                logger.info(f"Model {self.name} training completed. MSE: {mse:.3f}")
            
            self.is_trained = True
            self.version = f"1.0.{int(datetime.now().timestamp())}"
            
            # Save model to storage
            if self.storage_manager:
                model_path = f"models/{self.name}/latest"
                success = await self.save(model_path)
                if success:
                    logger.info(f"Model {self.name} saved to storage at {model_path}")
                else:
                    logger.error(f"Failed to save model {self.name} to storage")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model {self.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model"""
        try:
            if not self.is_trained:
                return {"error": "Model not trained"}
            
            features = test_data.get('features', [])
            targets = test_data.get('targets', [])
            
            if not features or not targets:
                return {"error": "No test data provided"}
            
            X = np.array([list(f.values()) if isinstance(f, dict) else f for f in features])
            y = np.array(targets)
            
            # Apply same preprocessing
            if self.label_encoder and self.model_type == 'classification':
                y = self.label_encoder.transform(y)
            
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            if self.model_type == 'classification':
                accuracy = accuracy_score(y, y_pred)
                return {"accuracy": accuracy}
            else:
                mse = mean_squared_error(y, y_pred)
                return {"mse": mse}
                
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}
    
    async def save(self, path: str) -> bool:
        """Save the model to storage"""
        try:
            if not self.storage_manager:
                logger.error("No storage manager available")
                return False
            
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model
                model_file = os.path.join(temp_dir, "model.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(self.model, f)
                
                # Save scaler
                if self.scaler:
                    scaler_file = os.path.join(temp_dir, "scaler.pkl")
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(self.scaler, f)
                
                # Save label encoder
                if self.label_encoder:
                    encoder_file = os.path.join(temp_dir, "label_encoder.pkl")
                    with open(encoder_file, 'wb') as f:
                        pickle.dump(self.label_encoder, f)
                
                # Save metadata
                metadata = {
                    "name": self.name,
                    "version": self.version,
                    "algorithm": self.algorithm,
                    "model_type": self.model_type,
                    "is_trained": self.is_trained,
                    "config": self.config,
                    "saved_at": datetime.utcnow().isoformat()
                }
                
                metadata_file = os.path.join(temp_dir, "metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Upload all files to storage
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    object_key = f"{path}/{filename}"
                    
                    await self.storage_manager.upload_file(file_path, object_key)
                    logger.debug(f"Uploaded {filename} to {object_key}")
            
            logger.info(f"Model {self.name} saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def load(self, path: str) -> bool:
        """Load the model from storage"""
        try:
            if not self.storage_manager:
                logger.error("No storage manager available")
                return False
            
            # Create temporary directory for downloaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download all files from storage
                files_to_download = ["model.pkl", "metadata.json", "scaler.pkl", "label_encoder.pkl"]
                
                for filename in files_to_download:
                    object_key = f"{path}/{filename}"
                    file_path = os.path.join(temp_dir, filename)
                    
                    try:
                        if await self.storage_manager.file_exists(object_key):
                            await self.storage_manager.download_file(object_key, file_path)
                            logger.debug(f"Downloaded {object_key}")
                    except Exception as e:
                        if filename == "model.pkl":
                            logger.error(f"Failed to download required file {filename}: {e}")
                            return False
                        else:
                            logger.warning(f"Optional file {filename} not found: {e}")
                
                # Load metadata
                metadata_file = os.path.join(temp_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.version = metadata.get('version', self.version)
                        self.algorithm = metadata.get('algorithm', self.algorithm)
                        self.model_type = metadata.get('model_type', self.model_type)
                        self.is_trained = metadata.get('is_trained', False)
                
                # Load model
                model_file = os.path.join(temp_dir, "model.pkl")
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                else:
                    logger.error("Model file not found")
                    return False
                
                # Load scaler
                scaler_file = os.path.join(temp_dir, "scaler.pkl")
                if os.path.exists(scaler_file):
                    with open(scaler_file, 'rb') as f:
                        self.scaler = pickle.load(f)
                
                # Load label encoder
                encoder_file = os.path.join(temp_dir, "label_encoder.pkl")
                if os.path.exists(encoder_file):
                    with open(encoder_file, 'rb') as f:
                        self.label_encoder = pickle.load(f)
            
            logger.info(f"Model {self.name} loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False