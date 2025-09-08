# ml_service/app/models/tensorflow_model.py
import os
import tempfile
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from core.model_interface import ModelInterface
from core.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class TensorflowModel(ModelInterface):
    """TensorFlow model implementation"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.version = config.get('model', {}).get('version', '1.0.0')
        self.model = None
        self.storage_manager = StorageManager()
    
    def get_name(self) -> str:
        return self.name
    
    def get_version(self) -> str:
        return self.version
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the model"""
        if not self.model:
            logger.error(f"Model {self.name} not loaded")
            return {}
        
        try:
            # Extract features according to the model configuration
            features = self.config.get('training', {}).get('features', [])
            
            # Create feature vector
            input_values = []
            for feature in features:
                if feature in input_data:
                    input_values.append(input_data[feature])
                else:
                    logger.warning(f"Missing feature {feature} in input data")
                    return {}
            
            # Convert to numpy array
            input_array = np.array([input_values], dtype=np.float32)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            
            # Get confidence (depends on the model type)
            confidence = 0.0
            output_format = self.config.get('inference', {}).get('output_format', 'classification')
            
            if output_format == 'classification':
                # For classification, get the max probability
                confidence = float(np.max(prediction))
                # Get the predicted class
                prediction_value = int(np.argmax(prediction))
                
                # Map to class labels if available
                class_labels = self.config.get('inference', {}).get('class_labels', [])
                if class_labels and prediction_value < len(class_labels):
                    prediction_value = class_labels[prediction_value]
            else:
                # For regression, just use the value
                prediction_value = float(prediction[0])
                confidence = 0.95  # Default confidence for regression
            
            # Format output according to model configuration
            result = {
                "prediction": prediction_value,
                "confidence": confidence,
                "model_name": self.name,
                "model_version": self.version,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add any additional output fields from configuration
            for key, value in self.config.get('inference', {}).get('output_fields', {}).items():
                result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with {self.name}: {e}")
            return {}
    
    async def train(self, training_data: Dict[str, Any]) -> bool:
        """Train the model using provided data"""
        try:
            # Import TensorFlow here to avoid dependency issues
            import tensorflow as tf
            
            # Extract configuration
            features = self.config.get('training', {}).get('features', [])
            target = self.config.get('training', {}).get('target')
            model_architecture = self.config.get('training', {}).get('architecture', [])
            parameters = self.config.get('training', {}).get('parameters', {})
            
            if not features or not target or not model_architecture:
                logger.error(f"Missing required training configuration for {self.name}")
                return False
            
            # Extract training data
            X_data = training_data.get('features', [])
            y_data = training_data.get('targets', [])
            
            if not X_data or not y_data:
                logger.error("Missing features or targets in training data")
                return False
            
            # Convert to numpy arrays
            X = np.array(X_data, dtype=np.float32)
            y = np.array(y_data)
            
            # Determine if classification or regression
            output_format = self.config.get('inference', {}).get('output_format', 'classification')
            
            # Create model based on architecture config
            model = tf.keras.Sequential()
            
            # Add input layer
            input_shape = (len(features),)
            model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
            
            # Add hidden layers
            for layer in model_architecture:
                layer_type = layer.get('type', 'dense')
                units = layer.get('units', 64)
                activation = layer.get('activation', 'relu')
                
                if layer_type.lower() == 'dense':
                    model.add(tf.keras.layers.Dense(units, activation=activation))
                elif layer_type.lower() == 'dropout':
                    rate = layer.get('rate', 0.2)
                    model.add(tf.keras.layers.Dropout(rate))
            
            # Add output layer
            if output_format == 'classification':
                # Get number of classes
                num_classes = len(set(y.flatten()))
                activation = 'softmax' if num_classes > 2 else 'sigmoid'
                output_units = num_classes if num_classes > 2 else 1
                model.add(tf.keras.layers.Dense(output_units, activation=activation))
                
                # Compile for classification
                loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
                model.compile(
                    optimizer=parameters.get('optimizer', 'adam'),
                    loss=loss,
                    metrics=['accuracy']
                )
            else:
                # Regression output
                model.add(tf.keras.layers.Dense(1))
                
                # Compile for regression
                model.compile(
                    optimizer=parameters.get('optimizer', 'adam'),
                    loss=parameters.get('loss', 'mse'),
                    metrics=[parameters.get('metrics', ['mae'])]
                )
            
            # Train the model
            epochs = parameters.get('epochs', 50)
            batch_size = parameters.get('batch_size', 32)
            validation_split = parameters.get('validation_split', 0.2)
            
            model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            # Save the model
            self.model = model
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            self.version = version
            
            model_path = f"models/{self.name}/{version}"
            await self.save(model_path)
            
            # Also save as latest
            await self.save(f"models/{self.name}/latest")
            
            logger.info(f"Successfully trained model {self.name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model {self.name}: {e}")
            return False
    
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model using test data"""
        if not self.model:
            logger.error(f"Model {self.name} not loaded")
            return {}
        
        try:
            # Extract test data
            X_test = test_data.get('features', [])
            y_test = test_data.get('targets', [])
            
            if not X_test or not y_test:
                logger.error("Missing features or targets in test data")
                return {}
            
            # Convert to numpy arrays
            X = np.array(X_test, dtype=np.float32)
            y = np.array(y_test)
            
            # Evaluate model
            scores = self.model.evaluate(X, y, verbose=0)
            
            # Get metrics names
            metric_names = self.model.metrics_names
            
            # Create metrics dictionary
            metrics = {
                metric_names[i]: float(scores[i]) 
                for i in range(len(metric_names))
            }
            
            # Add sample count
            metrics["n_samples"] = len(y)
            
            # Add prediction time metrics
            import time
            start_time = time.time()
            self.model.predict(X[:1])  # Warm up
            
            # Measure prediction time
            times = []
            for _ in range(10):
                t_start = time.time()
                self.model.predict(X[:1])
                times.append(time.time() - t_start)
            
            avg_prediction_time = sum(times) / len(times)
            metrics["avg_prediction_time_ms"] = avg_prediction_time * 1000
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {self.name}: {e}")
            return {}
    
    async def save(self, path: str) -> bool:
        """Save the model to storage"""
        if not self.model:
            logger.error(f"Cannot save model {self.name}: model not initialized")
            return False
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            model_dir = os.path.join(temp_dir, "model")
            
            # Save TensorFlow model
            self.model.save(model_dir)
            
            # Save model configuration
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f)
            
            # Save metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            metadata = {
                "name": self.name,
                "version": self.version,
                "framework": "tensorflow",
                "created_at": datetime.utcnow().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Upload files to storage
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, temp_dir)
                    object_key = f"{path}/{rel_path}"
                    
                    await self.storage_manager.upload_file(file_path, object_key)
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info(f"Saved model {self.name} v{self.version} to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {self.name}: {e}")
            return False
    
    async def load(self, path: str) -> bool:
        """Load the model from storage"""
        try:
            # Import TensorFlow here to avoid dependency issues
            import tensorflow as tf
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            model_dir = os.path.join(temp_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            # List all files in the path
            files = await self.storage_manager.list_files(path)
            
            # Download all files
            for file_key in files:
                if file_key.startswith(path):
                    rel_path = file_key[len(path)+1:]  # Remove path/ prefix
                    local_path = os.path.join(temp_dir, rel_path)
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file
                    await self.storage_manager.download_file(file_key, local_path)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.version = metadata.get("version", self.version)
            
            # Load configuration
            config_path = os.path.join(temp_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Load model
            model_dir = os.path.join(temp_dir, "model")
            if os.path.exists(model_dir):
                self.model = tf.keras.models.load_model(model_dir)
                
                logger.info(f"Loaded model {self.name} v{self.version} from {path}")
                
                # Clean up
                import shutil
                shutil.rmtree(temp_dir)
                
                return True
            else:
                logger.error(f"Model directory not found at {model_dir}")
                return False
            
        except Exception as e:
            logger.error(f"Error loading model {self.name} from {path}: {e}")
            return False