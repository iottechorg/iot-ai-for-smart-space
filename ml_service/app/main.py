


# app/main.py - Clean orchestration
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Dict, Any
import json

import uvicorn
from .core.ml_service import MLService
from .api.ml_api import MLServiceAPI
from .messaging.mqtt_handler import MQTTHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLServiceOrchestrator:
    """Orchestrates all service components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ml_service = None
        self.api = None
        self.mqtt_handler = None
        self.is_running = False
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing ML Service Orchestrator...")
            
            # Initialize ML service
            self.ml_service = MLService(self.config)
            if not await self.ml_service.initialize():
                return False
            
            # Initialize API wrapper
            self.api = MLServiceAPI(self.ml_service)
            
            # Initialize MQTT handler
            self.mqtt_handler = MQTTHandler(self.config)
            self.mqtt_handler.set_message_handler(self._handle_mqtt_message)
            
            if not await self.mqtt_handler.connect():
                logger.warning("MQTT connection failed, continuing without MQTT")
            
            self.is_running = True
            logger.info("ML Service Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all components gracefully"""
        logger.info("Shutting down ML Service Orchestrator...")
        self.is_running = False
        
        if self.mqtt_handler:
            self.mqtt_handler.disconnect()
        
        logger.info("ML Service Orchestrator shutdown complete")
    
    async def _handle_mqtt_message(self, message: Dict[str, Any]):
        """Handle incoming MQTT messages"""
        try:
            # Make prediction
            prediction = await self.ml_service.predict(
                message['device_type'],
                message['device_id'],
                message['data']
            )
            
            # Publish prediction result
            if self.mqtt_handler:
                prediction_topic = f"smartcity/{message['device_type']}/{message['device_id']}/predictions"
                self.mqtt_handler.publish(prediction_topic, prediction)
                
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")

async def main():
    """Main entry point"""
    # Build configuration
    config = {
        'clickhouse_password': os.environ.get('CLICKHOUSE_PASSWORD', ''),
        'clickhouse_db': os.environ.get('CLICKHOUSE_DB', 'smartcity'),
        'mqtt_broker': os.environ.get('MQTT_BROKER', 'emqx'),
        'mqtt_port': int(os.environ.get('MQTT_PORT', '1883')),
        'mqtt_username': os.environ.get('MQTT_USERNAME'),
        'mqtt_password': os.environ.get('MQTT_PASSWORD'),
        'api_port': int(os.environ.get('API_PORT', '8000')),
        'model_config_path': os.environ.get('MODEL_CONFIG_PATH', '/app/config'),
        'training_interval_hours': int(os.environ.get('TRAINING_INTERVAL_HOURS', '24')),
        'storage_type': os.environ.get('STORAGE_TYPE', 'minio'),
        'minio_endpoint': os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
        'minio_access_key': os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
        'minio_secret_key': os.environ.get('MINIO_SECRET_KEY', 'minioadmin123'),
        'minio_bucket': os.environ.get('MINIO_BUCKET', 'models')
    }
    
    # Create orchestrator
    orchestrator = MLServiceOrchestrator(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize services
        if not await orchestrator.initialize():
            logger.error("Failed to initialize services")
            return
        
        # Start API server
        config_uvicorn = uvicorn.Config(
            app=orchestrator.api.app,
            host="0.0.0.0",
            port=config['api_port'],
            log_level="info"
        )
        server = uvicorn.Server(config_uvicorn)
        
        logger.info(f"Starting ML Service API on port {config['api_port']}")
        logger.info(f"API Documentation: http://0.0.0.0:{config['api_port']}/docs")
        logger.info(f"Health Check: http://0.0.0.0:{config['api_port']}/health")
        
        await server.serve()
        
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())