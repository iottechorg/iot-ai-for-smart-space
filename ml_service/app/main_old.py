# ml_service/app/main.py
import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, logging_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to run the ML service"""
    logger.info("Starting Smart City ML Service")
    
    # Import here to avoid circular imports
    from app.core.ml_service import MLService
    
    # Create ML service
    ml_service = MLService(
        config_path=os.getenv('CONFIG_PATH', '/app/config'),
        mqtt_broker=os.getenv('MQTT_BROKER', 'emqx'),
        mqtt_port=int(os.getenv('MQTT_PORT', '1883')),
        mqtt_topic_prefix=os.getenv('MQTT_TOPIC_PREFIX', 'smartcity')
    )
    
    # Run the service
    await ml_service.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("ML Service shutdown by user")
    except Exception as e:
        logger.error(f"ML Service failed with error: {e}")