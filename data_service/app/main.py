
# data_service/app/main_fixed.py - Fixed main file with better error handling

import asyncio
import logging
import os
from app.core.data_service import DataServiceFixed

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def wait_for_services():
    """Wait for required services with detailed logging"""
    import clickhouse_connect
    import paho.mqtt.client as mqtt
    
    startup_wait = int(os.environ.get('STARTUP_WAIT_SECONDS', '20'))
    logger.info(f"Waiting {startup_wait} seconds for services to start...")
    await asyncio.sleep(startup_wait)
    
    # Wait for ClickHouse with detailed logging
    clickhouse_host = os.environ.get('CLICKHOUSE_HOST', 'clickhouse')
    clickhouse_port = int(os.environ.get('CLICKHOUSE_PORT', '8123'))
    
    logger.info(f"Testing ClickHouse connection to {clickhouse_host}:{clickhouse_port}")
    
    for attempt in range(15):
        try:
            logger.info(f"ClickHouse connection attempt {attempt + 1}")
            client = clickhouse_connect.get_client(
                host=clickhouse_host,
                port=clickhouse_port,
                username='default',
                password='',
                connect_timeout=5
            )
            result = client.query("SELECT version()")
            version = result.result_rows[0][0] if result.result_rows else "unknown"
            logger.info(f"ClickHouse connection verified: {version}")
            client.close()
            break
        except Exception as e:
            logger.warning(f"ClickHouse not ready: {e}")
            if attempt < 14:
                await asyncio.sleep(5)
            else:
                logger.error("Failed to connect to ClickHouse")
                raise
    
    # Wait for EMQX with detailed logging
    mqtt_broker = os.environ.get('MQTT_BROKER', 'emqx')
    mqtt_port = int(os.environ.get('MQTT_PORT', '1883'))
    
    logger.info(f"Testing EMQX connection to {mqtt_broker}:{mqtt_port}")
    
    for attempt in range(15):
        try:
            logger.info(f"EMQX connection attempt {attempt + 1}")
            
            test_client = mqtt.Client(client_id="startup_test")
            
            connected = False
            def on_connect(client, userdata, flags, rc):
                nonlocal connected
                if rc == 0:
                    connected = True
                    logger.info(f"EMQX test connection successful")
                else:
                    logger.warning(f"EMQX test connection failed: {rc}")
            
            test_client.on_connect = on_connect
            test_client.connect(mqtt_broker, mqtt_port, 5)
            test_client.loop_start()
            
            # Wait for connection
            for i in range(10):
                if connected:
                    break
                await asyncio.sleep(0.5)
            
            test_client.loop_stop()
            test_client.disconnect()
            
            if connected:
                logger.info("EMQX connection verified")
                break
            else:
                logger.warning("EMQX connection test failed")
                
        except Exception as e:
            logger.warning(f"EMQX not ready: {e}")
            
        if attempt < 14:
            await asyncio.sleep(5)
        else:
            logger.error("Failed to connect to EMQX")
            raise

async def main():
    """Main entry point with enhanced error handling"""
    try:
        # Log environment variables for debugging
        logger.info("Starting Data Service with configuration:")
        for key, value in os.environ.items():
            if key.startswith(('CLICKHOUSE_', 'MQTT_', 'BATCH_')):
                logger.info(f"  {key}={value}")
        
        # Wait for dependencies
        await wait_for_services()
        
        # Build configuration
        config = {
            'clickhouse_host': os.environ.get('CLICKHOUSE_HOST', 'clickhouse'),
            'clickhouse_port': int(os.environ.get('CLICKHOUSE_PORT', '8123')),
            'clickhouse_user': os.environ.get('CLICKHOUSE_USER', 'default'),
            'clickhouse_password': os.environ.get('CLICKHOUSE_PASSWORD', ''),
            'clickhouse_db': os.environ.get('CLICKHOUSE_DB', 'smartcity'),
            'mqtt_broker': os.environ.get('MQTT_BROKER', 'emqx'),
            'mqtt_port': int(os.environ.get('MQTT_PORT', '1883')),
            'mqtt_username': os.environ.get('MQTT_USERNAME'),
            'mqtt_password': os.environ.get('MQTT_PASSWORD'),
            'batch_size': int(os.environ.get('BATCH_SIZE', '100')),
            'batch_timeout': int(os.environ.get('BATCH_TIMEOUT', '5'))
        }
        
        logger.info("Starting Data Service...")
        service = DataServiceFixed(config)
        
        await service.run()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Data Service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())

