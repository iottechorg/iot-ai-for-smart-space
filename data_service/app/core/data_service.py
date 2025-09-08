# data_service/app/core/data_service_fixed.py 

import asyncio
import logging
import json
import time
import queue
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import paho.mqtt.client as mqtt
import clickhouse_connect
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceMessage:
    """Data structure for device messages"""
    timestamp: datetime
    device_type: str
    device_id: str
    data: Dict[str, Any]
    topic: str
    
@dataclass
class PredictionMessage:
    """Data structure for prediction messages"""
    timestamp: datetime
    device_type: str
    device_id: str
    model_name: str
    prediction: str
    confidence: float
    metadata: Dict[str, Any]

class DataServiceFixed:
    """
    Fixed Data Service with enhanced debugging and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Enable debug logging
        logging.getLogger("paho.mqtt.client").setLevel(logging.INFO)
        
        # ClickHouse client
        self.db_client = None
        
        # MQTT client for consuming data
        self.mqtt_client = mqtt.Client(
            client_id=f"data-service-{int(time.time())}",
            clean_session=False,
            protocol=mqtt.MQTTv311
        )
        
        # MQTT authentication
        mqtt_username = config.get('mqtt_username')
        mqtt_password = config.get('mqtt_password')
        if mqtt_username and mqtt_password:
            self.mqtt_client.username_pw_set(mqtt_username, mqtt_password)
            logger.info(f"MQTT authentication set for user: {mqtt_username}")
        else:
            logger.info("No MQTT authentication configured")
        
        # MQTT callbacks
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_subscribe = self._on_mqtt_subscribe
        self.mqtt_client.on_log = self._on_mqtt_log
        
        # Threading and queues
        self.is_running = False
        self.device_message_queue = queue.Queue(maxsize=10000)
        self.prediction_message_queue = queue.Queue(maxsize=5000)
        
        # Worker threads
        self.device_data_worker = None
        self.prediction_data_worker = None
        
        # Batch processing settings
        self.batch_size = config.get('batch_size', 100)
        self.batch_timeout = config.get('batch_timeout', 5)
        
        # Enhanced statistics
        self.stats = {
            'messages_received': 0,
            'device_records_written': 0,
            'prediction_records_written': 0,
            'errors': 0,
            'last_write_time': None,
            'mqtt_connected': False,
            'clickhouse_connected': False,
            'subscriptions_confirmed': 0
        }
        
        logger.info(f"Data Service initialized with config: {config}")
    
    def _on_mqtt_log(self, client, userdata, level, buf):
        """MQTT log callback for debugging"""
        logger.debug(f"MQTT Log [{level}]: {buf}")
    
    def _on_mqtt_subscribe(self, client, userdata, mid, granted_qos):
        """MQTT subscribe callback"""
        self.stats['subscriptions_confirmed'] += 1
        logger.info(f"MQTT subscription confirmed: mid={mid}, qos={granted_qos}")
    
    async def initialize(self):
        """Initialize the data service with enhanced error handling"""
        logger.info("Initializing Data Service...")
        
        # Initialize ClickHouse connection
        await self._initialize_clickhouse()
        
        # Initialize MQTT connection
        await self._initialize_mqtt()
        
        # Create database tables if they don't exist
        await self._ensure_tables_exist()
        
        logger.info("Data Service initialized successfully")
        return True
    
    async def _initialize_clickhouse(self):
        """Initialize ClickHouse connection with retry logic"""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to ClickHouse (attempt {attempt + 1})")
                
                ch_host = self.config.get('clickhouse_host', 'clickhouse')
                ch_port = self.config.get('clickhouse_port', 8123)
                ch_db = self.config.get('clickhouse_db', 'smartcity')
                
                logger.info(f"ClickHouse target: {ch_host}:{ch_port}/{ch_db}")
                
                self.db_client = clickhouse_connect.get_client(
                    host=ch_host,
                    port=ch_port,
                    username=self.config.get('clickhouse_user', 'default'),
                    password=self.config.get('clickhouse_password', ''),
                    database=ch_db,
                    connect_timeout=10
                )
                
                # Test connection
                result = self.db_client.query("SELECT version()")
                version = result.result_rows[0][0] if result.result_rows else "unknown"
                logger.info(f"ClickHouse connection successful: {version}")
                
                self.stats['clickhouse_connected'] = True
                break
                
            except Exception as e:
                logger.warning(f"ClickHouse connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                else:
                    raise Exception("Failed to connect to ClickHouse after all retries")
    
    async def _initialize_mqtt(self):
        """Initialize MQTT connection with enhanced debugging"""
        max_retries = 10
        
        mqtt_broker = self.config.get('mqtt_broker', 'emqx')
        mqtt_port = self.config.get('mqtt_port', 1883)
        
        logger.info(f"MQTT target: {mqtt_broker}:{mqtt_port}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to EMQX (attempt {attempt + 1})")
                
                self.mqtt_client.connect(mqtt_broker, mqtt_port, 60)
                self.mqtt_client.loop_start()
                
                # Wait for connection
                await asyncio.sleep(5)
                
                if self.mqtt_client.is_connected():
                    logger.info("EMQX connection successful")
                    self.stats['mqtt_connected'] = True
                    break
                else:
                    logger.warning(f"EMQX connection attempt {attempt + 1} failed - not connected")
                    
            except Exception as e:
                logger.warning(f"EMQX connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                else:
                    raise Exception("Failed to connect to EMQX after all retries")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connect callback with enhanced logging"""
        if rc == 0:
            logger.info("Connected to EMQX broker successfully!")
            logger.info(f"Session present: {flags.get('session present', False)}")
            logger.info(f"Client ID: {client._client_id}")
            
            # Subscribe to device data and predictions with detailed logging
            topics = [
                ("smartcity/+/+/data", 1),
                ("smartcity/+/+/predictions", 1)
            ]
            
            for topic, qos in topics:
                result, mid = client.subscribe(topic, qos)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Subscribed to topic '{topic}' with QoS {qos} (mid: {mid})")
                else:
                    logger.error(f"Failed to subscribe to '{topic}': {result}")
            
            self.stats['mqtt_connected'] = True
            
        else:
            logger.error(f"Failed to connect to EMQX broker with code {rc}")
            self.stats['mqtt_connected'] = False
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback with enhanced debugging"""
        try:
            self.stats['messages_received'] += 1
            
            logger.debug(f"Received MQTT message #{self.stats['messages_received']}")
            logger.debug(f"  Topic: {msg.topic}")
            logger.debug(f"  QoS: {msg.qos}")
            logger.debug(f"  Retain: {msg.retain}")
            logger.debug(f"  Payload length: {len(msg.payload)}")
            logger.debug(f"  Payload preview: {msg.payload.decode()[:100]}...")
            
            # Parse topic with validation
            topic_parts = msg.topic.split('/')
            if len(topic_parts) < 4:
                logger.warning(f"Invalid topic format: {msg.topic} (expected: smartcity/device_type/device_id/message_type)")
                return
            
            if topic_parts[0] != 'smartcity':
                logger.warning(f"Unexpected topic prefix: {topic_parts[0]} (expected: smartcity)")
                return
            
            device_type = topic_parts[1]
            device_id = topic_parts[2]
            message_type = topic_parts[3]
            
            logger.debug(f"Parsed topic: device_type={device_type}, device_id={device_id}, message_type={message_type}")
            
            # Parse payload with error handling
            try:
                data = json.loads(msg.payload.decode())
                logger.debug(f"Parsed JSON payload: {list(data.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}")
                logger.error(f"Raw payload: {msg.payload}")
                self.stats['errors'] += 1
                return
            
            # Route message based on type
            if message_type == "data":
                logger.debug(f"Queuing device data message")
                self._queue_device_message(device_type, device_id, data, msg.topic)
            elif message_type == "predictions":
                logger.debug(f"Queuing prediction message")
                self._queue_prediction_message(device_type, device_id, data, msg.topic)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            logger.error(f"Message details: topic={msg.topic}, payload={msg.payload}")
            self.stats['errors'] += 1
    
    def _queue_device_message(self, device_type: str, device_id: str, data: Dict[str, Any], topic: str):
        """Queue device data message with enhanced logging"""
        try:
            # Extract timestamp with better error handling
            timestamp_str = data.get('timestamp')
            if timestamp_str:
                try:
                    if timestamp_str.endswith('Z'):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    logger.debug(f"Parsed timestamp: {timestamp}")
                except ValueError as e:
                    logger.warning(f"Invalid timestamp format '{timestamp_str}': {e}, using current time")
                    timestamp = datetime.utcnow()
            else:
                logger.debug("No timestamp in message, using current time")
                timestamp = datetime.utcnow()
            
            message = DeviceMessage(
                timestamp=timestamp,
                device_type=device_type,
                device_id=device_id,
                data=data,
                topic=topic
            )
            
            # Add to queue with error handling
            try:
                self.device_message_queue.put_nowait(message)
                logger.debug(f"Queued device message, queue size: {self.device_message_queue.qsize()}")
            except queue.Full:
                logger.warning("Device message queue full, dropping message")
                self.stats['errors'] += 1
                
        except Exception as e:
            logger.error(f"Error queuing device message: {e}")
            self.stats['errors'] += 1
    
    def _queue_prediction_message(self, device_type: str, device_id: str, data: Dict[str, Any], topic: str):
        """Queue prediction message with enhanced logging"""
        try:
            # Extract timestamp
            timestamp_str = data.get('timestamp')
            if timestamp_str:
                try:
                    if timestamp_str.endswith('Z'):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()
            
            message = PredictionMessage(
                timestamp=timestamp,
                device_type=device_type,
                device_id=device_id,
                model_name=data.get('model_name', 'unknown'),
                prediction=str(data.get('prediction', '')),
                confidence=float(data.get('confidence', 0.0)),
                metadata=data
            )
            
            # Add to queue
            try:
                self.prediction_message_queue.put_nowait(message)
                logger.debug(f"Queued prediction message, queue size: {self.prediction_message_queue.qsize()}")
            except queue.Full:
                logger.warning("Prediction message queue full, dropping message")
                self.stats['errors'] += 1
                
        except Exception as e:
            logger.error(f"Error queuing prediction message: {e}")
            self.stats['errors'] += 1
    
    async def _ensure_tables_exist(self):
        """Ensure ClickHouse tables exist"""
        try:
            logger.info("Checking/creating ClickHouse tables...")
            
            # Create device_data table
            device_data_table = """
            CREATE TABLE IF NOT EXISTS device_data (
                timestamp DateTime64(3),
                device_type String,
                device_id String,
                metric_name String,
                metric_value Float64,
                metadata String DEFAULT ''
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (device_type, device_id, metric_name, timestamp)
            TTL timestamp + INTERVAL 1 YEAR
            """
            
            # Create predictions table
            predictions_table = """
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp DateTime64(3),
                device_type String,
                device_id String,
                model_name String,
                prediction String,
                confidence Float64,
                metadata String DEFAULT ''
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (device_type, device_id, model_name, timestamp)
            TTL timestamp + INTERVAL 1 YEAR
            """
            
            self.db_client.command(device_data_table)
            self.db_client.command(predictions_table)
            
            # Verify tables were created
            tables = self.db_client.query("SHOW TABLES")
            table_names = [row[0] for row in tables.result_rows]
            logger.info(f"ClickHouse tables verified: {table_names}")
            
        except Exception as e:
            logger.error(f"Error creating ClickHouse tables: {e}")
            raise
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.stats['mqtt_connected'] = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection from EMQX broker: {rc}")
        else:
            logger.info("Disconnected from EMQX broker")
    
    def _start_device_data_worker(self):
        """Start worker thread for processing device data"""
        def device_data_worker():
            logger.info("Started device data worker thread")
            batch = []
            last_batch_time = time.time()
            
            while self.is_running:
                try:
                    # Get message with timeout
                    try:
                        message = self.device_message_queue.get(timeout=1.0)
                        batch.append(message)
                        logger.debug(f"Added message to batch, batch size: {len(batch)}")
                    except queue.Empty:
                        # Check if we need to flush partial batch
                        if batch and (time.time() - last_batch_time) > self.batch_timeout:
                            logger.debug(f"Batch timeout reached, flushing {len(batch)} messages")
                            self._write_device_data_batch(batch)
                            batch = []
                            last_batch_time = time.time()
                        continue
                    
                    # Process batch when full or timeout reached
                    if (len(batch) >= self.batch_size or 
                        (batch and (time.time() - last_batch_time) > self.batch_timeout)):
                        
                        logger.debug(f"Writing batch of {len(batch)} device messages")
                        self._write_device_data_batch(batch)
                        batch = []
                        last_batch_time = time.time()
                    
                except Exception as e:
                    logger.error(f"Error in device data worker: {e}")
                    time.sleep(1)
            
            # Process remaining batch on shutdown
            if batch:
                logger.info(f"Processing final batch of {len(batch)} messages")
                self._write_device_data_batch(batch)
            
            logger.info("Device data worker thread stopped")
        
        self.device_data_worker = threading.Thread(target=device_data_worker, daemon=True)
        self.device_data_worker.start()
    
    def _write_device_data_batch(self, batch: List[DeviceMessage]):
        """Write batch of device data to ClickHouse with enhanced logging"""
        if not batch:
            return
        
        try:
            logger.debug(f"Processing batch of {len(batch)} device messages")
            
            # Prepare data for batch insert
            rows = []
            
            for message in batch:
                # Extract numeric fields from device data
                for key, value in message.data.items():
                    # Skip non-numeric fields and metadata
                    if key in ['device_type', 'device_id', 'timestamp', 'gateway_id', 'measurement']:
                        continue
                    
                    # Try to convert to float
                    try:
                        numeric_value = float(value)
                        rows.append([
                            message.timestamp,
                            message.device_type,
                            message.device_id,
                            key,
                            numeric_value,
                            json.dumps(message.data)
                        ])
                        logger.debug(f"Added row: {message.device_type}/{message.device_id}/{key} = {numeric_value}")
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping non-numeric value: {key} = {value}")
                        continue
            
            if rows:
                logger.info(f"Writing {len(rows)} device data records to ClickHouse")
                
                # Batch insert to ClickHouse
                self.db_client.insert(
                    'device_data',
                    rows,
                    column_names=['timestamp', 'device_type', 'device_id', 'metric_name', 'metric_value', 'metadata']
                )
                
                self.stats['device_records_written'] += len(rows)
                self.stats['last_write_time'] = datetime.utcnow().isoformat()
                
                logger.info(f"Successfully wrote {len(rows)} device data records")
            else:
                logger.warning(f"No valid numeric data found in batch of {len(batch)} messages")
            
            # Mark all messages as processed
            for _ in batch:
                self.device_message_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error writing device data batch: {e}")
            logger.error(f"Batch details: {len(batch)} messages")
            self.stats['errors'] += len(batch)
    
    def _start_prediction_data_worker(self):
        """Start worker thread for processing prediction data"""
        def prediction_data_worker():
            logger.info("Started prediction data worker thread")
            batch = []
            last_batch_time = time.time()
            
            while self.is_running:
                try:
                    # Get message with timeout
                    try:
                        message = self.prediction_message_queue.get(timeout=1.0)
                        batch.append(message)
                    except queue.Empty:
                        # Check if we need to flush partial batch
                        if batch and (time.time() - last_batch_time) > self.batch_timeout:
                            self._write_prediction_data_batch(batch)
                            batch = []
                            last_batch_time = time.time()
                        continue
                    
                    # Process batch when full or timeout reached
                    if (len(batch) >= self.batch_size or 
                        (batch and (time.time() - last_batch_time) > self.batch_timeout)):
                        
                        self._write_prediction_data_batch(batch)
                        batch = []
                        last_batch_time = time.time()
                    
                except Exception as e:
                    logger.error(f"Error in prediction data worker: {e}")
                    time.sleep(1)
            
            # Process remaining batch on shutdown
            if batch:
                self._write_prediction_data_batch(batch)
            
            logger.info("Prediction data worker thread stopped")
        
        self.prediction_data_worker = threading.Thread(target=prediction_data_worker, daemon=True)
        self.prediction_data_worker.start()
    
    def _write_prediction_data_batch(self, batch: List[PredictionMessage]):
        """Write batch of prediction data to ClickHouse"""
        if not batch:
            return
        
        try:
            logger.info(f"Writing {len(batch)} prediction records to ClickHouse")
            
            # Prepare data for batch insert
            rows = []
            
            for message in batch:
                rows.append([
                    message.timestamp,
                    message.device_type,
                    message.device_id,
                    message.model_name,
                    message.prediction,
                    message.confidence,
                    json.dumps(message.metadata)
                ])
            
            if rows:
                # Batch insert to ClickHouse
                self.db_client.insert(
                    'predictions',
                    rows,
                    column_names=['timestamp', 'device_type', 'device_id', 'model_name', 'prediction', 'confidence', 'metadata']
                )
                
                self.stats['prediction_records_written'] += len(rows)
                self.stats['last_write_time'] = datetime.utcnow().isoformat()
                
                logger.info(f"Successfully wrote {len(rows)} prediction records")
            
            # Mark all messages as processed
            for _ in batch:
                self.prediction_message_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error writing prediction data batch: {e}")
            self.stats['errors'] += len(batch)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced service statistics"""
        return {
            **self.stats,
            'device_queue_size': self.device_message_queue.qsize(),
            'prediction_queue_size': self.prediction_message_queue.qsize(),
        }
    
    async def run(self):
        """Run the data service with enhanced monitoring"""
        if not await self.initialize():
            logger.error("Failed to initialize data service")
            return
        
        self.is_running = True
        
        # Start worker threads
        self._start_device_data_worker()
        self._start_prediction_data_worker()
        
        logger.info("Data Service is running...")
        logger.info(f"Batch size: {self.batch_size}, Batch timeout: {self.batch_timeout}s")
        logger.info(f"MQTT connected: {self.stats['mqtt_connected']}")
        logger.info(f"ClickHouse connected: {self.stats['clickhouse_connected']}")
        
        # Keep service running and log statistics periodically
        try:
            while self.is_running:
                await asyncio.sleep(30)  # Log stats every 30 seconds
                stats = self.get_stats()
                logger.info(f"Stats: {stats['messages_received']} msgs received, "
                           f"{stats['device_records_written']} device records, "
                           f"{stats['prediction_records_written']} prediction records, "
                           f"{stats['errors']} errors, "
                           f"MQTT: {stats['mqtt_connected']}, "
                           f"CH: {stats['clickhouse_connected']}")
        
        except asyncio.CancelledError:
            logger.info("Data service cancelled")
        except Exception as e:
            logger.error(f"Error in data service: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the data service"""
        logger.info("Stopping Data Service...")
        self.is_running = False
        
        # Wait for queues to be processed
        logger.info("Waiting for queues to be processed...")
        start_time = time.time()
        while ((not self.device_message_queue.empty() or not self.prediction_message_queue.empty()) 
               and (time.time() - start_time) < 30):
            await asyncio.sleep(0.5)
        
        # Stop worker threads
        if self.device_data_worker and self.device_data_worker.is_alive():
            self.device_data_worker.join(timeout=30)
        
        if self.prediction_data_worker and self.prediction_data_worker.is_alive():
            self.prediction_data_worker.join(timeout=30)
        
        # Disconnect MQTT
        if self.mqtt_client.is_connected():
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        logger.info("Data Service stopped")
