# app/messaging/mqtt_handler.py - Separate MQTT concerns
import asyncio
import logging
import json
import queue
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import time
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class MQTTHandler:
    """Handles MQTT messaging separately from ML logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = mqtt.Client(
            client_id=f"ml-service-{int(time.time())}",
            clean_session=False,
            protocol=mqtt.MQTTv311
        )
        
        # Authentication
        username = config.get('mqtt_username')
        password = config.get('mqtt_password')
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Configuration
        self.client.max_inflight_messages_set(20)
        self.client.max_queued_messages_set(100)
        
        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        # Message processing
        self.message_queue = queue.Queue(maxsize=1000)
        self.message_handler: Optional[Callable] = None
        self.is_running = False
        self.processing_thread = None
    
    def set_message_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Set the handler for incoming messages"""
        self.message_handler = handler
    
    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client.connect(
                self.config.get('mqtt_broker', 'emqx'),
                self.config.get('mqtt_port', 1883),
                60
            )
            self.client.loop_start()
            
            # Wait for connection
            await asyncio.sleep(3)
            
            if self.client.is_connected():
                self.is_running = True
                self._start_message_processing()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=30)
        
        if self.client.is_connected():
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 1) -> bool:
        """Publish message to MQTT"""
        try:
            message = json.dumps(payload)
            msg_info = self.client.publish(topic, message, qos=qos)
            return msg_info.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to device data
            client.subscribe("smartcity/+/+/data", qos=1)
            logger.info("Subscribed to device data topics")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            # Parse topic: smartcity/{device_type}/{device_id}/data
            topic_parts = msg.topic.split('/')
            if len(topic_parts) >= 4:
                device_type = topic_parts[1]
                device_id = topic_parts[2]
                
                data = json.loads(msg.payload.decode())
                
                # Queue message for processing
                try:
                    self.message_queue.put_nowait({
                        'device_type': device_type,
                        'device_id': device_id,
                        'data': data,
                        'topic': msg.topic,
                        'qos': msg.qos,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    logger.debug(f"Queued message from {device_type}/{device_id}")
                except queue.Full:
                    logger.warning("Message queue full, dropping message")
                    
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection: {rc}")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logger.debug(f"Message {mid} published successfully")
    
    def _start_message_processing(self):
        """Start message processing thread"""
        def processing_loop():
            logger.info("Started MQTT message processing thread")
            while self.is_running:
                try:
                    try:
                        message = self.message_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # Process message with handler
                    if self.message_handler:
                        try:
                            asyncio.run(self.message_handler(message))
                        except Exception as e:
                            logger.error(f"Message handler error: {e}")
                    
                    self.message_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in message processing loop: {e}")
            
            logger.info("MQTT message processing thread stopped")
        
        self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()