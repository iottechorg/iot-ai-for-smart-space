# gateway/app/core/gateway.py - COMPLETE BIDIRECTIONAL VERSION
import asyncio
import logging
from typing import Dict, List, Any
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time

from .device_interface import DeviceInterface
from .device_registry import DeviceRegistry

logger = logging.getLogger(__name__)

class IoTGateway:
    """
    MQTT-based IoT Gateway with bidirectional communication support
    Single Responsibility: Manages device communication and data publishing/receiving
    Open/Closed: Extensible for new device types without modifying core code
    Liskov Substitution: Any device implementing DeviceInterface can be used
    Interface Segregation: Clean interfaces for device operations
    Dependency Inversion: Depends on abstractions, not concrete implementations
    """
    
    def __init__(self, mqtt_broker: str, mqtt_port: int, mqtt_topic_prefix: str = "smartcity"):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic_prefix = mqtt_topic_prefix
        self.device_registry = DeviceRegistry()
        self.client = mqtt.Client()
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.is_connected = False
        self.connection_event = asyncio.Event()
        
        # 🔥 BIDIRECTIONAL COMMUNICATION COMPONENTS
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.command_processor_task = None
        self.command_processor_running = False  # Separate flag for command processing
        
    async def initialize(self):
        """Initialize the gateway and connect to MQTT broker"""
        logger.info("Initializing IoT Gateway with bidirectional communication...")
        
        # Setup MQTT client callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_message = self._on_message  # 🔥 THIS IS KEY - RECEIVE COMMANDS
        
        # Enable logging for MQTT client
        self.client.enable_logger(logger)
        
        try:
            logger.info(f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            
            # Wait for connection with timeout
            try:
                await asyncio.wait_for(self.connection_event.wait(), timeout=10.0)
                logger.info("Successfully connected to MQTT broker")
                
                # Subscribe to command topics for all devices
                await self._subscribe_to_command_topics()
                
                # Start command processor
                self.command_processor_running = True
                self.command_processor_task = asyncio.create_task(self._process_commands())
                logger.info("🚀 Command processor started - ready to receive MQTT commands")
                
                return True
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for MQTT connection")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            self.is_connected = True
            self.connection_event.set()
        else:
            logger.error(f"Failed to connect to MQTT broker with code {rc}")
            self.is_connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        logger.warning(f"Disconnected from MQTT broker with code {rc}")
        self.is_connected = False
        self.connection_event.clear()
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logger.debug(f"Message published successfully with mid: {mid}")
    
    def _on_message(self, client, userdata, msg):
        """🔥 MQTT message received callback - handles incoming commands"""
        logger.info(f"📨 MQTT Message Received: {msg.topic}")
        logger.debug(f"📨 Payload: {msg.payload.decode()}")
        
        try:
            # Parse the topic to extract device info
            topic_parts = msg.topic.split('/')
            logger.debug(f"🔍 Topic parts: {topic_parts}")
            
            if len(topic_parts) >= 4 and topic_parts[-1] == 'command':
                # Expected format: smartcity/{device_type}/{device_id}/command
                topic_prefix = topic_parts[0]  # smartcity
                device_type = topic_parts[-3]  # device_type
                device_id = topic_parts[-2]    # device_id
                command_type = topic_parts[-1] # command
                
                logger.info(f"🎯 Command received for device: {device_id} (type: {device_type})")
                
                # Verify device exists and supports write
                device = self.device_registry.get_device(device_id)
                if not device:
                    logger.error(f"❌ Device {device_id} not found in registry")
                    return
                
                if not device.supports_write():
                    logger.error(f"❌ Device {device_id} does not support write operations")
                    # Send error response
                    asyncio.create_task(self._publish_command_response(device_id, {
                        "success": False,
                        "error": f"Device {device_id} is read-only",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    return
                
                # Parse the command payload
                try:
                    command_data = json.loads(msg.payload.decode())
                    logger.info(f"📝 Parsed command: {command_data}")
                    
                    # Add to command queue for processing
                    command = {
                        'device_id': device_id,
                        'device_type': device_type,
                        'command_data': command_data,
                        'topic': msg.topic,
                        'timestamp': datetime.utcnow()
                    }
                    
                    # Put command in queue (non-blocking)
                    try:
                        self.command_queue.put_nowait(command)
                        logger.info(f"✅ Command queued for device {device_id}")
                        logger.info(f"📊 Queue size: {self.command_queue.qsize()}")
                    except asyncio.QueueFull:
                        logger.warning(f"⚠️ Command queue full, dropping command for {device_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in command message: {e}")
                    # Send error response
                    asyncio.create_task(self._publish_command_response(device_id, {
                        "success": False,
                        "error": f"Invalid JSON: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"❌ Error processing command message: {e}")
            else:
                logger.debug(f"📋 Received message on non-command topic: {msg.topic}")
                
        except Exception as e:
            logger.error(f"💥 Critical error handling MQTT message: {e}")
            import traceback
            traceback.print_exc()
    
    async def _subscribe_to_command_topics(self):
        """Subscribe to command topics for all registered devices"""
        if not self.is_connected:
            logger.warning("Cannot subscribe: MQTT not connected")
            return
        
        writable_devices = []
        for device_id in self.device_registry.get_all_device_ids():
            device = self.device_registry.get_device(device_id)
            if device and device.supports_write():
                command_topic = f"{self.mqtt_topic_prefix}/{device.get_type()}/{device_id}/command"
                result, mid = self.client.subscribe(command_topic, qos=1)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    writable_devices.append(device_id)
                    logger.info(f"✅ Subscribed to command topic: {command_topic}")
                else:
                    logger.error(f"❌ Failed to subscribe to {command_topic} (error: {result})")
        
        if writable_devices:
            logger.info(f"📡 Total command subscriptions: {len(writable_devices)} devices")
            logger.info(f"🎛️ Writable devices: {writable_devices}")
        else:
            logger.info("📖 No writable devices found - all devices are read-only sensors")
    
    async def _process_commands(self):
        """🔥 Process incoming commands from the queue"""
        logger.info("🚀 Command processor started - waiting for commands...")
        
        # Log processor status every 60 seconds
        last_status_log = time.time()
        
        while self.command_processor_running:  # Use separate flag
            try:
                # Log status periodically
                current_time = time.time()
                if current_time - last_status_log >= 60:  # Every 60 seconds
                    logger.info(f"💓 Command processor alive - Queue size: {self.command_queue.qsize()}")
                    last_status_log = current_time
                
                # Wait for commands with timeout
                try:
                    command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                    logger.info(f"⚡ Processing command from queue: {command['device_id']}")
                except asyncio.TimeoutError:
                    continue
                
                # Process the command
                await self._execute_device_command(command)
                
            except Exception as e:
                logger.error(f"💥 Error in command processor: {e}")
                await asyncio.sleep(1)
        
        logger.info("🛑 Command processor stopped")
    
    async def _execute_device_command(self, command: Dict[str, Any]):
        """🔥 Execute a command on a specific device"""
        device_id = command['device_id']
        command_data = command['command_data']
        
        logger.info(f"🎯 Executing command on device {device_id}")
        logger.debug(f"📝 Command details: {command_data}")
        
        device = self.device_registry.get_device(device_id)
        if not device:
            logger.error(f"❌ Device {device_id} not found for command execution")
            await self._publish_command_response(device_id, {
                "success": False,
                "error": f"Device {device_id} not found",
                "timestamp": datetime.utcnow().isoformat()
            })
            return
        
        if not device.supports_write():
            logger.error(f"❌ Device {device_id} does not support write operations")
            await self._publish_command_response(device_id, {
                "success": False,
                "error": f"Device {device_id} does not support commands",
                "timestamp": datetime.utcnow().isoformat()
            })
            return
        
        try:
            logger.info(f"🔧 Calling device.write_data() for {device_id}")
            
            # 🔥 THIS IS WHERE THE ACTUAL DEVICE COMMAND IS EXECUTED
            result = await device.write_data(command_data)
            
            # Add metadata
            result.update({
                "device_id": device_id,
                "device_type": device.get_type(),
                "timestamp": datetime.utcnow().isoformat(),
                "command": command_data
            })
            
            success = result.get('success', False)
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{status} Command executed on {device_id}")
            logger.debug(f"📤 Command result: {result}")
            
            # Publish response
            await self._publish_command_response(device_id, result)
            
        except Exception as e:
            logger.error(f"💥 Error executing command on device {device_id}: {e}")
            await self._publish_command_response(device_id, {
                "success": False,
                "error": str(e),
                "device_id": device_id,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _publish_command_response(self, device_id: str, response: Dict[str, Any]):
        """Publish command execution response"""
        device = self.device_registry.get_device(device_id)
        if not device:
            return
        
        response_topic = f"{self.mqtt_topic_prefix}/{device.get_type()}/{device_id}/response"
        
        try:
            message = json.dumps(response, default=str)
            result = self.client.publish(response_topic, message, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"📤 Published command response for device {device_id}")
                logger.debug(f"Response: {message}")
            else:
                logger.error(f"❌ Failed to publish command response for {device_id}")
                
        except Exception as e:
            logger.error(f"💥 Exception while publishing command response for {device_id}: {e}")
    
    def register_device(self, device: DeviceInterface):
        """Register a new device and subscribe to its command topic if it supports write"""
        self.device_registry.register_device(device)
        logger.info(f"Registered device: {device.get_id()} ({device.get_type()})")
        
        # Check if device supports write operations
        if device.supports_write():
            logger.info(f"Device {device.get_id()} supports WRITE operations")
            
            # Subscribe to command topic if connected
            if self.is_connected:
                command_topic = f"{self.mqtt_topic_prefix}/{device.get_type()}/{device.get_id()}/command"
                result, mid = self.client.subscribe(command_topic, qos=1)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"✅ Subscribed to command topic: {command_topic}")
                    logger.info(f"📝 Writable properties: {device.get_writable_properties()}")
                else:
                    logger.error(f"❌ Failed to subscribe to command topic: {command_topic} (error: {result})")
            else:
                logger.warning(f"⚠️ MQTT not connected, will subscribe to {device.get_id()} commands when connected")
        else:
            logger.info(f"Device {device.get_id()} is READ-ONLY (sensor)")
    
    async def unregister_device(self, device_id: str):
        """Unregister a device and unsubscribe from its command topic"""
        device = self.device_registry.get_device(device_id)
        if device and device.supports_write() and self.is_connected:
            command_topic = f"{self.mqtt_topic_prefix}/{device.get_type()}/{device_id}/command"
            self.client.unsubscribe(command_topic)
            logger.info(f"Unsubscribed from command topic: {command_topic}")
        
        if self.device_registry.unregister_device(device_id):
            logger.info(f"Unregistered device: {device_id}")
        
        # Stop monitoring task if exists
        if device_id in self.monitoring_tasks:
            await self.stop_device_monitoring(device_id)
    
    async def start_device_monitoring(self, device_id: str, interval: int = 30):
        """Start monitoring a specific device"""
        device = self.device_registry.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        task = asyncio.create_task(self._monitor_device(device, interval))
        self.monitoring_tasks[device_id] = task
        logger.info(f"Started monitoring device: {device_id}")
    
    async def start_all_monitoring(self, interval: int = 30):
        """Start monitoring all registered devices"""
        for device_id in self.device_registry.get_all_device_ids():
            await self.start_device_monitoring(device_id, interval)
        
        self.is_running = True
        logger.info(f"Started monitoring {len(self.device_registry.get_all_device_ids())} devices")
    
    async def _monitor_device(self, device: DeviceInterface, interval: int):
        """Monitor a device and publish its readings"""
        logger.info(f"Starting monitoring loop for device: {device.get_id()}")
        
        while self.is_running:
            try:
                # Check if MQTT is connected
                if not self.is_connected:
                    logger.warning(f"MQTT not connected, waiting for connection...")
                    await self.connection_event.wait()
                
                reading = await device.read_data()
                if reading:
                    await self._publish_reading(reading)
                else:
                    logger.warning(f"No reading received from device {device.get_id()}")
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring device {device.get_id()}: {e}")
                await asyncio.sleep(interval)
    
    async def _publish_reading(self, reading: Dict[str, Any]):
        """Publish device reading to MQTT"""
        device_id = reading.get("device_id")
        device_type = reading.get("device_type")
        
        if not device_id or not device_type:
            logger.error(f"Invalid reading format, missing device_id or device_type: {reading}")
            return
        
        # Ensure timestamp exists and is in correct format
        if "timestamp" not in reading:
            reading["timestamp"] = datetime.utcnow().isoformat()
        
        # Ensure timestamp has microseconds for consistency
        timestamp = reading["timestamp"]
        if isinstance(timestamp, str) and "." not in timestamp:
            reading["timestamp"] = datetime.utcnow().isoformat()
        
        # Create MQTT topic based on device type and ID
        topic = f"{self.mqtt_topic_prefix}/{device_type}/{device_id}/data"
        
        # Convert reading to JSON
        try:
            message = json.dumps(reading, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize reading to JSON: {e}")
            return
        
        # Check connection before publishing
        if not self.is_connected:
            logger.warning(f"Cannot publish: MQTT not connected")
            return
        
        # Publish to MQTT
        try:
            result = self.client.publish(topic, message, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published reading for device {device_id} to topic {topic}")
                logger.debug(f"Message content: {message}")
            else:
                logger.error(f"Failed to publish reading for device {device_id} with error code {result.rc}")
                
        except Exception as e:
            logger.error(f"Exception while publishing reading for device {device_id}: {e}")
    
    async def stop_device_monitoring(self, device_id: str):
        """Stop monitoring a specific device"""
        if device_id in self.monitoring_tasks:
            self.monitoring_tasks[device_id].cancel()
            del self.monitoring_tasks[device_id]
            logger.info(f"Stopped monitoring device: {device_id}")
    
    async def stop_all_monitoring(self):
        """Stop all device monitoring"""
        logger.info("Stopping all device monitoring...")
        self.is_running = False
        
        for device_id in list(self.monitoring_tasks.keys()):
            await self.stop_device_monitoring(device_id)
        
        # Stop command processor
        self.command_processor_running = False
        if self.command_processor_task:
            self.command_processor_task.cancel()
            try:
                await self.command_processor_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from MQTT
        if self.is_connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from MQTT broker")
    
    async def run(self):
        """Run the gateway service"""
        if not await self.initialize():
            logger.error("Failed to initialize gateway, exiting")
            return
        
        # Add a small delay to ensure connection is stable
        await asyncio.sleep(2)
        
        await self.start_all_monitoring()
        
        # Keep the service running
        try:
            while self.is_running:
                await asyncio.sleep(60)
                await self._perform_health_check()
        except KeyboardInterrupt:
            logger.info("Shutting down gateway...")
        finally:
            await self.stop_all_monitoring()
    
    async def _perform_health_check(self):
        """Perform health check on all devices"""
        logger.debug("Performing health check on all devices...")
        unhealthy_devices = []
        
        for device_id in self.device_registry.get_all_device_ids():
            device = self.device_registry.get_device(device_id)
            try:
                is_healthy = await device.health_check()
                if not is_healthy:
                    unhealthy_devices.append(device_id)
            except Exception as e:
                logger.error(f"Health check failed for device {device_id}: {e}")
                unhealthy_devices.append(device_id)
        
        if unhealthy_devices:
            logger.warning(f"Unhealthy devices detected: {unhealthy_devices}")
        else:
            logger.debug("All devices are healthy")
    
    async def publish_test_message(self):
        """Publish a test message to verify MQTT connectivity"""
        test_message = {
            "device_id": "test_device",
            "device_type": "test",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"test": True, "message": "Gateway connectivity test"},
            "status": "online"
        }
        
        logger.info("Publishing test message...")
        await self._publish_reading(test_message)