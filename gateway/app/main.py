# gateway/app/main.py (Updated for Physical Device Support)
import asyncio
import logging
import os
import sys
import math
from dotenv import load_dotenv

from core.gateway import IoTGateway
from core.device_discovery import EnhancedDeviceDiscovery
from core.connection_factory import ConnectionFactory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function with enhanced physical device support"""
    logger.info("🚀 Starting Smart City IoT Gateway with Physical Device Support")
    
    # Create connection factory and log available types
    factory = ConnectionFactory()
    logger.info(f"📡 Available connections: {factory.get_available_connections()}")
    logger.info(f"🔄 Available processors: {factory.get_available_processors()}")
    
    # Create gateway
    gateway = IoTGateway(
        mqtt_broker=os.getenv('MQTT_BROKER', 'emqx'),
        mqtt_port=int(os.getenv('MQTT_PORT', '1883')),
        mqtt_topic_prefix=os.getenv('MQTT_TOPIC_PREFIX', 'smartcity')
    )
    
    # Initialize gateway
    if not await gateway.initialize():
        logger.error("❌ Failed to initialize gateway, exiting")
        return
    
    # Test connectivity
    logger.info("🔍 Testing MQTT connectivity...")
    await gateway.publish_test_message()
    await asyncio.sleep(2)
    
    # Create enhanced device discovery
    device_discovery = EnhancedDeviceDiscovery(gateway=gateway)
    
    # Scan for physical devices first
    logger.info("🔎 Scanning for physical devices...")
    scan_results = await device_discovery.scan_for_physical_devices()
    
    # Discover all device types
    logger.info("🔍 Discovering device types...")
    discovered_devices = device_discovery.discover_devices()
    
    if not discovered_devices:
        logger.warning("⚠️ No devices discovered. Add device files to the devices/ folder.")
        logger.info("The gateway will monitor for new devices and automatically add them.")
    else:
        # Categorize discovered devices
        physical_devices = []
        simulated_devices = []
        
        for device_type, device_class in discovered_devices.items():
            from core.physical_device import PhysicalDevice
            if issubclass(device_class, PhysicalDevice):
                physical_devices.append(device_type)
            else:
                simulated_devices.append(device_type)
        
        logger.info(f"📱 Discovered {len(discovered_devices)} device types:")
        if physical_devices:
            logger.info(f"  🔌 Physical: {physical_devices}")
        if simulated_devices:
            logger.info(f"  💻 Simulated: {simulated_devices}")
    
    # Load/generate configuration
    config = device_discovery.generate_dynamic_config()

    # Create device instances
    device_instances = await device_discovery.create_device_instances(config)
    logger.info(f"✅ Created {len(device_instances)} device instances")
    
    # Log instance summary
    physical_count = 0
    simulated_count = 0
    for device in device_instances:
        gateway.register_device(device)
        from core.physical_device import PhysicalDevice
        if isinstance(device, PhysicalDevice):
            physical_count += 1
            logger.info(f"  🔌 Physical: {device.get_id()} ({device.get_type()})")
        else:
            simulated_count += 1
            logger.info(f"  💻 Simulated: {device.get_id()} ({device.get_type()})")
    
    logger.info(f"📊 Device summary: {physical_count} physical, {simulated_count} simulated")
    
    # Start file monitoring for hot-plugging
    device_discovery.start_file_monitoring()
    
    # Start monitoring if we have devices
    if device_instances:
        monitoring_interval = config.get("global_settings", {}).get("monitoring_interval", 30)
        logger.info(f"⏱️ Starting device monitoring with {monitoring_interval}s interval...")
        await gateway.start_all_monitoring(interval=monitoring_interval)
    
    # Keep the service running
    try:
        logger.info("🟢 Gateway is running with physical device support!")
        logger.info("   • Add new device files to auto-register them")
        logger.info("   • Connect physical devices to auto-discover them")
        logger.info("   • Send MQTT commands to control actuators")
        logger.info("   • Press Ctrl+C to stop")
        
        while gateway.is_running or not device_instances:
            await asyncio.sleep(60)
            
            # Perform health checks
            if gateway.is_running:
                await gateway._perform_health_check()
                
                # Check physical device connections
                for device in device_instances:
                    from core.physical_device import PhysicalDevice
                    if isinstance(device, PhysicalDevice):
                        conn_info = device.get_connection_info()
                        if conn_info['connection_status'] == 'error':
                            logger.warning(f"⚠️ Physical device {device.get_id()} connection error: {conn_info['last_error']}")
            
            # If we started with no devices, check if we have some now
            if not device_instances and device_discovery.active_instances:
                device_instances = list(device_discovery.active_instances.values())
                if device_instances:
                    monitoring_interval = config.get("global_settings", {}).get("monitoring_interval", 30)
                    await gateway.start_all_monitoring(interval=monitoring_interval)
                    logger.info("✅ Started monitoring newly discovered devices")
                    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gateway...")
    finally:
        # Cleanup
        device_discovery.stop_file_monitoring()
        
        # Disconnect physical devices
        for device in device_instances:
            from core.physical_device import PhysicalDevice
            if isinstance(device, PhysicalDevice):
                await device.disconnect()
                logger.info(f"🔌 Disconnected physical device: {device.get_id()}")
        
        await gateway.stop_all_monitoring()


if __name__ == "__main__":
    try:
        # Import math for dew point calculation in ESP32 device
        import math
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Gateway shutdown by user")
    except Exception as e:
        logger.error(f"💥 Gateway failed with error: {e}")
        import traceback
        traceback.print_exc()
