# gateway/app/core/device_discovery.py (Updated for Physical Devices)
import os
import sys
import importlib
import inspect
import logging
import asyncio
from typing import Dict, List, Type, Any, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .device_interface import DeviceInterface
from .physical_device import PhysicalDevice
from .connection_factory import ConnectionFactory

logger = logging.getLogger(__name__)

class DeviceFileHandler(FileSystemEventHandler):
    """Handle file system events for device discovery"""
    
    def __init__(self, discovery_manager):
        self.discovery_manager = discovery_manager
        self.debounce_delay = 2.0  # seconds
        self.pending_changes = set()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"New device file detected: {event.src_path}")
            self._schedule_rediscovery(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"Device file modified: {event.src_path}")
            self._schedule_rediscovery(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"Device file deleted: {event.src_path}")
            self._schedule_rediscovery(event.src_path)
    
    def _schedule_rediscovery(self, file_path):
        """Schedule a rediscovery with debouncing"""
        self.pending_changes.add(file_path)
        
        async def delayed_rediscovery():
            await asyncio.sleep(self.debounce_delay)
            if file_path in self.pending_changes:
                self.pending_changes.remove(file_path)
                await self.discovery_manager.rediscover_devices()
        
        # Schedule the coroutine
        asyncio.create_task(delayed_rediscovery())



class EnhancedDeviceDiscovery:
    """
    Enhanced device discovery with physical device support
    Automatically discovers and loads both simulated and physical device classes
    """
    
    def __init__(self, devices_package: str = "devices", gateway=None):
        self.devices_package = devices_package
        self.gateway = gateway
        self.discovered_devices: Dict[str, Type[DeviceInterface]] = {}
        self.active_instances: Dict[str, DeviceInterface] = {}
        self.file_observer = None
        self.is_monitoring = False
        self.connection_factory = ConnectionFactory()
        
        # Get the devices directory path
        self.devices_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            self.devices_package
        )
    
    def start_file_monitoring(self):
        """Start monitoring the devices directory for changes"""
        if not os.path.exists(self.devices_dir):
            logger.warning(f"Devices directory not found: {self.devices_dir}")
            return
        
        if self.is_monitoring:
            return
        
        event_handler = DeviceFileHandler(self)
        self.file_observer = Observer()
        self.file_observer.schedule(event_handler, self.devices_dir, recursive=False)
        self.file_observer.start()
        self.is_monitoring = True
        logger.info(f"Started monitoring devices directory: {self.devices_dir}")
    
    def stop_file_monitoring(self):
        """Stop monitoring the devices directory"""
        if self.file_observer and self.is_monitoring:
            self.file_observer.stop()
            self.file_observer.join()
            self.is_monitoring = False
            logger.info("Stopped monitoring devices directory")
    
    def discover_devices(self) -> Dict[str, Type[DeviceInterface]]:
        """
        Discover all device classes including physical devices
        """
        logger.info(f"Discovering devices in package: {self.devices_package}")
        
        if not os.path.exists(self.devices_dir):
            logger.warning(f"Devices directory not found: {self.devices_dir}")
            return {}
        
        new_discovered = {}
        
        # Get all Python files in the devices directory
        device_files = [
            f[:-3] for f in os.listdir(self.devices_dir) 
            if f.endswith('.py') and f != '__init__.py' and not f.startswith('_')
        ]
        
        logger.info(f"Found device files: {device_files}")
        
        for module_name in device_files:
            try:
                # Import or reload the module
                module_path = f"{self.devices_package}.{module_name}"
                
                # Reload module if it was already imported
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                
                module = importlib.import_module(module_path)
                
                # Find all classes in the module that implement DeviceInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (obj != DeviceInterface and obj != PhysicalDevice and
                        issubclass(obj, DeviceInterface) and 
                        obj.__module__ == module_path):
                        
                        # Validate that the class implements required methods
                        if self._validate_device_class(obj):
                            device_type = obj.get_device_info()['type']
                            new_discovered[device_type] = obj
                            
                            # Log device type and capabilities
                            device_info = obj.get_device_info()
                            is_physical = issubclass(obj, PhysicalDevice)
                            device_category = "Physical" if is_physical else "Simulated"
                            
                            logger.info(f"Discovered {device_category} device: {device_type} -> {obj.__name__}")
                            
                            if is_physical:
                                conn_type = device_info.get('connection_type', 'unknown')
                                protocol = device_info.get('protocol', 'unknown')
                                logger.info(f"  Connection: {conn_type}, Protocol: {protocol}")
                        else:
                            logger.warning(f"Device class {obj.__name__} does not implement required methods")
                        
            except Exception as e:
                logger.error(f"Failed to load device module {module_name}: {e}")
        
        self.discovered_devices = new_discovered
        logger.info(f"Total devices discovered: {len(self.discovered_devices)}")
        
        # Log summary by type
        physical_count = sum(1 for cls in self.discovered_devices.values() 
                           if issubclass(cls, PhysicalDevice))
        simulated_count = len(self.discovered_devices) - physical_count
        logger.info(f"Device summary: {physical_count} physical, {simulated_count} simulated")
        
        return self.discovered_devices
    
    def _validate_device_class(self, device_class: Type[DeviceInterface]) -> bool:
        """Validate that device class implements all required methods"""
        required_methods = ['get_default_config', 'get_device_info']
        
        for method in required_methods:
            if not hasattr(device_class, method):
                logger.error(f"Device class {device_class.__name__} missing required method: {method}")
                return False
        
        # Additional validation for physical devices
        if issubclass(device_class, PhysicalDevice):
            device_info = device_class.get_device_info()
            if 'connection_type' not in device_info:
                logger.warning(f"Physical device {device_class.__name__} missing connection_type in device_info")
        
        return True
    
    def generate_dynamic_config(self) -> Dict[str, Any]:
        """
        Generate configuration for both physical and simulated devices
        """
        if not self.discovered_devices:
            logger.warning("No devices discovered, generating empty configuration")
            return {
                "devices": {},
                "global_settings": {
                    "monitoring_interval": 30,
                    "default_instance_count": 1
                },
                "connection_settings": {
                    "serial_scan_ports": ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"],
                    "wifi_network_scan": True,
                    "i2c_bus_scan": True
                }
            }
        
        config = {
            "devices": {},
            "global_settings": {
                "monitoring_interval": 30,
                "default_instance_count": 1
            },
            "connection_settings": {
                "serial_scan_ports": ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"],
                "wifi_default_network": "192.168.1.0/24",
                "i2c_default_bus": 1
            }
        }
        
        # Generate configuration for each discovered device
        for device_type, device_class in self.discovered_devices.items():
            try:
                device_info = device_class.get_device_info()
                default_config = device_class.get_default_config()
                
                # Determine if this is a physical device
                is_physical = issubclass(device_class, PhysicalDevice)
                
                device_config = {
                    "enabled": True,
                    "instance_count": default_config.get("default_instances", 1),
                    "is_physical": is_physical,
                    "info": device_info,
                    "config": default_config
                }
                
                # Add physical device specific configuration
                if is_physical:
                    device_config["connection_required"] = True
                    device_config["auto_discovery"] = True
                
                config["devices"][device_type] = device_config
                
                device_category = "Physical" if is_physical else "Simulated"
                logger.info(f"Generated config for {device_category} {device_type}: {device_info.get('name', device_type)}")
                
            except Exception as e:
                logger.error(f"Failed to get configuration for {device_type}: {e}")
        
        return config
    
    async def create_device_instances(self, config: Dict[str, Any]) -> List[DeviceInterface]:
        """
        Create instances of both physical and simulated devices
        """
        instances = []
        
        for device_type, device_config in config.get("devices", {}).items():
            if not device_config.get("enabled", True):
                logger.info(f"Skipping disabled device type: {device_type}")
                continue
            
            if device_type not in self.discovered_devices:
                logger.warning(f"Device type {device_type} not found in discovered devices")
                continue
            
            device_class = self.discovered_devices[device_type]
            instance_count = device_config.get("instance_count", 1)
            device_specific_config = device_config.get("config", {})
            is_physical = device_config.get("is_physical", False)
            
            logger.info(f"Creating {instance_count} instances of {device_type} ({'Physical' if is_physical else 'Simulated'})")
            
            for i in range(instance_count):
                try:
                    if is_physical:
                        instance = await self._create_physical_device_instance(
                            device_class, device_type, i + 1, device_specific_config
                        )
                    else:
                        instance = await self._create_simulated_device_instance(
                            device_class, device_type, i + 1, device_specific_config
                        )
                    
                    if instance:
                        instances.append(instance)
                        self.active_instances[instance.get_id()] = instance
                        logger.info(f"Created device instance: {instance.get_id()}")
                    
                except Exception as e:
                    logger.error(f"Failed to create instance {i+1} of {device_type}: {e}")
        
        return instances
    
    async def _create_physical_device_instance(self, device_class: Type[PhysicalDevice], 
                                             device_type: str, instance_num: int, 
                                             config: Dict[str, Any]) -> Optional[PhysicalDevice]:
        """Create a physical device instance with connection auto-configuration"""
        device_id = f"{device_type}_{instance_num}"
        
        try:
            # Calculate location with offset
            location = self._calculate_instance_location(config, instance_num)
            
            # Auto-configure connection parameters
            updated_config = await self._auto_configure_connection(config, device_type, instance_num)
            
            # Create device instance
            if location:
                instance = device_class(device_id=device_id, location=location)
            else:
                instance = device_class(device_id=device_id)
            
            # Update instance configuration
            await instance.configure(updated_config)
            
            # Test connection
            if await instance.connection.connect():
                logger.info(f"Physical device {device_id} connected successfully")
                return instance
            else:
                logger.error(f"Failed to connect to physical device {device_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create physical device instance {device_id}: {e}")
            return None
    
    async def _create_simulated_device_instance(self, device_class: Type[DeviceInterface], 
                                              device_type: str, instance_num: int, 
                                              config: Dict[str, Any]) -> Optional[DeviceInterface]:
        """Create a simulated device instance (original logic)"""
        device_id = f"{device_type}_{instance_num}"
        
        try:
            # Calculate location with offset
            location = self._calculate_instance_location(config, instance_num)
            
            # Create device instance
            if location:
                instance = device_class(device_id=device_id, location=location)
            else:
                instance = device_class(device_id=device_id)
            
            # Apply configuration
            await instance.configure(config)
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create simulated device instance {device_id}: {e}")
            return None
    
    def _calculate_instance_location(self, config: Dict[str, Any], instance_num: int) -> Optional[Dict[str, float]]:
        """Calculate location for device instance"""
        if "location_base" not in config:
            return None
        
        base_location = config["location_base"]
        location_offset = config.get("location_offset", 0.01)
        
        return {
            "lat": base_location["lat"] + (instance_num - 1) * location_offset,
            "lon": base_location["lon"] + (instance_num - 1) * location_offset
        }
    
    async def _auto_configure_connection(self, config: Dict[str, Any], device_type: str, 
                                       instance_num: int) -> Dict[str, Any]:
        """Auto-configure connection parameters for physical devices"""
        updated_config = config.copy()
        
        if 'connection' in updated_config:
            connection_config = updated_config['connection'].copy()
            
            # Auto-configure serial port
            if 'port' in connection_config and connection_config['port'] == "/dev/ttyUSB0":
                # Try different ports for different instances
                ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]
                if instance_num <= len(ports):
                    connection_config['port'] = ports[instance_num - 1]
                    logger.info(f"Auto-configured serial port for {device_type}_{instance_num}: {connection_config['port']}")
            
            # Auto-configure WiFi IP
            if 'host' in connection_config and connection_config['host'] == "192.168.1.100":
                # Generate unique IP for each instance
                base_ip = "192.168.1."
                ip_suffix = 100 + instance_num - 1
                connection_config['host'] = f"{base_ip}{ip_suffix}"
                logger.info(f"Auto-configured WiFi IP for {device_type}_{instance_num}: {connection_config['host']}")
            
            # Auto-configure I2C address
            if 'address' in connection_config and connection_config['address'] == 0x48:
                # Generate unique I2C address for each instance
                connection_config['address'] = 0x48 + instance_num - 1
                logger.info(f"Auto-configured I2C address for {device_type}_{instance_num}: 0x{connection_config['address']:02X}")
            
            updated_config['connection'] = connection_config
        
        return updated_config
    
    async def scan_for_physical_devices(self) -> Dict[str, List[str]]:
        """Scan for available physical devices on various interfaces"""
        scan_results = {
            "serial_ports": [],
            "wifi_devices": [],
            "i2c_devices": []
        }
        
        # Scan serial ports
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            scan_results["serial_ports"] = [port.device for port in ports]
            logger.info(f"Found serial ports: {scan_results['serial_ports']}")
        except ImportError:
            logger.warning("Serial port scanning not available")
        
        # Scan I2C bus
        try:
            from connections.i2c_connection import I2CConnection
            # This would require actual I2C scanning implementation
            logger.info("I2C scanning capability available")
        except ImportError:
            logger.warning("I2C scanning not available")
        
        # WiFi device discovery could be added here
        # (network scanning, mDNS discovery, etc.)
        
        return scan_results
    
    async def rediscover_devices(self):
        """Rediscover devices with enhanced physical device support"""
        logger.info("Rediscovering devices (including physical devices)...")
        
        # Store current device instances
        old_instances = self.active_instances.copy()
        
        # Scan for physical devices
        scan_results = await self.scan_for_physical_devices()
        logger.info(f"Physical device scan results: {scan_results}")
        
        # Discover devices again
        self.discover_devices()
        
        # Generate new configuration
        config = self.generate_dynamic_config()
        
        # Create new instances
        new_instances = await self.create_device_instances(config)
        
        # Update gateway if available
        if self.gateway:
            # Remove old devices that are no longer present
            for device_id in old_instances:
                if device_id not in self.active_instances:
                    old_device = old_instances[device_id]
                    await self.gateway.stop_device_monitoring(device_id)
                    
                    # Disconnect physical devices
                    if isinstance(old_device, PhysicalDevice):
                        await old_device.disconnect()
                    
                    logger.info(f"Removed device: {device_id}")
            
            # Add new devices
            for instance in new_instances:
                if instance.get_id() not in old_instances:
                    self.gateway.register_device(instance)
                    await self.gateway.start_device_monitoring(instance.get_id())
                    
                    device_type = "Physical" if isinstance(instance, PhysicalDevice) else "Simulated"
                    logger.info(f"Added new {device_type} device: {instance.get_id()}")
        
        logger.info(f"Rediscovery complete. Active devices: {len(self.active_instances)}")


