# # dashboard/app/dashboard_config.py

# from typing import Dict, Any

# class DashboardConfig:
#     """Configuration class for dynamic dashboard customization"""
    
#     METRIC_DISPLAY_CONFIG = {
#         'temperature': {'icon': '🌡️', 'unit': '°C', 'format': '.1f', 'chart_color': '#FF6B6B'},
#         'humidity': {'icon': '💧', 'unit': '%', 'format': '.1f', 'chart_color': '#4ECDC4'},
#         'average_speed': {'icon': '🏃', 'unit': 'km/h', 'format': '.1f', 'chart_color': '#45B7D1'},
#         'vehicle_count': {'icon': '🚗', 'unit': '', 'format': '.0f', 'chart_color': '#FDCB6E'},
#         'water_level': {'icon': '🌊', 'unit': 'm', 'format': '.2f', 'chart_color': '#00B894'},
#         'lane_occupancy': {'icon': '🛣️', 'unit': '%', 'format': '.1f', 'chart_color': '#FF7675'},
#         'power': {'icon': '⚡', 'unit': 'W', 'format': '.1f', 'chart_color': '#FDCB6E'},
#         'brightness': {'icon': '💡', 'unit': '%', 'format': '.0f', 'chart_color': '#FFF200'},
#         'pressure': {'icon': '💨', 'unit': 'hPa', 'format': '.1f', 'chart_color': '#A8E6CF'},
#         'flow_rate': {'icon': '💧', 'unit': 'm³/s', 'format': '.2f', 'chart_color': '#4ECDC4'},
#         'air_quality': {'icon': '🌬️', 'unit': 'AQI', 'format': '.0f', 'chart_color': '#6BCF7F'},
#         'noise_level': {'icon': '🔊', 'unit': 'dB', 'format': '.1f', 'chart_color': '#FF6B9D'},
#         'pm25': {'icon': '🌫️', 'unit': 'μg/m³', 'format': '.1f', 'chart_color': '#B19CD9'},
#         'pm10': {'icon': '🌫️', 'unit': 'μg/m³', 'format': '.1f', 'chart_color': '#C79DD7'},
#         'co2': {'icon': '💨', 'unit': 'ppm', 'format': '.0f', 'chart_color': '#D1C4E9'},
#         'precipitation': {'icon': '🌧️', 'unit': 'mm', 'format': '.1f', 'chart_color': '#00CEC9'},
#         'battery': {'icon': '🔋', 'unit': '%', 'format': '.0f', 'chart_color': '#00E676'},
#         'default': {'icon': '📊', 'unit': '', 'format': '.2f', 'chart_color': '#A8A8A8'}
#     }
    
#     DEVICE_TYPE_EMOJIS = {
#         'traffic_sensor': '🚦',
#         'water_level_sensor': '🌊',
#         'air_quality_sensor': '🌬️',
#         'smart_light': '💡',
#         'smart_thermostat': '🌡️',
#         'smart_irrigation': '🚿',
#         'noise_sensor': '🔊',
#         'parking_sensor': '🅿️',
#         'default': '📡'
#     }
    
#     ALERT_THRESHOLDS = {
#         'temperature': {'critical_high': 40, 'warning_high': 35},
#         'water_level': {'critical_high': 4.0, 'warning_high': 3.0},
#         'vehicle_count': {'critical_high': 100, 'warning_high': 60},
#         'average_speed': {'critical_low': 15, 'warning_low': 30},
#         'lane_occupancy': {'critical_high': 95, 'warning_high': 80},
#         'noise_level': {'critical_high': 85, 'warning_high': 70},
#         'battery': {'critical_low': 10, 'warning_low': 20},
#     }
    
#     @classmethod
#     def get_metric_config(cls, metric_name: str) -> Dict[str, str]:
#         metric_lower = metric_name.lower()
#         for key, config in cls.METRIC_DISPLAY_CONFIG.items():
#             if key in metric_lower:
#                 return config
#         return cls.METRIC_DISPLAY_CONFIG['default']
    
#     @classmethod
#     def get_device_emoji(cls, device_type: str) -> str:
#         device_type_lower = device_type.lower()
#         return cls.DEVICE_TYPE_EMOJIS.get(device_type_lower, cls.DEVICE_TYPE_EMOJIS['default'])
    
#     @classmethod
#     def get_alert_config(cls, metric_name: str) -> Dict[str, float]:
#         metric_lower = metric_name.lower()
#         for key, thresholds in cls.ALERT_THRESHOLDS.items():
#             if key in metric_lower:
#                 return thresholds
#         return {}

# dashboard/app/dashboard_config.py

class DashboardConfig:
    """Configuration class for Smart City Dashboard display settings."""
    
    # Device type emoji mappings
    DEVICE_EMOJIS = {
        'traffic_sensor': '🚦',
        'air_quality': '🌬️',
        'noise_sensor': '🔊',
        'weather_station': '🌤️',
        'parking_sensor': '🅿️',
        'energy_meter': '⚡',
        'water_sensor': '💧',
        'waste_sensor': '🗑️',
        'camera': '📹',
        'street_light': '💡',
        'emergency_button': '🚨',
        'bike_station': '🚲',
        'ev_charger': '🔌',
        'smart_bin': '♻️',
        'flood_sensor': '🌊',
        'radiation_sensor': '☢️',
        'soil_sensor': '🌱',
        'default': '📊'
    }
    
    # Metric configuration with icons, units, formatting, and chart colors
    METRIC_CONFIGS = {
        # Traffic sensors
        'vehicle_count': {
            'icon': '🚗',
            'unit': ' vehicles',
            'format': '.0f',
            'chart_color': '#E74C3C'
        },
        'speed_avg': {
            'icon': '🏃',
            'unit': ' km/h',
            'format': '.1f',
            'chart_color': '#3498DB'
        },
        'congestion_level': {
            'icon': '🚦',
            'unit': '%',
            'format': '.1f',
            'chart_color': '#F39C12'
        },
        
        # Air quality sensors
        'pm25': {
            'icon': '💨',
            'unit': ' μg/m³',
            'format': '.1f',
            'chart_color': '#9B59B6'
        },
        'pm10': {
            'icon': '🌫️',
            'unit': ' μg/m³',
            'format': '.1f',
            'chart_color': '#8E44AD'
        },
        'co2': {
            'icon': '💨',
            'unit': ' ppm',
            'format': '.0f',
            'chart_color': '#E67E22'
        },
        'aqi': {
            'icon': '🌬️',
            'unit': '',
            'format': '.0f',
            'chart_color': '#2ECC71'
        },
        'no2': {
            'icon': '🏭',
            'unit': ' ppb',
            'format': '.1f',
            'chart_color': '#D35400'
        },
        'o3': {
            'icon': '☀️',
            'unit': ' ppb',
            'format': '.1f',
            'chart_color': '#F1C40F'
        },
        
        # Weather stations
        'temperature': {
            'icon': '🌡️',
            'unit': '°C',
            'format': '.1f',
            'chart_color': '#E74C3C'
        },
        'humidity': {
            'icon': '💧',
            'unit': '%',
            'format': '.1f',
            'chart_color': '#3498DB'
        },
        'pressure': {
            'icon': '📊',
            'unit': ' hPa',
            'format': '.1f',
            'chart_color': '#9B59B6'
        },
        'wind_speed': {
            'icon': '💨',
            'unit': ' m/s',
            'format': '.1f',
            'chart_color': '#1ABC9C'
        },
        'wind_direction': {
            'icon': '🧭',
            'unit': '°',
            'format': '.0f',
            'chart_color': '#F39C12'
        },
        'rainfall': {
            'icon': '🌧️',
            'unit': ' mm',
            'format': '.1f',
            'chart_color': '#2980B9'
        },
        'uv_index': {
            'icon': '☀️',
            'unit': '',
            'format': '.1f',
            'chart_color': '#F1C40F'
        },
        
        # Noise sensors
        'noise_level': {
            'icon': '🔊',
            'unit': ' dB',
            'format': '.1f',
            'chart_color': '#E74C3C'
        },
        'peak_noise': {
            'icon': '📢',
            'unit': ' dB',
            'format': '.1f',
            'chart_color': '#C0392B'
        },
        
        # Parking sensors
        'occupancy': {
            'icon': '🅿️',
            'unit': '%',
            'format': '.1f',
            'chart_color': '#3498DB'
        },
        'available_spots': {
            'icon': '🟢',
            'unit': ' spots',
            'format': '.0f',
            'chart_color': '#2ECC71'
        },
        'total_spots': {
            'icon': '🔢',
            'unit': ' spots',
            'format': '.0f',
            'chart_color': '#95A5A6'
        },
        
        # Energy meters
        'power_consumption': {
            'icon': '⚡',
            'unit': ' kW',
            'format': '.2f',
            'chart_color': '#F39C12'
        },
        'voltage': {
            'icon': '🔌',
            'unit': ' V',
            'format': '.1f',
            'chart_color': '#E67E22'
        },
        'current': {
            'icon': '⚡',
            'unit': ' A',
            'format': '.2f',
            'chart_color': '#D35400'
        },
        'power_factor': {
            'icon': '📊',
            'unit': '',
            'format': '.3f',
            'chart_color': '#8E44AD'
        },
        'frequency': {
            'icon': '📡',
            'unit': ' Hz',
            'format': '.1f',
            'chart_color': '#2C3E50'
        },
        
        # Water sensors
        'flow_rate': {
            'icon': '💧',
            'unit': ' L/min',
            'format': '.1f',
            'chart_color': '#3498DB'
        },
        'water_level': {
            'icon': '📏',
            'unit': ' cm',
            'format': '.1f',
            'chart_color': '#2980B9'
        },
        'water_quality': {
            'icon': '🧪',
            'unit': '',
            'format': '.1f',
            'chart_color': '#1ABC9C'
        },
        'ph_level': {
            'icon': '⚗️',
            'unit': ' pH',
            'format': '.2f',
            'chart_color': '#16A085'
        },
        'turbidity': {
            'icon': '🌊',
            'unit': ' NTU',
            'format': '.1f',
            'chart_color': '#148F77'
        },
        
        # Waste sensors
        'fill_level': {
            'icon': '🗑️',
            'unit': '%',
            'format': '.1f',
            'chart_color': '#27AE60'
        },
        'weight': {
            'icon': '⚖️',
            'unit': ' kg',
            'format': '.1f',
            'chart_color': '#229954'
        },
        'compaction_ratio': {
            'icon': '📦',
            'unit': '',
            'format': '.2f',
            'chart_color': '#1E8449'
        },
        
        # Street lights
        'brightness': {
            'icon': '💡',
            'unit': '%',
            'format': '.0f',
            'chart_color': '#F1C40F'
        },
        'energy_consumption': {
            'icon': '🔋',
            'unit': ' W',
            'format': '.1f',
            'chart_color': '#F39C12'
        },
        'operating_hours': {
            'icon': '⏰',
            'unit': ' hrs',
            'format': '.1f',
            'chart_color': '#E67E22'
        },
        
        # Generic/default metrics
        'status': {
            'icon': '📊',
            'unit': '',
            'format': '.0f',
            'chart_color': '#95A5A6'
        },
        'battery_level': {
            'icon': '🔋',
            'unit': '%',
            'format': '.0f',
            'chart_color': '#2ECC71'
        },
        'signal_strength': {
            'icon': '📶',
            'unit': ' dBm',
            'format': '.0f',
            'chart_color': '#3498DB'
        },
        'uptime': {
            'icon': '⏱️',
            'unit': ' hrs',
            'format': '.1f',
            'chart_color': '#9B59B6'
        }
    }
    
    # Alert thresholds for different metrics
    ALERT_CONFIGS = {
        # Air quality thresholds (WHO guidelines)
        'pm25': {
            'warning_high': 35,
            'critical_high': 75
        },
        'pm10': {
            'warning_high': 50,
            'critical_high': 150
        },
        'co2': {
            'warning_high': 1000,
            'critical_high': 2000
        },
        'aqi': {
            'warning_high': 100,
            'critical_high': 200
        },
        'no2': {
            'warning_high': 40,
            'critical_high': 100
        },
        'o3': {
            'warning_high': 60,
            'critical_high': 120
        },
        
        # Weather thresholds
        'temperature': {
            'warning_low': 0,
            'warning_high': 35,
            'critical_low': -10,
            'critical_high': 40
        },
        'wind_speed': {
            'warning_high': 15,
            'critical_high': 25
        },
        'uv_index': {
            'warning_high': 6,
            'critical_high': 8
        },
        
        # Noise thresholds (WHO guidelines)
        'noise_level': {
            'warning_high': 55,
            'critical_high': 70
        },
        'peak_noise': {
            'warning_high': 70,
            'critical_high': 85
        },
        
        # Infrastructure thresholds
        'fill_level': {
            'warning_high': 80,
            'critical_high': 95
        },
        'battery_level': {
            'warning_low': 20,
            'critical_low': 10
        },
        'signal_strength': {
            'warning_low': -80,
            'critical_low': -100
        },
        
        # Traffic thresholds
        'congestion_level': {
            'warning_high': 70,
            'critical_high': 90
        },
        'vehicle_count': {
            'warning_high': 100,
            'critical_high': 150
        },
        
        # Energy thresholds
        'power_consumption': {
            'warning_high': 500,
            'critical_high': 1000
        },
        'voltage': {
            'warning_low': 210,
            'warning_high': 250,
            'critical_low': 200,
            'critical_high': 260
        },
        
        # Water quality thresholds
        'ph_level': {
            'warning_low': 6.5,
            'warning_high': 8.5,
            'critical_low': 6.0,
            'critical_high': 9.0
        },
        'turbidity': {
            'warning_high': 4,
            'critical_high': 10
        }
    }
    
    @classmethod
    def get_device_emoji(cls, device_type: str) -> str:
        """Get emoji for device type."""
        return cls.DEVICE_EMOJIS.get(device_type, cls.DEVICE_EMOJIS['default'])
    
    @classmethod
    def get_metric_config(cls, metric_name: str) -> dict:
        """Get configuration for a metric."""
        default_config = {
            'icon': '📊',
            'unit': '',
            'format': '.2f',
            'chart_color': '#95A5A6'
        }
        return cls.METRIC_CONFIGS.get(metric_name, default_config)
    
    @classmethod
    def get_alert_config(cls, metric_name: str) -> dict:
        """Get alert thresholds for a metric."""
        return cls.ALERT_CONFIGS.get(metric_name, {})
    
    @classmethod
    def get_chart_colors(cls) -> list:
        """Get list of chart colors for multi-series plots."""
        return [
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#1ABC9C', '#E67E22', '#F1C40F', '#2980B9', '#8E44AD',
            '#27AE60', '#D35400', '#C0392B', '#2C3E50', '#7F8C8D'
        ]
    
    @classmethod
    def get_severity_color(cls, severity: str) -> str:
        """Get color for alert severity."""
        colors = {
            'critical': '#E74C3C',
            'warning': '#F39C12',
            'info': '#3498DB',
            'success': '#2ECC71'
        }
        return colors.get(severity, '#95A5A6')
    
    @classmethod
    def format_metric_value(cls, metric_name: str, value: float) -> str:
        """Format metric value with proper units and precision."""
        config = cls.get_metric_config(metric_name)
        try:
            formatted = f"{value:{config['format']}}{config['unit']}"
        except (ValueError, TypeError):
            formatted = f"{value:.2f}{config['unit']}"
        return formatted
    
    @classmethod
    def get_status_color(cls, value: float, metric_name: str) -> str:
        """Get status color based on metric value and thresholds."""
        thresholds = cls.get_alert_config(metric_name)
        
        if not thresholds:
            return '#2ECC71'  # Green if no thresholds defined
        
        # Check critical thresholds first
        if 'critical_high' in thresholds and value > thresholds['critical_high']:
            return '#E74C3C'  # Red
        if 'critical_low' in thresholds and value < thresholds['critical_low']:
            return '#E74C3C'  # Red
            
        # Check warning thresholds
        if 'warning_high' in thresholds and value > thresholds['warning_high']:
            return '#F39C12'  # Orange
        if 'warning_low' in thresholds and value < thresholds['warning_low']:
            return '#F39C12'  # Orange
            
        return '#2ECC71'  # Green (normal)
    
    @classmethod
    def get_device_types_info(cls) -> dict:
        """Get comprehensive information about all device types."""
        return {
            device_type: {
                'emoji': emoji,
                'display_name': device_type.replace('_', ' ').title(),
                'category': cls._get_device_category(device_type)
            }
            for device_type, emoji in cls.DEVICE_EMOJIS.items()
            if device_type != 'default'
        }
    
    @classmethod
    def _get_device_category(cls, device_type: str) -> str:
        """Get category for device type."""
        categories = {
            'traffic_sensor': 'Transportation',
            'parking_sensor': 'Transportation',
            'bike_station': 'Transportation',
            'ev_charger': 'Transportation',
            
            'air_quality': 'Environment',
            'noise_sensor': 'Environment',
            'weather_station': 'Environment',
            'radiation_sensor': 'Environment',
            'soil_sensor': 'Environment',
            'flood_sensor': 'Environment',
            
            'energy_meter': 'Utilities',
            'water_sensor': 'Utilities',
            'street_light': 'Utilities',
            
            'waste_sensor': 'Waste Management',
            'smart_bin': 'Waste Management',
            
            'camera': 'Security',
            'emergency_button': 'Security'
        }
        return categories.get(device_type, 'Other')
    
    @classmethod
    def get_dashboard_theme(cls) -> dict:
        """Get dashboard theme configuration."""
        return {
            'primary_color': '#2C3E50',
            'secondary_color': '#3498DB',
            'success_color': '#2ECC71',
            'warning_color': '#F39C12',
            'danger_color': '#E74C3C',
            'info_color': '#3498DB',
            'background_color': '#FFFFFF',
            'card_background': '#F8F9FA',
            'text_color': '#2C3E50',
            'border_color': '#E9ECEF'
        }