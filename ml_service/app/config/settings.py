# ml_service/app/config/settings.py
import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # MQTT settings
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC_PREFIX: str = "smartcity"
    
    # Storage settings
    STORAGE_TYPE: str = "filesystem"
    STORAGE_PATH: str = "/app/storage"
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "models"
    
    # InfluxDB settings
    INFLUX_URL: str = "http://influxdb:8086"
    INFLUX_TOKEN: str = "my-token"
    INFLUX_ORG: str = "smartcity"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Model settings
    CONFIG_PATH: str = "/app/config"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Singleton instance
settings = Settings()