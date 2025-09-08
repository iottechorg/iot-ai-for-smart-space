# ml_service/app/core/storage_manager.py
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Storage manager for ML models and data
    Uses the StorageFactory to get appropriate storage implementation
    """
    
    def __init__(self):
        from app.storage.factory import StorageFactory
        self.storage = StorageFactory.create_storage()
    
    async def initialize(self):
        """Initialize storage"""
        await self.storage.initialize()
        logger.info("Storage initialized")
        return True
    
    async def upload_file(self, file_path: str, object_key: str, metadata: Optional[Dict] = None):
        """Upload file to storage"""
        try:
            await self.storage.upload_file(file_path, object_key, metadata)
            logger.info(f"Uploaded {file_path} to {object_key}")
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    async def download_file(self, object_key: str, file_path: str):
        """Download file from storage"""
        try:
            await self.storage.download_file(object_key, file_path)
            logger.info(f"Downloaded {object_key} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {object_key}: {e}")
            raise
    
    async def delete_file(self, object_key: str):
        """Delete file from storage"""
        try:
            await self.storage.delete_file(object_key)
            logger.info(f"Deleted {object_key}")
        except Exception as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            raise
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage"""
        try:
            return await self.storage.list_files(prefix)
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
    
    async def file_exists(self, object_key: str) -> bool:
        """Check if file exists in storage"""
        try:
            return await self.storage.file_exists(object_key)
        except Exception as e:
            logger.error(f"Failed to check if file exists: {e}")
            return False
    
    async def get_file_metadata(self, object_key: str) -> Optional[Dict]:
        """Get file metadata from storage"""
        try:
            return await self.storage.get_file_metadata(object_key)
        except Exception as e:
            logger.error(f"Failed to get metadata for {object_key}: {e}")
            return None