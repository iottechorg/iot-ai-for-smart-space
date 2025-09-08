# ml_service/app/storage/factory.py
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class StorageFactory:
    @staticmethod
    def create_storage(storage_type: Optional[str] = None):
        """Create storage instance based on configuration"""
        if storage_type is None:
            storage_type = os.environ.get('STORAGE_TYPE', 'filesystem')
        
        # Try MinIO first if configured, fall back to filesystem if it fails
        if storage_type.lower() == 'minio':
            try:
                from app.storage.minio_storage import MinIOStorage
                
                endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
                access_key = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
                secret_key = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
                bucket_name = os.environ.get('MINIO_BUCKET', 'models')
                
                storage = MinIOStorage(
                    endpoint=endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    bucket_name=bucket_name
                )
                
                logger.info(f"Using MinIO storage with endpoint {endpoint}")
                return storage
                
            except Exception as e:
                logger.error(f"MinIO storage failed, falling back to filesystem: {e}")
                from app.storage.filesystem_storage import FilesystemStorage
                return FilesystemStorage(base_path="/app/storage")
        
        elif storage_type.lower() == 'filesystem':
            from app.storage.filesystem_storage import FilesystemStorage
            base_path = os.environ.get('STORAGE_PATH', '/app/storage')
            logger.info(f"Using filesystem storage at {base_path}")
            return FilesystemStorage(base_path=base_path)
        
        else:
            logger.error(f"Unsupported storage type: {storage_type}, using filesystem")
            from app.storage.filesystem_storage import FilesystemStorage
            return FilesystemStorage(base_path="/app/storage")

    @staticmethod
    def create_fallback_storage():
        """Create a fallback filesystem storage"""
        from app.storage.filesystem_storage import FilesystemStorage
        return FilesystemStorage(base_path="/app/storage")