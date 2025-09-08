# ml_service/app/storage/minio_storage.py - FIXED VERSION
from minio import Minio
from minio.error import S3Error
from .storage_interface import StorageInterface
from typing import Optional, List, Dict, Any
import asyncio
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

class MinIOStorage(StorageInterface):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.client = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize MinIO client and create bucket if needed"""
        try:
            logger.info(f"Initializing MinIO connection to {self.endpoint}")
            
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=False  # Set to True for HTTPS
            )
            
            # Test connection
            loop = asyncio.get_event_loop()
            bucket_exists = await loop.run_in_executor(
                None, self.client.bucket_exists, self.bucket_name
            )
            
            if not bucket_exists:
                await loop.run_in_executor(
                    None, self.client.make_bucket, self.bucket_name
                )
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists")
            
            self.is_initialized = True
            logger.info("MinIO storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            self.client = None
            self.is_initialized = False
            raise
    
    def _check_initialized(self):
        """Check if MinIO is properly initialized"""
        if not self.is_initialized or self.client is None:
            raise Exception("MinIO client not initialized")
    
    async def upload_file(self, file_path: str, object_key: str, metadata: Optional[Dict] = None):
        """Upload file to MinIO"""
        try:
            self._check_initialized()
            
            # Ensure file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_sync,
                file_path,
                object_key,
                metadata
            )
            logger.info(f"Uploaded {file_path} to {object_key}")
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to {object_key}: {e}")
            raise
    
    def _upload_sync(self, file_path: str, object_key: str, metadata: Optional[Dict] = None):
        """Synchronous upload for executor"""
        self.client.fput_object(
            self.bucket_name,
            object_key,
            file_path,
            metadata=metadata
        )
    
    async def download_file(self, object_key: str, file_path: str):
        """Download file from MinIO"""
        try:
            self._check_initialized()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.client.fget_object,
                self.bucket_name,
                object_key,
                file_path
            )
            logger.info(f"Downloaded {object_key} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {object_key}: {e}")
            raise
    
    async def delete_file(self, object_key: str):
        """Delete file from MinIO"""
        try:
            self._check_initialized()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.client.remove_object,
                self.bucket_name,
                object_key
            )
            logger.info(f"Deleted {object_key}")
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                logger.warning(f"File not found for deletion: {object_key}")
            else:
                logger.error(f"Failed to delete {object_key}: {e}")
                raise
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in MinIO"""
        try:
            self._check_initialized()
            
            loop = asyncio.get_event_loop()
            objects = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
            )
            return [obj.object_name for obj in objects]
            
        except S3Error as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    async def file_exists(self, object_key: str) -> bool:
        """Check if file exists in MinIO"""
        try:
            self._check_initialized()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.client.stat_object,
                self.bucket_name,
                object_key
            )
            return True
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            else:
                logger.error(f"Error checking file existence: {e}")
                return False
    
    async def get_file_metadata(self, object_key: str) -> Optional[Dict]:
        """Get file metadata from MinIO"""
        try:
            self._check_initialized()
            
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                None,
                self.client.stat_object,
                self.bucket_name,
                object_key
            )
            return {
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": stat.metadata
            }
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return None
            logger.error(f"Failed to get metadata for {object_key}: {e}")
            return None


