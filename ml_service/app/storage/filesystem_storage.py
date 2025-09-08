# ml_service/app/storage/filesystem_storage.py
import os
import shutil
import logging
from typing import Optional, List, Dict, Any
from .storage_interface import StorageInterface

logger = logging.getLogger(__name__)

class FilesystemStorage(StorageInterface):
    """Filesystem storage implementation as fallback"""
    
    def __init__(self, base_path: str = "/app/storage"):
        self.base_path = base_path
    
    async def initialize(self):
        """Initialize filesystem storage"""
        try:
            os.makedirs(self.base_path, exist_ok=True)
            logger.info(f"Filesystem storage initialized at {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize filesystem storage: {e}")
            raise
    
    async def upload_file(self, file_path: str, object_key: str, metadata: Optional[Dict] = None):
        """Upload file to filesystem"""
        try:
            dest_path = os.path.join(self.base_path, object_key)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = f"{dest_path}.metadata"
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump(metadata, f)
            
            logger.debug(f"Uploaded {file_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    async def download_file(self, object_key: str, file_path: str):
        """Download file from filesystem"""
        try:
            src_path = os.path.join(self.base_path, object_key)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"File not found: {object_key}")
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy2(src_path, file_path)
            logger.debug(f"Downloaded {object_key} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {object_key}: {e}")
            raise
    
    async def delete_file(self, object_key: str):
        """Delete file from filesystem"""
        try:
            file_path = os.path.join(self.base_path, object_key)
            if os.path.exists(file_path):
                os.remove(file_path)
                
                # Remove metadata if exists
                metadata_path = f"{file_path}.metadata"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    
                logger.debug(f"Deleted {object_key}")
        except Exception as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            raise
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in filesystem"""
        try:
            files = []
            search_path = os.path.join(self.base_path, prefix) if prefix else self.base_path
            
            if not os.path.exists(search_path):
                return files
            
            for root, _, filenames in os.walk(search_path):
                for filename in filenames:
                    if not filename.endswith('.metadata'):
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, self.base_path)
                        files.append(rel_path)
            
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
    
    async def file_exists(self, object_key: str) -> bool:
        """Check if file exists"""
        try:
            file_path = os.path.join(self.base_path, object_key)
            return os.path.exists(file_path)
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            return False
    
    async def get_file_metadata(self, object_key: str) -> Optional[Dict]:
        """Get file metadata"""
        try:
            file_path = os.path.join(self.base_path, object_key)
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            metadata = {
                "size": stat.st_size,
                "last_modified": stat.st_mtime,
                "path": file_path
            }
            
            # Load custom metadata if exists
            metadata_path = f"{file_path}.metadata"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    import json
                    custom_metadata = json.load(f)
                    metadata.update(custom_metadata)
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None


