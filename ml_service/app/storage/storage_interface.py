# ml_service/app/storage/storage_interface.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import asyncio

class StorageInterface(ABC):
    """Abstract interface for storage backends"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize storage backend"""
        pass
    
    @abstractmethod
    async def upload_file(self, file_path: str, object_key: str, metadata: Optional[Dict] = None):
        """Upload file to storage"""
        pass
    
    @abstractmethod
    async def download_file(self, object_key: str, file_path: str):
        """Download file from storage"""
        pass
    
    @abstractmethod
    async def delete_file(self, object_key: str):
        """Delete file from storage"""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage"""
        pass
    
    @abstractmethod
    async def file_exists(self, object_key: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, object_key: str) -> Optional[Dict]:
        """Get file metadata"""
        pass