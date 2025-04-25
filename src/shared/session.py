"""
Session management for data persistence.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages session data for the application.
    Provides thread-safe access to shared data storage.
    """
    
    def __init__(self):
        """Initialize the session manager with empty storage."""
        self._file_storage: Dict[str, str] = {}
        self._data_storage: Dict[str, Any] = {}
        self._preprocessed_data_storage: Dict[str, Any] = {}
        self._transformed_data_storage: Dict[str, Any] = {}
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def store_file(self, file_id: str, file_path: str) -> None:
        """
        Store a file path in the session.
        
        Args:
            file_id: Unique identifier for the file
            file_path: Path to the file on disk
        """
        async with self._lock:
            self._file_storage[file_id] = file_path
            self._session_metadata[file_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "type": "file"
            }
            logger.info(f"Stored file with ID: {file_id}")
    
    async def store_data(self, file_id: str, data: Any) -> None:
        """
        Store data in the session.
        
        Args:
            file_id: Unique identifier for the data
            data: The data to store
        """
        async with self._lock:
            self._data_storage[file_id] = data
            self._session_metadata[file_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "type": "data"
            }
            logger.info(f"Stored data with ID: {file_id}")
    
    async def store_preprocessed_data(self, file_id: str, data: Any) -> None:
        """
        Store preprocessed data in the session.
        
        Args:
            file_id: Unique identifier for the data
            data: The preprocessed data to store
        """
        async with self._lock:
            self._preprocessed_data_storage[file_id] = data
            self._session_metadata[file_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "type": "preprocessed_data"
            }
            logger.info(f"Stored preprocessed data with ID: {file_id}")
    
    async def store_transformed_data(self, file_id: str, data: Any) -> None:
        """
        Store transformed data in the session.
        
        Args:
            file_id: Unique identifier for the data
            data: The transformed data to store
        """
        async with self._lock:
            self._transformed_data_storage[file_id] = data
            self._session_metadata[file_id] = {
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "type": "transformed_data"
            }
            logger.info(f"Stored transformed data with ID: {file_id}")
    
    async def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get a file path from the session.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            Path to the file or None if not found
        """
        async with self._lock:
            if file_id in self._file_storage:
                self._session_metadata[file_id]["last_accessed"] = datetime.now()
                return self._file_storage[file_id]
            return None
    
    async def get_data(self, file_id: str) -> Optional[Any]:
        """
        Get data from the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            The stored data or None if not found
        """
        async with self._lock:
            if file_id in self._data_storage:
                self._session_metadata[file_id]["last_accessed"] = datetime.now()
                return self._data_storage[file_id]
            return None
    
    async def get_preprocessed_data(self, file_id: str) -> Optional[Any]:
        """
        Get preprocessed data from the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            The stored preprocessed data or None if not found
        """
        async with self._lock:
            if file_id in self._preprocessed_data_storage:
                self._session_metadata[file_id]["last_accessed"] = datetime.now()
                return self._preprocessed_data_storage[file_id]
            return None
    
    async def get_transformed_data(self, file_id: str) -> Optional[Any]:
        """
        Get transformed data from the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            The stored transformed data or None if not found
        """
        async with self._lock:
            if file_id in self._transformed_data_storage:
                self._session_metadata[file_id]["last_accessed"] = datetime.now()
                return self._transformed_data_storage[file_id]
            return None
    
    async def has_data(self, file_id: str) -> bool:
        """
        Check if data exists in the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            True if data exists, False otherwise
        """
        async with self._lock:
            return file_id in self._data_storage
    
    async def has_preprocessed_data(self, file_id: str) -> bool:
        """
        Check if preprocessed data exists in the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            True if preprocessed data exists, False otherwise
        """
        async with self._lock:
            return file_id in self._preprocessed_data_storage
    
    async def has_transformed_data(self, file_id: str) -> bool:
        """
        Check if transformed data exists in the session.
        
        Args:
            file_id: Unique identifier for the data
            
        Returns:
            True if transformed data exists, False otherwise
        """
        async with self._lock:
            return file_id in self._transformed_data_storage
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """
        Remove old sessions that haven't been accessed for a while.
        
        Args:
            max_age_hours: Maximum age in hours before a session is removed
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        async with self._lock:
            for file_id, metadata in self._session_metadata.items():
                if metadata["last_accessed"] < cutoff_time:
                    to_remove.append(file_id)
            
            for file_id in to_remove:
                if file_id in self._file_storage:
                    del self._file_storage[file_id]
                if file_id in self._data_storage:
                    del self._data_storage[file_id]
                if file_id in self._preprocessed_data_storage:
                    del self._preprocessed_data_storage[file_id]
                if file_id in self._transformed_data_storage:
                    del self._transformed_data_storage[file_id]
                del self._session_metadata[file_id]
                
            logger.info(f"Cleaned up {len(to_remove)} old sessions")
