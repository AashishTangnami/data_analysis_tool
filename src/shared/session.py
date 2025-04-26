"""
Session management for data persistence.
"""
import asyncio
import logging
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages session data for the application.
    Provides thread-safe access to shared data storage.
    """

    def __init__(self):
        """Initialize the session manager with consolidated storage."""
        # Consolidated data store with metadata
        self.data_store: Dict[str, Dict[str, Any]] = {}
        self.current_cache_size: int = 0
        self.cache_size_limit: int = 1024 * 1024 * 1024  # 1GB default cache limit
        self._lock = asyncio.Lock()

        # Operation history
        self.operation_history: Dict[str, List[Dict[str, Any]]] = {}

    async def store_file(self, file_id: str, file_path: str) -> None:
        """
        Store a file path in the session.

        Args:
            file_id: Unique identifier for the file
            file_path: Path to the file on disk
        """
        async with self._lock:
            self.data_store[file_id] = {
                "data": file_path,
                "type": "file",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "size": 0  # File paths don't count toward cache size
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
            # Calculate data size (approximate)
            data_size = self._estimate_data_size(data)

            # Check if adding this data would exceed cache limit
            if self.current_cache_size + data_size > self.cache_size_limit:
                # Implement LRU eviction strategy
                self._evict_least_recently_used()

            # Store data in the consolidated data store
            self.data_store[file_id] = {
                "data": data,
                "type": "data",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "size": data_size
            }
            self.current_cache_size += data_size

            logger.info(f"Stored data with ID: {file_id}, size: {data_size} bytes")

    async def replace_data(self, file_id: str, data: Any) -> None:
        """
        Replace existing data for a file_id to avoid memory duplication.

        Args:
            file_id: File identifier
            data: New data to store
        """
        async with self._lock:
            # Calculate data size
            data_size = self._estimate_data_size(data)

            # Remove old data and update cache size
            if file_id in self.data_store:
                old_size = self.data_store[file_id].get('size', 0)
                self.current_cache_size -= old_size

                # Preserve the data type and creation time if possible
                data_type = self.data_store[file_id].get('type', 'data')
                created_at = self.data_store[file_id].get('created_at', datetime.now())
            else:
                data_type = 'data'
                created_at = datetime.now()

            # Store new data in consolidated data store
            self.data_store[file_id] = {
                "data": data,
                "type": data_type,
                "created_at": created_at,
                "last_accessed": datetime.now(),
                "size": data_size
            }
            self.current_cache_size += data_size

            # Log the replacement
            logger.info(f"Replaced data for file_id: {file_id}, new size: {data_size} bytes")


    async def store_preprocessed_data(self, file_id: str, data: Any) -> None:
        """
        Store preprocessed data in the session.

        Args:
            file_id: Unique identifier for the data
            data: The preprocessed data to store
        """
        async with self._lock:
            # Calculate data size
            data_size = self._estimate_data_size(data)

            # Check if adding this data would exceed cache limit
            if self.current_cache_size + data_size > self.cache_size_limit:
                # Implement LRU eviction strategy
                self._evict_least_recently_used()

            # Store in consolidated data store
            self.data_store[file_id + "_preprocessed"] = {
                "data": data,
                "type": "preprocessed_data",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "size": data_size
            }
            self.current_cache_size += data_size

            logger.info(f"Stored preprocessed data with ID: {file_id}, size: {data_size} bytes")

    async def store_transformed_data(self, file_id: str, data: Any) -> None:
        """
        Store transformed data in the session.

        Args:
            file_id: Unique identifier for the data
            data: The transformed data to store
        """
        async with self._lock:
            # Calculate data size
            data_size = self._estimate_data_size(data)

            # Check if adding this data would exceed cache limit
            if self.current_cache_size + data_size > self.cache_size_limit:
                # Implement LRU eviction strategy
                self._evict_least_recently_used()

            # Store in consolidated data store
            self.data_store[file_id + "_transformed"] = {
                "data": data,
                "type": "transformed_data",
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "size": data_size
            }
            self.current_cache_size += data_size

            logger.info(f"Stored transformed data with ID: {file_id}, size: {data_size} bytes")

    async def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get a file path from the session.

        Args:
            file_id: Unique identifier for the file

        Returns:
            Path to the file or None if not found
        """
        async with self._lock:
            if file_id in self.data_store and self.data_store[file_id]["type"] == "file":
                # Update last accessed time
                self.data_store[file_id]["last_accessed"] = datetime.now()
                return self.data_store[file_id]["data"]
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
            if file_id in self.data_store:
                # Update last accessed time
                self.data_store[file_id]["last_accessed"] = datetime.now()
                return self.data_store[file_id]["data"]
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
            preprocessed_id = file_id + "_preprocessed"
            if preprocessed_id in self.data_store:
                # Update last accessed time
                self.data_store[preprocessed_id]["last_accessed"] = datetime.now()
                return self.data_store[preprocessed_id]["data"]
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
            transformed_id = file_id + "_transformed"
            if transformed_id in self.data_store:
                # Update last accessed time
                self.data_store[transformed_id]["last_accessed"] = datetime.now()
                return self.data_store[transformed_id]["data"]
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
            return file_id in self.data_store and self.data_store[file_id]["type"] == "data"

    async def has_preprocessed_data(self, file_id: str) -> bool:
        """
        Check if preprocessed data exists in the session.

        Args:
            file_id: Unique identifier for the data

        Returns:
            True if preprocessed data exists, False otherwise
        """
        async with self._lock:
            preprocessed_id = file_id + "_preprocessed"
            return preprocessed_id in self.data_store

    async def has_transformed_data(self, file_id: str) -> bool:
        """
        Check if transformed data exists in the session.

        Args:
            file_id: Unique identifier for the data

        Returns:
            True if transformed data exists, False otherwise
        """
        async with self._lock:
            transformed_id = file_id + "_transformed"
            return transformed_id in self.data_store

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """
        Remove old sessions that haven't been accessed for a while.

        Args:
            max_age_hours: Maximum age in hours before a session is removed
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        async with self._lock:
            # Find old sessions
            for file_id, item in self.data_store.items():
                if "last_accessed" in item and item["last_accessed"] < cutoff_time:
                    to_remove.append(file_id)

            # Remove old sessions and update cache size
            for file_id in to_remove:
                if file_id in self.data_store:
                    # Reduce cache size
                    self.current_cache_size -= self.data_store[file_id].get("size", 0)
                    # Remove from data store
                    del self.data_store[file_id]

            logger.info(f"Cleaned up {len(to_remove)} old sessions")

    async def get_operation_history(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Get the operation history for a file_id.

        Args:
            file_id: File identifier

        Returns:
            List of operations applied to this file
        """
        # No need to check hasattr since we initialize it in __init__
        return self.operation_history.get(file_id, [])

    async def add_to_operation_history(self, file_id: str, operation: Dict[str, Any]) -> None:
        """
        Add an operation to the history for a file_id.

        Args:
            file_id: File identifier
            operation: Operation to add to history
        """
        # No need to check hasattr since we initialize it in __init__
        if file_id not in self.operation_history:
            self.operation_history[file_id] = []

        self.operation_history[file_id].append(operation)

    async def set_operation_history(self, file_id: str, operations: List[Dict[str, Any]]) -> None:
        """
        Set the operation history for a file_id.

        Args:
            file_id: File identifier
            operations: List of operations
        """
        # No need to check hasattr since we initialize it in __init__
        self.operation_history[file_id] = operations

    async def remove_last_operation(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Remove and return the last operation from history.

        Args:
            file_id: File identifier

        Returns:
            Last operation or None if history is empty
        """
        # No need to check hasattr since we initialize it in __init__
        if file_id not in self.operation_history:
            return None

        if not self.operation_history[file_id]:
            return None

        return self.operation_history[file_id].pop()

    def _estimate_data_size(self, data: Any) -> int:
        """
        Estimate the memory size of data in bytes.

        Args:
            data: The data to estimate size for

        Returns:
            Estimated size in bytes
        """
        try:
            # For pandas DataFrames
            if hasattr(data, 'memory_usage'):
                try:
                    # For pandas DataFrame
                    return int(data.memory_usage(deep=True).sum())
                except:
                    pass

            # For numpy arrays
            if hasattr(data, 'nbytes'):
                return data.nbytes

            # For dictionaries, lists, etc.
            return sys.getsizeof(data)

        except Exception as e:
            logger.warning(f"Error estimating data size: {str(e)}")
            # Return a default size if estimation fails
            return 1024 * 1024  # 1MB default

    def _evict_least_recently_used(self):
        """
        Evict least recently used items from cache until we're under the limit.
        """
        if not self.data_store:
            return

        # Sort by last accessed time
        sorted_items = sorted(
            self.data_store.items(),
            key=lambda x: x[1].get('last_accessed', datetime.min)
        )

        # Remove items until we're under the limit
        for file_id, item in sorted_items:
            if self.current_cache_size <= self.cache_size_limit * 0.8:  # 80% of limit
                break

            # Remove item and update cache size
            item_size = item.get('size', 0)
            del self.data_store[file_id]
            self.current_cache_size -= item_size
            logger.info(f"Evicted {file_id} from cache (size: {item_size} bytes)")