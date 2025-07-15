#!/usr/bin/env python3
"""
LLM Connection Pool
==================
Simple connection pooling for LLM instances to reduce initialization overhead.
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class LLMConnectionPool:
    """
    Connection pool for LLM instances to reduce initialization overhead.
    """
    
    def __init__(self, max_connections: int = 3, connection_timeout: float = 30.0):
        """
        Initialize the connection pool.
        
        Args:
            max_connections: Maximum number of connections per model type
            connection_timeout: Timeout for getting connections from pool
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pools: Dict[str, Queue] = {}
        self.connection_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.active_connections: Dict[str, List[Any]] = {}
        
    def _get_pool_key(self, model_path: str, model_type: str) -> str:
        """Generate a unique key for the pool"""
        return f"{model_type}:{model_path}"
    
    def _create_connection(self, model_path: str, model_type: str, **kwargs) -> Any:
        """Create a new LLM connection"""
        logger.info(f"Creating new {model_type} connection for {model_path}")
        
        if model_type == "qwq":
            from models.qwq_model import QwQModel
            return QwQModel(model_path, **kwargs)
        elif model_type == "attention":
            from models.attention_extractor import EnhancedAttentionExtractor
            return EnhancedAttentionExtractor(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_connection(self, model_path: str, model_type: str, **kwargs) -> Any:
        """
        Get a connection from the pool or create a new one.
        
        Args:
            model_path: Path to the model
            model_type: Type of model ("ollama", "attention", etc.)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            LLM connection instance
        """
        pool_key = self._get_pool_key(model_path, model_type)
        
        with self.lock:
            # Initialize pool if it doesn't exist
            if pool_key not in self.pools:
                self.pools[pool_key] = Queue(maxsize=self.max_connections)
                self.connection_counts[pool_key] = 0
                self.active_connections[pool_key] = []
        
        # Try to get connection from pool
        try:
            connection = self.pools[pool_key].get(timeout=0.1)
            logger.info(f"Reusing pooled connection for {pool_key}")
            return connection
        except Empty:
            pass
        
        # Create new connection if pool is empty and we haven't reached max
        with self.lock:
            if self.connection_counts[pool_key] < self.max_connections:
                try:
                    connection = self._create_connection(model_path, model_type, **kwargs)
                    self.connection_counts[pool_key] += 1
                    self.active_connections[pool_key].append(connection)
                    logger.info(f"Created new connection for {pool_key} ({self.connection_counts[pool_key]}/{self.max_connections})")
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create connection for {pool_key}: {e}")
                    raise
        
        # Wait for connection to become available
        try:
            connection = self.pools[pool_key].get(timeout=self.connection_timeout)
            logger.info(f"Got connection from pool after waiting for {pool_key}")
            return connection
        except Empty:
            raise RuntimeError(f"Connection pool timeout for {pool_key}")
    
    def return_connection(self, connection: Any, model_path: str, model_type: str):
        """
        Return a connection to the pool.
        
        Args:
            connection: The connection to return
            model_path: Path to the model
            model_type: Type of model
        """
        pool_key = self._get_pool_key(model_path, model_type)
        
        if pool_key in self.pools:
            try:
                self.pools[pool_key].put_nowait(connection)
                logger.debug(f"Returned connection to pool for {pool_key}")
            except:
                logger.warning(f"Pool full for {pool_key}, discarding connection")
        else:
            logger.warning(f"No pool found for {pool_key}, discarding connection")
    
    @contextmanager
    def get_connection_context(self, model_path: str, model_type: str, **kwargs):
        """
        Context manager for getting and returning connections.
        
        Args:
            model_path: Path to the model
            model_type: Type of model
            **kwargs: Additional arguments for model initialization
            
        Yields:
            LLM connection instance
        """
        connection = self.get_connection(model_path, model_type, **kwargs)
        try:
            yield connection
        finally:
            self.return_connection(connection, model_path, model_type)
    
    def close_all(self):
        """Close all connections in all pools"""
        with self.lock:
            for pool_key, pool in self.pools.items():
                logger.info(f"Closing connections for {pool_key}")
                while not pool.empty():
                    try:
                        connection = pool.get_nowait()
                        if hasattr(connection, 'close'):
                            connection.close()
                    except Empty:
                        break
                
                # Close active connections
                for connection in self.active_connections.get(pool_key, []):
                    if hasattr(connection, 'close'):
                        connection.close()
            
            self.pools.clear()
            self.connection_counts.clear()
            self.active_connections.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about the connection pools"""
        stats = {}
        
        with self.lock:
            for pool_key in self.pools:
                stats[pool_key] = {
                    'total_connections': self.connection_counts.get(pool_key, 0),
                    'pooled_connections': self.pools[pool_key].qsize(),
                    'active_connections': len(self.active_connections.get(pool_key, [])),
                    'max_connections': self.max_connections
                }
        
        return stats


# Global connection pool instance
_global_pool: Optional[LLMConnectionPool] = None


def get_global_pool() -> LLMConnectionPool:
    """Get the global connection pool instance"""
    global _global_pool
    if _global_pool is None:
        _global_pool = LLMConnectionPool()
    return _global_pool


def close_global_pool():
    """Close the global connection pool"""
    global _global_pool
    if _global_pool:
        _global_pool.close_all()
        _global_pool = None
