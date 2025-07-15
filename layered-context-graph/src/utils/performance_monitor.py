#!/usr/bin/env python3
"""
Performance Monitoring Utilities
================================
Provides tools for timing code execution and tracking memory usage.
"""

import time
import torch
import psutil
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class Timer:
    """A simple timer context manager."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {elapsed:.3f} seconds")

def timed(func):
    """Decorator to time a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def profile_memory():
    """Profiles current CPU and GPU memory usage."""
    stats = {}
    # CPU Memory
    process = psutil.Process()
    stats['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
    
    # GPU Memory
    if torch.cuda.is_available():
        stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return stats

def gpu_memory_tracked(func):
    """Decorator to track GPU memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        torch.cuda.reset_peak_memory_stats()
        before_mem = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        after_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        
        logger.info(f"Function '{func.__name__}' GPU Memory Usage:")
        logger.info(f"  Before: {before_mem / 1e6:.2f} MB")
        logger.info(f"  After:  {after_mem / 1e6:.2f} MB")
        logger.info(f"  Peak:   {peak_mem / 1e6:.2f} MB")
        logger.info(f"  Delta:  {(after_mem - before_mem) / 1e6:.2f} MB")
        
        return result
    return wrapper
