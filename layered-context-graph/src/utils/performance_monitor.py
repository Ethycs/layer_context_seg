#!/usr/bin/env python3
"""
Performance Monitoring Utilities
================================
Decorators and utilities for monitoring performance and timing.
"""

import time
import functools
import logging
from typing import Callable, Any, Dict, Optional
import torch
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Global performance monitoring singleton"""
    
    _instance = None
    _metrics = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = {}
        return cls._instance
    
    def record_metric(self, name: str, value: float, category: str = 'general'):
        """Record a performance metric"""
        if category not in self._metrics:
            self._metrics[category] = {}
        
        if name not in self._metrics[category]:
            self._metrics[category][name] = []
        
        self._metrics[category][name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for category, metrics in self._metrics.items():
            summary[category] = {}
            for name, values in metrics.items():
                if values:
                    summary[category][name] = {
                        'count': len(values),
                        'total': sum(values),
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self._metrics = {}


def timed(func: Optional[Callable] = None, *, 
          name: Optional[str] = None, 
          category: str = 'timing',
          log_result: bool = True) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        name: Custom name for the metric
        category: Category for grouping metrics
        log_result: Whether to log the timing
    
    Usage:
        @timed
        def my_function():
            pass
            
        @timed(name="custom_name", category="llm_calls")
        def my_other_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        metric_name = name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                
                # Record metric
                monitor = PerformanceMonitor()
                monitor.record_metric(metric_name, elapsed, category)
                
                if log_result:
                    logger.info(f"{metric_name} took {elapsed:.3f} seconds")
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def gpu_memory_tracked(func: Optional[Callable] = None, *, 
                      name: Optional[str] = None,
                      log_result: bool = True) -> Callable:
    """
    Decorator to track GPU memory usage.
    
    Args:
        func: Function to track
        name: Custom name for the metric
        log_result: Whether to log the memory usage
    """
    def decorator(f: Callable) -> Callable:
        metric_name = name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                return f(*args, **kwargs)
            
            # Clear cache and get initial memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            
            try:
                result = f(*args, **kwargs)
                
                # Get final memory
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / (1024 ** 2)  # MB
                
                # Record metric
                monitor = PerformanceMonitor()
                monitor.record_metric(f"{metric_name}_gpu_memory_mb", memory_used, "gpu_memory")
                
                if log_result:
                    logger.info(f"{metric_name} used {memory_used:.2f} MB GPU memory")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {metric_name}: {e}")
                raise
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def batch_processing_tracked(func: Callable) -> Callable:
    """
    Decorator specifically for tracking batch processing performance.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract batch size from arguments
        batch_size = None
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        elif len(args) > 0 and isinstance(args[0], (list, torch.Tensor)):
            batch_size = len(args[0])
        
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        result = func(*args, **kwargs)
        
        elapsed = time.perf_counter() - start_time
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - start_memory) / (1024 ** 2)
        else:
            memory_used = 0
        
        # Record metrics
        monitor = PerformanceMonitor()
        monitor.record_metric(f"{func.__name__}_time", elapsed, "batch_processing")
        monitor.record_metric(f"{func.__name__}_memory_mb", memory_used, "batch_processing")
        
        if batch_size:
            monitor.record_metric(f"{func.__name__}_time_per_item", elapsed / batch_size, "batch_processing")
            logger.info(f"{func.__name__}: {batch_size} items in {elapsed:.3f}s ({elapsed/batch_size:.3f}s per item)")
        else:
            logger.info(f"{func.__name__} took {elapsed:.3f} seconds")
        
        return result
    
    return wrapper


def profile_memory(log_cpu: bool = True, log_gpu: bool = True) -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    stats = {}
    
    if log_cpu:
        process = psutil.Process(os.getpid())
        stats['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
        stats['cpu_memory_percent'] = process.memory_percent()
    
    if log_gpu and torch.cuda.is_available():
        stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        stats['gpu_memory_percent'] = (torch.cuda.memory_allocated() / 
                                      torch.cuda.get_device_properties(0).total_memory * 100)
    
    return stats


def log_performance_summary():
    """Log a summary of all collected performance metrics"""
    monitor = PerformanceMonitor()
    summary = monitor.get_summary()
    
    logger.info("="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    for category, metrics in summary.items():
        logger.info(f"\n{category.upper()}:")
        for name, stats in metrics.items():
            logger.info(f"  {name}:")
            logger.info(f"    Total: {stats['total']:.3f}")
            logger.info(f"    Count: {stats['count']}")
            logger.info(f"    Average: {stats['mean']:.3f}")
            logger.info(f"    Min: {stats['min']:.3f}")
            logger.info(f"    Max: {stats['max']:.3f}")
    
    logger.info("="*60)


# Context manager for timing blocks of code
class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "Block", log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        
        monitor = PerformanceMonitor()
        monitor.record_metric(self.name, self.elapsed, "timer")
        
        if self.log:
            logger.info(f"{self.name} took {self.elapsed:.3f} seconds")