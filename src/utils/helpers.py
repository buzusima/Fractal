"""
Helper utilities for XAUUSD EA
Common utility functions
"""

import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class PerformanceMonitor:
    """Simple performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """Create a performance checkpoint"""
        self.checkpoints[name] = time.time()
    
    def get_elapsed(self, checkpoint_name: str = None) -> float:
        """Get elapsed time since start or checkpoint"""
        if checkpoint_name and checkpoint_name in self.checkpoints:
            return time.time() - self.checkpoints[checkpoint_name]
        return time.time() - self.start_time
    
    def report(self) -> Dict[str, float]:
        """Get performance report"""
        now = time.time()
        report = {"total_elapsed": now - self.start_time}
        
        for name, timestamp in self.checkpoints.items():
            report[f"{name}_elapsed"] = now - timestamp
        
        return report
