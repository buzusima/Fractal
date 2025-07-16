import logging
import logging.handlers
import os
import sys
import traceback
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import gzip
import shutil
from enum import Enum
from dataclasses import dataclass, asdict
import threading
import queue
import time

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"

@dataclass
class LogEntry:
    """Enhanced log entry structure"""
    timestamp: datetime
    level: str
    logger_name: str
    module: str
    function: str
    line_number: int
    message: str
    extra_data: Dict = None
    thread_id: int = None
    thread_name: str = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line": self.line_number,
            "message": self.message,
            "extra": self.extra_data or {},
            "thread_id": self.thread_id,
            "thread_name": self.thread_name
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

class LogFormatter(logging.Formatter):
    """Enhanced log formatter with colors and structure"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'TRADE': '\033[92m',     # Bright Green
        'SIGNAL': '\033[94m',    # Bright Blue
        'RISK': '\033[91m',      # Bright Red
        'PERFORMANCE': '\033[96m', # Bright Cyan
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_thread: bool = True, json_format: bool = False):
        self.use_colors = use_colors
        self.include_thread = include_thread
        self.json_format = json_format
        
        if json_format:
            super().__init__()
        else:
            if include_thread:
                fmt = "[%(asctime)s] [%(levelname)-8s] [%(name)-15s] [%(threadName)-10s] %(funcName)s:%(lineno)d - %(message)s"
            else:
                fmt = "[%(asctime)s] [%(levelname)-8s] [%(name)-15s] %(funcName)s:%(lineno)d - %(message)s"
            
            super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        if self.json_format:
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record) -> str:
        """Format log record as JSON"""
        try:
            # Extract extra data
            extra_data = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 'processName', 
                              'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    try:
                        json.dumps(value)  # Test if serializable
                        extra_data[key] = value
                    except (TypeError, ValueError):
                        extra_data[key] = str(value)
            
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                message=record.getMessage(),
                extra_data=extra_data if extra_data else None,
                thread_id=record.thread,
                thread_name=record.threadName
            )
            
            return log_entry.to_json()
            
        except Exception as e:
            # Fallback to simple format if JSON fails
            return f'{{"timestamp": "{datetime.now().isoformat()}", "level": "ERROR", "message": "Log formatting error: {str(e)}"}}'
    
    def _format_text(self, record) -> str:
        """Format log record as colored text"""
        # Get base formatted message
        formatted = super().format(record)
        
        # Add colors if enabled and we're outputting to a terminal
        if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            
            # Color the level name
            formatted = formatted.replace(
                f'[{record.levelname}]',
                f'[{level_color}{record.levelname}{reset_color}]'
            )
        
        return formatted

class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler that compresses old log files"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False, 
                 compress_old_files=True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress_old_files = compress_old_files
    
    def doRollover(self):
        """Override to add compression"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            self.rotate(self.baseFilename, dfn)
            
            # Compress the rotated file
            if self.compress_old_files and os.path.exists(dfn):
                self._compress_file(dfn)
        
        if not self.delay:
            self.stream = self._open()
    
    def _compress_file(self, filename: str):
        """Compress a log file using gzip"""
        try:
            compressed_filename = filename + '.gz'
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file after compression
            os.remove(filename)
            
        except Exception as e:
            # If compression fails, just keep the original file
            print(f"Failed to compress log file {filename}: {e}")

class TradingLogManager:
    """Centralized log management for trading application"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 10,
                 console_level: str = "WARNING",
                 file_level: str = "INFO",
                 json_logs: bool = True,
                 compress_old_logs: bool = True,
                 separate_log_files: bool = True):
        
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.json_logs = json_logs
        self.compress_old_logs = compress_old_logs
        self.separate_log_files = separate_log_files
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Log statistics
        self.log_stats = {
            "total_logs": 0,
            "error_count": 0,
            "warning_count": 0,
            "trade_logs": 0,
            "signal_logs": 0,
            "start_time": datetime.now()
        }
        
        # Thread-safe logging queue
        self.log_queue = queue.Queue(maxsize=10000)
        self.queue_handler = None
        self.queue_listener = None
        
        # Performance monitoring
        self.performance_logs = []
        self.max_performance_logs = 1000
        
        # Setup logging
        self._setup_logging()
        self._setup_specialized_loggers()
        self._start_queue_listener()
        
        # Get main logger
        self.logger = logging.getLogger("TradingSystem")
        # self.logger.info("Enhanced logging system initialized")  # Reduced
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")
        self.logger.info(f"JSON logs: {self.json_logs}, Compression: {self.compress_old_logs}")
    
    def _setup_logging(self):
        """Setup main logging configuration"""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root level to DEBUG to capture everything
        root_logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_formatter = LogFormatter(
            use_colors=True, 
            include_thread=True, 
            json_format=False
        )
        console_handler.setFormatter(console_formatter)
        
        # Main log file handler
        main_log_file = self.log_dir / "trading_main.log"
        main_file_handler = CompressedRotatingFileHandler(
            filename=str(main_log_file),
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            compress_old_files=self.compress_old_logs
        )
        main_file_handler.setLevel(self.file_level)
        
        if self.json_logs:
            main_formatter = LogFormatter(json_format=True)
        else:
            main_formatter = LogFormatter(use_colors=False, json_format=False)
        main_file_handler.setFormatter(main_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(main_file_handler)
        
        # Setup queue handler for thread-safe logging
        self.queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.queue_handler.setLevel(logging.DEBUG)
        
        # Store original handlers for queue listener
        self.handlers = [console_handler, main_file_handler]
    
    def _setup_specialized_loggers(self):
        """Setup specialized loggers for different components"""
        specialized_configs = {
            "trades": {
                "filename": "trades.log",
                "level": logging.INFO,
                "format_type": "json"
            },
            "signals": {
                "filename": "signals.log", 
                "level": logging.INFO,
                "format_type": "json"
            },
            "risk": {
                "filename": "risk.log",
                "level": logging.WARNING,
                "format_type": "json"
            },
            "performance": {
                "filename": "performance.log",
                "level": logging.INFO,
                "format_type": "json"
            },
            "errors": {
                "filename": "errors.log",
                "level": logging.ERROR,
                "format_type": "text"
            },
            "mt5": {
                "filename": "mt5_connection.log",
                "level": logging.DEBUG,
                "format_type": "text"
            }
        }
        
        if self.separate_log_files:
            for logger_name, config in specialized_configs.items():
                self._create_specialized_logger(logger_name, config)
    
    def _create_specialized_logger(self, name: str, config: Dict):
        """Create a specialized logger"""
        logger = logging.getLogger(f"TradingSystem.{name}")
        logger.setLevel(config["level"])
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        # File handler
        log_file = self.log_dir / config["filename"]
        file_handler = CompressedRotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.max_file_size // 2,  # Smaller files for specialized logs
            backupCount=self.backup_count,
            compress_old_files=self.compress_old_logs
        )
        file_handler.setLevel(config["level"])
        
        # Formatter
        if config["format_type"] == "json":
            formatter = LogFormatter(json_format=True)
        else:
            formatter = LogFormatter(use_colors=False, json_format=False)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        # Also add to main handlers for console output
        if config["level"] >= self.console_level:
            logger.addHandler(self.handlers[0])  # Console handler
    
    def _start_queue_listener(self):
        """Start the queue listener for thread-safe logging"""
        self.queue_listener = logging.handlers.QueueListener(
            self.log_queue, 
            *self.handlers,
            respect_handler_level=True
        )
        self.queue_listener.start()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        logger = logging.getLogger(f"TradingSystem.{name}")
        
        # Add queue handler for thread safety
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
            logger.addHandler(self.queue_handler)
        
        return logger
    
    def log_trade_event(self, event_type: str, trade_data: Dict):
        """Log trade events with structured data"""
        trade_logger = self.get_logger("trades")
        
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **trade_data
        }
        
        trade_logger.info(f"Trade Event: {event_type}", extra=log_data)
        self.log_stats["trade_logs"] += 1
    
    def log_signal_event(self, signal_type: str, signal_data: Dict):
        """Log signal events with structured data"""
        signal_logger = self.get_logger("signals")
        
        log_data = {
            "signal_type": signal_type,
            "timestamp": datetime.now().isoformat(),
            **signal_data
        }
        
        signal_logger.info(f"Signal: {signal_type}", extra=log_data)
        self.log_stats["signal_logs"] += 1
    
    def log_risk_event(self, risk_level: str, risk_data: Dict):
        """Log risk events"""
        risk_logger = self.get_logger("risk")
        
        log_data = {
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat(),
            **risk_data
        }
        
        if risk_level.upper() in ["HIGH", "CRITICAL"]:
            risk_logger.error(f"Risk Alert: {risk_level}", extra=log_data)
        else:
            risk_logger.warning(f"Risk Update: {risk_level}", extra=log_data)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", metadata: Dict = None):
        """Log performance metrics"""
        perf_logger = self.get_logger("performance")
        
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        perf_logger.info(f"Performance: {metric_name}={value}{unit}", extra=metric_data)
        
        # Store for trend analysis
        self.performance_logs.append(metric_data)
        if len(self.performance_logs) > self.max_performance_logs:
            self.performance_logs.pop(0)
    
    def log_mt5_event(self, event_type: str, details: str, error_code: int = None):
        """Log MT5 connection events"""
        mt5_logger = self.get_logger("mt5")
        
        log_data = {
            "event_type": event_type,
            "details": details,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if error_code and error_code != 0:
            mt5_logger.error(f"MT5 Error: {event_type}", extra=log_data)
        else:
            mt5_logger.info(f"MT5: {event_type}", extra=log_data)
    
    def log_exception(self, exception: Exception, context: str = "", extra_data: Dict = None):
        """Log exceptions with full context"""
        error_logger = self.get_logger("errors")
        
        exc_data = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "extra_data": extra_data or {}
        }
        
        error_logger.error(f"Exception in {context}: {str(exception)}", extra=exc_data)
        self.log_stats["error_count"] += 1
    
    def get_log_statistics(self) -> Dict:
        """Get logging statistics"""
        uptime = datetime.now() - self.log_stats["start_time"]
        
        return {
            **self.log_stats,
            "uptime_seconds": uptime.total_seconds(),
            "logs_per_minute": self.log_stats["total_logs"] / max(1, uptime.total_seconds() / 60),
            "error_rate": (self.log_stats["error_count"] / max(1, self.log_stats["total_logs"])) * 100,
            "queue_size": self.log_queue.qsize() if self.log_queue else 0
        }
    
    def export_logs(self, start_time: datetime = None, end_time: datetime = None, 
                   log_types: List[str] = None, output_format: str = "json") -> str:
        """Export logs to file"""
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=24)
            if end_time is None:
                end_time = datetime.now()
            
            export_filename = f"export_{start_time.strftime('%Y%m%d_%H%M%S')}_to_{end_time.strftime('%Y%m%d_%H%M%S')}.{output_format}"
            export_path = self.log_dir / "exports" / export_filename
            export_path.parent.mkdir(exist_ok=True)
            
            # Collect logs from specified files
            log_files = []
            if log_types:
                for log_type in log_types:
                    log_file = self.log_dir / f"{log_type}.log"
                    if log_file.exists():
                        log_files.append(log_file)
            else:
                log_files = list(self.log_dir.glob("*.log"))
            
            exported_logs = []
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if output_format == "json":
                                try:
                                    log_entry = json.loads(line.strip())
                                    log_time = datetime.fromisoformat(log_entry.get("timestamp", ""))
                                    if start_time <= log_time <= end_time:
                                        exported_logs.append(log_entry)
                                except (json.JSONDecodeError, ValueError):
                                    continue
                            else:
                                # For text format, just include all lines (simple export)
                                exported_logs.append(line.strip())
                
                except Exception as e:
                    self.logger.error(f"Error reading log file {log_file}: {e}")
            
            # Write export file
            with open(export_path, 'w', encoding='utf-8') as f:
                if output_format == "json":
                    json.dump(exported_logs, f, indent=2, default=str)
                else:
                    f.write('\n'.join(exported_logs))
            
            self.logger.info(f"Logs exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Log export failed: {e}")
            return ""
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = 0
            for log_file in self.log_dir.rglob("*.log*"):
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not delete old log file {log_file}: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} old log files")
            
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")
    
    def get_recent_errors(self, hours: int = 24) -> List[Dict]:
        """Get recent error logs"""
        try:
            errors = []
            error_file = self.log_dir / "errors.log"
            
            if not error_file.exists():
                return errors
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with open(error_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        if self.json_logs:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry.get("timestamp", ""))
                            if log_time >= cutoff_time:
                                errors.append(log_entry)
                        else:
                            # For text logs, just include recent lines
                            errors.append({"message": line.strip(), "timestamp": datetime.now().isoformat()})
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            return sorted(errors, key=lambda x: x.get("timestamp", ""), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent errors: {e}")
            return []
    
    def get_performance_trends(self, metric_name: str = None, hours: int = 24) -> List[Dict]:
        """Get performance metric trends"""
        if metric_name:
            return [
                metric for metric in self.performance_logs 
                if metric["metric_name"] == metric_name and 
                datetime.fromisoformat(metric["timestamp"]) >= datetime.now() - timedelta(hours=hours)
            ]
        else:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                metric for metric in self.performance_logs 
                if datetime.fromisoformat(metric["timestamp"]) >= cutoff_time
            ]
    
    def create_log_report(self, hours: int = 24) -> Dict:
        """Create comprehensive log report"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            report = {
                "report_period": f"Last {hours} hours",
                "generated_at": datetime.now().isoformat(),
                "statistics": self.get_log_statistics(),
                "recent_errors": self.get_recent_errors(hours),
                "performance_summary": {},
                "log_file_sizes": {},
                "disk_usage": {}
            }
            
            # Performance summary
            perf_trends = self.get_performance_trends(hours=hours)
            if perf_trends:
                metrics_summary = {}
                for metric in perf_trends:
                    name = metric["metric_name"]
                    if name not in metrics_summary:
                        metrics_summary[name] = {"values": [], "unit": metric.get("unit", "")}
                    metrics_summary[name]["values"].append(metric["value"])
                
                for name, data in metrics_summary.items():
                    values = data["values"]
                    report["performance_summary"][name] = {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "unit": data["unit"]
                    }
            
            # Log file sizes
            for log_file in self.log_dir.glob("*.log"):
                try:
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    report["log_file_sizes"][log_file.name] = f"{size_mb:.2f} MB"
                except Exception:
                    report["log_file_sizes"][log_file.name] = "Unknown"
            
            # Disk usage
            try:
                total_size = sum(
                    f.stat().st_size for f in self.log_dir.rglob("*") if f.is_file()
                )
                report["disk_usage"] = {
                    "total_log_size_mb": total_size / (1024 * 1024),
                    "log_directory": str(self.log_dir.absolute())
                }
            except Exception:
                report["disk_usage"] = {"error": "Could not calculate disk usage"}
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating log report: {e}")
            return {"error": f"Report generation failed: {e}"}
    
    def shutdown(self):
        """Gracefully shutdown the logging system"""
        try:
            self.logger.info("Shutting down logging system...")
            
            # Stop queue listener
            if self.queue_listener:
                self.queue_listener.stop()
            
            # Close all handlers
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            
            print("Logging system shutdown complete")
            
        except Exception as e:
            print(f"Error during logging shutdown: {e}")

# Global log manager instance
_log_manager = None

def initialize_logging(log_dir: str = "logs", **kwargs) -> TradingLogManager:
    """Initialize the global logging system"""
    global _log_manager
    
    if _log_manager is None:
        _log_manager = TradingLogManager(log_dir=log_dir, **kwargs)
    
    return _log_manager

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    global _log_manager
    
    if _log_manager is None:
        _log_manager = initialize_logging()
    
    return _log_manager.get_logger(name)

def log_trade(event_type: str, trade_data: Dict):
    """Convenience function for logging trades"""
    global _log_manager
    if _log_manager:
        _log_manager.log_trade_event(event_type, trade_data)

def log_signal(signal_type: str, signal_data: Dict):
    """Convenience function for logging signals"""
    global _log_manager
    if _log_manager:
        _log_manager.log_signal_event(signal_type, signal_data)

def log_risk(risk_level: str, risk_data: Dict):
    """Convenience function for logging risk events"""
    global _log_manager
    if _log_manager:
        _log_manager.log_risk_event(risk_level, risk_data)

def log_performance(metric_name: str, value: float, unit: str = "", metadata: Dict = None):
    """Convenience function for logging performance metrics"""
    global _log_manager
    if _log_manager:
        _log_manager.log_performance_metric(metric_name, value, unit, metadata)

def log_exception(exception: Exception, context: str = "", extra_data: Dict = None):
    """Convenience function for logging exceptions"""
    global _log_manager
    if _log_manager:
        _log_manager.log_exception(exception, context, extra_data)

def shutdown_logging():
    """Shutdown the logging system"""
    global _log_manager
    if _log_manager:
        _log_manager.shutdown()
        _log_manager = None

# Example usage and testing
if __name__ == "__main__":
    # Initialize logging
    log_manager = initialize_logging(
        log_dir="test_logs",
        console_level="INFO",
        file_level="DEBUG",
        json_logs=True,
        separate_log_files=True
    )
    
    # Get loggers
    main_logger = get_logger("main")
    test_logger = get_logger("test")
    
    # Test different log types
    main_logger.info("System started")
    main_logger.debug("Debug message")
    main_logger.warning("Warning message")
    main_logger.error("Error message")
    
    # Test structured logging
    log_trade("POSITION_OPENED", {
        "symbol": "XAUUSD",
        "type": "BUY",
        "volume": 0.01,
        "price": 2000.50
    })
    
    log_signal("BUY_SIGNAL", {
        "rsi": 25.5,
        "fractal": True,
        "strength": 85.2
    })
    
    log_performance("execution_time", 150.5, "ms", {"function": "analyze_signals"})
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_exception(e, "testing_exceptions", {"test_data": "value"})
    
    # Generate report
    import time
    time.sleep(1)  # Allow logs to be processed
    
    report = log_manager.create_log_report(hours=1)
    print("\nLog Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Cleanup
    shutdown_logging()