# utils/logging_utils.py

import os
import sys
import logging
import time
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: Optional[str] = None, 
                 log_level: int = logging.INFO) -> None:
    """
    Sets up logging configuration for the application.
    Creates a logs directory and configures both file and console logging.
    """
    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = 'logs'
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # File handler with rotation
    log_file = os.path.join(
        log_dir, 
        f'sales_agent_{time.strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    logging.info("Logging system initialized")

def log_conversation_event(event_type: str, details: dict) -> None:
    """Log conversation events with structured data"""
    logging.info(f"Conversation event: {event_type}", extra={
        "event_type": event_type,
        "details": details,
        "timestamp": time.time()
    })

def log_error_with_context(error: Exception, context: dict) -> None:
    """Log errors with additional context"""
    logging.error(
        f"Error occurred: {str(error)}",
        extra={
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": time.time()
        },
        exc_info=True
    )

def setup_performance_logging() -> None:
    """Set up performance monitoring logging"""
    perf_logger = logging.getLogger('performance')
    
    # Create performance log directory
    perf_log_dir = 'logs/performance'
    if not os.path.exists(perf_log_dir):
        os.makedirs(perf_log_dir)
    
    # Set up performance log handler
    perf_handler = RotatingFileHandler(
        os.path.join(perf_log_dir, 'performance.log'),
        maxBytes=5242880,  # 5MB
        backupCount=3
    )
    perf_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)

def log_performance_metric(metric_name: str, value: float, 
                         additional_info: Optional[dict] = None) -> None:
    """Log performance metrics"""
    perf_logger = logging.getLogger('performance')
    perf_logger.info(
        f"Performance metric: {metric_name}",
        extra={
            "metric_name": metric_name,
            "value": value,
            "additional_info": additional_info or {},
            "timestamp": time.time()
        }
    )