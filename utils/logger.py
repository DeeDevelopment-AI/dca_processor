import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger with file and console handlers"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        # File handler
        file_handler = RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)

    return logger