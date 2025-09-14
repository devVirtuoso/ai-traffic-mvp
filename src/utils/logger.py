"""
Helper function to configure and return a basic logger instance.
"""

import logging

def get_logger(name: str = "ai-traffic-mvp") -> logging.Logger:
    """Return a configured logger that logs to the console."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
