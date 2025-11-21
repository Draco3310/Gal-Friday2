import logging
import sys
from pathlib import Path

def setup_logger(name: str = "GalFriday", level: str = "INFO") -> logging.Logger:
    """
    Sets up a simple logger with console output.
    """
    logger = logging.getLogger(name)

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logger.setLevel(numeric_level)

    # Prevent adding multiple handlers if setup is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
