import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(
    log_name="app_logger",
    log_file="logs/app.log",
    log_level=logging.INFO,
    max_bytes=5 * 1024 * 1024,  # 5 MB
    backup_count=5
):
    """
    Sets up rotating file logging.
    """

    # Create logs folder if not exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # Avoid duplicate logs
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
