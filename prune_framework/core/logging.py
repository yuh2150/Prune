import logging
import sys

_LOGGER_NAME = "prune_framework"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Returns a configured logger instance for prune_framework."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
