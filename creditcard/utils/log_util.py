import logging


def get_logger() -> logging.Logger:
    # Get root logger
    logger = logging.getLogger()
    # Set new log level
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)

    return logger
