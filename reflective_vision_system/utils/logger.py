import logging
import os
from datetime import datetime
from rich.logging import RichHandler


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)

    return logger
