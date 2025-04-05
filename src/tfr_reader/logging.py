import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
