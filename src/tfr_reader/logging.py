import logging


class Logger:
    def __init__(self, name: str = __name__, verbose: bool = True):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(name)
        self._verbose = verbose

    def info(self, message: str, *args, **kwargs):
        if self._verbose:
            self._logger.info(message, *args, **kwargs)
