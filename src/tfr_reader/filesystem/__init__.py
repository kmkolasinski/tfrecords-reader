import os.path

from tfr_reader.filesystem.base import BaseFile, BaseFileSystem  # noqa: F401
from tfr_reader.filesystem.gcs import GCSFile, GCSFileSystem  # noqa: F401
from tfr_reader.filesystem.local import LocalFile, LocalFileSystem  # noqa: F401


def get_file_system(path: str) -> BaseFileSystem:
    """Get the appropriate file system for the given path."""
    if str(path).startswith("gs://"):
        return GCSFileSystem()
    if os.path.exists(path):
        return LocalFileSystem()
    raise FileNotFoundError(f"Path {path} does not exist.")
