import os.path

from tfr_reader.filesystem.base import BaseFile, BaseFileSystem

try:
    from tfr_reader.filesystem.gcs import GCSFileSystem
except ModuleNotFoundError:

    def GCSFileSystem():  # noqa: N802
        raise ImportError(
            "Google storage features are disabled. "
            "Please install it with `pip install tfr-reader[google]`."
        )


from tfr_reader.filesystem.local import LocalFile, LocalFileSystem


def get_file_system(path: str) -> BaseFileSystem:
    """Get the appropriate file system for the given path."""
    if str(path).startswith("gs://"):
        return GCSFileSystem()
    if os.path.exists(path):
        return LocalFileSystem()
    raise FileNotFoundError(f"Path {path} does not exist.")


__all__ = [
    "BaseFile",
    "BaseFileSystem",
    "GCSFileSystem",
    "LocalFile",
    "LocalFileSystem",
]
