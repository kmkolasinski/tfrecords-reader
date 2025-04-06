import os
from typing import IO, Any

from tfr_reader.filesystem import base


class LocalFile(base.BaseFile):
    """Local file system implementation."""

    def __init__(self, path: str, mode: str = "rb"):
        self.path = path
        self.mode = mode
        self.file: IO[Any] | None = None

    def open(self):
        """Open the file."""
        self.file = open(self.path, self.mode)

    def read(self, size: int = -1) -> bytes:
        """Read data from the file."""
        if self.file is None:
            raise ValueError("File is not open!")
        return self.file.read(size)

    def get_bytes(self, start: int, end: int) -> bytes:
        """Read data from the file between start and end offsets."""
        if self.file is None:
            raise ValueError("File is not open!")
        self.file.seek(start)
        return self.file.read(end - start)

    def close(self) -> None:
        """Close the file."""
        if self.file:
            self.file.close()
            self.file = None


class LocalFileSystem(base.BaseFileSystem[LocalFile]):
    """Local file system implementation."""

    def open(self, path: str, mode: str = "rb") -> LocalFile:
        """Open a file in the specified mode."""
        return LocalFile(path, mode)

    def listdir(self, path: str) -> list[str]:
        """List files and directories in the specified path."""

        filenames = os.listdir(path)
        return [os.path.join(path, filename) for filename in filenames]

    def exists(self, path: str) -> bool:
        """Check if a path exists in the file system."""
        return os.path.exists(path)
