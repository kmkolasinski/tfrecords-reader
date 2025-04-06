import abc
from typing import Generic, TypeVar

GenericFile = TypeVar("GenericFile", bound="BaseFile")


class BaseFile(abc.ABC):
    """Abstract base class for file operations."""

    @abc.abstractmethod
    def read(self, size: int = -1) -> bytes:
        """Read data from the file."""

    @abc.abstractmethod
    def get_bytes(self, start: int, end: int) -> bytes:
        """Read data from the file between start and end offsets."""

    @abc.abstractmethod
    def open(self):
        """Open the file."""

    @abc.abstractmethod
    def close(self):
        """Close the file."""

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        self.close()
        return False


class BaseFileSystem(Generic[GenericFile], abc.ABC):
    """Abstract base class for file system operations."""

    @abc.abstractmethod
    def open(self, path: str, mode: str = "rb") -> GenericFile:
        """Open a file in the specified mode."""

    @abc.abstractmethod
    def listdir(self, path: str) -> list[str]:
        """List files and directories in the specified path."""

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists in the file system."""
