import abc
from typing import Generic, TypeVar

GenericFile = TypeVar("GenericFile", bound="BaseFile")


class BaseFile(abc.ABC):
    """Abstract base class for file operations."""

    @abc.abstractmethod
    def read(self, size: int = -1) -> bytes:
        """Read data from the file."""

    @abc.abstractmethod
    def seek(self, offset: int):
        """Move the file pointer to a new location."""

    @abc.abstractmethod
    def close(self):
        """Close the file."""


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
