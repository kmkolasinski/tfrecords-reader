import gcsfs
from gcsfs import core

from tfr_reader.filesystem import base


class GCSFile(base.BaseFile):
    """Google Cloud Storage file implementation."""

    def __init__(self, path: str, mode: str = "rb"):
        self.path = path
        self.mode = mode
        self.file: core.GCSFile | None = None

    def open(self):
        """Open the file."""
        self.file = gcsfs.GCSFileSystem().open(self.path, self.mode)

    def read(self, size: int = -1) -> bytes:
        """Read data from the file."""
        if self.file is None:
            raise ValueError("File is not open!")
        return self.file.read(size)

    def seek(self, offset: int):
        """Move the file pointer to a new location."""
        if self.file is None:
            raise ValueError("File is not open!")
        self.file.seek(offset)

    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()
            self.file = None


class GCSFileSystem(base.BaseFileSystem[GCSFile]):
    """Google Cloud Storage file system implementation."""

    def __init__(self):
        self.fs = gcsfs.GCSFileSystem()

    def open(self, path: str, mode: str = "rb") -> GCSFile:
        """Open a file in the specified mode."""
        gcs_file = GCSFile(path, mode)
        gcs_file.open()
        return gcs_file

    def listdir(self, path: str) -> list[str]:
        """List files and directories in the specified path."""
        paths = self.fs.listdir(path, detail=False)
        # add missing gs:// prefix
        return [f"gs://{path}" for path in paths]

    def exists(self, path: str) -> bool:
        return self.fs.exists(path)
