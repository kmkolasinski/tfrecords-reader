import hashlib
import warnings
from threading import Lock

import diskcache
import gcsfs
from gcsfs import core
from google.cloud import storage
from requests.adapters import HTTPAdapter

from tfr_reader import logging
from tfr_reader.filesystem import base

CACHE_DIR = "/tmp/tfr-reader-cache"  # noqa: S108
CACHE = diskcache.Cache(CACHE_DIR)
LOGGER = logging.Logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="google")


def hash_path(path: str) -> str:
    """Generate a hash for the given path."""
    return hashlib.sha224(path.encode()).hexdigest()


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
        if size == -1 and hash_path(self.path) in CACHE:
            LOGGER.info("Loaded from cache %s", self.path)
            return CACHE[hash_path(self.path)]
        if self.file is None:
            raise ValueError("File is not open!")
        data = self.file.read(size)
        if size == -1:
            CACHE[hash_path(self.path)] = data
        return data

    def get_bytes(self, start: int, end: int) -> bytes:
        """Read data from the file between start and end offsets."""
        # this way is faster
        blob = _blob_from_uri(self.path)
        return blob.download_as_bytes(start=start, end=end - 1, checksum=None)

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
        return GCSFile(path, mode)

    def listdir(self, path: str) -> list[str]:
        """List files and directories in the specified path."""
        paths = self.fs.listdir(path, detail=False)
        # add missing gs:// prefix
        return [f"gs://{path}" for path in paths]

    def exists(self, path: str) -> bool:
        return self.fs.exists(path)


def _blob_from_uri(uri: str) -> storage.Blob:
    if hasattr(storage.Blob, "from_uri"):
        # https://github.com/googleapis/python-storage/blob/main/CHANGELOG.md#300-2025-01-28
        return storage.Blob.from_uri(uri, client=_StorageClient.get())
    return storage.Blob.from_string(uri, client=_StorageClient.get())


class _Client(storage.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        # to hide: WARNING Connection pool is full, discarding connection
        self._http.mount("https://", adapter)
        self._http._auth_request.session.mount("https://", adapter)  # noqa: SLF001


class _StorageClient:
    _storage: _Client | None = None
    _lock = Lock()

    @classmethod
    def get(cls) -> _Client:
        if cls._storage is None:
            with cls._lock:
                if cls._storage is None:
                    cls._storage = _Client()

        return cls._storage
