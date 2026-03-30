import fnmatch
import hashlib
import struct
from collections.abc import Iterable
from concurrent import futures
from pathlib import Path
from typing import overload

import polars as pl
from tqdm import tqdm

from tfr_reader import example, indexer, logging
from tfr_reader import filesystem as fs

LOGGER = logging.Logger(__name__)


class TFRecordFileReader:
    """A low-level reader for fetching individual examples from a single TFRecord file.

    This class handles opening a TFRecord file (local or remote), reading byte
    ranges, and decoding them into raw feature data. It is primarily used as
    an underlying reader by the dataset-level abstractions.
    """

    def __init__(self, filepath: str):
        """Initializes the dataset with the TFRecord file reader.

        Args:
            filepath: Path to the TFRecord file.
        """
        self.tfrecord_filepath = filepath
        self.storage = fs.get_file_system(filepath)
        self._file: fs.BaseFile | None = None

    def get_example(self, start: int, end: int) -> example.Feature:
        """Retrieves the raw TFRecord data at the specified byte offsets.

        Args:
            start: The start byte index of the record to retrieve.
            end: The end byte index of the record to retrieve.

        Returns:
            feature: The raw serialized record data as a Feature object.
        """
        if self._file is None:
            raise OSError("File is not open. Use context manager!")

        example_data = self._file.get_bytes(start, end)
        length = start - end
        if not example_data or len(example_data) < length:
            raise OSError(f"Failed to read data from {(start, end)}!")

        # dropping the length and length_crc bytes for simplicity
        data = example_data[8 + 4 : -4 :]
        return example.decode(data)

    def _open(self):
        """Opens the TFRecord file for reading."""
        if self._file is None:
            self._file = self.storage.open(self.tfrecord_filepath, "rb")
            self._file.open()

    def _close(self):
        """Closes the TFRecord file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False


class TFRecordDatasetReader:
    """A dataset reader for indexed TFRecord datasets.

    Provides mechanisms to load an indexed TFRecord dataset (either local or from cloud storage).
    It supports querying by SQL, retrieving chunks or individual features, and caching
    dataset indices using polars as the underlying query engine.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        index_df: pl.DataFrame | None = None,
        verbose: bool = True,
        index_cache_dir: str | Path | None = None,
    ):
        """Initializes the dataset with the TFRecord files and their index.

        Args:
            dataset_dir: Directory containing the .tfrecord files and the index.
            index_df: Optional pre-loaded polars DataFrame representing the dataset index.
            verbose: If True, prints loading progression.
            index_cache_dir: Optional path to a directory where the downloaded dataset
                index will be cached locally to speed up subsequent loads.
        """

        self.storage = fs.get_file_system(dataset_dir)
        self.dataset_dir = str(dataset_dir)
        self.verbose = verbose
        self.logger = logging.Logger(self.__class__.__name__, verbose)
        self.index_cache_dir = Path(index_cache_dir) if index_cache_dir is not None else None

        if index_df is None:
            index_path = join_path(dataset_dir, indexer.INDEX_FILENAME)
            index_df = self._load_or_cache_index(index_path)

        self.index_df = index_df.with_row_index("_row_id")
        self.ctx = pl.SQLContext(index=self.index_df, eager=True)
        self.logger.info(f"Loaded dataset index with N={self.index_df.height} records ...")

    def __len__(self) -> int:
        """Returns the number of records in the dataset."""
        return self.size

    @property
    def size(self) -> int:
        """Returns the size of the dataset."""
        return self.index_df.height

    @classmethod
    def build_index_from_dataset_dir(
        cls,
        dataset_dir: str | Path,
        index_fn: example.IndexFunc | None = None,
        filepattern: str = "*.tfrecord",
        processes: int = 1,
        index_cache_dir: str | Path | None = None,
    ) -> "TFRecordDatasetReader":
        """Creates an index for all TFRecord files in a directory.

        Args:
            dataset_dir: Path to the directory containing tfrecord files.
            index_fn: function to create additional columns in the index
            filepattern: Pattern to match TFRecord files.
            processes: Number of processes to use for parallel processing.

        Returns:
            dataset: indexed TFRecord dataset reader.
        """

        storage = fs.get_file_system(dataset_dir)
        if not isinstance(storage, fs.LocalFileSystem):
            raise TypeError("Only local file system is supported for now.")

        data = indexer.create_index_for_directory(
            dataset_dir,
            index_fn=index_fn,
            filepattern=filepattern,
            processes=processes,
        )
        ds = pl.DataFrame(data).sort(by=["tfrecord_filename", "tfrecord_start"])
        ds.write_parquet(Path(dataset_dir) / indexer.INDEX_FILENAME)
        return cls(str(dataset_dir), index_df=ds, index_cache_dir=index_cache_dir)

    @overload
    def __getitem__(self, idx: int) -> example.Feature: ...

    @overload
    def __getitem__(self, idx: Iterable[int]) -> list[example.Feature]: ...

    def __getitem__(self, idx: int | Iterable[int]) -> example.Feature | list[example.Feature]:
        """Retrieves the TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            feature: The raw serialized record data as a Feature object.
        """
        if isinstance(idx, Iterable):
            return [self[i] for i in idx]  # type: ignore[misc]
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx=} out of bounds, dataset size={self.size}")
        offsets = self.index_df.row(idx, named=True)
        path = join_path(self.dataset_dir, offsets["tfrecord_filename"])
        with TFRecordFileReader(path) as reader:
            return reader.get_example(offsets["tfrecord_start"], offsets["tfrecord_end"])

    def select(self, sql_query: str) -> tuple[pl.DataFrame, list[example.Feature]]:
        """Executes an SQL query against the dataset index and loads the matched records.

        Args:
            sql_query: SQL statement querying the internal dataset index (table is `index`).

        Returns:
            A tuple containing:
              - A polars DataFrame representing the index selection.
              - A list of decoded Feature objects corresponding to the selection.
        """
        selection = self.ctx.execute(sql_query)
        self.logger.info(f"Selected N={selection.height} records ...")
        return selection, self.load_records(selection)

    def query(self, sql_query: str) -> pl.DataFrame:
        """Executes an SQL query against the dataset index.

        Args:
            sql_query: SQL statement querying the internal dataset index (table is `index`).

        Returns:
            A polars DataFrame containing the retrieved rows.
        """
        return self.ctx.execute(sql_query)

    def load_records(
        self, selection: pl.DataFrame, max_workers: int | None = None
    ) -> list[example.Feature]:
        """Loads physical TFRecord examples based on a provided index selection.

        Args:
            selection: A polars DataFrame containing exactly `tfrecord_filename`,
                `tfrecord_start`, and `tfrecord_end` columns.
            max_workers: The thread pool max workers for concurrent retrieval.

        Returns:
            A list of decoded Feature objects in the order of the selection dataframe.
        """
        index_cols = ["tfrecord_filename", "tfrecord_start", "tfrecord_end"]
        selection = selection[index_cols]
        pbar = {
            "total": selection.height,
            "desc": "Loading records ...",
            "disable": not self.verbose,
        }

        example_items = [
            {
                "filepath": join_path(self.dataset_dir, row["tfrecord_filename"]),
                "start": row["tfrecord_start"],
                "end": row["tfrecord_end"],
            }
            for row in selection.iter_rows(named=True)
        ]

        def get_single(row: dict) -> example.Feature:
            with TFRecordFileReader(row["filepath"]) as reader:
                return reader.get_example(row["start"], row["end"])

        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(tqdm(pool.map(get_single, example_items), **pbar))

    def _load_or_cache_index(self, index_path: str) -> pl.DataFrame:
        """Loads dataset index from local cache if available, or downloads and caches it.

        Args:
            index_path: The remote or local path to the index file in the storage.

        Returns:
            index_df: Polars dataframe containing the dataset index.

        Raises:
            FileNotFoundError: If the index file does not exist in the storage.
        """
        if self.index_cache_dir is None:
            if not self.storage.exists(index_path):
                raise FileNotFoundError(
                    f"Index file {index_path} does not exist. Please create the index first.",
                )
            self.logger.info("Loading dataset index from %s ...", index_path)
            with self.storage.open(index_path, "rb") as file:
                return pl.read_parquet(file.read())

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        path_hash = hashlib.sha256(index_path.encode("utf-8")).hexdigest()
        cached_index_path = self.index_cache_dir / f"{path_hash}_{indexer.INDEX_FILENAME}"

        if cached_index_path.exists():
            self.logger.info("Loading dataset index from cache %s ...", cached_index_path)
            return pl.read_parquet(cached_index_path)

        if not self.storage.exists(index_path):
            raise FileNotFoundError(
                f"Index file {index_path} does not exist. Please create the index first.",
            )
        self.logger.info("Loading dataset index from %s ...", index_path)
        with self.storage.open(index_path, "rb") as file:
            index_bytes = file.read()

        self.logger.info("Caching dataset index to %s ...", cached_index_path)
        with open(cached_index_path, "wb") as f_out:
            f_out.write(index_bytes)

        return pl.read_parquet(index_bytes)


def inspect_dataset_example(
    dataset_dir: str,
    filepattern: str = "*.tfrecord",
) -> tuple[example.Feature, list[dict[str, str]]]:
    """Inspects the TFRecord dataset and returns an example and its feature types."""
    storage = fs.get_file_system(dataset_dir)
    paths = storage.listdir(dataset_dir)
    paths = sorted([path for path in paths if fnmatch.fnmatch(path, filepattern)])
    LOGGER.info("Found N=%s TFRecord files ...", len(paths))

    with storage.open(paths[0], "rb") as file:
        length_bytes = file.read(8)
        if not length_bytes:
            raise IndexError("Failed to read length bytes")
        length = struct.unpack("<Q", length_bytes)[0]
        file.read(4)  # Skip length CRC
        data = file.read(length)
        if not data or len(data) < length:
            raise OSError("Failed to read data!")
        feature = example.decode(data)

    keys = list(feature.feature)
    feature_types = [
        {
            "key": key,
            "type": feature.feature[key].WhichOneof("kind"),
            "length": len(feature[key].value),
        }
        for key in keys
    ]

    return feature, feature_types


def load_from_directory(
    dataset_dir: str | Path,
    *,
    # index options
    filepattern: str = "*.tfrecord",
    index_fn: example.IndexFunc | None = None,
    processes: int = 1,
    override: bool = False,
    index_cache_dir: str | Path | None = None,
) -> TFRecordDatasetReader:
    """Loads an existing TFRecord dataset or creates an index for one if missing.

    Args:
        dataset_dir: Path/URI to the directory containing tfrecord files.
        filepattern: Glob pattern to identify TFRecord files (default: "*.tfrecord").
        index_fn: Optional parsing function to extract custom columns/fields into the index.
        processes: The number of processes to use if generating the index from scratch.
        override: If True, forces creating a new dataset index over an existing one.
        index_cache_dir: Optional location to persist remote index parquets locally.

    Returns:
        An instantiated TFRecordDatasetReader.
    """
    if (Path(dataset_dir) / indexer.INDEX_FILENAME).exists() and not override:
        LOGGER.info(
            "Index file already exists. Loading the dataset from the index ..."
            "If you want to override the index, set override=True.",
        )
        return TFRecordDatasetReader(dataset_dir, index_cache_dir=index_cache_dir)
    return TFRecordDatasetReader.build_index_from_dataset_dir(
        dataset_dir, index_fn, filepattern, processes, index_cache_dir=index_cache_dir
    )


def join_path(base_path: str | Path, suffix: str) -> str:
    """Properly joins a base directory URI/path with a filename suffix.

    Args:
        base_path: Base directory path (handles cloud GS scheme syntax gracefully).
        suffix: Appended filename component.

    Returns:
        The concatenated string representing the full URI.
    """
    base_str = str(base_path)
    if not base_str.endswith("/"):
        base_str += "/"
    return base_str + suffix
