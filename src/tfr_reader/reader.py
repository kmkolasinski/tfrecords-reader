import fnmatch
import struct
from collections.abc import Iterable
from concurrent import futures
from pathlib import Path

import polars as pl
from tqdm import tqdm

from tfr_reader import example, indexer, logging
from tfr_reader import filesystem as fs

LOGGER = logging.Logger(__name__)


class TFRecordFileReader:
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
    def __init__(
        self,
        dataset_dir: str,
        index_df: pl.DataFrame | None = None,
        verbose: bool = True,
    ):
        """Initializes the dataset with the TFRecord files and their index."""

        self.storage = fs.get_file_system(dataset_dir)
        self.dataset_dir = dataset_dir
        self.verbose = verbose
        self.logger = logging.Logger(self.__class__.__name__, verbose)

        if index_df is None:
            index_path = join_path(dataset_dir, indexer.INDEX_FILENAME)
            if not self.storage.exists(index_path):
                raise FileNotFoundError(
                    f"Index file {index_path} does not exist. Please create the index first.",
                )
            self.logger.info("Loading dataset index from %s ...", index_path)
            with self.storage.open(index_path, "rb") as file:
                index_df = pl.read_parquet(file.read())
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
        dataset_dir: str,
        index_fn: example.IndexFunc | None = None,
        filepattern: str = "*.tfrecord",
        processes: int = 1,
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
        return cls(str(dataset_dir), index_df=ds)

    def __getitem__(self, idx: int | Iterable[int]) -> example.Feature | list[example.Feature]:
        """Retrieves the TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            feature: The raw serialized record data as a Feature object.
        """
        if isinstance(idx, Iterable):
            return [self[i] for i in idx]
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx=} out of bounds, dataset size={self.size}")
        offsets = self.index_df.row(idx, named=True)
        path = join_path(self.dataset_dir, offsets["tfrecord_filename"])
        with TFRecordFileReader(path) as reader:
            return reader.get_example(offsets["tfrecord_start"], offsets["tfrecord_end"])

    def select(self, sql_query: str) -> tuple[pl.DataFrame, list[example.Feature]]:
        selection = self.ctx.execute(sql_query)
        self.logger.info(f"Selected N={selection.height} records ...")
        return selection, self.load_records(selection)

    def query(self, sql_query: str) -> pl.DataFrame:
        return self.ctx.execute(sql_query)

    def load_records(
        self, selection: pl.DataFrame, max_workers: int | None = None
    ) -> list[example.Feature]:
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
    dataset_dir: str,
    *,
    # index options
    filepattern: str = "*.tfrecord",
    index_fn: example.IndexFunc | None = None,
    processes: int = 1,
    override: bool = False,
) -> TFRecordDatasetReader:
    """Creates an index for the TFRecord dataset."""
    if (Path(dataset_dir) / indexer.INDEX_FILENAME).exists() and not override:
        LOGGER.info(
            "Index file already exists. Loading the dataset from the index ..."
            "If you want to override the index, set override=True.",
        )
        return TFRecordDatasetReader(dataset_dir)
    return TFRecordDatasetReader.build_index_from_dataset_dir(
        dataset_dir, index_fn, filepattern, processes
    )


def join_path(base_path: str, suffix: str) -> str:
    if not base_path.endswith("/"):
        base_path += "/"
    return base_path + suffix
