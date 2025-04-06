import struct
from pathlib import Path

import polars as pl
from tqdm import tqdm

from tfr_reader import example, indexer, logging
from tfr_reader import filesystem as fs

LOGGER = logging.get_logger(__name__)
INDEX_FILENAME = "tfrds-reader-index.parquet"


class TFRecordFileReader:
    def __init__(self, filepath: str):
        """Initializes the dataset with the TFRecord file and its index.

        Args:
            filepath: Path to the TFRecord file.
        """
        self.tfrecord_filepath = filepath
        self.storage = fs.get_file_system(filepath)
        self.file: fs.BaseFile | None = None

    def __getitem__(self, offset: int) -> example.Feature:
        """Retrieves the raw TFRecord at the specified index.

        Args:
            offset (int): The byte offset of the record to retrieve.

        Returns:
            feature: The raw serialized record data as a Feature object.
        """
        if self.file is None:
            raise ValueError("File is not open. Use context manager!")

        self.file.seek(offset)
        length_bytes = self.file.read(8)
        if not length_bytes:
            raise IndexError("Failed to read length bytes")
        length = struct.unpack("<Q", length_bytes)[0]
        self.file.read(4)  # Skip length CRC
        data = self.file.read(length)
        if not data or len(data) < length:
            raise OSError(f"Failed to read data at offset {offset}")
        self.file.read(4)  # Skip data CRC
        return indexer.decode(data)

    def open(self):
        """Opens the TFRecord file for reading."""
        if self.file is None:
            self.file = self.storage.open(self.tfrecord_filepath, "rb")

    def close(self):
        """Closes the TFRecord file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Context manager entry method."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.close()
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

        if index_df is None:
            index_path = join_path(dataset_dir, INDEX_FILENAME)
            if not self.storage.exists(index_path):
                raise FileNotFoundError(
                    f"Index file {index_path} does not exist. Please create the index first.",
                )
            file = self.storage.open(index_path, "rb")
            index_df = pl.read_parquet(file)
        self.index_df = index_df
        self.ctx = pl.SQLContext(index=self.index_df, eager=True)

        if self.verbose:
            print(f"Loaded dataset index with N={self.index_df.height} records ...")

    @classmethod
    def build_index_from_dataset_dir(
        cls,
        dataset_dir: str,
        feature_parse_fn: indexer.FeatureParseFunc,
        processes: int = 1,
    ) -> "TFRecordDatasetReader":
        storage = fs.get_file_system(dataset_dir)
        if not isinstance(storage, fs.LocalFileSystem):
            raise TypeError("Only local file system is supported for now.")

        data = indexer.create_index_for_directory(
            dataset_dir,
            feature_parse_fn,
            processes,
        )
        ds = pl.DataFrame(data).sort(by=["tfrecord_filename", "tfrecord_offset"])
        ds.write_parquet(Path(dataset_dir) / INDEX_FILENAME)
        return cls(dataset_dir, index_df=ds)

    def select(self, sql_query: str) -> tuple[pl.DataFrame, list[example.Feature]]:
        selection = self.ctx.execute(sql_query)
        if self.verbose:
            print(f"Selected N={selection.height} records ...")
        return selection, self.load_records(selection)

    def query(self, sql_query: str) -> pl.DataFrame:
        return self.ctx.execute(sql_query)

    def load_records(self, selection: pl.DataFrame) -> list[example.Feature]:
        examples = []
        index_cols = ["tfrecord_filename", "tfrecord_offset"]
        grouped = selection[index_cols].group_by("tfrecord_filename")
        num_groups = grouped.len().height

        iterator = grouped
        if self.verbose:
            iterator = tqdm(grouped, total=num_groups, desc="Loading records ...")  # type: ignore  # noqa: PGH003
            print(f"Getting examples from N={num_groups} TFRecord files ...")

        for (filename,), group in iterator:
            offsets = group["tfrecord_offset"].to_list()
            path = join_path(self.dataset_dir, str(filename))
            with TFRecordFileReader(path) as reader:
                examples.extend([reader[offset] for offset in offsets])

        return examples


def inspect_dataset_example(
    dataset_dir: str,
) -> tuple[example.Feature, list[dict[str, str]]]:
    """Inspects the TFRecord dataset and returns an example and its feature types."""
    storage = fs.get_file_system(dataset_dir)
    paths = storage.listdir(dataset_dir)
    paths = sorted([path for path in paths if path.endswith(".tfrecord")])
    LOGGER.info("Found N=%s TFRecord files ...", len(paths))
    with TFRecordFileReader(paths[0]) as reader:
        feature = reader[0]

    keys = list(feature.feature)
    feature_types = [{"key": key, "type": feature.feature[key].WhichOneof("kind")} for key in keys]

    return feature, feature_types


def join_path(base_path: str, suffix: str) -> str:
    if not base_path.endswith("/"):
        base_path += "/"
    return base_path + suffix
