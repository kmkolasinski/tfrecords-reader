import struct
from collections import defaultdict
from io import BufferedReader
from pathlib import Path

import polars as pl

from tfr_reader import indexer
from tfr_reader.example import example_pb2

INDEX_FILENAME = "tfrds-reader-index.parquet"


class TFRecordFileReader:
    def __init__(self, tfrecord_filename: str | Path):
        """Initializes the dataset with the TFRecord file and its index.

        Args:
            tfrecord_filename (str): Path to the TFRecord file.
        """
        self.tfrecord_filename = tfrecord_filename
        self.file: BufferedReader | None = None

    def __getitem__(self, offset: int) -> example_pb2.Example:
        """Retrieves the raw TFRecord at the specified index.

        Args:
            offset (int): The byte offset of the record to retrieve.

        Returns:
            bytes: The raw serialized record data.
        """
        if self.file is None:
            raise ValueError("File is not open. Use context manager!")
        return load_and_decode(self.file, offset)

    def open(self):
        """Opens the TFRecord file for reading."""
        if self.file is None:
            self.file = open(self.tfrecord_filename, "rb")  # noqa:SIM115

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
    def __init__(self, dataset_dir: str | Path, index_df: pl.DataFrame | None = None):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

        if index_df is None:
            index_path = self.dataset_dir / INDEX_FILENAME
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index file {index_path} does not exist. Please create the index first.",
                )
            index_df = pl.read_parquet(index_path)
        self.index_df = index_df
        self.ctx = pl.SQLContext(index=self.index_df, eager=True)

    @classmethod
    def build_index_from_dataset_dir(
        cls,
        dataset_dir: str | Path,
        feature_parse_fn: indexer.FeatureParseFunc,
        processes: int = 1,
    ) -> "TFRecordDatasetReader":
        data = indexer.create_index_for_directory(
            dataset_dir,
            feature_parse_fn,
            processes,
        )
        ds = pl.DataFrame(data).sort(by=["tfrecord_filename", "tfrecord_offset"])
        ds.write_parquet(Path(dataset_dir) / INDEX_FILENAME)
        return cls(dataset_dir, index_df=ds)

    def select(self, sql_query: str) -> tuple[pl.DataFrame, list[example_pb2.Example]]:
        result = self.ctx.execute(sql_query)
        return result, self.load_records(result)

    def query(self, sql_query: str) -> pl.DataFrame:
        return self.ctx.execute(sql_query)

    def load_records(self, rows: pl.DataFrame) -> list[example_pb2.Example]:
        examples = []
        grouped = rows[["tfrecord_filename", "tfrecord_offset"]].group_by(
            "tfrecord_filename",
        )
        for (filename,), group in grouped:
            offsets = group["tfrecord_offset"].to_list()
            path = Path(self.dataset_dir) / str(filename)
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist.")
            if not path.is_file():
                raise ValueError(f"Path {path} is not a file.")
            if not path.suffix == ".tfrecord":
                raise ValueError(f"File {path} is not a TFRecord file.")

            with TFRecordFileReader(path) as reader:
                examples.extend([reader[offset] for offset in offsets])

        return examples


def inspect_dataset_example(
    dataset_dir: str,
) -> tuple[example_pb2.Example, dict[str, list[str]]]:
    """Inspects the TFRecord dataset and returns an example and its feature types."""
    paths = sorted(Path(dataset_dir).glob("*.tfrecord"))
    print(f"Found N={len(paths)} TFRecord files ...")
    with TFRecordFileReader(paths[0]) as reader:
        example = reader[0]

    feature = example.features.feature
    keys = list(feature)
    feature_types = defaultdict(list)
    for key in keys:
        feature_types["name"].append(key)
        feature_types["type"].append(feature[key].WhichOneof("kind"))
    return example, feature_types


def load_and_decode(file: BufferedReader, offset: int) -> example_pb2.Example:
    """Reads a TFRecord file and decodes the example at the specified offset."""
    file.seek(offset)
    length_bytes = file.read(8)
    if not length_bytes:
        raise IndexError("Failed to read length bytes")
    length = struct.unpack("<Q", length_bytes)[0]
    file.read(4)  # Skip length CRC
    data = file.read(length)
    file.read(4)  # Skip data CRC
    return indexer.decode(data)
