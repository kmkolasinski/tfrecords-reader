from importlib import metadata

from tfr_reader.reader import (
    TFRecordDatasetReader,
    TFRecordFileReader,
    inspect_dataset_example,
    join_path,
)

__all__ = [
    "TFRecordDatasetReader",
    "TFRecordFileReader",
    "inspect_dataset_example",
    "join_path",
]

__version__ = metadata.version(__package__ or __name__)
