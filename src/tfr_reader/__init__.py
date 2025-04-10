from importlib import metadata

from tfr_reader.example import Feature, set_decoder_type
from tfr_reader.reader import (
    TFRecordDatasetReader,
    TFRecordFileReader,
    inspect_dataset_example,
    join_path,
    load_from_directory,
)

__all__ = [
    "Feature",
    "TFRecordDatasetReader",
    "TFRecordFileReader",
    "inspect_dataset_example",
    "join_path",
    "load_from_directory",
    "set_decoder_type",
]

__version__ = metadata.version(__package__ or __name__)
