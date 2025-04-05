from importlib import metadata

from tfr_reader.reader import TFRecordDatasetReader, inspect_dataset_example  # noqa: F401

__version__ = metadata.version(__package__ or __name__)
