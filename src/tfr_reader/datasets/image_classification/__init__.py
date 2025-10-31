"""
Image classification dataset components.
"""

from tfr_reader.datasets.image_classification.dataset import TFRecordsImageDataset
from tfr_reader.datasets.image_classification.tfrecord_processor import TFRecordProcessor

__all__ = ["TFRecordProcessor", "TFRecordsImageDataset"]
