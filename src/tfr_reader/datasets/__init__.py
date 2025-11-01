try:
    from tfr_reader.datasets.image_classification.dataset import TFRecordsImageDataset

    __all__ = ["TFRecordsImageDataset"]
except ImportError as error:
    msg = (
        "Image classification dataset features are not available. "
        "To use this feature, install the package with datasets support:\n"
        "    pip install tfr_reader[datasets]\n"
    )
    raise ImportError(msg) from error
