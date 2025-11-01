"""
TFRecord datasets module.
"""

try:
    from tfr_reader.datasets.image_classification.dataset import TFRecordsImageDataset

    __all__ = ["TFRecordsImageDataset"]
except ImportError:
    # Provide helpful error message
    def _raise_import_error(*_args, **_kwargs):
        msg = (
            "Image classification dataset features are not available. "
            "To use this feature, install the package with datasets support:\n\n"
            "    pip install tfr_reader[datasets]\n\n"
            "Or install the required dependencies manually:\n"
            "    pip install numpy cython opencv-python-headless"
        )
        raise ImportError(msg)

    # Create a mock class that raises the error when instantiated
    class TFRecordsImageDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            _raise_import_error()

    __all__ = ["TFRecordsImageDataset"]
