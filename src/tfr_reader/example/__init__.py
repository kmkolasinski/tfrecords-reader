from typing import Literal

from tfr_reader.example import feature
from tfr_reader.example.feature import Feature, FeatureDecodeFunc, decode


def set_decoder_type(decoder_type: Literal["google", "cython"]) -> None:
    """Set the decoder type for the example module.

    Args:
        decoder_type: The type of decoder to use. Can be "python" or "cython".
    """
    feature.TFRECORD_READER_DECODER_IMP = decoder_type


__all__ = [
    "Feature",
    "FeatureDecodeFunc",
    "decode",
    "set_decoder_type",
]
