from typing import Literal

from tfr_reader.example import feature
from tfr_reader.example.feature import Feature, IndexFunc, decode


def set_decoder_type(decoder_type: Literal["protobuf", "cython"]) -> None:
    """Set the decoder type for the example module.

    * protobuf - Use the protobuf decoder from the google.protobuf library.
    * cython - Use the custom decoder from the tfr_reader.cython.decoder module.

    Args:
        decoder_type: The type of decoder to use. Can be "protobuf" or "cython".
    """
    feature.TFRECORD_READER_DECODER_IMP = decoder_type


__all__ = [
    "Feature",
    "IndexFunc",
    "decode",
    "set_decoder_type",
]
