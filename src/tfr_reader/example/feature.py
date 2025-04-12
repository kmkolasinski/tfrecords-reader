import abc
import io
import warnings
from collections.abc import Callable
from typing import Any, Generic, Literal, TypeVar

from tfr_reader.cython import decoder

T = TypeVar("T")

IndexFunc = Callable[["Feature"], dict[str, Any]]


class BaseFeature(Generic[T], abc.ABC):
    def __init__(self, feature):
        """Initialize the FloatList object with a protobuf FloatList."""
        self.feature = feature

    @property
    @abc.abstractmethod
    def value(self) -> list[T]:
        """Get the value of the feature."""


class FloatList(BaseFeature[float]):
    @property
    def value(self) -> list[float]:
        """Get the value of the float list."""
        return self.feature.float_list.value


class Int64List(BaseFeature[int]):
    @property
    def value(self) -> list[int]:
        """Get the value of the int64 list."""
        return self.feature.int64_list.value


class BytesList(BaseFeature[bytes]):
    @property
    def value(self) -> list[bytes]:
        """Get the value of the bytes list."""
        return self.feature.bytes_list.value

    @property
    def bytes_io(self) -> list[io.BytesIO]:
        """Get the value of the bytes list as BytesIO objects."""
        return [io.BytesIO(b) for b in self.value]


class Feature:
    __slots__ = ("feature",)

    def __init__(self, feature):
        """Initialize the Feature object with a protobuf Feature."""
        self.feature = feature

    def __len__(self) -> int:
        """Get the number of features."""
        return len(self.feature)

    def __repr__(self):
        """Get the string representation of the Feature object."""
        return f"Feature({set(self.feature.keys())})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return False
        return self.as_dict == other.as_dict

    @property
    def as_dict(self) -> dict[str, list[Any]]:
        """Get the feature values as a dictionary."""
        return {key: self[key].value for key in self.feature}

    @property
    def fields_names(self) -> list[str]:
        """Get the names of the fields in the feature."""
        return list(self.feature.keys())

    @property
    def fields(self) -> list[tuple[str, str]]:
        """Get the types of the fields in the feature."""
        return [(key, self.feature[key].WhichOneof("kind")) for key in self.feature]

    def __getitem__(self, key: str) -> BaseFeature:
        """Get the feature by key."""
        if key not in self.feature:
            raise KeyError(
                f"Feature '{key}' not found in the example, "
                f"expected one of {list(self.feature)}"
            )
        feature = self.feature[key]
        kind = feature.WhichOneof("kind")
        if kind == "float_list":
            return FloatList(feature)
        if kind == "int64_list":
            return Int64List(feature)
        if kind == "bytes_list":
            return BytesList(feature)
        raise ValueError(f"Unknown feature kind: '{kind}' for '{key}'!")


def _cython_decode_fn(raw_record: bytes) -> Feature:
    proto = decoder.example_from_bytes(raw_record)
    return Feature(proto.features.feature)


_google_decode_fn = None
TFRECORD_READER_DECODER_IMP: Literal["protobuf", "cython"] = "cython"

try:
    # https://github.com/protocolbuffers/protobuf/blob/main/python/README.md
    from google import protobuf
    from google.protobuf.internal import api_implementation

    from tfr_reader.example import tfr_example_pb2

    def _google_decode_fn(raw_record: bytes) -> Feature:
        proto = tfr_example_pb2.Example()
        proto.ParseFromString(raw_record)
        return Feature(proto.features.feature)

    if api_implementation.Type() == "python":
        warnings.warn(
            f"Detected 'Python' protobuf API implementation type ({protobuf.__version__}). "
            "Fallback to custom Cython decoder, maybe unsafe! To force official "
            "Google decoder, you can set tfr.set_decoder_type('protobuf')"
        )
        TFRECORD_READER_DECODER_IMP = "cython"

except ImportError:
    warnings.warn(
        "Google protobuf library not found. "
        "Falling back to Cython decoder for TFRecord files, which may be unsafe. "
    )
    TFRECORD_READER_DECODER_IMP = "cython"
except Exception as e:  # noqa: BLE001
    warnings.warn(
        f"Failed to import Google protobuf library: {e}. "
        "Falling back to Cython decoder for TFRecord files, which may be unsafe. "
    )
    TFRECORD_READER_DECODER_IMP = "cython"


def decode(raw_record: bytes) -> Feature:
    if TFRECORD_READER_DECODER_IMP == "protobuf":
        return _google_decode_fn(raw_record)
    if TFRECORD_READER_DECODER_IMP == "cython":
        return _cython_decode_fn(raw_record)
    raise ValueError(f"Unknown decoder type: {TFRECORD_READER_DECODER_IMP}!")
