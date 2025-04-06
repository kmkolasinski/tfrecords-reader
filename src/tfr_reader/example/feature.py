import abc
import io
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from tfr_reader.example import example_pb2

T = TypeVar("T")

FeatureDecodeFunc = Callable[["Feature"], dict[str, Any]]


class BaseFeature(Generic[T], abc.ABC):
    def __init__(self, feature: example_pb2.Feature):
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
    def bytes(self) -> list[io.BytesIO]:
        """Get the value of the bytes list as BytesIO objects."""
        return [io.BytesIO(b) for b in self.value]


class Feature:
    __slots__ = ("feature",)

    def __init__(self, feature: example_pb2.Feature):
        """Initialize the Feature object with a protobuf Feature."""
        self.feature = feature

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
