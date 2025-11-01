from typing import Literal

class BytesList:
    value: list[bytes]
    def __init__(self, value: list[bytes]) -> None: ...
    def __getitem__(self, item: int) -> bytes: ...
    def __len__(self) -> int: ...

class FloatList:
    value: list[float]
    def __init__(self, value: list[float]) -> None: ...
    def __getitem__(self, item: int) -> float: ...

class Int64List:
    value: list[int]
    def __init__(self, value: list[int]) -> None: ...
    def __getitem__(self, item: int) -> int: ...

class Feature:
    key: str
    kind: Literal["float_list", "int64_list", "bytes_list"]
    float_list: FloatList
    int64_list: Int64List
    bytes_list: BytesList

    def __init__(
        self,
        key: str,
        kind: Literal["float_list", "int64_list", "bytes_list"],
        float_list: FloatList = ...,
        int64_list: Int64List = ...,
        bytes_list: BytesList = ...,
    ) -> None: ...
    def WhichOneof(self, kind: str) -> str: ...

class Features:
    feature: dict[str, Feature]
    def __init__(self, feature: dict[str, Feature]) -> None: ...

class Example:
    features: Features
    def __init__(self, features: Features) -> None: ...

def example_from_bytes(buffer: bytes) -> Example: ...
