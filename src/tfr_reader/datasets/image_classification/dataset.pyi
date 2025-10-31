from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

class TFRecordsImageDataset:
    def __init__(
        self,
        tfrecord_paths: list[str],
        input_size: tuple[int, int] = (320, 320),
        batch_size: int = 32,
        num_threads: int = 4,
        shuffle: bool = True,
        interleave_files: bool = True,
        repeat: int = -1,
        prefetch: int = 3,
        max_open_files: int = 16,
        interleave_block_size: int | None = None,
        seed: int | None = None,
        save_index: bool = True,
        image_feature_key: str = "image/encoded",
        label_feature_key: str = "image/object/bbox/label",
    ) -> None: ...
    def __iter__(self) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.int32]]]: ...
    def __next__(self) -> tuple[NDArray[np.uint8], NDArray[np.int32]]: ...
    def reset(self) -> None: ...
    def shuffle(self) -> None: ...
    def set_epoch(self, epoch: int) -> None: ...
    def __len__(self) -> int: ...
    @property
    def batches_per_epoch(self) -> int: ...
    def close(self) -> None: ...
