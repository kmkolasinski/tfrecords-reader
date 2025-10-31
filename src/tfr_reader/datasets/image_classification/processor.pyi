from typing import Any

import numpy as np
from numpy.typing import NDArray

class ImageProcessor:
    def __init__(
        self,
        target_width: int = 320,
        target_height: int = 320,
        num_threads: int = 4,
        image_feature_key: str = "image/encoded",
        label_feature_key: str = "image/object/bbox/label",
    ) -> None: ...
    def decode_example(self, example_bytes: bytes) -> dict[str, Any]: ...
    def decode_and_resize_image(self, image_bytes: bytes) -> NDArray[np.uint8]: ...
    def process_batch(
        self, examples: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.int32]]: ...
