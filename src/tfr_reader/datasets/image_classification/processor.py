"""
ImageProcessor: Extracted image processing logic for reusability.
Handles parallel decoding, resizing, and color conversion.
"""

from typing import Any

import cv2  # type: ignore[import-not-found]
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.tfr_reader.cython import decoder, image
    from cython.parallel import parallel, prange
else:
    from tfr_reader.cython import decoder, image


class ImageProcessor:
    """
    Handles image decoding and processing.

    This class provides efficient image decoding, resizing, and batch processing.
    Note: The parallel processing feature requires Cython compilation. This pure
    Python version processes data sequentially.
    """

    def __init__(  # noqa: PLR0913
        self,
        target_width: int = 320,
        target_height: int = 320,
        num_threads: int = 4,
        image_feature_key: str = "image/encoded",
        label_feature_key: str = "image/object/bbox/label",
        processing_backend: str = "cython",
    ):
        """
        Initialize the ImageProcessor.

        Args:
            target_width: Target image width for resizing.
            target_height: Target image height for resizing.
            num_threads: Number of threads (only used in Cython-compiled version).
            image_feature_key: Feature key for image data in TFRecord.
            label_feature_key: Feature key for label data in TFRecord.
        """
        self.target_width = target_width
        self.target_height = target_height
        self.num_threads = num_threads  # Not used in this implementation
        self.image_feature_key = image_feature_key
        self.label_feature_key = label_feature_key
        self.processing_backend = processing_backend

    def decode_example(self, example_bytes: bytes) -> dict[str, Any]:
        """
        Decode a TFRecord example and extract image bytes and label.

        Args:
            example_bytes: Raw TFRecord example bytes.

        Returns:
            A dictionary with 'image_bytes' and 'label' keys.
        """
        example: decoder.Example = decoder.example_from_bytes(example_bytes)

        label_feature = example.features.feature[self.label_feature_key]
        label = label_feature.int64_list.value[0]

        image_feature = example.features.feature[self.image_feature_key]
        image_data = image_feature.bytes_list.value[0]

        return {"image_bytes": image_data, "label": label}

    def process_batch(self, raw_data_list: list[bytes]) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Process a batch of raw TFRecord examples sequentially.

        Args:
            raw_data_list: A list of raw TFRecord example bytes.

        Returns:
            A tuple of (images_array, labels_array), or None if the input is empty.
        """
        if cython.compiled:
            if self.processing_backend == "opencv":
                return self.cy_process_batch_opencv(raw_data_list)
            if self.processing_backend == "cython":
                return self.cy_process_batch_cython(raw_data_list)
            raise ValueError(f"Unknown processing backend: {self.processing_backend}")

        batch_size = len(raw_data_list)
        if batch_size == 0:
            return None

        # Pre-allocate output arrays
        images = np.empty((batch_size, self.target_height, self.target_width, 3), dtype=np.uint8)
        labels = np.empty(batch_size, dtype=np.int32)

        # Decode all examples first
        decoded_data = [self.decode_example(data) for data in raw_data_list]

        for i, data in enumerate(decoded_data):
            image_bytes = data["image_bytes"]
            labels[i] = data["label"]
            images[i] = self.decode_and_resize_image(image_bytes)

        return images, labels

    def cy_process_batch_opencv(self, raw_data_list):
        batch_size = len(raw_data_list)
        if batch_size == 0:
            return None

        # Pre-allocate output arrays
        images = np.empty((batch_size, self.target_height, self.target_width, 3), dtype=np.uint8)
        labels = np.empty(batch_size, dtype=np.int32)

        # Decode all examples first (must be sequential due to Python object access)
        decoded_data = []
        for i in range(batch_size):
            decoded = self.decode_example(raw_data_list[i])
            decoded_data.append(decoded)

        with cython.nogil, parallel(num_threads=self.num_threads):
            for i in prange(batch_size, schedule="dynamic"):
                with cython.gil:
                    data = decoded_data[i]
                    image_bytes = data["image_bytes"]
                    labels[i] = data["label"]
                    images[i] = self.decode_and_resize_image(image_bytes)

        return images, labels

    def cy_process_batch_cython(self, raw_data_list):
        batch_size = len(raw_data_list)
        if batch_size == 0:
            return None

        labels = np.empty(batch_size, dtype=np.int32)

        # Decode all examples first (must be sequential due to Python object access)
        images_bytes = []
        for i in range(batch_size):
            data = self.decode_example(raw_data_list[i])
            labels[i] = data["label"]
            images_bytes.append(data["image_bytes"])

        return image.decode_jpegs_to_array(
            images_bytes, self.target_height, self.target_width
        ), labels

    def decode_and_resize_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode and resize a single image.

        Args:
            image_bytes: Raw image bytes (e.g., JPEG, PNG).

        Returns:
            A resized image as a numpy array (H, W, 3) in RGB format.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.target_width, self.target_height))
