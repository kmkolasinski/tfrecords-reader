"""
ImageProcessor: Extracted image processing logic for reusability.
Handles parallel decoding, resizing, and color conversion.
"""

import cv2
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.numpy import import_array
    from cython.cimports.tfr_reader.cython import decoder
    from cython.cimports.tfr_reader.cython.decoder import Example, Feature
    from cython.parallel import parallel, prange

    import_array()
else:
    from tfr_reader.cython import decoder
    from tfr_reader.cython.decoder import Example, Feature


@cython.cclass
class ImageProcessor:
    """
    Handles parallel image decoding and processing.

    This class provides efficient image decoding, resizing, and batch processing
    with configurable parallelization.
    """

    target_width: cython.int
    target_height: cython.int
    num_threads: cython.int
    image_feature_key: str
    label_feature_key: str

    def __init__(
        self,
        target_width: cython.int = 320,
        target_height: cython.int = 320,
        num_threads: cython.int = 4,
        image_feature_key: str = "image/encoded",
        label_feature_key: str = "image/object/bbox/label",
    ):
        """
        Initialize the ImageProcessor.

        Args:
            target_width: Target image width for resizing
            target_height: Target image height for resizing
            num_threads: Number of threads for parallel processing (use 1 for sequential)
            image_feature_key: Feature key for image data in TFRecord
            label_feature_key: Feature key for label data in TFRecord
        """
        self.target_width = target_width
        self.target_height = target_height
        self.num_threads = num_threads
        self.image_feature_key = image_feature_key
        self.label_feature_key = label_feature_key

    @cython.cfunc
    @cython.locals(
        example=Example,
        label_feature=Feature,
        image_feature=Feature,
        label=cython.int,
        image_data=bytes,
    )
    def _decode_example_impl(self, example_bytes: bytes) -> dict:
        """C-level implementation of example decoding."""
        # Decode using C-level decoder
        example = decoder.example_from_bytes(example_bytes)

        # Extract features at C level
        label_feature = example.features.feature[self.label_feature_key]
        label = label_feature.int64_list.value[0]

        image_feature = example.features.feature[self.image_feature_key]
        image_data = image_feature.bytes_list.value[0]

        return {"image_bytes": image_data, "label": label}

    @cython.ccall
    def decode_example(self, example_bytes: bytes) -> dict:
        """
        Decode a TFRecord example and extract image bytes and label.

        Args:
            example_bytes: Raw TFRecord example bytes

        Returns:
            Dictionary with 'image_bytes' and 'label' keys
        """
        return self._decode_example_impl(example_bytes)

    @cython.ccall
    def decode_and_resize_image(self, image_bytes: bytes):
        """
        Decode and resize a single image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Resized image as numpy array (H, W, 3) in RGB format
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.target_width, self.target_height))

    @cython.ccall
    @cython.locals(
        batch_size=cython.int,
        i=cython.int,
    )
    def process_batch(self, raw_data_list: list) -> tuple:
        """
        Process a batch of raw TFRecord examples in parallel.

        Args:
            raw_data_list: List of raw TFRecord example bytes

        Returns:
            Tuple of (images array, labels array)
            - images: (batch_size, target_height, target_width, 3) uint8 array
            - labels: (batch_size,) int32 array
        """
        batch_size = len(raw_data_list)
        i: cython.int

        if batch_size == 0:
            return None, None

        # Pre-allocate output arrays
        images = np.empty((batch_size, self.target_height, self.target_width, 3), dtype=np.uint8)
        labels = np.empty(batch_size, dtype=np.int32)

        # Decode all examples first (must be sequential due to Python object access)
        decoded_data = []
        for i in range(batch_size):
            decoded = self.decode_example(raw_data_list[i])
            decoded_data.append(decoded)

        # Process images in parallel
        if cython.compiled:
            with cython.nogil, parallel(num_threads=self.num_threads):
                for i in prange(batch_size, schedule="dynamic"):
                    with cython.gil:
                        # Re-acquire GIL for Python object access and cv2 operations
                        data = decoded_data[i]
                        image_bytes = data["image_bytes"]
                        labels[i] = data["label"]

                        # Decode and resize
                        images[i] = self.decode_and_resize_image(image_bytes)
        else:
            # Fallback for non-compiled mode
            for i in range(batch_size):
                data = decoded_data[i]
                image_bytes = data["image_bytes"]
                labels[i] = data["label"]
                images[i] = self.decode_and_resize_image(image_bytes)

        return images, labels
