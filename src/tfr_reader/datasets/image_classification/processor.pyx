# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# distutils: language=c++

"""
ImageProcessor: Extracted image processing logic for reusability.
Handles parallel decoding, resizing, and color conversion.
"""

import numpy as np
# noinspection PyUnresolvedReferences
cimport numpy as cnp
import cv2
from tfr_reader.cython cimport decoder
from tfr_reader.cython.decoder cimport Example, Feature
# noinspection PyUnresolvedReferences
from cython.parallel cimport prange, parallel
from cython cimport nogil

cnp.import_array()


cdef class ImageProcessor:
    """
    Handles parallel image decoding and processing.

    This class provides efficient image decoding, resizing, and batch processing
    with configurable parallelization.
    """

    def __init__(
        self,
        int target_width=320,
        int target_height=320,
        int num_threads=4,
        str image_feature_key='image/encoded',
        str label_feature_key='image/object/bbox/label'
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

    cpdef dict decode_example(self, bytes example_bytes):
        """
        Decode a TFRecord example and extract image bytes and label.

        Args:
            example_bytes: Raw TFRecord example bytes

        Returns:
            Dictionary with 'image_bytes' and 'label' keys
        """
        cdef Example example
        cdef Feature label_feature
        cdef Feature image_feature
        cdef int label
        cdef bytes image_data

        # Decode using C-level decoder
        example = decoder.example_from_bytes(example_bytes)

        # Extract features at C level
        label_feature = example.features.feature[self.label_feature_key]
        label = label_feature.int64_list.value[0]

        image_feature = example.features.feature[self.image_feature_key]
        image_data = image_feature.bytes_list.value[0]

        return {"image_bytes": image_data, "label": label}

    cpdef cnp.ndarray decode_and_resize_image(self, bytes image_bytes):
        """
        Decode and resize a single image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Resized image as numpy array (H, W, 3) in RGB format
        """
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] nparr
        cdef cnp.ndarray image

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_width, self.target_height))

        return image

    cpdef tuple process_batch(self, list raw_data_list):
        """
        Process a batch of raw TFRecord examples in parallel.

        Args:
            raw_data_list: List of raw TFRecord example bytes

        Returns:
            Tuple of (images array, labels array)
            - images: (batch_size, target_height, target_width, 3) uint8 array
            - labels: (batch_size,) int32 array
        """
        cdef int batch_size = len(raw_data_list)
        cdef int i

        if batch_size == 0:
            return None, None

        # Pre-allocate output arrays
        cdef cnp.ndarray[cnp.uint8_t, ndim=4] images = np.empty(
            (batch_size, self.target_height, self.target_width, 3), dtype=np.uint8
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.empty(batch_size, dtype=np.int32)

        # Decode all examples first (must be sequential due to Python object access)
        cdef list decoded_data = []
        for i in range(batch_size):
            decoded = self.decode_example(raw_data_list[i])
            decoded_data.append(decoded)

        # Process images in parallel
        with nogil, parallel(num_threads=self.num_threads):
            for i in prange(batch_size, schedule='dynamic'):
                with gil:
                    # Re-acquire GIL for Python object access and cv2 operations
                    data = decoded_data[i]
                    image_bytes = data['image_bytes']
                    labels[i] = data['label']

                    # Decode and resize
                    images[i] = self.decode_and_resize_image(image_bytes)

        return images, labels
