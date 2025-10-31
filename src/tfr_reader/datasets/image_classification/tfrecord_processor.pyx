# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
import cv2
import random
from tfr_reader.cython import indexer
from tfr_reader import example as tfr_example
# noinspection PyUnresolvedReferences
from cython.parallel cimport prange, parallel
from cython cimport nogil
cnp.import_array()

cdef class TFRecordProcessor:
    """
    Cython class for efficient parallel reading and processing of TFRecord files.

    """
    cdef object reader
    cdef list indices
    cdef int current_idx
    cdef int total_examples
    cdef int target_width
    cdef int target_height
    cdef int num_threads

    def __init__(
            self,
            str tfrecord_path,
            int target_width=320,
            int target_height=320,
            int num_threads=4
    ):
        """
        Initialize the TFRecord processor.

        Args:
            tfrecord_path: Path to the TFRecord file
            target_width: Target image width for resizing
            target_height: Target image height for resizing
            num_threads: Number of threads for parallel processing (default: 4, use 1 for sequential)
        """
        self.reader = indexer.TFRecordFileReader(tfrecord_path)
        self.total_examples = len(self.reader)
        self.indices = list(range(self.total_examples))
        random.shuffle(self.indices)
        self.current_idx = 0
        self.target_width = target_width
        self.target_height = target_height
        self.num_threads = num_threads

    def shuffle(self):
        """Shuffle the reading order."""
        random.shuffle(self.indices)
        self.current_idx = 0

    def reset(self):
        """Reset the reader to the beginning."""
        self.current_idx = 0

    cdef dict _read_example(self, int index):
        """
        Read a single example from the TFRecord file.

        Args:
            index: Index of the example to read

        Returns:
            Dictionary with image_bytes and label
        """
        cdef object example, feature
        cdef int label
        cdef bytes image_data

        example = self.reader.get_example(index)
        feature = tfr_example.decode(example)
        label = feature['image/object/bbox/label'].value[0]
        image_data = feature['image/encoded'].value[0]

        return {"image_bytes": image_data, "label": label}

    cpdef cnp.ndarray decode_and_resize_image(self, bytes image_bytes):
        """
        Decode and resize a single image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Resized image as numpy array
        """
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] nparr
        cdef cnp.ndarray image

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_width, self.target_height))

        return image

    def read_batch(self, int batch_size):
        """
        Read and process a batch of examples.

        Args:
            batch_size: Number of examples to read

        Returns:
            Tuple of (images array, labels array)
        """
        cdef int i, idx
        cdef int actual_size = min(batch_size, self.total_examples - self.current_idx)

        if actual_size <= 0:
            return None, None

        # Pre-allocate arrays
        cdef cnp.ndarray[cnp.uint8_t, ndim=4] images = np.empty(
            (actual_size, self.target_height, self.target_width, 3), dtype=np.uint8
        )
        cdef cnp.ndarray[cnp.int32_t, ndim=1] labels = np.empty(actual_size, dtype=np.int32)

        # Pre-fetch all the raw data (must be done sequentially due to Python objects)
        cdef list data_list = []
        for i in range(actual_size):
            idx = self.indices[self.current_idx + i]
            data = self._read_example(idx)
            data_list.append(data)

        # Process images in parallel (if num_threads > 1) or sequentially (if num_threads == 1)
        cdef int thread_id
        with nogil, parallel(num_threads=self.num_threads):
            for i in prange(actual_size, schedule='dynamic'):
                with gil:
                    # Re-acquire GIL for Python object access and cv2 operations
                    data = data_list[i]
                    image_bytes = data['image_bytes']
                    labels[i] = data['label']

                    # Decode and resize using the optimized method
                    images[i] = self.decode_and_resize_image(image_bytes)

        self.current_idx += actual_size
        return images, labels

    def __len__(self):
        """Return the total number of examples."""
        return self.total_examples

    def __iter__(self):
        """Make the processor iterable."""
        return self

    def __next__(self):
        """
        Get the next processed example.

        Returns:
            Tuple of (image, label)
        """
        if self.current_idx >= self.total_examples:
            raise StopIteration

        cdef int idx = self.indices[self.current_idx]
        cdef dict data = self._read_example(idx)

        self.current_idx += 1

        image = self.decode_and_resize_image(data['image_bytes'])
        return image, data['label']
