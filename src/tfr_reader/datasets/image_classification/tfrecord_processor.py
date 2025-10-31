"""
Cython class for efficient parallel reading and processing of TFRecord files.
This is for backward compatibility.
"""

import random

import cv2
import cython
import numpy as np

if cython.compiled:
    from cython.cimports.numpy import import_array
    from cython.cimports.tfr_reader.cython import decoder
    from cython.cimports.tfr_reader.cython.decoder import Example, Feature
    from cython.cimports.tfr_reader.cython.indexer import TFRecordFileReader
    from cython.parallel import parallel, prange

    import_array()
else:
    from tfr_reader.cython import decoder
    from tfr_reader.cython.decoder import Example, Feature
    from tfr_reader.cython.indexer import TFRecordFileReader


@cython.cclass
class TFRecordProcessor:
    """
    Cython class for efficient parallel reading and processing of TFRecord files.
    """

    reader: TFRecordFileReader
    indices: list
    current_idx: cython.int
    total_examples: cython.int
    target_width: cython.int
    target_height: cython.int
    num_threads: cython.int

    def __init__(
        self,
        tfrecord_path: str,
        target_width: cython.int = 320,
        target_height: cython.int = 320,
        num_threads: cython.int = 4,
    ):
        """
        Initialize the TFRecord processor.

        Args:
            tfrecord_path: Path to the TFRecord file
            target_width: Target image width for resizing
            target_height: Target image height for resizing
            num_threads: Number of threads for parallel processing (default: 4)
        """
        self.reader = TFRecordFileReader(tfrecord_path)
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

    @cython.cfunc
    @cython.locals(
        example_bytes=bytes,
        example=Example,
        label_feature=Feature,
        image_feature=Feature,
        label=cython.int,
        image_data=bytes,
    )
    def _read_example(self, index: cython.int) -> dict:
        """
        Read a single example from the TFRecord file using fast C-level calls.

        Args:
            index: Index of the example to read

        Returns:
            Dictionary with image_bytes and label
        """
        # Use fast C-level method to get raw bytes
        example_bytes = self.reader.get_example_fast(index)

        # Decode using C-level decoder
        example = decoder.example_from_bytes(example_bytes)

        # Extract features at C level
        label_feature = example.features.feature["image/object/bbox/label"]
        label = label_feature.int64_list.value[0]

        image_feature = example.features.feature["image/encoded"]
        image_data = image_feature.bytes_list.value[0]

        return {"image_bytes": image_data, "label": label}

    @cython.ccall
    def decode_and_resize_image(self, image_bytes: bytes):
        """
        Decode and resize a single image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Resized image as numpy array
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.target_width, self.target_height))

    def read_batch(self, batch_size: cython.int):
        """
        Read and process a batch of examples.

        Args:
            batch_size: Number of examples to read

        Returns:
            Tuple of (images array, labels array)
        """
        i: cython.int
        idx: cython.int
        actual_size: cython.int = min(batch_size, self.total_examples - self.current_idx)

        if actual_size <= 0:
            return None, None

        # Pre-allocate arrays
        images = np.empty((actual_size, self.target_height, self.target_width, 3), dtype=np.uint8)
        labels = np.empty(actual_size, dtype=np.int32)

        # Pre-fetch all the raw data (must be done sequentially due to Python objects)
        data_list = []
        for i in range(actual_size):
            idx = self.indices[self.current_idx + i]
            data = self._read_example(idx)
            data_list.append(data)

        # Process images in parallel (if num_threads > 1) or sequentially (if num_threads == 1)
        if cython.compiled:
            with cython.nogil, parallel(num_threads=self.num_threads):
                for i in prange(actual_size, schedule="dynamic"):
                    with cython.gil:
                        # Re-acquire GIL for Python object access and cv2 operations
                        data = data_list[i]
                        image_bytes = data["image_bytes"]
                        labels[i] = data["label"]

                        # Decode and resize using the optimized method
                        images[i] = self.decode_and_resize_image(image_bytes)
        else:
            # Fallback for non-compiled mode
            for i in range(actual_size):
                data = data_list[i]
                image_bytes = data["image_bytes"]
                labels[i] = data["label"]
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

        idx: cython.int = self.indices[self.current_idx]
        data: dict = self._read_example(idx)

        self.current_idx += 1

        image = self.decode_and_resize_image(data["image_bytes"])
        return image, data["label"]
