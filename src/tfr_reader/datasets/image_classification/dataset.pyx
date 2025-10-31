# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# distutils: language=c++

"""
TFRecordsImageDataset: High-level API for multi-file TFRecord image datasets.
"""

import numpy as np
cimport numpy as cnp
from tfr_reader.datasets.image_classification.multi_file_reader cimport MultiFileReader
from tfr_reader.datasets.image_classification.processor cimport ImageProcessor
from tfr_reader.datasets.image_classification.sampler cimport BatchSampler
from tfr_reader.datasets.image_classification.prefetch import PrefetchBuffer

cnp.import_array()


cdef class TFRecordsImageDataset:
    """
    High-level dataset API for reading multiple TFRecord files with image data.

    Provides a tf.data.Dataset-like interface with support for shuffling,
    interleaving, repeating, and prefetching.
    """

    def __init__(
        self,
        list tfrecord_paths,
        tuple input_size=(320, 320),
        int batch_size=32,
        int num_threads=4,
        bint shuffle=True,
        bint interleave_files=True,
        int repeat=-1,
        int prefetch=3,
        int max_open_files=16,
        interleave_block_size=None,
        seed=None,
        bint save_index=True,
        str image_feature_key='image/encoded',
        str label_feature_key='image/object/bbox/label'
    ):
        """
        Initialize the TFRecordsImageDataset.

        Args:
            tfrecord_paths: List of paths to TFRecord files
            input_size: (height, width) tuple for resized images
            batch_size: Number of examples per batch
            num_threads: Number of threads for parallel image processing
            shuffle: Whether to shuffle examples
            interleave_files: Whether to interleave reading across files
            repeat: Number of epochs (-1 for infinite, 1 for single pass)
            prefetch: Number of batches to prefetch (0 to disable)
            max_open_files: Maximum number of files to keep open simultaneously
            interleave_block_size: Block size for interleaving (None = batch_size)
            seed: Random seed for reproducibility (None for random)
            save_index: Whether to save/load indices to/from disk
            image_feature_key: Feature key for image data in TFRecord
            label_feature_key: Feature key for label data in TFRecord
        """
        cdef list file_lengths
        cdef int block_size

        if not tfrecord_paths:
            raise ValueError("tfrecord_paths cannot be empty")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Determine interleave block size
        if interleave_block_size is None:
            block_size = batch_size
        else:
            block_size = interleave_block_size

        # Initialize components
        print("Initializing TFRecordsImageDataset...")

        # 1. MultiFileReader - loads indices and manages file handles
        self.file_reader = MultiFileReader(
            tfrecord_paths,
            max_open_files=max_open_files,
            save_index=save_index
        )

        # 2. ImageProcessor - handles image decoding and resizing
        self.image_processor = ImageProcessor(
            target_width=input_size[1],   # (height, width) -> width
            target_height=input_size[0],  # (height, width) -> height
            num_threads=num_threads,
            image_feature_key=image_feature_key,
            label_feature_key=label_feature_key
        )

        # 3. BatchSampler - handles shuffling and batch generation
        file_lengths = self.file_reader.get_file_lengths()
        self.sampler = BatchSampler(
            file_lengths=file_lengths,
            shuffle=shuffle,
            interleave=interleave_files,
            repeat=repeat,
            interleave_block_size=block_size,
            drop_remainder=True,
            seed=seed
        )

        self.batch_size = batch_size
        self.prefetch_size = prefetch
        self.use_prefetch = prefetch > 0
        self.prefetch_buffer = None
        self._iterator_started = False

        print(f"Dataset ready: {len(self.sampler)} examples, "
              f"{len(tfrecord_paths)} files, batch_size={batch_size}")

    def __iter__(self):
        """Make the dataset iterable."""
        self.reset()
        return self

    def __next__(self):
        """
        Get the next batch.

        Returns:
            Tuple of (images, labels)
            - images: (batch_size, height, width, 3) uint8 array
            - labels: (batch_size,) int32 array

        Raises:
            StopIteration: When no more data is available
        """
        if self.use_prefetch:
            return self._next_with_prefetch()
        else:
            return self._next_without_prefetch()

    def _next_without_prefetch(self):
        """Get next batch without prefetching."""
        batch = self._generate_batch()
        if batch is None:
            raise StopIteration
        return batch

    def _next_with_prefetch(self):
        """Get next batch with prefetching."""
        # Start prefetch buffer on first iteration
        if not self._iterator_started:
            self.prefetch_buffer = PrefetchBuffer(
                batch_generator=self._generate_batch,
                buffer_size=self.prefetch_size,
                num_workers=1
            )
            self._iterator_started = True

        batch = self.prefetch_buffer.get(timeout=30.0)
        if batch is None:
            raise StopIteration

        return batch

    cdef object _generate_batch(self):
        """
        Generate a single batch.

        Returns:
            Tuple of (images, labels) or None if no more data
        """
        cdef list indices
        cdef list raw_examples
        cdef tuple batch

        # Get next batch of indices from sampler
        indices = self.sampler.next_batch(self.batch_size)

        if indices is None or len(indices) == 0:
            return None

        # Read raw examples from files
        raw_examples = self.file_reader.get_examples_batch(indices)

        # Process batch (decode and resize images)
        batch = self.image_processor.process_batch(raw_examples)

        return batch

    cpdef void reset(self):
        """Reset the dataset to the beginning."""
        # Stop prefetch buffer if active
        if self.prefetch_buffer is not None:
            self.prefetch_buffer.stop()
            self.prefetch_buffer = None

        self._iterator_started = False
        self.sampler.reset()

    cpdef void shuffle(self):
        """Re-shuffle the dataset (if shuffle was enabled)."""
        self.sampler._shuffle_indices()

    cpdef void set_epoch(self, int epoch):
        """
        Set the current epoch (useful for distributed training).

        Args:
            epoch: Epoch number
        """
        self.sampler.set_epoch(epoch)

    def __len__(self):
        """Return the total number of examples."""
        return len(self.sampler)

    @property
    def batches_per_epoch(self):
        """Return the number of batches per epoch."""
        total_examples = len(self.sampler)
        return (total_examples + self.batch_size - 1) // self.batch_size

    @property
    def num_open_files(self):
        """Return the number of currently open files."""
        return self.file_reader.num_open_files

    def close(self):
        """Close all resources."""
        if self.prefetch_buffer is not None:
            self.prefetch_buffer.stop()
            self.prefetch_buffer = None

        if self.file_reader is not None:
            self.file_reader.close_all()

    def __dealloc__(self):
        """Cleanup on destruction."""
        if self.file_reader is not None:
            self.close()
