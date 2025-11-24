"""
TFRecordsImageDataset: High-level API for multi-file TFRecord image datasets.
"""

from tfr_reader.datasets.image_classification.multi_file_reader import MultiFileReader
from tfr_reader.datasets.image_classification.prefetch import PrefetchBuffer
from tfr_reader.datasets.image_classification.processor import ImageProcessor
from tfr_reader.datasets.image_classification.sampler import BatchSampler


class TFRecordsImageDataset:
    """
    High-level dataset API for reading multiple TFRecord files with image data.

    Provides a tf.data.Dataset-like interface with support for shuffling,
    interleaving, repeating, and prefetching.
    """

    def __init__(  # noqa: PLR0913
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
        seed: int | None = None,
        save_index: bool = True,
        image_feature_key: str = "image/encoded",
        label_feature_key: str = "image/object/bbox/label",
        processing_backend: str = "cython",
        verbose: bool = False,
    ):
        """
        Initialize the TFRecordsImageDataset.

        Args:
            tfrecord_paths: list of paths to TFRecord files.
            input_size: (height, width) tuple for resized images.
            batch_size: Number of examples per batch.
            num_threads: Number of threads for parallel image processing.
            shuffle: Whether to shuffle examples.
            interleave_files: Whether to interleave reading across files.
            repeat: Number of epochs (-1 for infinite, 1 for single pass).
            prefetch: Number of batches to prefetch (0 to disable).
            max_open_files: Maximum number of files to keep open simultaneously.
            interleave_block_size: Block size for interleaving.
            seed: Random seed for reproducibility.
            save_index: Whether to save/load indices to/from disk.
            image_feature_key: Feature key for image data in TFRecord.
            label_feature_key: Feature key for label data in TFRecord.
            processing_backend: Backend to use for image processing ("cython" or "opencv").
        """

        if not tfrecord_paths:
            raise ValueError("tfrecord_paths cannot be empty")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        print("Initializing TFRecordsImageDataset...")

        self.file_reader = MultiFileReader(
            tfrecord_paths,
            max_open_files=max_open_files,
            save_index=save_index,
            verbose=verbose,
        )
        self.image_processor = ImageProcessor(
            target_width=input_size[1],
            target_height=input_size[0],
            num_threads=num_threads,
            image_feature_key=image_feature_key,
            label_feature_key=label_feature_key,
            processing_backend=processing_backend,
        )

        self.sampler = BatchSampler(
            file_lengths=self.file_reader.get_file_lengths(),
            shuffle=shuffle,
            interleave=interleave_files,
            repeat=repeat,
            interleave_block_size=max_open_files,
            drop_remainder=True,
            seed=seed,
        )

        self.batch_size = batch_size
        self.prefetch_size = prefetch
        self.use_prefetch = prefetch > 0
        self.prefetch_buffer: PrefetchBuffer | None = None
        self._iterator_started = False

        print(
            f"Dataset ready: {len(self.sampler)} examples, "
            f"{len(tfrecord_paths)} files, batch_size={batch_size}"
        )

    def __iter__(self):
        """Make the dataset iterable."""
        self.reset()
        return self

    def __next__(self):
        """
        Get the next batch.

        Returns:
            A tuple of (images, labels).

        Raises:
            StopIteration: When no more data is available.
        """
        if self.use_prefetch:
            return self._next_with_prefetch()
        return self._next_without_prefetch()

    def _next_without_prefetch(self):
        """Get the next batch without prefetching."""
        batch = self._generate_batch()
        if batch is None:
            raise StopIteration
        return batch

    def _next_with_prefetch(self):
        """Get the next batch with prefetching."""
        if not self._iterator_started:
            self.prefetch_buffer = PrefetchBuffer(
                batch_generator=self._generate_batch,
                buffer_size=self.prefetch_size,
                num_workers=1,
            )
            self._iterator_started = True

        if self.prefetch_buffer is None:
            raise RuntimeError("Prefetch buffer not initialized")

        batch = self.prefetch_buffer.get(timeout=30.0)
        if batch is None:
            raise StopIteration
        return batch

    def _generate_batch(self) -> tuple | None:
        """
        Generate a single batch.

        Returns:
            A tuple of (images, labels) or None if no more data is available.
        """
        indices = self.sampler.next_batch(self.batch_size)
        if indices is None or len(indices) == 0:
            return None

        raw_examples: list = self.file_reader.get_examples_batch(indices)
        return self.image_processor.process_batch(raw_examples)

    def reset(self):
        """Reset the dataset to the beginning."""
        if self.prefetch_buffer is not None:
            self.prefetch_buffer.stop()
            self.prefetch_buffer = None
        self._iterator_started = False
        self.sampler.reset()

    def shuffle(self):
        """Re-shuffle the dataset if shuffle was enabled."""
        self.sampler.shuffle_indices()

    def set_epoch(self, epoch: int):
        """
        Set the current epoch.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)

    def __len__(self):
        """Return the total number of examples."""
        return len(self.sampler)

    @property
    def batches_per_epoch(self) -> int:
        """Return the number of batches per epoch."""
        total_examples = len(self)
        return (total_examples + self.batch_size - 1) // self.batch_size

    @property
    def num_open_files(self) -> int:
        """Return the number of currently open files."""
        return self.file_reader.num_open_files

    def close(self):
        """Close all resources."""
        if hasattr(self, "prefetch_buffer") and self.prefetch_buffer is not None:
            self.prefetch_buffer.stop()
        if hasattr(self, "file_reader") and self.file_reader is not None:
            self.file_reader.close_all()

    def __del__(self):
        """Cleanup on destruction."""
        self.close()
