"""
BatchSampler: Handles shuffling, interleaving, and batch sampling logic.

This file is written in pure Python with type hints for clarity.
"""

import random
import time


class BatchSampler:
    """
    Manages batch sampling with shuffling, interleaving, and repeat logic.

    Generates batches of (file_idx, example_idx) tuples for efficient
    multi-file reading.
    """

    def __init__(  # noqa: PLR0913
        self,
        file_lengths: list[int],
        shuffle: bool = True,
        interleave: bool = True,
        repeat: int = 1,
        interleave_block_size: int = 32,
        drop_remainder: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the BatchSampler.

        Args:
            file_lengths: list of the number of examples in each file.
            shuffle: Whether to shuffle the indices.
            interleave: Whether to interleave across files.
            repeat: Number of epochs (-1 for infinite, 1 for single pass).
            interleave_block_size: Block size for interleaving.
            drop_remainder: Whether to drop the last batch if it's smaller than the batch size.
            seed: Random seed for reproducibility.
        """
        self.shuffle: bool = shuffle
        self.interleave: bool = interleave
        self.drop_remainder: bool = drop_remainder
        self.max_epochs: int = repeat
        self.interleave_block_size: int = interleave_block_size

        self.epoch_count: int = 0
        self.current_pos: int = 0
        self.total_examples: int = 0

        # Set random seed
        if seed is None:
            self.random_seed = int(time.time())
        else:
            self.random_seed = seed

        # Build and store the global index
        self.global_index = self._build_global_index(file_lengths)
        self.original_index = list(self.global_index)
        self.total_examples = len(self.global_index)

        # Initial shuffle if requested
        if self.shuffle:
            self._shuffle_indices()

    def _build_global_index(self, file_lengths: list[int]) -> list[tuple[int, int]]:
        """
        Build the global index of (file_idx, example_idx) pairs.

        Args:
            file_lengths: list of the number of examples in each file.

        Returns:
            A list of (file_idx, example_idx) tuples.
        """
        if self.interleave:
            return self._build_interleaved_index(file_lengths)
        # Sequential: all examples from file 0, then file 1, etc.
        index = []
        for file_idx, num_examples in enumerate(file_lengths):
            index.extend([(file_idx, example_idx) for example_idx in range(num_examples)])

        return index

    def _build_interleaved_index(self, file_lengths: list[int]) -> list[tuple[int, int]]:
        """
        Build an interleaved index with block-based sampling.

        Args:
            file_lengths: list of the number of examples in each file.

        Returns:
            An interleaved list of (file_idx, example_idx) tuples.
        """
        num_files = len(file_lengths)
        file_positions = [0] * num_files
        index = []

        all_exhausted = False
        while not all_exhausted:
            all_exhausted = True
            for file_idx in range(num_files):
                start_pos = file_positions[file_idx]
                if start_pos < file_lengths[file_idx]:
                    all_exhausted = False

                    # Determine the block size to sample from this file
                    remaining_examples = file_lengths[file_idx] - start_pos
                    examples_in_block = min(self.interleave_block_size, remaining_examples)

                    # Add examples from the block to the index
                    index.extend([(file_idx, start_pos + j) for j in range(examples_in_block)])

                    file_positions[file_idx] += examples_in_block
        return index

    def _shuffle_indices(self) -> None:
        """Shuffle the global index using Python's random module."""
        random.seed(self.random_seed)
        random.shuffle(self.global_index)
        # Update seed for the next shuffle to ensure different shuffling in subsequent epochs
        self.random_seed = random.randint(0, 2**32 - 1)  # noqa: S311

    def next_batch(self, batch_size) -> list[tuple[int, int]] | None:
        """
        Get the next batch of (file_idx, example_idx) tuples.

        Args:
            batch_size: The number of examples in the batch.

        Returns:
            A list of (file_idx, example_idx) tuples, or None if no more data is available.
        """
        if self.current_pos >= self.total_examples:
            # Check if we should start a new epoch
            if self.max_epochs == -1 or self.epoch_count < self.max_epochs - 1:
                self._reset_epoch()
            else:
                return None  # All epochs are complete

        # Determine the actual size of the next batch
        remaining_in_epoch = self.total_examples - self.current_pos
        actual_size: int = min(batch_size, remaining_in_epoch)

        if actual_size == 0:
            return None

        # If drop_remainder is true, and the batch is incomplete, skip it
        if self.drop_remainder and actual_size < batch_size:
            if self.max_epochs == -1 or self.epoch_count < self.max_epochs - 1:
                self._reset_epoch()
                # Recursively call to get the first full batch of the next epoch
                return self.next_batch(batch_size)
            return None

        # Collect and return the batch
        batch_start = self.current_pos
        batch_end = self.current_pos + actual_size
        batch = [self.global_index[i] for i in range(batch_start, batch_end)]
        self.current_pos += actual_size
        return batch

    def _reset_epoch(self):
        """Reset the sampler for a new epoch."""
        self.current_pos = 0
        self.epoch_count += 1
        if self.shuffle:
            self._shuffle_indices()

    def reset(self):
        """Reset the sampler to its initial state."""
        self.current_pos = 0
        self.epoch_count = 0
        self.global_index = list(self.original_index)
        if self.shuffle:
            self._shuffle_indices()

    def set_epoch(self, epoch: int):
        """
        Set the current epoch, useful for distributed training.

        Args:
            epoch: The epoch number to set.
        """
        self.epoch_count = epoch
        if self.shuffle:
            # Re-seed based on the original seed and epoch for reproducibility
            self.random_seed += epoch
            self._shuffle_indices()

    def __len__(self) -> int:
        """Return the total number of examples."""
        return self.total_examples

    @property
    def batches_per_epoch(self) -> int:
        """
        Approximation of batches per epoch.

        Note: This is dependent on batch_size which is not stored.
        Returning total examples is a more consistent measure.
        """
        return self.total_examples
