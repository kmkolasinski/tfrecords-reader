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
        interleave_block_size: int = 16,
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
        self.file_lengths = file_lengths

        # Initial shuffle if requested
        if self.shuffle:
            self.shuffle_indices()

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
        Build a global index that iterates through several files in an
        interleaved round-robin fashion, but at most `interleave_block_size`
        files are active at the same time.

        Args:
            file_lengths: Number of examples contained in every file (file 0, file 1 ...).

        Returns: A list of (file_id, position_inside_file) pairs describing the
            global ordering.
        """
        if self.interleave_block_size <= 0:
            raise ValueError("`interleave_block_size` must be a positive integer")

        num_files = len(file_lengths)
        # Pointer to the next record that must be read for every file
        offsets = [0] * num_files

        # Files currently participating in the round-robin
        active: list[int] = []
        next_file_to_activate = 0

        # Activate the first `interleave_block_size` files
        while len(active) < self.interleave_block_size and next_file_to_activate < num_files:
            active.append(next_file_to_activate)
            next_file_to_activate += 1

        result: list[tuple[int, int]] = []

        while active:
            finished: list[int] = []

            # One "round": produce one record from every active file
            for file_id in active:
                pos = offsets[file_id]
                if pos >= file_lengths[file_id]:
                    # Already exhausted - mark for removal
                    finished.append(file_id)
                    continue

                # Emit current record and advance file pointer
                result.append((file_id, pos))
                offsets[file_id] += 1

                # If that was the last record of this file, mark for removal
                if offsets[file_id] == file_lengths[file_id]:
                    finished.append(file_id)

            # Remove files that ran out of data
            if finished:
                active = [f for f in active if f not in finished]

            # Activate new files until the active set is full again
            while len(active) < self.interleave_block_size and next_file_to_activate < num_files:
                active.append(next_file_to_activate)
                next_file_to_activate += 1

        return result

    def shuffle_indices(self) -> None:
        """
        Shuffle the global index by shuffling example indices within each file.

        This approach maintains file locality, improving performance by reducing
        file switching overhead compared to shuffling across all files.
        """
        random.seed(self.random_seed)

        # Create shuffled index mappings for each file
        new_examples_order: list[list[int]] = []
        for num_examples in self.file_lengths:
            file_indices = list(range(num_examples))
            random.shuffle(file_indices)
            new_examples_order.append(file_indices)

        # Remap the global index using the shuffled file indices
        for i, (file_id, old_example_idx) in enumerate(self.global_index):
            new_example_idx = new_examples_order[file_id][old_example_idx]
            self.global_index[i] = (file_id, new_example_idx)

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
        actual_size = min(batch_size, remaining_in_epoch)

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
            self.shuffle_indices()

    def reset(self):
        """Reset the sampler to its initial state."""
        self.current_pos = 0
        self.epoch_count = 0
        self.global_index = list(self.original_index)
        if self.shuffle:
            self.shuffle_indices()

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
            self.shuffle_indices()

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
