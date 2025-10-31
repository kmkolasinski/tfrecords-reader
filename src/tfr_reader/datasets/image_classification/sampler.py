"""
BatchSampler: Handles shuffling, interleaving, and batch sampling logic.

This file uses Cython's pure Python mode for compilation.
Type checkers may report errors for Cython-specific syntax.
"""

import random

import cython

if cython.compiled:
    from cython.cimports.libc.stdint import uint64_t
    from cython.cimports.libc.stdlib import rand, srand
    from cython.cimports.libc.time import time
    from cython.cimports.libcpp.pair import pair
    from cython.cimports.libcpp.vector import vector
else:
    # Fallback types for non-compiled mode
    vector = list
    pair = tuple
    uint64_t = int


@cython.cclass
class BatchSampler:
    """
    Manages batch sampling with shuffling, interleaving, and repeat logic.

    Generates batches of (file_idx, example_idx) tuples for efficient
    multi-file reading.
    """

    global_index: vector[pair[cython.int, cython.int]]
    original_index: vector[pair[cython.int, cython.int]]
    current_pos: uint64_t
    epoch_count: cython.int
    max_epochs: cython.int
    shuffle: cython.bint
    interleave: cython.bint
    drop_remainder: cython.bint
    interleave_block_size: cython.int
    total_examples: uint64_t
    random_seed: cython.uint

    def __init__(  # noqa: PLR0913
        self,
        file_lengths: list,
        shuffle: cython.bint = True,
        interleave: cython.bint = True,
        repeat: cython.int = 1,
        interleave_block_size: cython.int = 32,
        drop_remainder: cython.bint = False,
        seed=None,
    ):
        """
        Initialize the BatchSampler.

        Args:
            file_lengths: List of number of examples in each file
            shuffle: Whether to shuffle the indices
            interleave: Whether to interleave across files
            repeat: Number of epochs (-1 for infinite, 1 for single pass)
            interleave_block_size: Block size for interleaving (default: 32)
            drop_remainder: Whether to drop the last batch if it contains fewer
                           elements than batch_size (default: False)
            seed: Random seed for reproducibility (None for random)
        """
        self.shuffle = shuffle
        self.interleave = interleave
        self.drop_remainder = drop_remainder
        self.max_epochs = repeat
        self.epoch_count = 0
        self.current_pos = 0
        self.interleave_block_size = interleave_block_size

        # Set random seed
        if seed is None:
            if cython.compiled:
                self.random_seed = cython.cast(cython.uint, time(cython.NULL))
            else:
                import time as time_module

                self.random_seed = int(time_module.time())
        else:
            self.random_seed = cython.cast(cython.uint, seed)

        # Build global index
        self._build_global_index(file_lengths)

        # Store original for reset
        if cython.compiled:
            self.original_index = self.global_index
        else:
            self.original_index = list(self.global_index)

        # Initial shuffle if requested
        if self.shuffle:
            self._shuffle_indices()

    @cython.cfunc
    @cython.locals(file_idx=cython.int, example_idx=cython.int, num_examples=cython.int)
    def _build_global_index(self, file_lengths: list):
        """
        Build the global index of (file_idx, example_idx) pairs.

        Args:
            file_lengths: List of number of examples in each file
        """
        if cython.compiled:
            self.global_index.clear()
        else:
            self.global_index = []

        if self.interleave:
            # Block interleaving: sample blocks from each file in round-robin
            self._build_interleaved_index(file_lengths)
        else:
            # Sequential: all examples from file 0, then file 1, etc.
            for file_idx in range(len(file_lengths)):
                num_examples = file_lengths[file_idx]
                for example_idx in range(num_examples):
                    if cython.compiled:
                        self.global_index.push_back(
                            pair[cython.int, cython.int](file_idx, example_idx)
                        )
                    else:
                        self.global_index.append((file_idx, example_idx))

        # Set total after building index
        if cython.compiled:
            self.total_examples = self.global_index.size()
        else:
            self.total_examples = len(self.global_index)

    @cython.cfunc
    @cython.locals(
        num_files=cython.int,
        file_idx=cython.int,
        i=cython.int,
        j=cython.int,
        all_exhausted=cython.bint,
        examples_in_block=cython.int,
    )
    def _build_interleaved_index(self, file_lengths: list):
        """
        Build interleaved index with block-based sampling.

        Args:
            file_lengths: List of number of examples in each file
        """
        num_files = len(file_lengths)
        if cython.compiled:
            file_positions: vector[cython.int] = vector[cython.int]()
        else:
            file_positions = []

        # Initialize positions
        for _i in range(num_files):
            if cython.compiled:
                file_positions.push_back(0)
            else:
                file_positions.append(0)

        # Round-robin through files, sampling blocks
        all_exhausted = False
        while not all_exhausted:
            all_exhausted = True

            for file_idx in range(num_files):
                if file_positions[file_idx] < file_lengths[file_idx]:
                    all_exhausted = False

                    # Sample a block from this file
                    examples_in_block = min(
                        self.interleave_block_size,
                        file_lengths[file_idx] - file_positions[file_idx],
                    )

                    for j in range(examples_in_block):
                        if cython.compiled:
                            self.global_index.push_back(
                                pair[cython.int, cython.int](file_idx, file_positions[file_idx] + j)
                            )
                        else:
                            self.global_index.append((file_idx, file_positions[file_idx] + j))

                    file_positions[file_idx] += examples_in_block

    @cython.cfunc
    @cython.locals(i=uint64_t, j=uint64_t, n=uint64_t)
    def _shuffle_indices(self):
        """Shuffle the global index using Fisher-Yates algorithm."""
        if cython.compiled:
            n = self.global_index.size()
            # Use the stored seed for reproducibility
            srand(self.random_seed)

            # Fisher-Yates shuffle
            temp: pair[cython.int, cython.int]
            for i in range(n - 1, 0, -1):
                j = rand() % (i + 1)
                # Swap
                temp = self.global_index[i]
                self.global_index[i] = self.global_index[j]
                self.global_index[j] = temp

            # Update seed for next shuffle
            self.random_seed = rand()
        else:
            # Use Python's random module
            random.seed(self.random_seed)
            random.shuffle(self.global_index)
            self.random_seed = random.randint(0, 2**32 - 1)  # noqa: S311

    @cython.ccall
    @cython.locals(actual_size=uint64_t, i=uint64_t, file_idx=cython.int, example_idx=cython.int)
    def next_batch(self, batch_size: cython.int) -> list:
        """
        Get the next batch of (file_idx, example_idx) tuples.

        Args:
            batch_size: Number of examples in the batch

        Returns:
            List of (file_idx, example_idx) tuples, or None if no more data
        """
        batch = []

        # Check if we've exhausted the current epoch
        if self.current_pos >= self.total_examples:
            # Handle repeat logic
            if self.max_epochs == -1 or self.epoch_count < self.max_epochs - 1:  # Infinite repeat
                self._reset_epoch()
            else:
                return None  # No more data

        # Determine actual batch size
        actual_size = min(batch_size, self.total_examples - self.current_pos)

        if actual_size == 0:
            return None

        # Drop remainder: skip incomplete batches
        if self.drop_remainder and actual_size < batch_size:
            # Move to next epoch or return None
            if self.max_epochs == -1 or self.epoch_count < self.max_epochs - 1:  # Infinite repeat
                self._reset_epoch()
                return self.next_batch(batch_size)
            return None  # No more data

        # Collect batch
        for i in range(actual_size):
            if cython.compiled:
                file_idx = self.global_index[self.current_pos + i].first
                example_idx = self.global_index[self.current_pos + i].second
            else:
                file_idx, example_idx = self.global_index[self.current_pos + i]
            batch.append((file_idx, example_idx))

        self.current_pos += actual_size

        return batch

    @cython.cfunc
    def _reset_epoch(self):
        """Reset to the beginning of a new epoch."""
        self.current_pos = 0
        self.epoch_count += 1

        if self.shuffle:
            self._shuffle_indices()

    @cython.ccall
    def reset(self):
        """Reset the sampler to the beginning."""
        self.current_pos = 0
        self.epoch_count = 0
        if cython.compiled:
            self.global_index = self.original_index
        else:
            self.global_index = list(self.original_index)

        if self.shuffle:
            self._shuffle_indices()

    @cython.ccall
    def set_epoch(self, epoch: cython.int):
        """
        Set the current epoch (useful for distributed training).

        Args:
            epoch: Epoch number
        """
        self.epoch_count = epoch
        if self.shuffle:
            # Re-seed based on epoch for reproducibility
            self.random_seed = cython.cast(cython.uint, self.random_seed + epoch)
            self._shuffle_indices()

    def __len__(self):
        """Return the total number of examples."""
        return self.total_examples

    @property
    def batches_per_epoch(self):
        """Return the number of batches per epoch (for a given batch size)."""
        # Note: This requires knowing batch_size, which is not stored here
        # Return total examples instead
        return self.total_examples
