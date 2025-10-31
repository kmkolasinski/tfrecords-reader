# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# distutils: language=c++

"""
BatchSampler: Handles shuffling, interleaving, and batch sampling logic.
"""

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from libc.stdint cimport uint64_t
from cython cimport nogil
import random


cdef class BatchSampler:
    """
    Manages batch sampling with shuffling, interleaving, and repeat logic.

    Generates batches of (file_idx, example_idx) tuples for efficient
    multi-file reading.
    """

    def __init__(
        self,
        list file_lengths,  # Number of examples in each file
        bint shuffle=True,
        bint interleave=True,
        int repeat=1,
        int interleave_block_size=32,
        bint drop_remainder=False,
        seed=None
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
            self.random_seed = <unsigned int>time(NULL)
        else:
            self.random_seed = <unsigned int>seed

        # Build global index
        self._build_global_index(file_lengths)

        # Store original for reset
        self.original_index = self.global_index

        # Initial shuffle if requested
        if self.shuffle:
            self._shuffle_indices()

    cdef void _build_global_index(self, list file_lengths):
        """
        Build the global index of (file_idx, example_idx) pairs.

        Args:
            file_lengths: List of number of examples in each file
        """
        cdef int file_idx
        cdef int example_idx
        cdef int num_examples

        self.global_index.clear()

        if self.interleave:
            # Block interleaving: sample blocks from each file in round-robin
            self._build_interleaved_index(file_lengths)
        else:
            # Sequential: all examples from file 0, then file 1, etc.
            for file_idx in range(len(file_lengths)):
                num_examples = file_lengths[file_idx]
                for example_idx in range(num_examples):
                    self.global_index.push_back(pair[int, int](file_idx, example_idx))

        # Set total after building index
        self.total_examples = self.global_index.size()

    cdef void _build_interleaved_index(self, list file_lengths):
        """
        Build interleaved index with block-based sampling.

        Args:
            file_lengths: List of number of examples in each file
        """
        cdef int num_files = len(file_lengths)
        cdef vector[int] file_positions  # Current position in each file
        cdef int file_idx
        cdef int i, j
        cdef int block_count
        cdef bint all_exhausted = False
        cdef int examples_in_block

        # Initialize positions
        for i in range(num_files):
            file_positions.push_back(0)

        # Round-robin through files, sampling blocks
        while not all_exhausted:
            all_exhausted = True

            for file_idx in range(num_files):
                if file_positions[file_idx] < file_lengths[file_idx]:
                    all_exhausted = False

                    # Sample a block from this file
                    examples_in_block = min(
                        self.interleave_block_size,
                        file_lengths[file_idx] - file_positions[file_idx]
                    )

                    for j in range(examples_in_block):
                        self.global_index.push_back(
                            pair[int, int](file_idx, file_positions[file_idx] + j)
                        )

                    file_positions[file_idx] += examples_in_block

    cdef void _shuffle_indices(self):
        """Shuffle the global index using Fisher-Yates algorithm."""
        cdef uint64_t i, j
        cdef pair[int, int] temp
        cdef uint64_t n = self.global_index.size()

        # Use the stored seed for reproducibility
        srand(self.random_seed)

        # Fisher-Yates shuffle
        for i in range(n - 1, 0, -1):
            j = rand() % (i + 1)
            # Swap
            temp = self.global_index[i]
            self.global_index[i] = self.global_index[j]
            self.global_index[j] = temp

        # Update seed for next shuffle
        self.random_seed = rand()

    cpdef list next_batch(self, int batch_size):
        """
        Get the next batch of (file_idx, example_idx) tuples.

        Args:
            batch_size: Number of examples in the batch

        Returns:
            List of (file_idx, example_idx) tuples, or None if no more data
        """
        cdef uint64_t actual_size
        cdef uint64_t i
        cdef list batch = []

        # Check if we've exhausted the current epoch
        if self.current_pos >= self.total_examples:
            # Handle repeat logic
            if self.max_epochs == -1:  # Infinite repeat
                self._reset_epoch()
            elif self.epoch_count < self.max_epochs - 1:
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
            if self.max_epochs == -1:  # Infinite repeat
                self._reset_epoch()
                return self.next_batch(batch_size)
            elif self.epoch_count < self.max_epochs - 1:
                self._reset_epoch()
                return self.next_batch(batch_size)
            else:
                return None  # No more data

        # Collect batch
        for i in range(actual_size):
            file_idx = self.global_index[self.current_pos + i].first
            example_idx = self.global_index[self.current_pos + i].second
            batch.append((file_idx, example_idx))

        self.current_pos += actual_size

        return batch

    cdef void _reset_epoch(self):
        """Reset to the beginning of a new epoch."""
        self.current_pos = 0
        self.epoch_count += 1

        if self.shuffle:
            self._shuffle_indices()

    cpdef void reset(self):
        """Reset the sampler to the beginning."""
        self.current_pos = 0
        self.epoch_count = 0
        self.global_index = self.original_index

        if self.shuffle:
            self._shuffle_indices()

    cpdef void set_epoch(self, int epoch):
        """
        Set the current epoch (useful for distributed training).

        Args:
            epoch: Epoch number
        """
        self.epoch_count = epoch
        if self.shuffle:
            # Re-seed based on epoch for reproducibility
            self.random_seed = <unsigned int>(self.random_seed + epoch)
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
