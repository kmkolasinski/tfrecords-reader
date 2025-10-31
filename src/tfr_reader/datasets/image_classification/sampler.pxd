# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t


cdef class BatchSampler:
    cdef vector[pair[int, int]] global_index
    cdef vector[pair[int, int]] original_index
    cdef uint64_t current_pos
    cdef int epoch_count
    cdef int max_epochs
    cdef bint shuffle
    cdef bint interleave
    cdef bint drop_remainder
    cdef int interleave_block_size
    cdef uint64_t total_examples
    cdef unsigned int random_seed

    cdef void _build_global_index(self, list file_lengths)
    cdef void _build_interleaved_index(self, list file_lengths)
    cdef void _shuffle_indices(self)
    cdef void _reset_epoch(self)

    cpdef list next_batch(self, int batch_size)
    cpdef void reset(self)
    cpdef void set_epoch(self, int epoch)
