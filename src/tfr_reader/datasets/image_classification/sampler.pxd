# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++

import cython
from cython.cimports.libc.stdint cimport uint64_t, int64_t
from cython.cimports.libcpp.vector cimport vector
from cython.cimports.libcpp.pair cimport pair

cdef class BatchSampler:
    cdef readonly list[tuple[int, int]] global_index
    cdef readonly list[tuple[int, int]] original_index
    cdef readonly list[int] file_lengths
    cdef public uint64_t current_pos
    cdef public int epoch_count
    cdef readonly int max_epochs
    cdef readonly bint shuffle
    cdef readonly bint interleave
    cdef readonly bint drop_remainder
    cdef readonly int interleave_block_size
    cdef public uint64_t total_examples
    cdef public unsigned int random_seed

    cdef list _build_global_index(self, list file_lengths)
    cdef list _build_interleaved_index(self, list file_lengths)
    cpdef void shuffle_indices(self)
    cdef void _reset_epoch(self)

    @cython.locals(max_epochs=uint64_t, remaining_in_epoch=uint64_t, actual_size=uint64_t)
    cpdef object next_batch(self, uint64_t batch_size)
    cpdef void reset(self)
