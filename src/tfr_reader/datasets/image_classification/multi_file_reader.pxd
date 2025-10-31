# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t
from tfr_reader.cython.indexer cimport TFRecordFileReader, example_pointer_t


cdef class MultiFileReader:
    cdef list file_paths  # Python list instead of vector[str]
    cdef vector[vector[example_pointer_t]] indices
    cdef dict active_readers  # Python dict instead of C++ map for Python objects
    cdef vector[int] lru_queue
    cdef int max_open_files
    cdef bint save_index
    cdef int num_files

    cdef uint64_t get_total_examples(self)
    cdef TFRecordFileReader _get_reader(self, int file_idx)
    cdef void _update_lru(self, int file_idx)
    cdef void _evict_reader(self, int file_idx)

    cpdef list get_file_lengths(self)
    cpdef bytes get_example(self, int file_idx, int example_idx)
    cpdef list get_examples_batch(self, list indices)
