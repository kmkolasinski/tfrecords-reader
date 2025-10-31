# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport FILE
from libc.stdint cimport uint64_t


cdef struct example_pointer_t:
    uint64_t start
    uint64_t end
    uint64_t example_size


cdef class TFRecordFileReader:
    cdef str tfrecord_filepath
    cdef vector[example_pointer_t] pointers
    cdef FILE* file

    cdef size_t size(self)
    cdef vector[example_pointer_t] _create_or_load_index(self, str tfrecord_filepath, bool save_index)
    cpdef example_pointer_t get_pointer(self, uint64_t idx)
    cdef bytes get_example_fast(self, uint64_t idx)
