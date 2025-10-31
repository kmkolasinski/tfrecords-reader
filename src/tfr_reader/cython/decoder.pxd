# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int64_t


cdef class BytesList:
    cdef public list[bytes] value


cdef class FloatList:
    cdef public vector[float] value


cdef class Int64List:
    cdef public vector[int64_t] value


cdef class Feature:
    cdef str key
    cdef str kind
    cdef FloatList _float_list
    cdef Int64List _int64_list
    cdef BytesList _bytes_list


cdef class Features:
    cdef public dict[str, Feature] feature


cdef class Example:
    cdef public Features features


cpdef Example example_from_bytes(const unsigned char[:] buffer)
cdef Features features_from_bytes(const unsigned char* buffer, uint64_t length)
cdef Feature parse_map_entry(const unsigned char* buffer, uint64_t length)
cdef Feature feature_from_bytes(str key, const unsigned char* buffer, int64_t length)
cdef BytesList bytes_list_from_bytes(const unsigned char* buffer, int64_t length)
cdef FloatList float32_list_from_bytes(const unsigned char* buffer, int64_t length)
cdef Int64List int64_list_from_bytes(const unsigned char* buffer, int64_t length)
