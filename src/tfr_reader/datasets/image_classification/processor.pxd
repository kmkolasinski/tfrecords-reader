# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++

import cython
from tfr_reader.cython.decoder cimport Example, Feature


cdef class ImageProcessor:
    cdef public int target_width
    cdef public int target_height
    cdef public int num_threads
    cdef public str image_feature_key
    cdef public str label_feature_key
    cdef public str processing_backend

    @cython.locals(
        batch_size=cython.int,
        i=cython.int,
    )
    cpdef object cy_process_batch_opencv(self, raw_data_list)

    @cython.locals(
        batch_size=cython.int,
        i=cython.int,
    )
    cpdef object cy_process_batch_cython(self, raw_data_list)

    @cython.locals(
        example=Example,
        label_feature=Feature,
        image_feature=Feature,
        label=cython.int,
        image_data=bytes
    )
    cpdef dict decode_example(self, bytes example_bytes)
