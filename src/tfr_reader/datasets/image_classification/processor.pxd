# cython: language_level=3
# distutils: language=c++

cimport numpy as cnp


cdef class ImageProcessor:
    cdef int target_width
    cdef int target_height
    cdef int num_threads
    cdef str image_feature_key
    cdef str label_feature_key

    cpdef dict decode_example(self, bytes example_bytes)
    cpdef cnp.ndarray decode_and_resize_image(self, bytes image_bytes)
    cpdef tuple process_batch(self, list raw_data_list)
