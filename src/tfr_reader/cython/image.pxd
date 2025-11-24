# cython: language_level=3
"""Cython header file for image module - allows efficient Cython-to-Cython calls."""

import numpy as np
cimport numpy as cnp

# C-level decode function (internal use)
cdef int decode_jpeg(
    const unsigned char* image_bytes,
    unsigned long image_size,
    unsigned char[:, :, ::1] output_array
) except -1 nogil


# Public cpdef functions that can be called from both Python and Cython
cpdef cnp.ndarray[cnp.uint8_t, ndim=4] decode_jpegs_to_array(
    list image_bytes_list,
    int target_height,
    int target_width,
    int target_channels=*
)
