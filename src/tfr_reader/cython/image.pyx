# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++
# distutils: extra_compile_args=-march=native -mtune=native -O3 -ffast-math -funroll-loops -fopt-info-vec-optimized -fopt-info-vec-missed -fopt-info-loop-optimized
# ##cython: linetrace=True


from libc.string cimport memcpy, memset
from libc.stdint cimport uint64_t, int64_t, uint8_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from cython.parallel cimport prange

import numpy as np
cimport numpy as cnp

# SIMD intrinsics for x86_64
cdef extern from "emmintrin.h" nogil:
    ctypedef struct __m128i:
        pass

    __m128i _mm_loadu_si128(const __m128i* p)
    void _mm_storeu_si128(__m128i* p, __m128i a)
    __m128i _mm_set1_epi8(char a)

# libjpeg-turbo C declarations
cdef extern from "turbojpeg.h" nogil:
    ctypedef void* tjhandle

    # Error handling
    char* tjGetErrorStr2(tjhandle handle)

    # Pixel formats
    enum TJPF:
        TJPF_RGB
        TJPF_BGR
        TJPF_RGBX
        TJPF_BGRX
        TJPF_XBGR
        TJPF_XRGB
        TJPF_GRAY

    # Subsampling options
    enum TJSAMP:
        TJSAMP_444
        TJSAMP_422
        TJSAMP_420
        TJSAMP_GRAY
        TJSAMP_440
        TJSAMP_411

    # Flags
    enum:
        TJFLAG_FASTDCT
        TJFLAG_FASTUPSAMPLE

    # Initialize decompressor
    tjhandle tjInitDecompress()

    # Destroy decompressor
    int tjDestroy(tjhandle handle)

    # Decompress JPEG header to get image dimensions
    int tjDecompressHeader3(
        tjhandle handle,
        const unsigned char* jpegBuf,
        unsigned long jpegSize,
        int* width,
        int* height,
        int* jpegSubsamp,
        int* jpegColorspace
    )

    # Decompress JPEG to buffer
    int tjDecompress2(
        tjhandle handle,
        const unsigned char* jpegBuf,
        unsigned long jpegSize,
        unsigned char* dstBuf,
        int width,
        int pitch,
        int height,
        int pixelFormat,
        int flags
    )

    # Scale image during decompression
    # Get scaling factors
    ctypedef struct tjscalingfactor:
        int num
        int denom

    tjscalingfactor* tjGetScalingFactors(int* numscalingfactors)


cdef void resize_image(
    const unsigned char* src_image,
    int src_width,
    int src_height,
    unsigned char[:, :, ::1] output_array,
    int target_width,
    int target_height,
    int target_channels
) noexcept nogil:
    """
    Highly optimized image resize using nearest-neighbor interpolation with SIMD operations.

    Parameters
    ----------
    src_image : const unsigned char*
        Source image buffer (height, width, channels)
    src_width : int
        Source image width
    src_height : int
        Source image height
    output_array : unsigned char[:, :, ::1]
        Output array memory view with shape (target_height, target_width, target_channels)
    target_width : int
        Target width
    target_height : int
        Target height
    target_channels : int
        Number of channels (1 or 3)
    """
    cdef int y, x, x_block
    cdef int src_x, src_y
    cdef int src_stride = src_width * target_channels
    cdef int src_offset, row_base
    cdef float scale_x, scale_y
    cdef int* src_x_coords
    cdef int* src_offsets
    cdef const unsigned char* src_row
    cdef unsigned char* dst_row
    cdef unsigned char* dst_ptr
    cdef const unsigned char* src_ptr

    # Pre-compute all source x coordinates and offsets for better cache locality
    src_x_coords = <int*>malloc(target_width * sizeof(int))
    src_offsets = <int*>malloc(target_width * sizeof(int))

    if src_x_coords == NULL or src_offsets == NULL:
        free(src_x_coords)
        free(src_offsets)
        return

    # Pre-compute scale factors and x-coordinates with offsets
    scale_x = <float>src_width / <float>target_width
    scale_y = <float>src_height / <float>target_height

    for x in range(target_width):
        src_x = <int>(x * scale_x)
        if src_x >= src_width:
            src_x = src_width - 1
        src_x_coords[x] = src_x
        src_offsets[x] = src_x * target_channels

    # Process row by row for better cache locality
    if target_channels == 3:
        # Highly optimized RGB path with explicit pointer arithmetic
        for y in range(target_height):
            src_y = <int>(y * scale_y)
            if src_y >= src_height:
                src_y = src_height - 1

            src_row = src_image + src_y * src_stride
            dst_row = &output_array[y, 0, 0]

            # Process pixels with manual loop unrolling
            x = 0
            # Process 8 pixels at a time for better throughput
            while x + 7 < target_width:
                # Pixel 0
                src_ptr = src_row + src_offsets[x]
                dst_ptr = dst_row + x * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 1
                src_ptr = src_row + src_offsets[x + 1]
                dst_ptr = dst_row + (x + 1) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 2
                src_ptr = src_row + src_offsets[x + 2]
                dst_ptr = dst_row + (x + 2) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 3
                src_ptr = src_row + src_offsets[x + 3]
                dst_ptr = dst_row + (x + 3) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 4
                src_ptr = src_row + src_offsets[x + 4]
                dst_ptr = dst_row + (x + 4) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 5
                src_ptr = src_row + src_offsets[x + 5]
                dst_ptr = dst_row + (x + 5) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 6
                src_ptr = src_row + src_offsets[x + 6]
                dst_ptr = dst_row + (x + 6) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                # Pixel 7
                src_ptr = src_row + src_offsets[x + 7]
                dst_ptr = dst_row + (x + 7) * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]

                x += 8

            # Handle remaining pixels
            while x < target_width:
                src_ptr = src_row + src_offsets[x]
                dst_ptr = dst_row + x * 3
                dst_ptr[0] = src_ptr[0]
                dst_ptr[1] = src_ptr[1]
                dst_ptr[2] = src_ptr[2]
                x += 1
    else:
        # Optimized grayscale path
        for y in range(target_height):
            src_y = <int>(y * scale_y)
            if src_y >= src_height:
                src_y = src_height - 1

            src_row = src_image + src_y * src_stride
            dst_row = &output_array[y, 0, 0]

            # Process with loop unrolling for grayscale
            x = 0
            while x + 15 < target_width:
                dst_row[x] = src_row[src_offsets[x]]
                dst_row[x + 1] = src_row[src_offsets[x + 1]]
                dst_row[x + 2] = src_row[src_offsets[x + 2]]
                dst_row[x + 3] = src_row[src_offsets[x + 3]]
                dst_row[x + 4] = src_row[src_offsets[x + 4]]
                dst_row[x + 5] = src_row[src_offsets[x + 5]]
                dst_row[x + 6] = src_row[src_offsets[x + 6]]
                dst_row[x + 7] = src_row[src_offsets[x + 7]]
                dst_row[x + 8] = src_row[src_offsets[x + 8]]
                dst_row[x + 9] = src_row[src_offsets[x + 9]]
                dst_row[x + 10] = src_row[src_offsets[x + 10]]
                dst_row[x + 11] = src_row[src_offsets[x + 11]]
                dst_row[x + 12] = src_row[src_offsets[x + 12]]
                dst_row[x + 13] = src_row[src_offsets[x + 13]]
                dst_row[x + 14] = src_row[src_offsets[x + 14]]
                dst_row[x + 15] = src_row[src_offsets[x + 15]]
                x += 16

            while x < target_width:
                dst_row[x] = src_row[src_offsets[x]]
                x += 1

    free(src_x_coords)
    free(src_offsets)


cdef int decode_jpeg(
    const unsigned char* image_bytes,
    unsigned long image_size,
    unsigned char[:, :, ::1] output_array
) except -1 nogil:
    """
    Decode JPEG image from bytes and write to output array with automatic resizing.

    Parameters
    ----------
    image_bytes : const unsigned char*
        Pointer to JPEG image bytes
    image_size : unsigned long
        Size of the JPEG image in bytes
    output_array : unsigned char[:, :, ::1]
        Memory view of output numpy array with shape (height, width, channels)

    Returns
    -------
    int
        0 on success, -1 on error
    """
    cdef tjhandle tj_handle = NULL
    cdef int ret = 0
    cdef int src_width, src_height, subsamp, colorspace
    cdef int target_height, target_width, target_channels
    cdef unsigned char* temp_buffer = NULL
    cdef unsigned char* row_src
    cdef unsigned char* row_dst
    cdef int x, y, c
    cdef float scale_x, scale_y
    cdef int src_x, src_y
    cdef int pixel_format

    # Get target dimensions from output array
    target_height = output_array.shape[0]
    target_width = output_array.shape[1]
    target_channels = output_array.shape[2]

    # Initialize TurboJPEG decompressor
    tj_handle = tjInitDecompress()
    if tj_handle == NULL:
        return -1

    # Get source image dimensions
    ret = tjDecompressHeader3(
        tj_handle,
        image_bytes,
        image_size,
        &src_width,
        &src_height,
        &subsamp,
        &colorspace
    )

    if ret != 0:
        tjDestroy(tj_handle)
        return -1

    # Determine pixel format based on target channels
    if target_channels == 3:
        pixel_format = TJPF_RGB
    elif target_channels == 1:
        pixel_format = TJPF_GRAY
    else:
        tjDestroy(tj_handle)
        return -1

    # Check if we need to resize
    if src_width == target_width and src_height == target_height:
        # Direct decompression to output buffer
        ret = tjDecompress2(
            tj_handle,
            image_bytes,
            image_size,
            &output_array[0, 0, 0],
            target_width,
            target_width * target_channels,  # pitch
            target_height,
            pixel_format,
            TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE
        )
    else:
        # Decompress to temporary buffer, then resize
        temp_buffer = <unsigned char*>malloc(src_width * src_height * target_channels)
        if temp_buffer == NULL:
            tjDestroy(tj_handle)
            return -1

        ret = tjDecompress2(
            tj_handle,
            image_bytes,
            image_size,
            temp_buffer,
            src_width,
            src_width * target_channels,  # pitch
            src_height,
            pixel_format,
            TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE
        )

        if ret == 0:
            # Use optimized resize function
            resize_image(
                temp_buffer,
                src_width,
                src_height,
                output_array,
                target_width,
                target_height,
                target_channels
            )

        free(temp_buffer)

    tjDestroy(tj_handle)

    if ret != 0:
        return -1

    return 0


cpdef cnp.ndarray[cnp.uint8_t, ndim=4] decode_jpegs_to_array(
    list image_bytes_list,
    int target_height,
    int target_width,
    int target_channels=3
):
    """
    Decodes a list of JPEG images from bytes list and returns a numpy array
    containing all decoded images resized to (target_height, target_width).

    Parameters
    ----------
    image_bytes_list : list[bytes]
        List of JPEG images as bytes
    target_height : int
        Target height for all decoded images
    target_width : int
        Target width for all decoded images
    target_channels : int, optional
        Number of channels (1 for grayscale, 3 for RGB), default is 3

    Returns
    -------
    np.ndarray[uint8, ndim=4]
        Output array with shape (num_images, target_height, target_width, target_channels)

    Raises
    ------
    ValueError
        If target_channels is not 1 or 3, or if image_bytes_list is empty
    RuntimeError
        If any JPEG decoding fails

    Examples
    --------
    >>> image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    >>> image_bytes_list = [open(f, 'rb').read() for f in image_files]
    >>> batch = decode_jpegs_to_array(image_bytes_list, 224, 224, 3)
    >>> assert batch.shape == (3, 224, 224, 3)
    """
    cdef int num_images = len(image_bytes_list)
    cdef int i
    cdef bytes img_bytes
    cdef const unsigned char* img_ptr
    cdef unsigned long img_size
    cdef int result

    if num_images == 0:
        raise ValueError("image_bytes_list cannot be empty")

    if target_channels not in (1, 3):
        raise ValueError(f"target_channels must be 1 or 3, got {target_channels}")

    # Allocate output array for all images
    cdef cnp.ndarray[cnp.uint8_t, ndim=4] output_array = np.zeros(
        (num_images, target_height, target_width, target_channels),
        dtype=np.uint8
    )

    # Create memory view for efficient access
    cdef unsigned char[:, :, :, ::1] output_view = output_array

    # Pre-extract pointers and sizes from Python objects before going parallel
    cdef const unsigned char** img_ptrs = <const unsigned char**>malloc(num_images * sizeof(unsigned char*))
    cdef unsigned long* img_sizes = <unsigned long*>malloc(num_images * sizeof(unsigned long))
    cdef int* results = <int*>malloc(num_images * sizeof(int))

    if img_ptrs == NULL or img_sizes == NULL or results == NULL:
        free(img_ptrs)
        free(img_sizes)
        free(results)
        raise MemoryError("Failed to allocate memory for parallel decoding")

    # Extract all pointers and sizes while we still have the GIL
    for i in range(num_images):
        img_bytes = image_bytes_list[i]
        img_ptrs[i] = <const unsigned char*>(<char*>img_bytes)
        img_sizes[i] = len(img_bytes)
        results[i] = 0

    # Decode all images in parallel
    with nogil:
        for i in prange(num_images, schedule='dynamic'):
            results[i] = decode_jpeg(img_ptrs[i], img_sizes[i], output_view[i])

    # Check for errors after parallel section
    cdef int first_error_idx = -1
    for i in range(num_images):
        if results[i] != 0:
            first_error_idx = i
            break

    # Clean up
    free(img_ptrs)
    free(img_sizes)
    free(results)

    if first_error_idx >= 0:
        raise RuntimeError(f"Failed to decode JPEG image at index {first_error_idx}")

    return output_array


def decode_jpeg_to_array(bytes image_bytes, cnp.ndarray[cnp.uint8_t, ndim=3] output_array):
    """
    Python wrapper for decode_jpeg function.

    Decodes a JPEG image from bytes and writes it to the provided numpy array.
    If the source image dimensions don't match the output array, the image
    will be resized using nearest-neighbor interpolation.

    Parameters
    ----------
    image_bytes : bytes
        JPEG image as bytes
    output_array : np.ndarray[uint8, ndim=3]
        Output numpy array with shape (height, width, channels)
        Must be contiguous and writable.
        Supported channels: 1 (grayscale) or 3 (RGB)

    Returns
    -------
    int
        0 on success, -1 on error

    Raises
    ------
    ValueError
        If output_array is not contiguous or has invalid shape
    RuntimeError
        If JPEG decoding fails

    Examples
    --------
    >>> import numpy as np
    >>> with open('image.jpg', 'rb') as f:
    ...     jpg_bytes = f.read()
    >>> output = np.zeros((224, 224, 3), dtype=np.uint8)
    >>> result = decode_jpeg_to_array(jpg_bytes, output)
    >>> assert result == 0
    >>> # output now contains the resized RGB image
    """
    if not output_array.flags.c_contiguous:
        raise ValueError("output_array must be C-contiguous")

    if output_array.shape[2] not in (1, 3):
        raise ValueError(f"output_array must have 1 or 3 channels, got {output_array.shape[2]}")

    cdef const unsigned char* img_ptr = <const unsigned char*>(<char*>image_bytes)
    cdef unsigned long img_size = len(image_bytes)
    cdef unsigned char[:, :, ::1] output_view = output_array

    cdef int result
    with nogil:
        result = decode_jpeg(img_ptr, img_size, output_view)

    if result != 0:
        raise RuntimeError("Failed to decode JPEG image")

    return result
