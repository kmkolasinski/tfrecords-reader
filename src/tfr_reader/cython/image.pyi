"""Type stubs for tfr_reader.cython.image module."""

import numpy as np
from numpy.typing import NDArray

def decode_jpeg_to_array(image_bytes: bytes, output_array: NDArray[np.uint8]) -> int:
    """
    Decodes a JPEG image from bytes and writes it to the provided numpy array.

    If the source image dimensions don't match the output array, the image
    will be resized using nearest-neighbor interpolation.

    Parameters
    ----------
    image_bytes : bytes
        JPEG image as bytes
    output_array : NDArray[np.uint8]
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

def decode_jpegs_to_array(
    image_bytes_list: list[bytes], target_height: int, target_width: int, target_channels: int = 3
) -> NDArray[np.uint8]:
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
    NDArray[np.uint8]
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
