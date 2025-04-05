from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_CUR
from libc.stdint cimport uint64_t

def create_tfrecord_index(str tfrecord_filename):
    cdef:
        list index = []
        FILE *f
        uint64_t current_offset, length
        cdef unsigned char length_bytes[8]
        size_t num_read
        int ret

    # Open the file using C's fopen
    f = fopen(tfrecord_filename.encode('utf-8'), b'rb')
    if not f:
        raise IOError("Cannot open file: {}".format(tfrecord_filename))

    while True:
        # Get the current offset
        current_offset = ftell(f)

        # Read 8 bytes for the length
        num_read = fread(length_bytes, 1, 8, f)
        if num_read != 8:
            break  # Reached EOF or error

        # Unpack length_bytes into uint64_t length (little-endian)
        length = 0
        length |= length_bytes[0]
        length |= length_bytes[1] << 8
        length |= length_bytes[2] << 16
        length |= length_bytes[3] << 24
        length |= length_bytes[4] << 32
        length |= length_bytes[5] << 40
        length |= length_bytes[6] << 48
        length |= length_bytes[7] << 56

        # Skip length CRC (4 bytes)
        ret = fseek(f, 4, SEEK_CUR)
        if ret != 0:
            break  # Error in seeking

        # Append the current offset to the index
        index.append(int(current_offset))

        # Skip data and data CRC
        ret = fseek(f, int(length) + 4, SEEK_CUR)
        if ret != 0:
            break  # Error in seeking

    # Close the file
    fclose(f)
    return index