import struct
import mmap


def create_tfrecord_index(tfrecord_filename):
    """
    Creates an index file containing the byte offsets of each record in a TFRecord file.
    Args:
        tfrecord_filename (str): Path to the TFRecord file.
    """
    index = []
    with open(tfrecord_filename, 'rb') as f:
        while True:
            current_offset = f.tell()
            length_bytes = f.read(8)
            if not length_bytes:
                break  # Reached EOF
            length = struct.unpack('<Q', length_bytes)[0]  # Little-endian unsigned long long
            f.read(4)  # Skip CRC of length
            f.seek(length + 4, 1)  # Skip the data and CRC of data
            index.append(current_offset)

    return index


def create_tfrecord_index_mmap(tfrecord_filename):
    """
    Creates an index file using a memory-mapped file for large datasets.

    Args:
        tfrecord_filename (str): Path to the TFRecord file.
    """
    offsets = []
    with open(tfrecord_filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0
        while offset < mm.size():
            offsets.append(offset)
            length_bytes = mm[offset: offset + 8]
            if not length_bytes:
                break
            length = struct.unpack('<Q', length_bytes)[0]
            offset += 8 + 4 + length + 4  # length + length_crc + data + data_crc
        mm.close()

    return offsets