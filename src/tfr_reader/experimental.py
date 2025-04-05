import mmap
import struct


def create_tfrecord_index(tfrecord_filename):
    """Creates an index file containing the byte offsets of each record in a TFRecord file.

    Args:
        tfrecord_filename (str): Path to the TFRecord file.
    """
    index = []
    with open(tfrecord_filename, "rb") as f:
        while True:
            current_offset = f.tell()
            length_bytes = f.read(8)
            if not length_bytes:
                break  # Reached EOF
            length = struct.unpack("<Q", length_bytes)[0]  # Little-endian unsigned long long
            f.read(4)  # Skip CRC of length
            f.seek(length + 4, 1)  # Skip the data and CRC of data
            index.append(current_offset)

    return index


def create_tfrecord_index_mmap(tfrecord_filename):
    """Creates an index file using a memory-mapped file for large datasets.

    Args:
        tfrecord_filename (str): Path to the TFRecord file.
    """
    offsets = []
    with open(tfrecord_filename, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0
        while offset < mm.size():
            offsets.append(offset)
            length_bytes = mm[offset : offset + 8]
            if not length_bytes:
                break
            length = struct.unpack("<Q", length_bytes)[0]
            offset += 8 + 4 + length + 4  # length + length_crc + data + data_crc
        mm.close()

    return offsets


class MyTFRecordDatasetMMap:
    def __init__(self, tfrecord_filename, index):
        self.tfrecord_filename = tfrecord_filename
        self.index = index
        self.file = open(tfrecord_filename, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.index):
            raise IndexError("Index out of bounds")
        offset = int(self.index[idx])
        length_bytes = self.mm[offset : offset + 8]
        length = struct.unpack("<Q", length_bytes)[0]
        data_offset = offset + 8 + 4  # Skip length and length_crc
        data = self.mm[data_offset : data_offset + length]
        return bytes(data)

    def close(self):
        self.mm.close()
        self.file.close()


class MyTFRecordDatasetGCS:
    def __init__(self, tfrecord_filename, index):
        """Initializes the dataset to work with TFRecords stored in GCS.

        Args:
            tfrecord_filename (str): GCS path to the TFRecord file (e.g., 'gs://bucket/path/to/file.tfrecord').
            index (list): List of byte offsets for each record in the TFRecord file.
        """
        self.tfrecord_filename = tfrecord_filename
        self.index = index
        # Use tf.io.gfile.GFile to open the file from GCS
        import tensorflow as tf

        self.file = tf.io.gfile.GFile(tfrecord_filename, "rb")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """Retrieves the raw TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            bytes: The raw serialized record data.
        """
        if idx < 0 or idx >= len(self.index):
            raise IndexError("Index out of bounds")
        offset = int(self.index[idx])
        self.file.seek(offset)
        # Read length (8 bytes)
        length_bytes = self.file.read(8)
        if not length_bytes or len(length_bytes) < 8:  # noqa: PLR2004
            raise OSError(f"Failed to read length bytes at offset {offset}")
        length = struct.unpack("<Q", length_bytes)[0]
        # Skip length CRC (4 bytes)
        self.file.read(4)
        # Read data
        data = self.file.read(length)
        if not data or len(data) < length:
            raise OSError(f"Failed to read data at offset {offset}")
        # Skip data CRC (4 bytes)
        self.file.read(4)
        return data

    def close(self):
        """Closes the GCS file."""
        self.file.close()
