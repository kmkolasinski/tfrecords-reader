from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_CUR, SEEK_SET
from libc.stdint cimport uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free


def create_tfrecord_offsets_index(str tfrecord_filename) -> list[int]:
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


cdef class TFRecordFileReader:

    cdef str tfrecord_filepath
    cdef list[int] offsets
    cdef FILE* file

    def __cinit__(self, str tfrecord_filepath):
        """
        Initializes the dataset with the TFRecord file and its offsets.

        Args:
            tfrecord_filepath: Path to the TFRecord file.
        """
        self.tfrecord_filepath = tfrecord_filepath
        self.offsets = create_tfrecord_offsets_index(tfrecord_filepath)
        self.file = fopen(tfrecord_filepath.encode('utf-8'), b'rb')
        if not self.file:
            raise IOError(f"Cannot open file: {tfrecord_filepath}")

    def __dealloc__(self):
        """
        Ensures the file is closed when the object is deallocated.
        """
        if self.file:
            fclose(self.file)

    def __len__(self) -> int:
        """
        Returns the number of records in the dataset.
        """
        return len(self.offsets)

    def get_offset(self, int idx) -> int:
        """
        Retrieves the offset of the record at the specified index.

        Args:
            idx: The index of the record.

        Returns:
            int: The offset of the record.
        """
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("Index out of bounds")
        return self.offsets[idx]

    def __getitem__(self, int idx) -> bytes:
        """
        Retrieves the raw TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            bytes: The raw serialized record data.
        """
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("Index out of bounds")

        cdef uint64_t offset = self.offsets[idx]
        cdef uint64_t initial_position
        cdef unsigned char length_bytes[8]
        cdef uint64_t length
        cdef unsigned char* data
        cdef int ret

        initial_position = ftell(self.file)

        # Seek to the record's offset
        ret = fseek(self.file, offset, SEEK_CUR)
        if ret != 0:
            raise IOError("Failed to seek to the record offset")

        # Read the length of the record
        if fread(length_bytes, 1, 8, self.file) != 8:
            raise IOError("Failed to read length bytes")

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
        ret = fseek(self.file, 4, SEEK_CUR)
        if ret != 0:
            raise IOError("Failed to skip length CRC")

        # Read the record data
        data = <unsigned char*>malloc(length)
        if not data:
            raise MemoryError("Failed to allocate memory for record data")
        if fread(data, 1, length, self.file) != length:
            free(data)
            raise IOError("Failed to read record data")

        # Skip data CRC (4 bytes)
        ret = fseek(self.file, 4, SEEK_CUR)
        if ret != 0:
            free(data)
            raise IOError("Failed to skip data CRC")

        # Convert the data to a Python bytes object
        py_data = PyBytes_FromStringAndSize(<char*>data, length)
        free(data)

        # Reset the file pointer to the initial position
        ret = fseek(self.file, initial_position, SEEK_SET)
        if ret != 0:
            raise IOError("Failed to reset file pointer to the initial position")
        return py_data

    def close(self):
        """
        Closes the TFRecord file.
        """
        if self.file:
            fclose(self.file)
            self.file = NULL
