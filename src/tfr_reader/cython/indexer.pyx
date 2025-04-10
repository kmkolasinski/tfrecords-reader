# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++
# ##cython: linetrace=True

from libcpp.vector cimport vector
from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_CUR, SEEK_SET
from libc.stdint cimport uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


cdef struct example_pointer_t:
    # bytes offset contains the offset of the example in the TFRecord file
    uint64_t start
    uint64_t end
    uint64_t example_size


cdef uint64_t unpack_bytes(unsigned char[8] length_bytes):
    """
    Unpack length_bytes into uint64_t length (little-endian)
    """
    cdef uint64_t value = 0
    memcpy(&value, &length_bytes[0], sizeof(value))
    return value

cdef vector[example_pointer_t] create_tfrecord_pointers_index(str tfrecord_filename):
    cdef:
        vector[example_pointer_t] pointers
        FILE *f
        uint64_t start, end, length
        unsigned char length_bytes[8]
        size_t num_read
        int ret

    f = fopen(tfrecord_filename.encode('utf-8'), b'rb')
    if not f:
        raise IOError("Cannot open file: {}".format(tfrecord_filename))

    while True:
        # Get the current offset
        start = ftell(f)

        # Read 8 bytes for the length
        num_read = fread(length_bytes, 1, 8, f)
        if num_read != 8:
            break  # Reached EOF or error

        # Unpack length_bytes into uint64_t length (little-endian)
        length = unpack_bytes(length_bytes)

        # Skip length CRC (4 bytes)
        ret = fseek(f, 4, SEEK_CUR)
        if ret != 0:
            break  # Error in seeking

        # end is 4 + 4 + 8 + length
        end = start + 4 + 4 + 8 + length
        pointers.push_back(example_pointer_t(start, end, length))

        # Skip data and data CRC
        ret = fseek(f, length + 4, SEEK_CUR)
        if ret != 0:
            break  # Error in seeking

    fclose(f)
    return pointers


cdef class TFRecordFileReader:

    cdef str tfrecord_filepath
    cdef vector[example_pointer_t] pointers
    cdef FILE* file

    def __cinit__(self, str tfrecord_filepath):
        """
        Initializes the dataset with the TFRecord file and its offsets.

        Args:
            tfrecord_filepath: Path to the TFRecord file.
        """
        self.tfrecord_filepath = tfrecord_filepath
        self.pointers = create_tfrecord_pointers_index(tfrecord_filepath)
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
        return self.pointers.size()

    cdef size_t size(self):
        """
        Returns the number of records in the dataset.
        """
        return self.pointers.size()

    cpdef example_pointer_t get_pointer(self, uint64_t idx):
        """
        Retrieves the offset of the record at the specified index.

        Args:
            idx: The index of the record.

        Returns:
            int: The offset of the record.
        """
        if idx < 0 or idx >= self.size():
            raise IndexError("Index out of bounds")
        return self.pointers[idx]

    def get_example(self, uint64_t idx) -> bytes:
        """
        Retrieves the raw TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            bytes: The raw serialized record data.
        """
        cdef example_pointer_t pointer
        cdef uint64_t initial_position
        cdef uint64_t offset
        cdef unsigned char * data
        cdef int ret

        if idx < 0 or idx >= self.size():
            raise IndexError("Index out of bounds")
        pointer = self.pointers[idx]

        initial_position = ftell(self.file)

        # Seek to the record's offset
        offset = pointer.start + 8 + 4  # Skip length bytes and length CRC
        ret = fseek(self.file, offset, SEEK_CUR)
        if ret != 0:
            raise IOError("Failed to seek to the record offset")

        # Read the record data
        data = <unsigned char *> malloc(pointer.example_size)
        if not data:
            raise MemoryError("Failed to allocate memory for record data")
        if fread(data, 1, pointer.example_size, self.file) != pointer.example_size:
            free(data)
            raise IOError("Failed to read record data")

        # Skip data CRC (4 bytes)
        ret = fseek(self.file, 4, SEEK_CUR)
        if ret != 0:
            free(data)
            raise IOError("Failed to skip data CRC")

        # Convert the data to a Python bytes object
        py_data = PyBytes_FromStringAndSize(<char *> data, pointer.example_size)
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
