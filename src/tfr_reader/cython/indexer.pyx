# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++
# ##cython: linetrace=True

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_CUR, SEEK_SET, fwrite
from libc.stdint cimport uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import os


cdef struct example_pointer_t:
    # bytes offset contains the offset of the example in the TFRecord file
    uint64_t start
    uint64_t end
    uint64_t example_size



cdef class TFRecordFileReader:

    def __cinit__(self, str tfrecord_filepath, bool save_index = True):
        """
        Initializes the dataset with the TFRecord file and its offsets.

        Args:
            tfrecord_filepath: Path to the TFRecord file.
        """
        self.tfrecord_filepath = tfrecord_filepath
        self.pointers = self._create_or_load_index(tfrecord_filepath, save_index)
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

    cpdef list[example_pointer_t] get_pointers(self):
        """
        Returns the list of example pointers.
        """
        return [self.pointers[i] for i in range(self.pointers.size())]

    cdef vector[example_pointer_t] _create_or_load_index(self, str tfrecord_filepath, bool save_index):
        """
        Creates index for the TFRecord file or loads it from a file if it exists and is up-to-date.

        Args:
            tfrecord_filepath: Path to the TFRecord file
            save_index: Whether to save/load the index to/from disk

        Returns:
            vector[example_pointer_t]: The index pointers
        """
        cdef:
            str index_filepath
            vector[example_pointer_t] pointers
            bool index_valid = False

        if not save_index:
            # Don't use caching, just create the index
            return create_tfrecord_pointers_index(tfrecord_filepath)

        index_filepath = get_index_filepath(tfrecord_filepath)

        # Check if index file exists and is newer than the TFRecord file
        if os.path.exists(index_filepath):
            try:
                tfrecord_mtime = os.path.getmtime(tfrecord_filepath)
                index_mtime = os.path.getmtime(index_filepath)

                # Index is valid if it's newer than the TFRecord file
                if index_mtime >= tfrecord_mtime:
                    index_valid = True
            except OSError:
                index_valid = False

        if index_valid:
            # Try to load the index from file
            try:
                pointers = load_index_from_file(index_filepath)
                return pointers
            except (IOError, MemoryError) as e:
                # If loading fails, fall back to creating the index
                pass

        # Create the index from scratch
        pointers = create_tfrecord_pointers_index(tfrecord_filepath)

        # Save the index to file
        try:
            if not save_index_to_file(index_filepath, pointers):
                # Saving failed, but we still have the index in memory
                pass
        except Exception:
            # Ignore errors during saving
            pass

        return pointers

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

    cdef bytes get_example_fast(self, uint64_t idx):
        """
        Fast C-level method to retrieve the raw TFRecord at the specified index.
        This method is designed for use from other Cython code.
        """
        cdef example_pointer_t pointer
        cdef uint64_t initial_position
        cdef uint64_t offset
        cdef unsigned char * data
        cdef int ret

        if idx >= self.size():
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

    def get_example(self, uint64_t idx) -> bytes:
        """
        Retrieves the raw TFRecord at the specified index.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            bytes: The raw serialized record data.
        """
        if idx < 0 or idx >= self.size():
            raise IndexError("Index out of bounds")
        return self.get_example_fast(idx)

    def close(self):
        """
        Closes the TFRecord file.
        """
        if self.file:
            fclose(self.file)
            self.file = NULL


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

    # print("Creating index for TFRecord file: {}".format(tfrecord_filename))
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


cdef str get_index_filepath(str tfrecord_filepath):
    """Generate the index file path for a given TFRecord file."""
    return tfrecord_filepath + ".idx"


cdef bool save_index_to_file(str index_filepath, vector[example_pointer_t]& pointers):
    """Save the index to a binary file."""
    cdef:
        FILE *f
        size_t num_pointers = pointers.size()
        size_t written

    f = fopen(index_filepath.encode('utf-8'), b'wb')
    if not f:
        return False

    # Write the number of pointers
    written = fwrite(&num_pointers, sizeof(size_t), 1, f)
    if written != 1:
        fclose(f)
        return False

    # Write all pointers
    if num_pointers > 0:
        written = fwrite(&pointers[0], sizeof(example_pointer_t), num_pointers, f)
        if written != num_pointers:
            fclose(f)
            return False

    fclose(f)
    return True


cdef vector[example_pointer_t] load_index_from_file(str index_filepath):
    """Load the index from a binary file."""
    cdef:
        FILE *f
        size_t num_pointers = 0
        size_t num_read
        vector[example_pointer_t] pointers
        example_pointer_t* temp_array
        size_t i

    f = fopen(index_filepath.encode('utf-8'), b'rb')
    if not f:
        raise IOError("Cannot open index file: {}".format(index_filepath))

    # Read the number of pointers
    num_read = fread(&num_pointers, sizeof(size_t), 1, f)
    if num_read != 1:
        fclose(f)
        raise IOError("Failed to read index file header")

    # Read all pointers
    if num_pointers > 0:
        temp_array = <example_pointer_t*>malloc(num_pointers * sizeof(example_pointer_t))
        if not temp_array:
            fclose(f)
            raise MemoryError("Failed to allocate memory for index")

        num_read = fread(temp_array, sizeof(example_pointer_t), num_pointers, f)
        if num_read != num_pointers:
            free(temp_array)
            fclose(f)
            raise IOError("Failed to read index pointers")

        # Copy to vector
        for i in range(num_pointers):
            pointers.push_back(temp_array[i])

        free(temp_array)

    fclose(f)
    return pointers
