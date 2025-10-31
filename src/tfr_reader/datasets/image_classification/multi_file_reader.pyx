# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# distutils: language=c++

"""
MultiFileReader: Efficient reading from multiple TFRecord files with file pooling.
"""

from libcpp.vector cimport vector
from libc.stdint cimport uint64_t
from tfr_reader.cython.indexer cimport TFRecordFileReader, example_pointer_t
import os


cdef class MultiFileReader:
    """
    Manages reading from multiple TFRecord files with efficient file handle pooling.

    Keeps only a limited number of files open at once using LRU caching,
    while maintaining all indices in memory for fast lookups.
    """

    def __init__(
        self,
        list tfrecord_paths,
        int max_open_files=16,
        bint save_index=True
    ):
        """
        Initialize the MultiFileReader.

        Args:
            tfrecord_paths: List of paths to TFRecord files
            max_open_files: Maximum number of files to keep open simultaneously
            save_index: Whether to save/load indices to/from disk
        """
        cdef int i
        cdef str path
        cdef TFRecordFileReader temp_reader
        cdef vector[example_pointer_t] file_index

        if not tfrecord_paths:
            raise ValueError("tfrecord_paths cannot be empty")

        if max_open_files < 1:
            raise ValueError("max_open_files must be at least 1")

        self.max_open_files = max_open_files
        self.save_index = save_index
        self.num_files = len(tfrecord_paths)
        self.active_readers = {}  # Python dict for storing readers
        self.file_paths = []  # Python list for file paths

        # Store file paths
        for path in tfrecord_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"TFRecord file not found: {path}")
            self.file_paths.append(path)

        # Load all indices (lightweight - just offsets, not full data)
        print(f"Loading indices for {self.num_files} files...")
        for i in range(self.num_files):
            # Create temporary reader just to load the index
            temp_reader = TFRecordFileReader(self.file_paths[i], save_index)

            # Copy the index
            file_index = temp_reader.pointers
            self.indices.push_back(file_index)

            # Reader will be closed when it goes out of scope
            # We'll open files on-demand later
            print(f"  File {i}: {len(file_index)} examples")

        print(f"Total examples: {self.get_total_examples()}")

    cdef uint64_t get_total_examples(self):
        """Get the total number of examples across all files."""
        cdef uint64_t total = 0
        cdef int i

        for i in range(self.num_files):
            total += self.indices[i].size()

        return total

    cpdef list get_file_lengths(self):
        """
        Get the number of examples in each file.

        Returns:
            List of example counts
        """
        cdef list lengths = []
        cdef int i

        for i in range(self.num_files):
            lengths.append(self.indices[i].size())

        return lengths

    cdef TFRecordFileReader _get_reader(self, int file_idx):
        """
        Get or create a reader for the specified file index.
        Implements LRU caching of file handles.

        Args:
            file_idx: Index of the file

        Returns:
            TFRecordFileReader instance
        """
        cdef TFRecordFileReader reader
        cdef int evict_idx

        # Check if reader is already active
        if file_idx in self.active_readers:
            # Move to end of LRU queue (most recently used)
            self._update_lru(file_idx)
            return self.active_readers[file_idx]

        # Need to open a new file
        # First check if we need to evict
        if len(self.active_readers) >= self.max_open_files:
            # Evict least recently used
            evict_idx = self.lru_queue[0]
            self._evict_reader(evict_idx)

        # Open new reader
        reader = TFRecordFileReader(self.file_paths[file_idx], self.save_index)
        self.active_readers[file_idx] = reader
        self.lru_queue.push_back(file_idx)

        return reader

    cdef void _update_lru(self, int file_idx):
        """
        Update LRU queue to mark file as most recently used.

        Args:
            file_idx: Index of the file
        """
        cdef int i
        cdef vector[int] new_queue

        # Remove file_idx from queue and re-add at end
        for i in range(self.lru_queue.size()):
            if self.lru_queue[i] != file_idx:
                new_queue.push_back(self.lru_queue[i])

        new_queue.push_back(file_idx)
        self.lru_queue = new_queue

    cdef void _evict_reader(self, int file_idx):
        """
        Evict a reader from the active pool.

        Args:
            file_idx: Index of the file to evict
        """
        cdef vector[int] new_queue
        cdef int i

        # Close the reader (Python will handle cleanup)
        if file_idx in self.active_readers:
            self.active_readers[file_idx].close()
            del self.active_readers[file_idx]

        # Remove from LRU queue
        for i in range(self.lru_queue.size()):
            if self.lru_queue[i] != file_idx:
                new_queue.push_back(self.lru_queue[i])

        self.lru_queue = new_queue

    cpdef bytes get_example(self, int file_idx, int example_idx):
        """
        Get a raw TFRecord example from a specific file.

        Args:
            file_idx: Index of the file
            example_idx: Index of the example within the file

        Returns:
            Raw TFRecord example bytes
        """
        cdef TFRecordFileReader reader

        if file_idx < 0 or file_idx >= self.num_files:
            raise IndexError(f"File index {file_idx} out of range [0, {self.num_files})")

        if example_idx < 0 or example_idx >= self.indices[file_idx].size():
            raise IndexError(
                f"Example index {example_idx} out of range [0, {self.indices[file_idx].size()})"
            )

        # Get reader (may open file or use cached)
        reader = self._get_reader(file_idx)

        # Read the example
        return reader.get_example_fast(example_idx)

    cpdef list get_examples_batch(self, list indices):
        """
        Get a batch of examples efficiently.

        Args:
            indices: List of (file_idx, example_idx) tuples

        Returns:
            List of raw TFRecord example bytes
        """
        cdef list batch = []
        cdef int file_idx, example_idx
        cdef bytes example_bytes

        for file_idx, example_idx in indices:
            example_bytes = self.get_example(file_idx, example_idx)
            batch.append(example_bytes)

        return batch

    def close_all(self):
        """Close all open file handles."""
        cdef int file_idx

        # Check if already cleaned up
        if self.active_readers is None:
            return

        # Close all active readers
        for file_idx in list(self.active_readers.keys()):
            self.active_readers[file_idx].close()

        self.active_readers.clear()
        self.lru_queue.clear()

    def __dealloc__(self):
        """Cleanup when object is destroyed."""
        self.close_all()

    def __len__(self):
        """Return total number of examples across all files."""
        return self.get_total_examples()

    @property
    def num_open_files(self):
        """Return the number of currently open files."""
        return len(self.active_readers)
