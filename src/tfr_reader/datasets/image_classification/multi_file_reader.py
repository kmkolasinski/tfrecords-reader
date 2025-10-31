"""
MultiFileReader: Efficient reading from multiple TFRecord files with file pooling.
"""

import os

import cython

if cython.compiled:
    from cython.cimports.libc.stdint import uint64_t
    from cython.cimports.libcpp.vector import vector
    from cython.cimports.tfr_reader.cython.indexer import (
        TFRecordFileReader,
        example_pointer_t,
    )
else:
    from tfr_reader.cython.indexer import TFRecordFileReader, example_pointer_t

    vector = list
    uint64_t = int


@cython.cclass
class MultiFileReader:
    """
    Manages reading from multiple TFRecord files with efficient file handle pooling.

    Keeps only a limited number of files open at once using LRU caching,
    while maintaining all indices in memory for fast lookups.
    """

    file_paths: list
    indices: vector[vector[example_pointer_t]]
    active_readers: dict
    lru_queue: vector[cython.int]
    max_open_files: cython.int
    save_index: cython.bint
    num_files: cython.int

    def __init__(
        self,
        tfrecord_paths: list,
        max_open_files: cython.int = 16,
        save_index: cython.bint = True,
    ):
        """
        Initialize the MultiFileReader.

        Args:
            tfrecord_paths: List of paths to TFRecord files
            max_open_files: Maximum number of files to keep open simultaneously
            save_index: Whether to save/load indices to/from disk
        """
        i: cython.int
        path: str
        temp_reader: TFRecordFileReader
        file_index: vector[example_pointer_t]

        if not tfrecord_paths:
            raise ValueError("tfrecord_paths cannot be empty")

        if max_open_files < 1:
            raise ValueError("max_open_files must be at least 1")

        self.max_open_files = max_open_files
        self.save_index = save_index
        self.num_files = len(tfrecord_paths)
        self.active_readers = {}  # Python dict for storing readers
        self.file_paths = []  # Python list for file paths

        if cython.compiled:
            self.lru_queue = vector[cython.int]()
        else:
            self.lru_queue = []
            self.indices = []

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
            if cython.compiled:
                file_index = temp_reader.pointers
                self.indices.push_back(file_index)
            else:
                file_index = list(temp_reader.pointers)
                self.indices.append(file_index)

            # Reader will be closed when it goes out of scope
            # We'll open files on-demand later
            print(f"  File {i}: {len(file_index)} examples")

        print(f"Total examples: {self.get_total_examples()}")

    @cython.cfunc
    @cython.locals(total=uint64_t, i=cython.int)
    def get_total_examples(self) -> uint64_t:
        """Get the total number of examples across all files."""
        total = 0

        for i in range(self.num_files):
            if cython.compiled:
                total += self.indices[i].size()
            else:
                total += len(self.indices[i])

        return total

    @cython.ccall
    @cython.locals(i=cython.int)
    def get_file_lengths(self) -> list:
        """
        Get the number of examples in each file.

        Returns:
            List of example counts
        """
        lengths = []

        for i in range(self.num_files):
            if cython.compiled:
                lengths.append(self.indices[i].size())
            else:
                lengths.append(len(self.indices[i]))

        return lengths

    @cython.cfunc
    @cython.locals(evict_idx=cython.int)
    def _get_reader(self, file_idx: cython.int) -> TFRecordFileReader:
        """
        Get or create a reader for the specified file index.
        Implements LRU caching of file handles.

        Args:
            file_idx: Index of the file

        Returns:
            TFRecordFileReader instance
        """
        reader: TFRecordFileReader

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
        if cython.compiled:
            self.lru_queue.push_back(file_idx)
        else:
            self.lru_queue.append(file_idx)

        return reader

    @cython.cfunc
    @cython.locals(i=cython.int)
    def _update_lru(self, file_idx: cython.int):
        """
        Update LRU queue to mark file as most recently used.

        Args:
            file_idx: Index of the file
        """
        if cython.compiled:
            new_queue: vector[cython.int] = vector[cython.int]()

            # Remove file_idx from queue and re-add at end
            for i in range(self.lru_queue.size()):
                if self.lru_queue[i] != file_idx:
                    new_queue.push_back(self.lru_queue[i])

            new_queue.push_back(file_idx)
            self.lru_queue = new_queue
        else:
            # Python fallback
            if file_idx in self.lru_queue:
                self.lru_queue.remove(file_idx)
            self.lru_queue.append(file_idx)

    @cython.cfunc
    @cython.locals(i=cython.int)
    def _evict_reader(self, file_idx: cython.int):
        """
        Evict a reader from the active pool.

        Args:
            file_idx: Index of the file to evict
        """
        # Close the reader (Python will handle cleanup)
        if file_idx in self.active_readers:
            self.active_readers[file_idx].close()
            del self.active_readers[file_idx]

        # Remove from LRU queue
        if cython.compiled:
            new_queue: vector[cython.int] = vector[cython.int]()
            for i in range(self.lru_queue.size()):
                if self.lru_queue[i] != file_idx:
                    new_queue.push_back(self.lru_queue[i])
            self.lru_queue = new_queue
        elif file_idx in self.lru_queue:
            self.lru_queue.remove(file_idx)

    @cython.ccall
    def get_example(self, file_idx: cython.int, example_idx: cython.int) -> bytes:
        """
        Get a raw TFRecord example from a specific file.

        Args:
            file_idx: Index of the file
            example_idx: Index of the example within the file

        Returns:
            Raw TFRecord example bytes
        """
        reader: TFRecordFileReader

        if file_idx < 0 or file_idx >= self.num_files:
            raise IndexError(f"File index {file_idx} out of range [0, {self.num_files})")

        index_size = (
            self.indices[file_idx].size() if cython.compiled else len(self.indices[file_idx])
        )
        if example_idx < 0 or example_idx >= index_size:
            raise IndexError(f"Example index {example_idx} out of range [0, {index_size})")

        # Get reader (may open file or use cached)
        reader = self._get_reader(file_idx)

        # Read the example
        return reader.get_example_fast(example_idx)

    @cython.ccall
    @cython.locals(file_idx=cython.int, example_idx=cython.int)
    def get_examples_batch(self, indices: list) -> list:
        """
        Get a batch of examples efficiently.

        Args:
            indices: List of (file_idx, example_idx) tuples

        Returns:
            List of raw TFRecord example bytes
        """
        batch = []
        example_bytes: bytes

        for file_idx, example_idx in indices:
            example_bytes = self.get_example(file_idx, example_idx)
            batch.append(example_bytes)

        return batch

    def close_all(self):
        """Close all open file handles."""
        file_idx: cython.int

        # Check if already cleaned up
        if self.active_readers is None:
            return

        # Close all active readers
        for file_idx in list(self.active_readers.keys()):
            self.active_readers[file_idx].close()

        self.active_readers.clear()
        if cython.compiled:
            self.lru_queue.clear()
        else:
            self.lru_queue = []

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
