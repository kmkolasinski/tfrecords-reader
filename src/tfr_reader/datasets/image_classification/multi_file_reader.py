"""
MultiFileReader: Efficient reading from multiple TFRecord files with file pooling.
"""

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tfr_reader.cython.indexer import TFRecordFileReader


class MultiFileReader:
    """
    Manages reading from multiple TFRecord files with efficient file handle pooling.

    Keeps only a limited number of files open at once using an LRU caching
    strategy, while maintaining all indices in memory for fast lookups.
    """

    def __init__(
        self,
        tfrecord_paths: list[str],
        max_open_files: int = 16,
        save_index: bool = True,
        *,
        max_workers: int | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the MultiFileReader.

        Args:
            tfrecord_paths: A list of paths to TFRecord files.
            max_open_files: The maximum number of file handles to keep open simultaneously.
            save_index: Whether to save/load indices to/from disk for faster startup.
            verbose: Whether to print status messages during initialization.
        """
        # LRU cache for active readers: OrderedDict is perfect for this
        self.active_readers: OrderedDict[int, TFRecordFileReader] = OrderedDict()

        if not tfrecord_paths:
            raise ValueError("tfrecord_paths cannot be empty.")
        if max_open_files < 1:
            raise ValueError("max_open_files must be at least 1.")

        self.file_paths = [str(p) for p in tfrecord_paths]
        self.max_open_files = max_open_files
        self.save_index = save_index
        self.num_files = len(self.file_paths)
        self.indices: list[list] = []
        self.verbose = verbose
        self.max_workers = max_workers
        self._load_all_indices()

    def _load_single_index(self, i: int, path: str) -> tuple[int, list, str, int]:
        """
        Load the index for a single TFRecord file.

        Args:
            i: The file index.
            path: The path to the TFRecord file.

        Returns:
            A tuple containing (file_index, index_data, filename, num_examples).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"TFRecord file not found: {path}")

        # A temporary reader is created just to load the index.
        # It will be closed automatically when it goes out of scope.
        temp_reader = TFRecordFileReader(path, self.save_index)
        file_index = temp_reader.get_pointers()
        return (i, file_index, Path(path).name, len(file_index))

    def _load_all_indices(self) -> None:
        """Load the indices for all TFRecord files into memory using parallel processing."""
        print(f"Loading indices for {self.num_files} files...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(
                lambda item: self._load_single_index(item[0], item[1]), enumerate(self.file_paths)
            )

            # Process results and build indices list
            for i, file_index, filename, num_examples in results:
                self.indices.append(file_index)
                if self.verbose:
                    print(f" [{i:>4}] {filename:<25} {num_examples:>10} examples")

    def get_total_examples(self) -> int:
        """Get the total number of examples across all files."""
        return sum(len(index) for index in self.indices)

    def get_file_lengths(self) -> list[int]:
        """Get the number of examples in each file."""
        return [len(index) for index in self.indices]

    def _get_reader(self, file_idx: int) -> TFRecordFileReader:
        """
        Get or create a reader for the specified file index using LRU caching.

        Args:
            file_idx: The index of the file to access.

        Returns:
            An active TFRecordFileReader instance.
        """
        if file_idx in self.active_readers:
            # Move the accessed reader to the end to mark it as most recently used
            self.active_readers.move_to_end(file_idx)
            return self.active_readers[file_idx]

        # If the cache is full, evict the least recently used reader
        if len(self.active_readers) >= self.max_open_files:
            # The popitem(last=False) method removes and returns the first item (LRU)
            oldest_idx, oldest_reader = self.active_readers.popitem(last=False)
            oldest_reader.close()

        # Open a new reader and add it to the cache
        path = self.file_paths[file_idx]
        reader = TFRecordFileReader(path, self.save_index)
        self.active_readers[file_idx] = reader
        return reader

    def get_example(self, file_idx: int, example_idx: int) -> bytes:
        """
        Retrieve a single raw TFRecord example.

        Args:
            file_idx: The index of the file.
            example_idx: The index of the example within that file.

        Returns:
            The raw TFRecord example as bytes.
        """
        if not 0 <= file_idx < self.num_files:
            raise IndexError(f"File index {file_idx} is out of range [0, {self.num_files}).")

        index_size = len(self.indices[file_idx])
        if not 0 <= example_idx < index_size:
            raise IndexError(f"Example index {example_idx} is out of range [0, {index_size}).")

        reader = self._get_reader(file_idx)
        return reader.get_example(example_idx)

    def get_examples_batch(self, indices: list[tuple[int, int]]) -> list[bytes]:
        """
        Retrieve a batch of examples efficiently.

        Args:
            indices: A list of (file_idx, example_idx) tuples.

        Returns:
            A list containing the raw TFRecord example bytes for each requested index.
        """
        return [self.get_example(file_idx, example_idx) for file_idx, example_idx in indices]

    def close_all(self):
        """Closes all currently open file handles."""
        for reader in self.active_readers.values():
            reader.close()
        self.active_readers.clear()

    def __len__(self) -> int:
        """Return the total number of examples across all files."""
        return self.get_total_examples()

    @property
    def num_open_files(self) -> int:
        """Return the number of currently open files."""
        return len(self.active_readers)

    def __del__(self):
        """Ensure all files are closed when the object is destroyed."""
        self.close_all()
