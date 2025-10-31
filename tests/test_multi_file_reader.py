"""Unit tests for MultiFileReader."""

import pytest

from tests import utils
from tfr_reader.datasets.image_classification.multi_file_reader import MultiFileReader


@pytest.fixture
def tfrecord_files(tmp_path):
    """Create multiple TFRecord files for testing."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"file_{i}.tfrecord"
        utils.write_image_tfrecord(file_path, num_records=10 + i * 5)
        files.append(str(file_path))
    return files


def test__multi_file_reader__initialization(tfrecord_files):
    """Test basic initialization."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)
    assert len(reader) == 10 + 15 + 20  # 45 total


def test__multi_file_reader__get_file_lengths(tfrecord_files):
    """Test getting file lengths."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)
    lengths = reader.get_file_lengths()
    assert lengths == [10, 15, 20]


def test__multi_file_reader__get_example(tfrecord_files):
    """Test reading a single example."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    # Read first example from first file
    example_bytes = reader.get_example(0, 0)
    assert isinstance(example_bytes, bytes)
    assert len(example_bytes) > 0

    # Read example from second file
    example_bytes = reader.get_example(1, 5)
    assert isinstance(example_bytes, bytes)
    assert len(example_bytes) > 0


def test__multi_file_reader__get_examples_batch(tfrecord_files):
    """Test reading a batch of examples."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    # Read batch from different files
    indices = [(0, 0), (0, 1), (1, 0), (2, 0)]
    batch = reader.get_examples_batch(indices)

    assert len(batch) == 4
    assert all(isinstance(example, bytes) for example in batch)


def test__multi_file_reader__file_pooling(tfrecord_files):
    """Test that file pooling limits open files."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    # Initially no files open
    assert reader.num_open_files == 0

    # Read from file 0
    reader.get_example(0, 0)
    assert reader.num_open_files == 1

    # Read from file 1
    reader.get_example(1, 0)
    assert reader.num_open_files == 2

    # Read from file 2 - should evict file 0 (LRU)
    reader.get_example(2, 0)
    assert reader.num_open_files == 2  # Still max 2


def test__multi_file_reader__invalid_file_index(tfrecord_files):
    """Test error handling for invalid file index."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    with pytest.raises(IndexError):
        reader.get_example(10, 0)


def test__multi_file_reader__invalid_example_index(tfrecord_files):
    """Test error handling for invalid example index."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    with pytest.raises(IndexError):
        reader.get_example(0, 1000)


def test__multi_file_reader__close_all(tfrecord_files):
    """Test closing all file handles."""
    reader = MultiFileReader(tfrecord_files, max_open_files=2, save_index=False)

    # Open some files
    reader.get_example(0, 0)
    reader.get_example(1, 0)
    assert reader.num_open_files > 0

    # Close all
    reader.close_all()
    assert reader.num_open_files == 0


def test__multi_file_reader__nonexistent_file(tmp_path):
    """Test error handling for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        MultiFileReader([str(tmp_path / "nonexistent.tfrecord")], save_index=False)


def test__multi_file_reader__empty_file_list():
    """Test error handling for empty file list."""
    with pytest.raises(ValueError):
        MultiFileReader([], save_index=False)
