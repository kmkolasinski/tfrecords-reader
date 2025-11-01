import os
from pathlib import Path

import pytest

from tests import utils
from tfr_reader.cython.indexer import TFRecordFileReader

NUM_RECORDS = 10


@pytest.fixture
def tfrecord_file(tmp_path):
    """Create a temporary TFRecord file for testing."""
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    utils.write_dummy_tfrecord(dummy_tfrecord_path, NUM_RECORDS)
    return str(dummy_tfrecord_path)


def test__index_caching_saves_to_disk(tfrecord_file: str):
    """Test that index is saved to disk when save_index=True."""
    index_file = tfrecord_file + ".idx"

    # Ensure no index file exists initially
    if os.path.exists(index_file):
        Path(index_file).unlink()

    # Create reader with save_index=True
    reader = TFRecordFileReader(tfrecord_file, save_index=True)
    num_records = len(reader)
    del reader

    # Verify index file was created
    assert os.path.exists(index_file), "Index file should be created"
    assert num_records == NUM_RECORDS


def test__index_caching_not_saved_when_disabled(tfrecord_file: str):
    """Test that index is not saved to disk when save_index=False."""
    index_file = tfrecord_file + ".idx"

    # Ensure no index file exists initially
    if os.path.exists(index_file):
        Path(index_file).unlink()

    # Create reader with save_index=False
    reader = TFRecordFileReader(tfrecord_file, save_index=False)
    num_records = len(reader)
    del reader

    # Verify index file was not created
    assert not os.path.exists(index_file), "Index file should not be created"
    assert num_records == NUM_RECORDS


def test__index_caching_loads_from_disk(tfrecord_file: str):
    """Test that index is loaded from disk on subsequent reads."""
    index_file = tfrecord_file + ".idx"

    # Ensure no index file exists initially
    if os.path.exists(index_file):
        Path(index_file).unlink()

    # First load - creates index
    reader1 = TFRecordFileReader(tfrecord_file, save_index=True)
    num_records1 = len(reader1)
    del reader1

    # Verify index file exists
    assert os.path.exists(index_file)
    index_mtime = Path(index_file).stat().st_mtime

    # Second load - should load from cache
    reader2 = TFRecordFileReader(tfrecord_file, save_index=True)
    num_records2 = len(reader2)
    del reader2

    # Verify index file was not modified (loaded from cache)
    assert Path(index_file).stat().st_mtime == index_mtime
    assert num_records1 == num_records2 == NUM_RECORDS


def test__cached_and_uncached_return_same_results(tfrecord_file: str):
    """Test that cached and uncached indexing return the same number of records."""
    # Create reader with caching
    reader_cached = TFRecordFileReader(tfrecord_file, save_index=True)
    num_cached = len(reader_cached)
    del reader_cached

    # Create reader without caching
    reader_uncached = TFRecordFileReader(tfrecord_file, save_index=False)
    num_uncached = len(reader_uncached)
    del reader_uncached

    assert num_cached == num_uncached == NUM_RECORDS
