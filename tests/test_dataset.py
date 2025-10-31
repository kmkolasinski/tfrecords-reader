"""Unit tests for TFRecordsImageDataset."""

import numpy as np
import pytest

from tests import utils
from tfr_reader.datasets.image_classification.dataset import TFRecordsImageDataset


@pytest.fixture
def tfrecord_files(tmp_path):
    """Create multiple TFRecord files for testing."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"file_{i}.tfrecord"
        utils.write_image_tfrecord(file_path, num_records=10)
        files.append(str(file_path))
    return files


def test__tfrecords_image_dataset__initialization(tfrecord_files):
    """Test basic initialization."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=8,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    assert len(dataset) == 30  # 3 files x 10 examples
    assert dataset.batches_per_epoch == 4  # ceil(30/8)


def test__tfrecords_image_dataset__iteration_single_epoch(tfrecord_files):
    """Test iterating through a single epoch."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    batches = []
    for images, labels in dataset:
        assert images.shape[1:] == (64, 64, 3)
        assert images.dtype == np.uint8
        assert labels.dtype == np.int32
        batches.append((images, labels))

    # Should have 3 batches (10, 10, 10)
    assert len(batches) == 3
    assert batches[0][0].shape[0] == 10
    assert batches[1][0].shape[0] == 10
    assert batches[2][0].shape[0] == 10


def test__tfrecords_image_dataset__partial_batch(tfrecord_files):
    """Test handling of partial batches."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=20,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    batches = list(dataset)

    # Should have 2 batches (20, 10)
    assert len(batches) == 2
    assert batches[0][0].shape[0] == 20
    assert batches[1][0].shape[0] == 10


def test__tfrecords_image_dataset__shuffle(tfrecord_files):
    """Test shuffling."""
    dataset1 = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=30,
        num_threads=2,
        shuffle=True,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        seed=42,
        save_index=False,
    )

    dataset2 = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=30,
        num_threads=2,
        shuffle=True,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        seed=43,  # Different seed
        save_index=False,
    )

    batch1 = next(iter(dataset1))
    batch2 = next(iter(dataset2))

    # Different seeds should give different order
    # (not guaranteed to be different, but very likely with 30 examples)
    labels1 = batch1[1]
    labels2 = batch2[1]
    assert not np.array_equal(labels1, labels2)


def test__tfrecords_image_dataset__interleave(tfrecord_files):
    """Test interleaved file reading."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=30,
        num_threads=2,
        shuffle=False,
        interleave_files=True,
        repeat=1,
        prefetch=0,
        interleave_block_size=5,
        save_index=False,
    )

    # Just verify it runs without errors
    batch = next(iter(dataset))
    assert batch[0].shape == (30, 64, 64, 3)


def test__tfrecords_image_dataset__repeat_infinite(tfrecord_files):
    """Test infinite repeat."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=-1,
        prefetch=0,
        save_index=False,
    )

    # Should be able to iterate beyond dataset size
    batches_collected = 0
    for _ in dataset:
        batches_collected += 1
        if batches_collected >= 10:  # 10 batches = >3 epochs
            break

    assert batches_collected == 10


def test__tfrecords_image_dataset__repeat_multiple_epochs(tfrecord_files):
    """Test finite repeat."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=2,
        prefetch=0,
        save_index=False,
    )

    batches = list(dataset)

    # Should have 6 batches (2 epochs x 3 batches)
    assert len(batches) == 6


def test__tfrecords_image_dataset__reset(tfrecord_files):
    """Test reset functionality."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    # Get first batch
    batch1 = next(iter(dataset))

    # Reset and get first batch again
    dataset.reset()
    batch2 = next(iter(dataset))

    # Should be identical
    np.testing.assert_array_equal(batch1[0], batch2[0])
    np.testing.assert_array_equal(batch1[1], batch2[1])


def test__tfrecords_image_dataset__different_input_sizes(tfrecord_files):
    """Test different input sizes."""
    sizes = [(32, 32), (64, 64), (128, 128), (320, 320)]

    for height, width in sizes:
        dataset = TFRecordsImageDataset(
            tfrecord_paths=tfrecord_files,
            input_size=(height, width),
            batch_size=5,
            num_threads=2,
            shuffle=False,
            interleave_files=False,
            repeat=1,
            prefetch=0,
            save_index=False,
        )

        batch = next(iter(dataset))
        assert batch[0].shape == (5, height, width, 3)


def test__tfrecords_image_dataset__single_file(tmp_path):
    """Test with a single file."""
    file_path = tmp_path / "single.tfrecord"
    utils.write_image_tfrecord(file_path, num_records=15)

    dataset = TFRecordsImageDataset(
        tfrecord_paths=[str(file_path)],
        input_size=(64, 64),
        batch_size=5,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    batches = list(dataset)
    assert len(batches) == 3
    assert len(dataset) == 15


def test__tfrecords_image_dataset__close(tfrecord_files):
    """Test closing resources."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=0,
        save_index=False,
    )

    # Get a batch
    next(iter(dataset))

    # Close - just verify it doesn't raise an error
    dataset.close()


def test__tfrecords_image_dataset__prefetch(tfrecord_files):
    """Test prefetching."""
    dataset = TFRecordsImageDataset(
        tfrecord_paths=tfrecord_files,
        input_size=(64, 64),
        batch_size=10,
        num_threads=2,
        shuffle=False,
        interleave_files=False,
        repeat=1,
        prefetch=2,  # Enable prefetching
        save_index=False,
    )

    batches = list(dataset)
    assert len(batches) == 3
    # Prefetch should not change results, just performance


def test__tfrecords_image_dataset__empty_file_list():
    """Test error handling for empty file list."""
    with pytest.raises(ValueError):
        TFRecordsImageDataset(
            tfrecord_paths=[],
            input_size=(64, 64),
            batch_size=10,
            save_index=False,
        )


def test__tfrecords_image_dataset__invalid_batch_size(tfrecord_files):
    """Test error handling for invalid batch size."""
    with pytest.raises(ValueError):
        TFRecordsImageDataset(
            tfrecord_paths=tfrecord_files,
            input_size=(64, 64),
            batch_size=0,
            save_index=False,
        )
