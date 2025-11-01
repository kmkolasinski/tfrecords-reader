"""Unit tests for BatchSampler."""

from tfr_reader.datasets.image_classification.sampler import BatchSampler


def test__batch_sampler__initialization():
    """Test basic initialization."""
    file_lengths = [100, 200, 150]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
    )
    assert len(sampler) == sum(file_lengths)


def test__batch_sampler__sequential_sampling():
    """Test sequential (non-interleaved) sampling."""
    file_lengths = [10, 20, 15]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
    )

    # First batch should be from file 0
    batch_size = 5
    batch = sampler.next_batch(batch_size)
    assert batch is not None
    assert len(batch) == batch_size
    assert all(file_idx == 0 for file_idx, _ in batch)
    assert [ex_idx for _, ex_idx in batch] == [0, 1, 2, 3, 4]


def test__batch_sampler__interleaved_sampling():
    """Test interleaved sampling."""
    file_lengths = [10, 10, 10]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
    )
    batch_size = 6
    batch = sampler.next_batch(batch_size)
    assert batch is not None
    assert len(batch) == batch_size

    # With block_size=2, should get 2 from file 0, 2 from file 1, 2 from file 2
    file_indices = [file_idx for file_idx, _ in batch]
    assert file_indices == [0, 0, 1, 1, 2, 2]


def test__batch_sampler__shuffle_deterministic():
    """Test that shuffle is deterministic with seed."""
    file_lengths = [10, 10, 10]

    sampler1 = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=False,
        repeat=1,
        seed=42,
    )

    sampler2 = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=False,
        repeat=1,
        seed=42,
    )

    batch1 = sampler1.next_batch(30)
    batch2 = sampler2.next_batch(30)

    assert batch1 == batch2


def test__batch_sampler__repeat_infinite():
    """Test infinite repeat (-1)."""
    file_lengths = [5]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=-1,
    )

    # Should be able to get more batches than total examples
    batch_size = 5
    for _ in range(3):  # 3 epochs
        batch = sampler.next_batch(batch_size)
        assert batch is not None
        assert len(batch) == batch_size

    # Should still have more data
    batch = sampler.next_batch(5)
    assert batch is not None


def test__batch_sampler__repeat_multiple_epochs():
    """Test finite repeat."""
    file_lengths = [5]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=2,
    )
    batch_size = 5
    # First epoch
    batch1 = sampler.next_batch(5)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # Second epoch
    batch2 = sampler.next_batch(5)
    assert batch2 is not None
    assert len(batch2) == batch_size

    # No more data
    batch3 = sampler.next_batch(5)
    assert batch3 is None


def test__batch_sampler__reset():
    """Test reset functionality."""
    file_lengths = [10]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
        seed=42,
    )

    # Get first batch
    batch1 = sampler.next_batch(5)

    # Reset
    sampler.reset()

    # Should get same batch again
    batch2 = sampler.next_batch(5)
    assert batch1 == batch2


def test__batch_sampler__partial_batch():
    """Test getting a partial batch at the end."""
    file_lengths = [7]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
    )

    # Get first batch of 5
    batch_size = 5
    batch1 = sampler.next_batch(5)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # Get remaining 2
    batch2 = sampler.next_batch(5)
    assert batch2 is not None
    assert len(batch2) == 2

    # No more data
    batch3 = sampler.next_batch(5)
    assert batch3 is None


def test__batch_sampler__drop_remainder_single_epoch():
    """Test drop_remainder drops incomplete batches in single epoch."""
    file_lengths = [7]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
        drop_remainder=True,
    )

    batch_size = 5
    # Get first batch of 5
    batch1 = sampler.next_batch(batch_size)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # Should return None since remaining batch has only 2 examples
    batch2 = sampler.next_batch(batch_size)
    assert batch2 is None


def test__batch_sampler__drop_remainder_multiple_epochs():
    """Test drop_remainder with multiple epochs."""
    file_lengths = [7]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=2,
        drop_remainder=True,
    )

    batch_size = 5
    # First epoch - first batch
    batch1 = sampler.next_batch(batch_size)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # First epoch - should skip remainder and move to second epoch
    batch2 = sampler.next_batch(batch_size)
    assert batch2 is not None
    assert len(batch2) == batch_size

    # Second epoch complete, no more data
    batch3 = sampler.next_batch(batch_size)
    assert batch3 is None


def test__batch_sampler__drop_remainder_infinite_repeat():
    """Test drop_remainder with infinite repeat."""
    file_lengths = [7]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=-1,
        drop_remainder=True,
    )

    batch_size = 5
    # Should continuously get full batches, skipping remainders
    for _ in range(10):  # Test 10 batches across multiple epochs
        batch = sampler.next_batch(batch_size)
        assert batch is not None
        assert len(batch) == batch_size


def test__batch_sampler__drop_remainder_with_exact_division():
    """Test drop_remainder when total examples divides evenly by batch_size."""
    file_lengths = [10]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
        drop_remainder=True,
    )

    batch_size = 5
    # Get first batch
    batch1 = sampler.next_batch(batch_size)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # Get second batch
    batch2 = sampler.next_batch(batch_size)
    assert batch2 is not None
    assert len(batch2) == batch_size

    # No more data
    batch3 = sampler.next_batch(batch_size)
    assert batch3 is None


def test__batch_sampler__drop_remainder_with_interleaving():
    """Test drop_remainder with interleaved sampling."""
    file_lengths = [7, 8]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
        drop_remainder=True,
    )

    batch_size = 10
    # Total examples = 15, should get 1 full batch of 10
    batch1 = sampler.next_batch(batch_size)
    assert batch1 is not None
    assert len(batch1) == batch_size

    # Remaining 5 examples should be dropped
    batch2 = sampler.next_batch(batch_size)
    assert batch2 is None
