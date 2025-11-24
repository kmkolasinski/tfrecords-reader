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


def test__batch_sampler__indices_generation_test_case1():
    file_lengths = [5, 3]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
        interleave_block_size=-1,  # not used when interleave is False
        drop_remainder=False,
    )

    expected_global_index = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
    ]

    assert sampler.global_index == expected_global_index

    batch_size = 3
    indices = sampler.next_batch(batch_size)
    assert indices == [(0, 0), (0, 1), (0, 2)]

    indices = sampler.next_batch(batch_size)
    assert indices == [(0, 3), (0, 4), (1, 0)]

    indices = sampler.next_batch(batch_size)
    assert indices == [(1, 1), (1, 2)]


def test__batch_sampler__indices_generation_global_index_with_interleave_block_size_1():
    file_lengths = [6, 3, 4]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=1,
        drop_remainder=False,
    )

    expected_global_index = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ]

    assert sampler.global_index == expected_global_index


def test__batch_sampler__indices_generation_global_index_with_interleave_block_size_2():
    file_lengths = [6, 3, 4]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
        drop_remainder=False,
    )

    expected_global_index = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (2, 0),
        (0, 4),
        (2, 1),
        (0, 5),
        (2, 2),
        (2, 3),
    ]

    assert sampler.global_index == expected_global_index


def test__batch_sampler__indices_generation_global_index_with_interleave_block_size_3():
    file_lengths = [6, 3, 4]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=3,
        drop_remainder=False,
    )

    expected_global_index = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (2, 3),
        (0, 4),
        (0, 5),
    ]

    assert sampler.global_index == expected_global_index


def test__batch_sampler__indices_generation_global_index_with_interleave_block_size_3_shuffle():
    file_lengths = [6, 3, 4]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=True,
        repeat=1,
        interleave_block_size=3,
        drop_remainder=False,
    )

    expected_global_index = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (2, 3),
        (0, 4),
        (0, 5),
    ]

    for gi, exp_gi in zip(sampler.global_index, expected_global_index, strict=False):
        assert gi[0] == exp_gi[0]  # file indices should match


def test__batch_sampler__shuffle_preserves_file_indices():
    """Test that shuffling preserves file indices (only shuffles within files)."""
    file_lengths = [10, 20, 15]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=False,
        repeat=1,
        seed=42,
    )

    # Group indices by file
    file_0_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 0]
    file_1_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 1]
    file_2_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 2]

    # Check that we have the correct number of indices per file
    assert len(file_0_indices) == 10
    assert len(file_1_indices) == 20
    assert len(file_2_indices) == 15

    # Check that each file's indices are a permutation of range(file_length)
    assert sorted(file_0_indices) == list(range(10))
    assert sorted(file_1_indices) == list(range(20))
    assert sorted(file_2_indices) == list(range(15))

    # Check that file order in global_index is preserved
    file_indices_order = [file_idx for file_idx, _ in sampler.global_index]
    expected_file_order = [0] * 10 + [1] * 20 + [2] * 15
    assert file_indices_order == expected_file_order


def test__batch_sampler__shuffle_with_interleave_preserves_file_indices():
    """Test that shuffling with interleaving preserves file indices."""
    file_lengths = [6, 3, 4]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
        seed=123,
    )

    # Count occurrences of each file
    file_counts = {0: 0, 1: 0, 2: 0}
    for file_idx, _ in sampler.global_index:
        file_counts[file_idx] += 1

    # Verify correct counts
    assert file_counts[0] == 6
    assert file_counts[1] == 3
    assert file_counts[2] == 4

    # Group indices by file and check they are valid permutations
    file_0_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 0]
    file_1_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 1]
    file_2_indices = [idx for file_idx, idx in sampler.global_index if file_idx == 2]

    assert sorted(file_0_indices) == list(range(6))
    assert sorted(file_1_indices) == list(range(3))
    assert sorted(file_2_indices) == list(range(4))


def test__batch_sampler__shuffle_changes_example_order():
    """Test that shuffling actually changes the order of examples within files."""
    file_lengths = [20]
    sampler_shuffled = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=False,
        repeat=1,
        seed=42,
    )

    sampler_unshuffled = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=False,
        repeat=1,
    )

    # Extract example indices
    shuffled_indices = [idx for _, idx in sampler_shuffled.global_index]
    unshuffled_indices = [idx for _, idx in sampler_unshuffled.global_index]

    # They should not be the same order (with high probability for 20 elements)
    assert shuffled_indices != unshuffled_indices

    # But both should contain all indices
    assert sorted(shuffled_indices) == sorted(unshuffled_indices)


def test__batch_sampler__shuffle_different_across_epochs():
    """Test that shuffling produces different orders across epochs."""
    file_lengths = [15]
    sampler = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=False,
        repeat=2,
        seed=42,
    )

    # Get first epoch indices
    first_epoch_indices = []
    for _ in range(3):  # Get 3 batches of 5
        batch = sampler.next_batch(5)
        if batch is not None:
            first_epoch_indices.extend(batch)

    # Get second epoch indices
    second_epoch_indices = []
    for _ in range(3):
        batch = sampler.next_batch(5)
        if batch is not None:
            second_epoch_indices.extend(batch)

    # Extract just the example indices
    first_epoch_ex_indices = [idx for _, idx in first_epoch_indices]
    second_epoch_ex_indices = [idx for _, idx in second_epoch_indices]

    # Orders should be different (with very high probability)
    assert first_epoch_ex_indices != second_epoch_ex_indices

    # But both should be complete permutations
    assert sorted(first_epoch_ex_indices) == list(range(15))
    assert sorted(second_epoch_ex_indices) == list(range(15))

    # File indices should all be 0
    assert all(file_idx == 0 for file_idx, _ in first_epoch_indices)
    assert all(file_idx == 0 for file_idx, _ in second_epoch_indices)


def test__batch_sampler__shuffle_deterministic_with_seed():
    """Test that shuffle with same seed produces identical results."""
    file_lengths = [10, 15, 8]

    sampler1 = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
        seed=999,
    )

    sampler2 = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=True,
        repeat=1,
        interleave_block_size=2,
        seed=999,
    )

    # Both should produce identical global indices
    assert sampler1.global_index == sampler2.global_index


def test__batch_sampler__shuffle_interleave_pattern_preserved():
    """Test that interleave pattern is preserved after shuffling."""
    file_lengths = [4, 4, 4]

    # First create unshuffled to see the pattern
    sampler_unshuffled = BatchSampler(
        file_lengths=file_lengths,
        shuffle=False,
        interleave=True,
        repeat=1,
        interleave_block_size=3,
        seed=42,
    )

    sampler_shuffled = BatchSampler(
        file_lengths=file_lengths,
        shuffle=True,
        interleave=True,
        repeat=1,
        interleave_block_size=3,
        seed=42,
    )

    # Extract file patterns (order of file_idx)
    unshuffled_file_pattern = [file_idx for file_idx, _ in sampler_unshuffled.global_index]
    shuffled_file_pattern = [file_idx for file_idx, _ in sampler_shuffled.global_index]

    # The file interleaving pattern should be identical
    assert unshuffled_file_pattern == shuffled_file_pattern

    # But the example indices within each file should be different
    unshuffled_examples = [idx for _, idx in sampler_unshuffled.global_index]
    shuffled_examples = [idx for _, idx in sampler_shuffled.global_index]

    # With high probability, they should differ
    assert unshuffled_examples != shuffled_examples
