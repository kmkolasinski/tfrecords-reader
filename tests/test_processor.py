"""Unit tests for ImageProcessor."""

import io

import numpy as np
from PIL import Image

from tests import utils
from tfr_reader.datasets.image_classification.processor import ImageProcessor


def test__image_processor__initialization():
    """Test basic initialization."""
    processor = ImageProcessor(
        target_width=320,
        target_height=320,
        num_threads=2,
    )
    # Just verify it initializes without error
    assert processor is not None


def test__image_processor__decode_example():
    """Test decoding a TFRecord example."""
    processor = ImageProcessor(target_width=64, target_height=64, num_threads=1)

    # Create a test example
    example = utils.create_image_example(0, width=128, height=128)
    example_bytes = example.SerializeToString()

    # Decode it
    decoded = processor.decode_example(example_bytes)

    assert "image_bytes" in decoded
    assert "label" in decoded
    assert isinstance(decoded["image_bytes"], bytes)
    assert decoded["label"] == 0  # First example has label 0


def test__image_processor__decode_and_resize_image():
    """Test decoding and resizing a single image."""
    processor = ImageProcessor(target_width=64, target_height=64, num_threads=1)

    # Create a test image
    image_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # noqa: NPY002
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # Decode and resize
    result = processor.decode_and_resize_image(image_bytes)

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test__image_processor__process_batch():
    """Test processing a batch of examples."""
    processor = ImageProcessor(target_width=64, target_height=64, num_threads=2)

    # Create test examples
    batch_size = 8
    raw_examples = []
    for i in range(batch_size):
        example = utils.create_image_example(i, width=128, height=128)
        raw_examples.append(example.SerializeToString())

    # Process batch
    images, labels = processor.process_batch(raw_examples)

    assert images.shape == (batch_size, 64, 64, 3)
    assert images.dtype == np.uint8
    assert labels.shape == (batch_size,)
    assert labels.dtype == np.int32

    # Check labels are correct (alternating 0, 1)
    for i in range(batch_size):
        assert labels[i] == i % 2


def test__image_processor__process_batch__empty():
    """Test processing an empty batch."""
    processor = ImageProcessor(target_width=64, target_height=64, num_threads=1)

    images, labels = processor.process_batch([])

    assert images is None
    assert labels is None


def test__image_processor__different_sizes():
    """Test resizing images to different target sizes."""
    test_sizes = [(32, 32), (128, 128), (320, 320), (640, 480)]

    for height, width in test_sizes:
        processor = ImageProcessor(target_width=width, target_height=height, num_threads=1)

        example = utils.create_image_example(0, width=256, height=256)
        raw_examples = [example.SerializeToString()]

        images, labels = processor.process_batch(raw_examples)

        assert images.shape == (1, height, width, 3)


def test__image_processor__parallel_processing():
    """Test parallel processing with multiple threads."""
    processor_single = ImageProcessor(target_width=64, target_height=64, num_threads=1)
    processor_multi = ImageProcessor(target_width=64, target_height=64, num_threads=4)

    # Create test batch
    batch_size = 16
    raw_examples = []
    for i in range(batch_size):
        example = utils.create_image_example(i, width=128, height=128)
        raw_examples.append(example.SerializeToString())

    # Process with both
    images_single, labels_single = processor_single.process_batch(raw_examples)
    images_multi, labels_multi = processor_multi.process_batch(raw_examples)

    # Results should be identical
    np.testing.assert_array_equal(images_single, images_multi)
    np.testing.assert_array_equal(labels_single, labels_multi)
