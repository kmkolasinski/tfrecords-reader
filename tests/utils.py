import io

import numpy as np
from PIL import Image

from tfr_reader.example import tfr_example_pb2 as pb2


def create_dummy_example(index: int) -> pb2.Example:
    features = pb2.Features(
        feature={
            "bytes_feature": pb2.Feature(bytes_list=pb2.BytesList(value=[f"A{index}".encode()])),
            "float_feature": pb2.Feature(
                float_list=pb2.FloatList(value=[1.1 * index, 2.2 * index, 3.3 * index])
            ),
            "int64_feature": pb2.Feature(
                int64_list=pb2.Int64List(value=[10 * index, 20 * index, 30 * index])
            ),
        }
    )
    return pb2.Example(features=features)


def write_dummy_tfrecord(output_path, num_records: int = 5):
    with open(output_path, "wb") as f:
        for i in range(num_records):
            example = create_dummy_example(i + 1)
            serialized_example = example.SerializeToString()
            # Write the length of the serialized example as a 8-byte integer (little-endian)
            f.write(len(serialized_example).to_bytes(8, byteorder="little"))
            # Write the length CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
            # Write the serialized example
            f.write(serialized_example)
            # Write the data CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")


def create_demo_example(index: int, label: int, name: str) -> pb2.Example:
    image_id = f"image-id-{index}".encode()
    features = pb2.Features(
        feature={
            "name": pb2.Feature(bytes_list=pb2.BytesList(value=[name.encode()])),
            "label": pb2.Feature(int64_list=pb2.Int64List(value=[label])),
            "image_id": pb2.Feature(bytes_list=pb2.BytesList(value=[image_id])),
        }
    )
    return pb2.Example(features=features)


def write_demo_tfrecord(output_path, num_records: int = 5):
    names = ["cat", "dog"]
    with open(output_path, "wb") as f:
        for i in range(num_records):
            name = names[i % len(names)]
            label = 1 if name == "cat" else 0
            example = create_demo_example(i, label, name)
            serialized_example = example.SerializeToString()
            # Write the length of the serialized example as a 8-byte integer (little-endian)
            f.write(len(serialized_example).to_bytes(8, byteorder="little"))
            # Write the length CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
            # Write the serialized example
            f.write(serialized_example)
            # Write the data CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")


def create_image_example(index: int, width: int = 64, height: int = 64) -> pb2.Example:
    """Create a TFRecord example with a random image and label."""
    # Create random image
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)  # noqa: NPY002
    image = Image.fromarray(image_array)

    # Encode as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    # Create label (alternating between 0 and 1)
    label = index % 2

    features = pb2.Features(
        feature={
            "image/encoded": pb2.Feature(bytes_list=pb2.BytesList(value=[image_bytes])),
            "image/object/bbox/label": pb2.Feature(int64_list=pb2.Int64List(value=[label])),
        }
    )
    return pb2.Example(features=features)


def write_image_tfrecord(output_path, num_records: int = 10, width: int = 64, height: int = 64):
    """Write a TFRecord file with image examples."""
    with open(output_path, "wb") as f:
        for i in range(num_records):
            example = create_image_example(i, width, height)
            serialized_example = example.SerializeToString()
            # Write the length of the serialized example as a 8-byte integer (little-endian)
            f.write(len(serialized_example).to_bytes(8, byteorder="little"))
            # Write the length CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
            # Write the serialized example
            f.write(serialized_example)
            # Write the data CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
