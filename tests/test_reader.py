import pytest

import tfr_reader as tfr
from tfr_reader.example import example_pb2 as pb2


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


@pytest.fixture(autouse=True)
def tfrecord_file(tmp_path):
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    write_dummy_tfrecord(dummy_tfrecord_path)
    return str(dummy_tfrecord_path)


def test__inspect_dataset_example(tmp_path: str):
    example, info = tfr.inspect_dataset_example(tmp_path)

    expected_info = [
        {"name": "bytes_feature", "type": "bytes_list"},
        {"name": "float_feature", "type": "float_list"},
        {"name": "int64_feature", "type": "int64_list"},
    ]
    assert info == expected_info
    feature = example.features.feature

    assert feature["bytes_feature"].bytes_list.value[0] == b"A1"
    assert feature["float_feature"].float_list.value == pytest.approx([1.1, 2.2, 3.3])
    assert feature["int64_feature"].int64_list.value == [10, 20, 30]


def test__tfrecord_file_reader(tfrecord_file: str):
    reader = tfr.TFRecordFileReader(tfrecord_file)
    assert reader.file is None
    with reader:
        assert reader.file is not None
        feature = reader[0].features.feature
        assert feature["bytes_feature"].bytes_list.value[0] == b"A1"

    assert reader.file is None
