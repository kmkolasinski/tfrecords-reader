import pytest

import tfr_reader as tfr
from tests import utils


@pytest.fixture(autouse=True)
def tfrecord_file(tmp_path):
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    utils.write_dummy_tfrecord(dummy_tfrecord_path)
    return str(dummy_tfrecord_path)


def test__inspect_dataset_example(tmp_path: str):
    feature, info = tfr.inspect_dataset_example(tmp_path)

    expected_info = [
        {"key": "bytes_feature", "type": "bytes_list"},
        {"key": "float_feature", "type": "float_list"},
        {"key": "int64_feature", "type": "int64_list"},
    ]
    assert info == expected_info

    assert feature["bytes_feature"].value[0] == b"A1"
    assert feature["float_feature"].value == pytest.approx([1.1, 2.2, 3.3])
    assert feature["int64_feature"].value == [10, 20, 30]


def test__tfrecord_file_reader(tfrecord_file: str):
    reader = tfr.TFRecordFileReader(tfrecord_file)
    assert reader.file is None
    with reader:
        assert reader.file is not None
        feature = reader[0]
        assert feature["bytes_feature"].value[0] == b"A1"

    assert reader.file is None
