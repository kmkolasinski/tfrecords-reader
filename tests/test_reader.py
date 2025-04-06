from pathlib import Path

import pytest

import tfr_reader as tfr
from tests import utils
from tfr_reader import indexer

NUM_RECORDS = 5


@pytest.fixture(autouse=True)
def tfrecord_file(tmp_path):
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    utils.write_dummy_tfrecord(dummy_tfrecord_path, NUM_RECORDS)
    return str(dummy_tfrecord_path)


def test__inspect_dataset_example(tmp_path: Path):
    feature, info = tfr.inspect_dataset_example(str(tmp_path))

    expected_info = [
        {"key": "bytes_feature", "type": "bytes_list", "length": 1},
        {"key": "float_feature", "type": "float_list", "length": 3},
        {"key": "int64_feature", "type": "int64_list", "length": 3},
    ]
    assert info == expected_info

    assert feature["bytes_feature"].value[0] == b"A1"
    assert feature["float_feature"].value == pytest.approx([1.1, 2.2, 3.3])
    assert feature["int64_feature"].value == [10, 20, 30]


def test__tfrecord_file_reader(tfrecord_file: str):
    index_data = indexer.create_index_for_tfrecord(tfrecord_file)
    reader = tfr.TFRecordFileReader(tfrecord_file)
    assert reader._file is None
    with reader:
        assert reader._file is not None
        start = index_data["tfrecord_start"][0]
        end = index_data["tfrecord_end"][0]
        feature = reader.get_example(start, end)
        assert feature["bytes_feature"].value[0] == b"A1"

    assert reader._file is None


def test__tfrecord_file_reader__invalid_offsets(tfrecord_file: str):
    reader = tfr.TFRecordFileReader(tfrecord_file)
    with reader, pytest.raises(Exception):  # noqa: B017, PT011
        reader.get_example(0, 20)

    with pytest.raises(OSError):  # noqa: PT011
        reader.get_example(0, 20)


def test__dataset_reader(tmp_path: Path):
    ds = tfr.TFRecordDatasetReader.build_index_from_dataset_dir(str(tmp_path), _decode_fn)
    assert ds.dataset_dir == str(tmp_path)
    assert ds.size == NUM_RECORDS
    assert len(ds) == NUM_RECORDS

    assert ds[0]["bytes_feature"].value[0] == b"A1"
    with pytest.raises(KeyError):
        _ = ds[0]["column"]

    assert ds[1]["bytes_feature"].value[0] == b"A2"

    with pytest.raises(IndexError):
        _ = ds[-1]

    with pytest.raises(IndexError):
        _ = ds[5]


def test__dataset_reader_from_index(tmp_path: Path):
    tfr.TFRecordDatasetReader.build_index_from_dataset_dir(str(tmp_path), _decode_fn)
    ds = tfr.TFRecordDatasetReader(str(tmp_path))
    assert ds.dataset_dir == str(tmp_path)
    assert ds.size == NUM_RECORDS
    assert len(ds) == NUM_RECORDS

    assert ds[0]["bytes_feature"].value[0] == b"A1"
    with pytest.raises(KeyError):
        _ = ds[0]["column"]

    assert ds[1]["bytes_feature"].value[0] == b"A2"

    with pytest.raises(IndexError):
        _ = ds[-1]

    with pytest.raises(IndexError):
        _ = ds[5]


def _decode_fn(feat: tfr.Feature) -> dict[str, tfr.Feature]:
    return {"column": feat["int64_feature"].value[0]}
