from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import tfr_reader as tfr
from tests import utils
from tfr_reader import indexer
from tfr_reader.example import decode
from tfr_reader.example import tfr_example_pb2 as pb2

NUM_RECORDS = 5


@pytest.fixture
def tfrecord_file(tmp_path):
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    utils.write_dummy_tfrecord(dummy_tfrecord_path, NUM_RECORDS)
    return str(dummy_tfrecord_path)


def test__inspect_dataset_example(tfrecord_file: str):
    dataset_dir = str(Path(tfrecord_file).parent)
    feature, info = tfr.inspect_dataset_example(dataset_dir)

    expected_info = [
        {"key": "bytes_feature", "type": "bytes_list", "length": 1},
        {"key": "float_feature", "type": "float_list", "length": 3},
        {"key": "int64_feature", "type": "int64_list", "length": 3},
    ]
    expected_info_dict = {info[i]["key"]: info[i] for i in range(len(expected_info))}
    info_dict = {info[i]["key"]: info[i] for i in range(len(info))}
    assert info_dict == expected_info_dict

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


def test__dataset_reader(tfrecord_file: str):
    dataset_dir = str(Path(tfrecord_file).parent)
    ds_created = tfr.TFRecordDatasetReader.build_index_from_dataset_dir(dataset_dir, _index_fn)
    ds_loaded = tfr.TFRecordDatasetReader(dataset_dir)
    for ds in [ds_created, ds_loaded]:
        assert ds.dataset_dir == dataset_dir
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


def test__dataset_reader_selecting_by_indices(tfrecord_file: str):
    reader = tfr.load_from_directory(
        Path(tfrecord_file).parent,
        index_fn=_index_fn,
    )
    assert reader[0]["int64_feature"].value == [10, 20, 30]
    assert reader[[]] == []
    assert reader[[0]] == [reader[0]]
    assert reader[[2, 1]] == [reader[2], reader[1]]
    indices = np.array([0, 1, 2, 3, 4])
    assert reader[indices] == [reader[i] for i in indices]


def test__dataset_reader_select(tfrecord_file: str):
    dataset_dir = str(Path(tfrecord_file).parent)
    tfr.TFRecordDatasetReader.build_index_from_dataset_dir(dataset_dir, _index_fn)
    ds = tfr.TFRecordDatasetReader(dataset_dir)
    selected_rows, examples = ds.select("SELECT * FROM index")
    assert len(examples) == NUM_RECORDS
    assert len(selected_rows) == NUM_RECORDS
    for row in range(NUM_RECORDS):
        assert examples[row]["bytes_feature"].value[0] == f"A{row + 1}".encode()
        int_col_value = examples[row]["int64_feature"].value[0]
        assert selected_rows.row(row, named=True)["column"] == int_col_value


def test__dataset_reader_demo(tmp_path: Path):
    num_records = 40
    utils.write_demo_tfrecord(tmp_path / "demo.tfrecord", num_records)
    tfr.TFRecordDatasetReader.build_index_from_dataset_dir(str(tmp_path), _decode_demo_fn)
    ds = tfr.TFRecordDatasetReader(str(tmp_path))
    assert ds.size == num_records
    for i in range(num_records):
        example = ds[i]
        assert example["name"].value[0] == b"cat" if i % 2 == 0 else b"dog"
        assert example["label"].value[0] == (1 if i % 2 == 0 else 0)
        assert example["image_id"].value[0] == f"image-id-{i}".encode()
        assert len(example) == 3  # noqa: PLR2004


def test__complex_bytes():
    random_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)  # noqa: NPY002
    random_image_bytes = Image.fromarray(random_image).tobytes()

    features = pb2.Features(
        feature={
            "image": pb2.Feature(bytes_list=pb2.BytesList(value=[random_image_bytes])),
        }
    )
    example = pb2.Example(features=features)
    example_bytes = example.SerializeToString()
    restored = decode(example_bytes)
    assert random_image_bytes == restored["image"].value[0]


def _decode_demo_fn(feat: tfr.Feature) -> dict[str, tfr.Feature]:
    return {
        "name": feat["name"].value[0].decode(),
        "label": feat["label"].value[0],
        "image_id": feat["image_id"].value[0].decode(),
    }


def _index_fn(feat: tfr.Feature) -> dict[str, tfr.Feature]:
    return {"column": feat["int64_feature"].value[0]}
