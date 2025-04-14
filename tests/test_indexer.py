from pathlib import Path

import pytest

from tests import utils
from tfr_reader import indexer
from tfr_reader.example import feature


@pytest.fixture(autouse=True)
def tfrecord_file(tmp_path):
    dummy_tfrecord_path = tmp_path / "dummy.tfrecord"
    utils.write_dummy_tfrecord(dummy_tfrecord_path, num_records=10)
    return str(dummy_tfrecord_path)


def test__inspect_dataset_example(tmp_path: str):
    def parse_fn(feat: feature.Feature) -> dict[str, feature.Feature]:
        return {"column": feat["int64_feature"].value[0]}

    filepath = str(Path(tmp_path) / "dummy.tfrecord")
    num_examples = 10
    num_columns = 4
    index_data = indexer.create_index_for_tfrecord(filepath, parse_fn)
    assert len(index_data) == num_columns
    assert len(index_data["tfrecord_start"]) == num_examples
    assert len(index_data["tfrecord_end"]) == num_examples
    assert len(index_data["tfrecord_filename"]) == num_examples
    assert len(index_data["column"]) == num_examples

    for start, end in zip(index_data["tfrecord_start"], index_data["tfrecord_end"], strict=False):
        assert start < end

    for start, end in zip(
        index_data["tfrecord_start"][1:], index_data["tfrecord_end"], strict=False
    ):
        assert start == end


def test__create_simple_index(tmp_path: str):
    label_field = "int64_feature"
    label_mapping = {10: {"name": "a"}, 20: {"name": "b"}}
    default_value = {"name": "none"}

    index_data = indexer.create_simple_index(
        tmp_path,
        label_field=label_field,
        label_mapping=label_mapping,
        default_value=default_value,
        processes=2,
    )
    num_examples = 10
    assert len(index_data) == num_examples
    expected_names = ["a", "b"] + ["none"] * (num_examples - 2)
    assert index_data["name"].to_list() == expected_names
    expected_labels = [(i + 1) * 10 for i in range(num_examples)]
    assert index_data["label"].to_list() == expected_labels
