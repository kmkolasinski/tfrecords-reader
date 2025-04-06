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
    num_columns = 3
    index_data = indexer.create_index_for_tfrecord(filepath, parse_fn)
    assert len(index_data) == num_columns
    assert len(index_data["tfrecord_offset"]) == num_examples
    assert len(index_data["tfrecord_filename"]) == num_examples
    assert len(index_data["column"]) == num_examples
