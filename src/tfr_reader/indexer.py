import functools
from collections import defaultdict
from collections.abc import Callable
from multiprocessing import pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tfr_reader.cython import indexer
from tfr_reader.example import example_pb2

FeatureParseFunc = Callable[[example_pb2.Feature], dict[str, Any]]


def decode(raw_record: bytes) -> example_pb2.Example:
    example = example_pb2.Example()
    example.ParseFromString(raw_record)
    return example


def create_index_for_tfrecord(
    tfrecord_path: str,
    feature_parse_fn: FeatureParseFunc,
) -> dict[str, list[Any]]:
    reader = indexer.TFRecordFileReader(tfrecord_path)
    filename = Path(tfrecord_path).name

    data = defaultdict(list)

    for i in range(len(reader)):
        offset = reader.get_offset(i)
        example = decode(reader[i])
        example_data = feature_parse_fn(example.features.feature)

        data["tfrecord_filename"].append(filename)
        data["tfrecord_offset"].append(offset)
        for key, vale in example_data.items():
            data[key].append(vale)

    reader.close()
    return data


def create_index_for_tfrecords(
    tfrecords_paths: list[str],
    feature_parse_fn: FeatureParseFunc,
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for a TFRecord files.

    Args:
        tfrecords_paths: List of TFRecord filenames to create an index for.
        feature_parse_fn: Function to parse the Features of the TFRecord example
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    map_fn = functools.partial(
        create_index_for_tfrecord,
        feature_parse_fn=feature_parse_fn,
    )

    with pool.Pool(processes) as p:
        results = list(
            tqdm(
                p.imap_unordered(map_fn, tfrecords_paths),
                total=len(tfrecords_paths),
                desc="Creating TFRecord Index",
            ),
        )

    data = defaultdict(list)
    for result in results:
        for key, value in result.items():
            data[key].extend(value)
    return data


def create_index_for_directory(
    directory: str | Path,
    feature_parse_fn: FeatureParseFunc,
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for all TFRecord files in a directory.

    Args:
        directory: Path to the directory containing *.tfrecord files.
        feature_parse_fn: Function to parse the Features of the TFRecord example
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    tfrecords_paths = [str(path) for path in Path(directory).glob("*.tfrecord")]
    if not tfrecords_paths:
        raise ValueError(f"No TFRecord files found in directory: {directory}")
    return create_index_for_tfrecords(
        tfrecords_paths,
        feature_parse_fn=feature_parse_fn,
        processes=processes,
    )
