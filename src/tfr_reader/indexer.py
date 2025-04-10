import functools
from collections import defaultdict
from multiprocessing import pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tfr_reader import example
from tfr_reader.cython import indexer


def create_index_for_tfrecord(
    tfrecord_path: str,
    feature_decode_fn: example.FeatureDecodeFunc | None = None,
) -> dict[str, list[Any]]:
    reader = indexer.TFRecordFileReader(tfrecord_path)
    filename = Path(tfrecord_path).name

    data = defaultdict(list)

    for i in range(len(reader)):
        pointer = reader.get_pointer(i)

        data["tfrecord_filename"].append(filename)
        data["tfrecord_start"].append(pointer["start"])
        data["tfrecord_end"].append(pointer["end"])

        if feature_decode_fn is not None:
            example_str = reader.get_example(i)
            example_data = feature_decode_fn(example.decode(example_str))
            for key, vale in example_data.items():
                data[key].append(vale)

    reader.close()
    return data


def create_index_for_tfrecords(
    tfrecords_paths: list[str],
    feature_decode_fn: example.FeatureDecodeFunc,
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for a TFRecord files.

    Args:
        tfrecords_paths: List of TFRecord filenames to create an index for.
        feature_decode_fn: Function to parse the Features of the TFRecord example
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    map_fn = functools.partial(
        create_index_for_tfrecord,
        feature_decode_fn=feature_decode_fn,
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
    feature_decode_fn: example.FeatureDecodeFunc,
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for all TFRecord files in a directory.

    Args:
        directory: Path to the directory containing *.tfrecord files.
        feature_decode_fn: Function to parse the Features of the TFRecord example
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    tfrecords_paths = [str(path) for path in Path(directory).glob("*.tfrecord")]
    if not tfrecords_paths:
        raise ValueError(f"No TFRecord files found in directory: {directory}")
    return create_index_for_tfrecords(
        tfrecords_paths,
        feature_decode_fn=feature_decode_fn,
        processes=processes,
    )
