import functools
from collections import defaultdict
from multiprocessing import pool
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from tfr_reader import example
from tfr_reader.cython import indexer

INDEX_FILENAME = "tfrds-reader-index.parquet"


def simple_index_fn(
    feature: example.Feature,
    label_field: str,
    label_mapping: dict[int, dict[str, Any]],
    default_value: dict[str, Any],
) -> dict[str, Any]:
    _label = feature[label_field].value[0]
    return {"label": _label, **label_mapping.get(_label, default_value)}


def create_simple_index(  # noqa: PLR0913
    directory: str | Path,
    label_field: str,
    label_mapping: dict[int, dict[str, Any]],
    default_value: dict[str, Any],
    *,
    filepattern: str = "*.tfrecord",
    processes: int = 1,
) -> pl.DataFrame:
    """
    Creates an index for a TFRecord dataset.
    Args:
        directory: Path to the directory containing *.tfrecord files.
        label_field: Name of the field to use as the label.
        label_mapping: Mapping of labels to their corresponding values.
        default_value: Default value for labels not in the mapping.
        filepattern: Pattern to match TFRecord files.
        processes: Number of processes to use for parallel processing.

    Returns:
        pl.DataFrame: DataFrame containing the index data.
    """
    index_fn = functools.partial(
        simple_index_fn,
        label_field=label_field,
        label_mapping=label_mapping,
        default_value=default_value,
    )

    data = create_index_for_directory(
        directory,
        index_fn=index_fn,
        filepattern=filepattern,
        processes=processes,
    )
    ds = pl.DataFrame(data).sort(by=["tfrecord_filename", "tfrecord_start"])
    ds.write_parquet(Path(directory) / INDEX_FILENAME)
    return ds


def create_index_for_tfrecord(
    tfrecord_path: str,
    index_fn: example.IndexFunc | None = None,
) -> dict[str, list[Any]]:
    reader = indexer.TFRecordFileReader(tfrecord_path)
    filename = Path(tfrecord_path).name

    data = defaultdict(list)

    for i in range(len(reader)):
        pointer = reader.get_pointer(i)

        data["tfrecord_filename"].append(filename)
        data["tfrecord_start"].append(pointer["start"])
        data["tfrecord_end"].append(pointer["end"])

        if index_fn is not None:
            example_str = reader.get_example(i)
            indexed_extra_data = index_fn(example.decode(example_str))
            for key, vale in indexed_extra_data.items():
                data[key].append(vale)

    reader.close()
    return data


def create_index_for_tfrecords(
    tfrecords_paths: list[str],
    index_fn: example.IndexFunc | None = None,
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for a TFRecord files.

    Args:
        tfrecords_paths: List of TFRecord filenames to create an index for.
        index_fn: function to create additional columns in the index
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    map_fn = functools.partial(
        create_index_for_tfrecord,
        index_fn=index_fn,
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
    index_fn: example.IndexFunc | None = None,
    filepattern: str = "*.tfrecord",
    processes: int = 1,
) -> dict[str, list[Any]]:
    """Creates an index for all TFRecord files in a directory.

    Args:
        directory: Path to the directory containing *.tfrecord files.
        index_fn: function to create additional columns in the index
        filepattern: Pattern to match TFRecord files.
        processes: Number of processes to use for parallel processing.

    Returns:
        dict: Dictionary containing the index data.
    """
    tfrecords_paths = [str(path) for path in Path(directory).glob(filepattern)]
    if not tfrecords_paths:
        raise ValueError(f"No TFRecord files found in directory: {directory}")
    return create_index_for_tfrecords(
        tfrecords_paths,
        index_fn=index_fn,
        processes=processes,
    )
