# tfrecords-reader

Tensorflow TFRecords reader for Python with Random access and streaming support.

## Installation

```bash
pip install git+https://github.com/kmkolasinski/tfrecords-reader.git
```

## Usage

### Dataset inspection
`inspect_dataset_example` function allows you to inspect the dataset and get a sample example
and its types.
```python
import tfr_reader as tfr
dataset_dir = "/path/to/directory/with/tfrecords"
example, types = tfr.inspect_dataset_example(dataset_dir)
types
>>> Out[1]:
[{'key': 'label', 'type': 'int64_list', 'length': 1},
 {'key': 'name', 'type': 'bytes_list', 'length': 1},
 {'key': 'image_id', 'type': 'bytes_list', 'length': 1},
 {'key': 'image', 'type': 'bytes_list', 'length': 1}]
```

### Dataset indexing
Create an index of the dataset for fast access. The index is a dictionary with keys as the
image IDs and values as the file names. The index is created by reading the dataset and
parsing the examples. The index is saved in the `dataset_dir` directory. You can use the
`indexed_cols_fn` function to specify the columns you want to index. The function should return
a dictionary with keys as the column names and values as the column values.

> [!NOTE]
> Indexing operation works only for local files, remote files are not supported.


```python
import tfr_reader as tfr
dataset_dir = "/path/to/directory/with/tfrecords"

def indexed_cols_fn(feature):
    return {
        "label": feature["label"].value[0],
        "name": feature["label"].value[0].decode(),
        "image_id": feature["image/id"].value[0].decode(),
    }

tfrds = tfr.TFRecordDatasetReader.build_index_from_dataset_dir(dataset_dir, indexed_cols_fn)

tfrds.index_df[:5]
>> Out[2]:
shape: (5, 6)
┌───────────────────┬────────────────┬──────────────┬────────┬───────┬───────────────┐
│ tfrecord_filename ┆ tfrecord_start ┆ tfrecord_end ┆ name   ┆ label ┆ image_id      │
│ ---               ┆ ---            ┆ ---          ┆ ---    ┆ ---   ┆ ---           │
│ str               ┆ i64            ┆ i64          ┆ binary ┆ i64   ┆ binary        │
╞═══════════════════╪════════════════╪══════════════╪════════╪═══════╪═══════════════╡
│ demo.tfrecord     ┆ 0              ┆ 79           ┆ b"cat" ┆ 1     ┆ b"image-id-0" │
│ demo.tfrecord     ┆ 79             ┆ 158          ┆ b"dog" ┆ 0     ┆ b"image-id-1" │
│ demo.tfrecord     ┆ 158            ┆ 237          ┆ b"cat" ┆ 1     ┆ b"image-id-2" │
│ demo.tfrecord     ┆ 237            ┆ 316          ┆ b"dog" ┆ 0     ┆ b"image-id-3" │
│ demo.tfrecord     ┆ 316            ┆ 395          ┆ b"cat" ┆ 1     ┆ b"image-id-4" │
└───────────────────┴────────────────┴──────────────┴────────┴───────┴───────────────┘
```
Explanation about the index format:
* **tfrecord_filename**: name of the tfrecord file
* **tfrecord_start**: start byte position of the example in the tfrecord file
* **tfrecord_end**: end byte position of the example in the tfrecord file
* other columns: indexed columns from the dataset with `indexed_cols_fn` function

### Dataset reading

```python
import tfr_reader as tfr

tfrds = tfr.TFRecordDatasetReader("/path/to/directory/with/tfrecords")
# assume that the dataset is indexed already
tfrds = tfr.TFRecordDatasetReader("gs://bucket/path/to/directory/with/tfrecords")
# selection API
selected_df, examples = tfrds.select("SELECT * FROM index WHERE label = 1 LIMIT 20")
# custom selection
selected_df = tfrds.index_df.sample(5)
examples = tfrds.load_records(selected_df)
# indexing API
for i in range(len(tfrds)):
    example = tfrds[i]
```
