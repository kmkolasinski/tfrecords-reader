# tfrecords-reader

Fast TensorFlow TFRecords reader for Python with Random access and Google Storage streaming support.

## Features
* No **TensorFlow** dependency - this library implement custom TFRecord Reader
* Protobuf is not required, this library contains cython decoder for TFRecord files
* Fast random access to TFRecords i.e. you can read any example from the dataset without
  reading the whole dataset e.g
    ```python
    import tfr_reader as tfr
    tfrds = tfr.TFRecordDatasetReader("/path/to/directory/with/tfrecords")
    example = tfrds[42]
    image_bytes: bytes = example["image/encoded"].value[0]
    ```

## Installation

* Base installation with minimum requirements:
```bash
pip install pip install "git+https://github.com/kmkolasinski/tfrecords-reader.git"
pip install .
```
* For extra Google Storage Cloud support use:
```bash
pip install pip install "git+https://github.com/kmkolasinski/tfrecords-reader.git#egg=[google]"
pip install ".[google]"
```

## Usage

### Dataset Inspection
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

### Dataset Indexing
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
        "name": feature["name"].value[0].decode(),
        "image_id": feature["image/id"].value[0].decode(),
    }

tfrds = tfr.TFRecordDatasetReader.build_index_from_dataset_dir(dataset_dir, indexed_cols_fn)

tfrds.index_df[:5]
>> Out[2]:
shape: (5, 6)
┌───────────────────┬────────────────┬──────────────┬──────┬───────┬────────────┐
│ tfrecord_filename ┆ tfrecord_start ┆ tfrecord_end ┆ name ┆ label ┆ image_id   │
│ ---               ┆ ---            ┆ ---          ┆ ---  ┆ ---   ┆ ---        │
│ str               ┆ i64            ┆ i64          ┆ str  ┆ i64   ┆ str        │
╞═══════════════════╪════════════════╪══════════════╪══════╪═══════╪════════════╡
│ demo.tfrecord     ┆ 0              ┆ 79           ┆ cat  ┆ 1     ┆ image-id-0 │
│ demo.tfrecord     ┆ 79             ┆ 158          ┆ dog  ┆ 0     ┆ image-id-1 │
│ demo.tfrecord     ┆ 158            ┆ 237          ┆ cat  ┆ 1     ┆ image-id-2 │
│ demo.tfrecord     ┆ 237            ┆ 316          ┆ dog  ┆ 0     ┆ image-id-3 │
│ demo.tfrecord     ┆ 316            ┆ 395          ┆ cat  ┆ 1     ┆ image-id-4 │
└───────────────────┴────────────────┴──────────────┴──────┴───────┴────────────┘
```
Explanation about the index format:
* **tfrecord_filename**: name of the tfrecord file
* **tfrecord_start**: start byte position of the example in the tfrecord file
* **tfrecord_end**: end byte position of the example in the tfrecord file
* other columns: indexed columns from the dataset with `indexed_cols_fn` function

### Dataset Reading

```python
import tfr_reader as tfr

tfrds = tfr.TFRecordDatasetReader("/path/to/directory/with/tfrecords")
# assume that the dataset is indexed already
tfrds = tfr.TFRecordDatasetReader("gs://bucket/path/to/directory/with/tfrecords")
# selection API
selected_df, examples = tfrds.select("SELECT * FROM index WHERE name = 'cat' LIMIT 20")
# custom selection
selected_df = tfrds.index_df.sample(5)
examples = tfrds.load_records(selected_df)
# indexing API
for i in range(len(tfrds)):
    example = tfrds[i]
    # assuming image is encoded as bytes at key "image/encoded"
    image_bytes = example["image/encoded"].value[0]
    # label is encoded as int64 at key "label"
    label = example["label"].value[0]
```


### Custom Protobuf Decoder for TFRecord files

If protobuf is not installed or it uses old and slow 'python' API
decoder, this library will use custom specialized protobuf decoder written in cython.
To enforce custom protobuf decoder for TFRecord files, run this command
```python
import tfr_reader as tfr
# to use custom protobuf decoder
tfr.set_decoder_type("cython")
# to use default protobuf decoder
tfr.set_decoder_type("protobuf")
```
