from tfr_reader.example import example_pb2 as pb2


def create_dummy_example(index: int) -> pb2.Example:
    features = pb2.Features(
        feature={
            "bytes_feature": pb2.Feature(bytes_list=pb2.BytesList(value=[f"A{index}".encode()])),
            "float_feature": pb2.Feature(
                float_list=pb2.FloatList(value=[1.1 * index, 2.2 * index, 3.3 * index])
            ),
            "int64_feature": pb2.Feature(
                int64_list=pb2.Int64List(value=[10 * index, 20 * index, 30 * index])
            ),
        }
    )
    return pb2.Example(features=features)


def write_dummy_tfrecord(output_path, num_records: int = 5):
    with open(output_path, "wb") as f:
        for i in range(num_records):
            example = create_dummy_example(i + 1)
            serialized_example = example.SerializeToString()
            # Write the length of the serialized example as a 8-byte integer (little-endian)
            f.write(len(serialized_example).to_bytes(8, byteorder="little"))
            # Write the length CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
            # Write the serialized example
            f.write(serialized_example)
            # Write the data CRC (not used, but required by TFRecord format)
            f.write(b"\x00\x00\x00\x00")
