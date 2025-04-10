# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False
# distutils: language=c++
# ##cython: linetrace=True
"""
Protobuf definition
https://protobuf.dev/programming-guides/encoding/
https://github.com/dogtopus/minipb/blob/master/minipb.py
"""

from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t
from libcpp.vector cimport vector


cdef:
    int WIRE_TYPE_VARINT = 0  # int32, int64, uint32, uint64, sint32, sint64, bool, enum
    int WIRE_TYPE_FIXED64 = 1  # double, fixed64, sfixed64
    int WIRE_TYPE_LENGTH_DELIMITED = 2  # string, bytes, embedded messages, packed repeated fields
    int WIRE_TYPE_FIXED32 = 5  # float, fixed32, sfixed32


cdef struct varint_t:
    int64_t value
    int64_t pos


cdef struct field_t:
    int64_t field_number
    int64_t wire_type
    int64_t length
    const unsigned char* buffer_pt


cdef varint_t decode_varint(const unsigned char* buffer, int64_t pos):
    """Reads a varint from the buffer starting at position pos."""
    cdef:
        int64_t result = 0
        int64_t shift = 0
        unsigned char b

    while True:
        b = buffer[pos]
        pos += 1
        result |= ((b & 0x7F) << shift)
        if not (b & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise Exception('Too many bytes when decoding varint.')
    return varint_t(result, pos)


cdef decode_message(
    vector[field_t] *fields,
    const unsigned char* buffer,
    int64_t pos,
    int64_t end
):
    """Parses a protobuf message from buffer starting at pos until end."""

    cdef:
        int64_t field_number
        int64_t wire_type
        int64_t length
        int64_t key
        field_t field
        const unsigned char* pointer

    while pos < end:
        variant = decode_varint(buffer, pos)
        key = variant.value
        pos = variant.pos

        field_number = key >> 3
        wire_type = key & 0x07

        if wire_type == WIRE_TYPE_FIXED64:
            if pos + 8 > end:
                raise Exception('Unexpected end of buffer when reading fixed64.')
            pointer = buffer + pos
            pos += 8
            field = field_t(field_number, wire_type, 8, pointer)
            fields.push_back(field)
        elif wire_type == WIRE_TYPE_LENGTH_DELIMITED:
            variant = decode_varint(buffer, pos)
            length = variant.value
            pos = variant.pos
            if pos + length > end:
                raise Exception('Unexpected end of buffer when reading length-delimited field.')
            pointer = buffer + pos

            pos += length
            field = field_t(field_number, wire_type, length, pointer)
            fields.push_back(field)

        elif wire_type == WIRE_TYPE_FIXED32:
            if pos + 4 > end:
                raise Exception('Unexpected end of buffer when reading fixed32.')
            pointer = buffer + pos
            pos += 4
            field = field_t(field_number, wire_type, 4, pointer)
            fields.push_back(field)
        else:
            raise Exception('Unsupported wire type: {}'.format(wire_type))


cpdef Example example_from_bytes(const unsigned char[:] buffer):
    cdef:
        int64_t pos = 0
        int64_t end = len(buffer)
        field_t field
        vector[field_t] fields

    decode_message(&fields, &buffer[0], pos, end)

    features = None
    for fi in range(fields.size()):
        field = fields[fi]
        if field.field_number == 1:  # features field
            if field.wire_type == WIRE_TYPE_LENGTH_DELIMITED:
                features = features_from_bytes(field.buffer_pt, field.length)
            else:
                raise Exception('Unexpected wire type for field features')
        else:
            # Ignore unknown fields
            pass
    return Example(features)


cdef Features features_from_bytes(const unsigned char* buffer, uint64_t length):
    cdef:
        int64_t pos = 0
        int64_t end = length
        field_t field
        vector[field_t] fields
        dict[str, Feature] feature = {}

    decode_message(&fields, buffer, pos, end)

    for fi in range(fields.size()):
        field = fields[fi]
        if field.field_number == 1:  # feature map
            if field.wire_type == WIRE_TYPE_LENGTH_DELIMITED:
                value = parse_map_entry(field.buffer_pt, field.length)
                feature[value.key] = value
            else:
                raise Exception('Unexpected wire type for field feature')
        else:
            pass
    return Features(feature)


cdef Feature parse_map_entry(const unsigned char* buffer, uint64_t length):
    """Parses a map entry with key and value parsers."""
    cdef:
        int64_t pos = 0
        int64_t end = length
        field_t field
        vector[field_t] fields
        str key

    decode_message(&fields, buffer, pos, end)
    field = fields[0]
    key = bytes(field.buffer_pt[:field.length]).decode('utf-8')
    field = fields[1]
    return feature_from_bytes(key, field.buffer_pt, field.length)


cdef Feature feature_from_bytes(str key, const unsigned char* buffer, int64_t length):
    cdef:
        int64_t pos = 0
        int64_t end = length
        field_t field
        vector[field_t] fields

    decode_message(&fields, buffer, pos, end)
    field = fields[0]

    if field.field_number == 1:  # bytes_list
        return Feature(
            key,
            'bytes_list',
            bytes_list=bytes_list_from_bytes(field.buffer_pt, field.length),
        )
    elif field.field_number == 2:  # float_list

        return Feature(
            key,
            'float_list',
            float_list=float32_list_from_bytes(field.buffer_pt, field.length),
        )
    elif field.field_number == 3:  # int64_list
        return Feature(
            key,
            'int64_list',
            int64_list=int64_list_from_bytes(field.buffer_pt, field.length),
        )
    else:
        raise Exception('Unexpected field number in Feature')



cdef BytesList bytes_list_from_bytes(const unsigned char* buffer, int64_t length):
    cdef:
        int64_t pos = 0
        int64_t end = length
        field_t field
        vector[field_t] fields
        list[bytes] value = []


    decode_message(&fields, buffer, pos, end)

    for fi in range(fields.size()):
        field = fields[fi]
        if field.field_number == 1:
            if field.wire_type == WIRE_TYPE_LENGTH_DELIMITED:
                value.append(bytes(field.buffer_pt[:field.length]))
            else:
                raise Exception('Unexpected wire type in BytesList')
        else:
            pass
    return BytesList(value)


cdef inline float unpack_float32(const unsigned char *float_bytes):
    """
    Unpack length_bytes into uint64_t length (little-endian)
    """
    cdef float value
    memcpy(&value, &float_bytes[0], sizeof(float))
    return value


cdef FloatList float32_list_from_bytes(const unsigned char* buffer, int64_t length):
    cdef:
        int64_t pos = 0
        int64_t end = length
        int64_t start
        int64_t num_floats
        int64_t i
        float float_value
        vector[float] value
        field_t field
        vector[field_t] fields

    decode_message(&fields, buffer, pos, end)

    for fi in range(fields.size()):
        field = fields[fi]

        if field.field_number == 1:
            if field.wire_type == WIRE_TYPE_LENGTH_DELIMITED:
                # packed repeated floats
                num_floats = field.length // 4
                for i in range(num_floats):
                    start = i * 4
                    float_value = unpack_float32(field.buffer_pt+start)
                    value.push_back(float_value)
            elif field.wire_type == WIRE_TYPE_FIXED32:
                float_value = unpack_float32(field.buffer_pt)
                value.push_back(float_value)
            else:
                raise Exception('Unexpected wire type in FloatList')
        else:
            pass
    return FloatList(value)


cdef Int64List int64_list_from_bytes(const unsigned char* buffer, int64_t length):
    cdef:
        int64_t pos = 0
        int64_t end = length
        int64_t int_value
        int64_t pos_inner
        vector[int64_t] value   # int64
        field_t field
        vector[field_t] fields

    decode_message(&fields, buffer, pos, end)

    for fi in range(fields.size()):
        field = fields[fi]
        if field.field_number == 1:
            if field.wire_type == WIRE_TYPE_LENGTH_DELIMITED:
                # packed repeated varints
                pos_inner = 0
                end_inner = field.length
                while pos_inner < end_inner:
                    variant = decode_varint(field.buffer_pt, pos_inner)
                    pos_inner = variant.pos
                    value.push_back(variant.value)
            elif field.wire_type == WIRE_TYPE_VARINT:
                variant = decode_varint(field.buffer_pt, 0)
                value.push_back(variant.value)
            else:
                raise Exception('Unexpected wire type in Int64List')
        else:
            pass
    return Int64List(value)


cdef class Example:
    cdef public Features features
    def __init__(self, Features features):
        self.features = features


cdef class Features:
    cdef public dict[str, Feature] feature
    def __init__(self, dict[str, Feature] feature):
        self.feature = feature  # dict mapping string to Feature


cdef class Feature:
    cdef str key
    cdef str kind
    # one of implementation
    cdef FloatList _float_list
    cdef Int64List _int64_list
    cdef BytesList _bytes_list

    def __init__(
        self,
        str key,
        str kind,
        FloatList float_list = FloatList(vector[float]()),
        Int64List int64_list = Int64List(vector[int64_t]()),
        BytesList bytes_list = BytesList([]),
    ):
        self.key = key
        self.kind = kind
        self._float_list = float_list
        self._int64_list = int64_list
        self._bytes_list = bytes_list

    def WhichOneof(self, kind: str) -> str:
        # this follows the Google protobug API
        return self.kind

    @property
    def float_list(self) -> FloatList:
        if self.kind != 'float_list':
            raise Exception('Feature is not a float_list')
        return self._float_list

    @property
    def int64_list(self) -> Int64List:
        if self.kind != 'int64_list':
            raise Exception('Feature is not an int64_list')
        return self._int64_list

    @property
    def bytes_list(self) -> BytesList:
        if self.kind != 'bytes_list':
            raise Exception('Feature is not a bytes_list')
        return self._bytes_list


cdef class BytesList:
    cdef public list[bytes] value

    def __init__(self, list[bytes] value):
        self.value = value

    def __getitem__(self, int item):
        return self.value[item]

    def __len__(self):
        return len(self.value)


cdef class FloatList:
    cdef public vector[float] value

    def __init__(self, vector[float] value):
        self.value = value

    def __getitem__(self, int item):
        return self.value[item]


cdef class Int64List:
    cdef public vector[int64_t] value

    def __init__(self, vector[int64_t] value):
        self.value = value

    def __getitem__(self, int item):
        return self.value[item]
