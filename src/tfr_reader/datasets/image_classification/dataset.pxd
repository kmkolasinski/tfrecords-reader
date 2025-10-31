# cython: language_level=3
# distutils: language=c++

from tfr_reader.datasets.image_classification.multi_file_reader cimport MultiFileReader
from tfr_reader.datasets.image_classification.processor cimport ImageProcessor
from tfr_reader.datasets.image_classification.sampler cimport BatchSampler


cdef class TFRecordsImageDataset:
    cdef MultiFileReader file_reader
    cdef ImageProcessor image_processor
    cdef BatchSampler sampler
    cdef int batch_size
    cdef int prefetch_size
    cdef object prefetch_buffer
    cdef bint use_prefetch
    cdef bint _iterator_started

    cdef object _generate_batch(self)

    cpdef void reset(self)
    cpdef void shuffle(self)
    cpdef void set_epoch(self, int epoch)
