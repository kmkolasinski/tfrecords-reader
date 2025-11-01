#cython: language_level=3

from tfr_reader.datasets.image_classification.multi_file_reader cimport MultiFileReader
from tfr_reader.datasets.image_classification.prefetch cimport PrefetchBuffer
from tfr_reader.datasets.image_classification.processor cimport ImageProcessor
from tfr_reader.datasets.image_classification.sampler cimport BatchSampler

cdef class TFRecordsImageDataset:
    cdef public MultiFileReader file_reader
    cdef public ImageProcessor image_processor
    cdef public BatchSampler sampler
    cdef public int batch_size
    cdef public int prefetch_size
    cdef public object prefetch_buffer
    cdef public bint use_prefetch
    cdef readonly bint _iterator_started

    cpdef _generate_batch(self)
    cpdef reset(self)
    cpdef shuffle(self)
    cpdef set_epoch(self, int epoch)
