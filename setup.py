"""Setup script to handle NumPy include directory for extensions.

The other Cython extensions (indexer, decoder) are defined in pyproject.toml.
This file is needed for extensions that require NumPy's include directory
which can only be determined at build time.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define all extensions that need NumPy
extensions = [
    # Original TFRecordProcessor (backward compatibility)
    Extension(
        name="tfr_reader.datasets.image_classification.tfrecord_processor",
        sources=["src/tfr_reader/datasets/image_classification/tfrecord_processor.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    # New multi-file dataset components
    Extension(
        name="tfr_reader.datasets.image_classification.processor",
        sources=["src/tfr_reader/datasets/image_classification/processor.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-finline-functions", "-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        name="tfr_reader.datasets.image_classification.sampler",
        sources=["src/tfr_reader/datasets/image_classification/sampler.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-finline-functions", "-O3"],
    ),
    Extension(
        name="tfr_reader.datasets.image_classification.multi_file_reader",
        sources=["src/tfr_reader/datasets/image_classification/multi_file_reader.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-finline-functions", "-O3"],
    ),
    Extension(
        name="tfr_reader.datasets.image_classification.dataset",
        sources=["src/tfr_reader/datasets/image_classification/dataset.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-finline-functions", "-O3"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)
