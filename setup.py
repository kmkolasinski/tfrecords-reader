"""Setup script to handle NumPy include directory for tfrecord_processor extension.

The other Cython extensions (indexer, decoder) are defined in pyproject.toml.
This file is only needed for the tfrecord_processor extension because it requires
NumPy's include directory which can only be determined at build time.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Only define the extension that needs NumPy
extensions = [
    Extension(
        name="tfr_reader.datasets.image_classification.tfrecord_processor",
        sources=["src/tfr_reader/datasets/image_classification/tfrecord_processor.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
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
