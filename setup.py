"""Setup script to handle all Cython extensions.

This file handles all Cython extensions including those that require NumPy's
include directory which can only be determined at build time.

Note: If numpy/cython are not installed, the extensions will not be built,
but the package will still install successfully.
"""

import multiprocessing

from setuptools import setup

# Try to import numpy and cython - only build extensions if available
try:
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension

    # Define all extensions with proper C++ and OpenMP configuration
    extensions = [
        # Basic Cython extensions (no numpy required at runtime)
        Extension(
            name="tfr_reader.cython.indexer",
            sources=["src/tfr_reader/cython/indexer.pyx"],
            extra_compile_args=["-finline-functions", "-O3"],
        ),
        Extension(
            name="tfr_reader.cython.decoder",
            sources=["src/tfr_reader/cython/decoder.pyx"],
            extra_compile_args=["-finline-functions", "-O3"],
        ),
        # Image classification extensions (require numpy)
        Extension(
            name="tfr_reader.datasets.image_classification.processor",
            sources=["src/tfr_reader/datasets/image_classification/processor.py"],
            include_dirs=[np.get_include(), "src"],
            extra_compile_args=["-finline-functions", "-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            language="c++",
        ),
        Extension(
            name="tfr_reader.datasets.image_classification.sampler",
            sources=["src/tfr_reader/datasets/image_classification/sampler.py"],
            include_dirs=[np.get_include(), "src"],
            extra_compile_args=["-finline-functions", "-O3"],
            language="c++",
        ),
        Extension(
            name="tfr_reader.cython.image",
            sources=["src/tfr_reader/cython/image.pyx"],
            include_dirs=[np.get_include(), "src"],
            extra_compile_args=["-finline-functions", "-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            libraries=["turbojpeg"],
            language="c++",
        ),
    ]

    ext_modules = cythonize(
        extensions,
        nthreads=multiprocessing.cpu_count(),  # Enable parallel compilation
        annotate=True,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
except ImportError:
    # If numpy or cython are not available, don't build the extensions
    # The package will still install, but image_classification features won't work
    ext_modules = []

setup(ext_modules=ext_modules)
