"""Setup script to handle NumPy include directory for extensions.

The other Cython extensions (indexer, decoder) are defined in pyproject.toml.
This file is needed for extensions that require NumPy's include directory
which can only be determined at build time.

All image_classification modules now use Cython Pure Python syntax (.py files)
and are compiled via cythonize with numpy include directories.

Note: If numpy/cython are not installed, the image_classification extensions
will not be built, but the package will still install successfully.
"""

import multiprocessing

from setuptools import setup

# Try to import numpy and cython - only build extensions if available
try:
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension

    # Define extensions with proper C++ and OpenMP configuration
    extensions = [
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
