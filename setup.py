# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Scaffolding build configuration.

Compiles two Cython extensions:
  1. ``scaffolding._tensor_ops``  — generic CPU hot-path kernels (nogil)
  2. ``scaffolding._mps_ops``     — Apple Accelerate-backed kernels (nogil)

The MPS extension is only built on macOS where Accelerate.framework is
available.  On other platforms only ``_tensor_ops`` is compiled.

Build
-----
    python setup.py build_ext --inplace       # development
    pip install -e .                          # editable install
    python setup.py bdist_wheel               # wheel

If CMake is preferred, see the top-level ``CMakeLists.txt``.
"""
import os
import sys
import platform
import numpy as np

from setuptools import setup, Extension, find_packages

# ── Try Cython; fall back to pre-generated .c files ──
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

# ── Common compile flags ──
EXTRA_COMPILE_ARGS: list[str] = ['-O3', '-ffast-math']
EXTRA_LINK_ARGS: list[str] = []

IS_MACOS = platform.system() == 'Darwin'

# ── Extension 1: _tensor_ops (generic CPU) ──
tensor_ops_src = '_tensor_ops.pyx' if USE_CYTHON else '_tensor_ops.c'
tensor_ops_ext = Extension(
    name='scaffolding._tensor_ops',
    sources=[tensor_ops_src],
    include_dirs=[np.get_include()],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    language='c',
)

# ── Extension 2: _mps_ops (macOS Accelerate) ──
extensions = [tensor_ops_ext]

if IS_MACOS:
    mps_ops_src = '_mps_ops.pyx' if USE_CYTHON else '_mps_ops.c'
    mps_compile_args = EXTRA_COMPILE_ARGS + [
        '-DACCELERATE_NEW_LAPACK',
        '-DACCELERATE_LAPACK_ILP64',
    ]
    mps_link_args = ['-framework', 'Accelerate']

    mps_ops_ext = Extension(
        name='scaffolding._mps_ops',
        sources=[mps_ops_src],
        include_dirs=[np.get_include()],
        extra_compile_args=mps_compile_args,
        extra_link_args=mps_link_args,
        language='c',
    )
    extensions.append(mps_ops_ext)

# ── Cythonize if Cython is available ──
if USE_CYTHON:
    compiler_directives = {
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'nonecheck': False,
        'initializedcheck': False,
    }
    extensions = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generates .html annotation files
    )

# ── Package metadata ──
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r',
          encoding='utf-8') as fh:
    try:
        long_description = fh.read()
    except FileNotFoundError:
        long_description = ''

setup(
    name='scaffolding',
    version='0.1.0',
    author='Pictofeed, LLC',
    author_email='engineering@helixtechnologies.dev',
    description=(
        'A deep learning framework written entirely in Python and Cython — '
        'NumPy + Apple Accelerate'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/helix-technologies/scaffolding',
    license='Proprietary',

    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks']),
    ext_modules=extensions,

    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24',
    ],
    extras_require={
        'dev': [
            'cython>=3.0',
            'pytest>=7.0',
            'pytest-benchmark',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False,
)
