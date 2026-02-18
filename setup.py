# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Scaffolding build configuration.

Compiles up to three Cython extensions:
  1. ``scaffolding._tensor_ops``  — generic CPU hot-path kernels (nogil)
  2. ``scaffolding._mps_ops``     — Apple Accelerate-backed kernels (macOS)
  3. ``scaffolding._cuda_ops``    — NVIDIA CUDA GPU kernels (Linux/Windows)

The MPS extension is only built on macOS where Accelerate.framework is
available.  The CUDA extension is only built when ``nvcc`` is found or
``CUDA_HOME`` / ``CUDA_PATH`` is set.

Build
-----
    python setup.py build_ext --inplace       # development
    pip install -e .                          # editable install
    python setup.py bdist_wheel               # wheel

CUDA-specific environment variables:
    CUDA_HOME / CUDA_PATH   — root of the CUDA toolkit
    SCAFFOLDING_CUDA_ARCH   — semicolon-separated arch list, e.g. "70;80;90"
                              (default: auto-detect or broad set)
    SCAFFOLDING_NO_CUDA     — set to 1 to skip CUDA entirely

If CMake is preferred, see the top-level ``CMakeLists.txt``.
"""
import os
import re
import shutil
import subprocess
import sys
import platform
import numpy as np

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

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
IS_WINDOWS = platform.system() == 'Windows'

# ══════════════════════════════════════════════════════════════════════
#  CUDA detection helpers
# ══════════════════════════════════════════════════════════════════════

def _has_cuda_files(path: str) -> bool:
    """Return True if *path* looks like a CUDA toolkit root.

    We check for the presence of an ``include/cuda.h`` header **or**
    an ``nvcc`` binary inside ``bin/``.  This avoids false-positives
    when the derived path is just ``/usr``.
    """
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, 'include', 'cuda.h')):
        return True
    if os.path.isfile(os.path.join(path, 'bin', 'nvcc')):
        return True
    return False


def _find_cuda_home() -> str | None:
    """Locate the CUDA toolkit root directory.

    Detection strategy (first match wins):
      1. ``CUDA_HOME`` or ``CUDA_PATH`` environment variable.
      2. ``nvcc`` on ``$PATH`` → derive toolkit root.  A special
         case handles the Ubuntu apt layout where ``nvcc`` lives in
         ``/usr/bin`` but CUDA headers/libs are elsewhere.
      3. ``/etc/alternatives/cuda`` symlink (Debian/Ubuntu).
      4. ``/usr/local/cuda`` symlink (NVIDIA .run / .deb installer).
      5. Versioned directories ``/usr/local/cuda-XX.Y`` (highest
         version wins).
      6. ``/opt/cuda``.
      7. The ``/usr`` prefix itself (Ubuntu ``nvidia-cuda-toolkit``
         package places headers in ``/usr/include`` and libs in
         ``/usr/lib``).
      8. Windows default paths.
    """
    import glob as _glob

    # 1. Explicit env var
    for var in ('CUDA_HOME', 'CUDA_PATH'):
        p = os.environ.get(var)
        if p and os.path.isdir(p):
            return p

    # 2. nvcc on PATH → derive CUDA_HOME
    nvcc = shutil.which('nvcc')
    if nvcc:
        real_nvcc = os.path.realpath(nvcc)
        derived = os.path.dirname(os.path.dirname(real_nvcc))
        # If derived is something like /usr/local/cuda — great.
        # If derived is /usr (apt install), check first before accepting;
        # prefer more specific paths found below.
        if derived != '/usr' and _has_cuda_files(derived):
            return derived

    # 3. /etc/alternatives/cuda (Debian / Ubuntu)
    alt = '/etc/alternatives/cuda'
    if os.path.isdir(alt):
        real = os.path.realpath(alt)
        if _has_cuda_files(real):
            return real

    # 4. /usr/local/cuda symlink
    if _has_cuda_files('/usr/local/cuda'):
        return '/usr/local/cuda'

    # 5. Versioned dirs /usr/local/cuda-XX.Y — pick the newest
    versioned = sorted(
        _glob.glob('/usr/local/cuda-*'),
        key=lambda p: list(map(int, re.findall(r'\d+', os.path.basename(p)))),
        reverse=True,
    )
    for candidate in versioned:
        if _has_cuda_files(candidate):
            return candidate

    # 6. /opt/cuda
    if _has_cuda_files('/opt/cuda'):
        return '/opt/cuda'

    # 7. /usr prefix (Ubuntu nvidia-cuda-toolkit apt package)
    if _has_cuda_files('/usr'):
        return '/usr'

    # 8. Windows default paths
    if IS_WINDOWS:
        for wver in ('12.8', '12.6', '12.4', '12.3', '12.2', '12.1',
                      '12.0', '11.8', '11.7'):
            wpath = (f'C:\\Program Files\\NVIDIA GPU Computing '
                     f'Toolkit\\CUDA\\v{wver}')
            if os.path.isdir(wpath):
                return wpath

    return None


def _cuda_version(cuda_home: str) -> tuple[int, int]:
    """Parse CUDA toolkit version from version.txt or nvcc --version."""
    # Try version.json (CUDA 11.8+)
    import json
    version_json = os.path.join(cuda_home, 'version.json')
    if os.path.isfile(version_json):
        with open(version_json) as f:
            data = json.load(f)
        v = data.get('cuda', {}).get('version', '')
        parts = v.split('.')
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
    # Try version.txt (older)
    version_txt = os.path.join(cuda_home, 'version.txt')
    if os.path.isfile(version_txt):
        with open(version_txt) as f:
            text = f.read()
        m = re.search(r'(\d+)\.(\d+)', text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    # Try nvcc --version
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    if os.path.isfile(nvcc):
        try:
            out = subprocess.check_output(
                [nvcc, '--version'], stderr=subprocess.STDOUT
            ).decode()
            m = re.search(r'release (\d+)\.(\d+)', out)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        except Exception:
            pass
    return (0, 0)


def _gencode_flags(cuda_home: str) -> list[str]:
    """Generate --gencode flags for nvcc.

    Covers SM 3.5 (Kepler) through SM 9.0 (Hopper), filtered by
    the toolkit version so we don't pass unsupported arch to old nvcc.
    """
    # User override
    env_arch = os.environ.get('SCAFFOLDING_CUDA_ARCH', '')
    if env_arch:
        arches = [a.strip() for a in env_arch.replace(',', ';').split(';')
                  if a.strip()]
    else:
        major, minor = _cuda_version(cuda_home)
        cuda_ver = major * 10 + minor  # e.g. 118 for 11.8

        # (min_cuda_ver_x10, sm)
        all_arches = [
            (0,   '35'),   # Kepler — CUDA 5+
            (0,   '37'),
            (0,   '50'),   # Maxwell
            (0,   '52'),
            (0,   '53'),
            (80,  '60'),   # Pascal — CUDA 8+
            (80,  '61'),
            (80,  '62'),
            (90,  '70'),   # Volta — CUDA 9+
            (100, '75'),   # Turing — CUDA 10+
            (110, '80'),   # Ampere — CUDA 11+
            (111, '86'),   # CUDA 11.1+
            (118, '89'),   # Ada Lovelace — CUDA 11.8+
            (118, '90'),   # Hopper — CUDA 11.8+
        ]
        arches = [
            sm for (min_ver, sm) in all_arches
            if cuda_ver >= min_ver
        ]
        # Drop sm_35/sm_37 if CUDA >= 12 (deprecated / removed)
        if cuda_ver >= 120:
            arches = [a for a in arches if a not in ('35', '37')]

    flags = []
    for sm in arches:
        flags.append(
            f'-gencode=arch=compute_{sm},code=sm_{sm}'
        )
    # PTX for the highest arch so future GPUs can JIT
    if arches:
        highest = arches[-1]
        flags.append(
            f'-gencode=arch=compute_{highest},code=compute_{highest}'
        )
    return flags


# ══════════════════════════════════════════════════════════════════════
#  Custom build_ext that compiles .cu with nvcc before linking
# ══════════════════════════════════════════════════════════════════════

class CudaBuildExt(_build_ext):
    """Extended build_ext that compiles CUDA sources with nvcc."""

    def build_extensions(self):
        for ext in self.extensions:
            cuda_sources = [
                s for s in ext.sources if s.endswith('.cu')
            ]
            if not cuda_sources:
                continue
            cuda_home = _find_cuda_home()
            if cuda_home is None:
                raise RuntimeError(
                    'CUDA sources found but CUDA toolkit not detected. '
                    'Set CUDA_HOME or install the CUDA toolkit.'
                )
            nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
            if IS_WINDOWS:
                nvcc += '.exe'

            # Ensure temp build dir exists
            build_temp = self.build_temp
            os.makedirs(build_temp, exist_ok=True)

            obj_files = []
            gencode = _gencode_flags(cuda_home)

            # K80-specific: allow env override for max register count
            # Default 64 optimizes for Kepler occupancy; set higher for
            # compute-heavy kernels on newer GPUs
            max_regs = int(os.environ.get('SCAFFOLDING_MAXREG', '64'))

            for cu_src in cuda_sources:
                obj_name = os.path.splitext(os.path.basename(cu_src))[0]
                obj_file = os.path.join(
                    build_temp, obj_name + ('.obj' if IS_WINDOWS else '.o')
                )
                cmd = [
                    nvcc, '-c', cu_src, '-o', obj_file,
                    '-O3',
                    '--use_fast_math',         # K80: enables __expf, __logf, etc.
                    '--compiler-options', "'-fPIC'" if not IS_WINDOWS else '',
                    '-DNDEBUG',
                    '-Xcompiler', '-O3',
                    '-Xptxas', '-v',           # Print register usage (for tuning)
                    '--maxrregcount=%d' % max_regs,  # K80: limit regs → higher occupancy
                ] + gencode

                print(f'[scaffolding] nvcc: {" ".join(cmd)}')
                subprocess.check_call(cmd)
                obj_files.append(obj_file)

                # Remove .cu from sources — the C compiler doesn't know it
                ext.sources.remove(cu_src)

            # Add compiled .o / .obj to extra_objects for the linker
            ext.extra_objects = getattr(ext, 'extra_objects', []) + obj_files

        # Now let setuptools handle the rest (Cython → C → .so)
        super().build_extensions()


# ══════════════════════════════════════════════════════════════════════
#  Extension 1: _tensor_ops (generic CPU) — macOS only
#  Extension 2: _mps_ops (macOS Accelerate) — macOS only
#
#  Both extensions depend on Cython + macOS toolchains.  On Linux the
#  pre-generated .c files are not shipped, so we skip them entirely;
#  the runtime imports in tensor.py / nn/ gracefully fall back to
#  pure-NumPy or CUDA paths.
# ══════════════════════════════════════════════════════════════════════

extensions: list[Extension] = []

if IS_MACOS:
    tensor_ops_src = '_tensor_ops.pyx' if USE_CYTHON else '_tensor_ops.c'
    # Only add if the source file actually exists
    if os.path.isfile(tensor_ops_src):
        tensor_ops_ext = Extension(
            name='scaffolding._tensor_ops',
            sources=[tensor_ops_src],
            include_dirs=[np.get_include()],
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
            language='c',
        )
        extensions.append(tensor_ops_ext)
    else:
        print(f'[scaffolding] Skipping _tensor_ops — {tensor_ops_src} not found')

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

# ══════════════════════════════════════════════════════════════════════
#  Extension 3: _cuda_ops (NVIDIA GPU)
# ══════════════════════════════════════════════════════════════════════

_NO_CUDA = os.environ.get('SCAFFOLDING_NO_CUDA', '0') == '1'
CUDA_HOME = _find_cuda_home() if not _NO_CUDA else None


def _build_cuda_extension():
    """Build the CUDA extension if toolkit is available."""
    if CUDA_HOME is None:
        return None

    cuda_major, cuda_minor = _cuda_version(CUDA_HOME)
    print(f'[scaffolding] Found CUDA {cuda_major}.{cuda_minor} '
          f'at {CUDA_HOME}')

    cuda_include = os.path.join(CUDA_HOME, 'include')
    # Determine CUDA library directory — differs across distros:
    #   /usr/local/cuda/lib64          (NVIDIA .run installer)
    #   /usr/lib/x86_64-linux-gnu      (Ubuntu apt, x86-64)
    #   /usr/lib/aarch64-linux-gnu     (Ubuntu apt, ARM64)
    #   CUDA_HOME/lib/x64              (Windows)
    #   CUDA_HOME/lib                  (generic fallback)
    cuda_lib_candidates = [
        os.path.join(CUDA_HOME, 'lib64'),
    ]
    # Ubuntu multiarch: /usr/lib/<triplet>
    machine = platform.machine()
    if machine in ('x86_64', 'AMD64'):
        cuda_lib_candidates.append(
            os.path.join(CUDA_HOME, 'lib', 'x86_64-linux-gnu'))
    elif machine == 'aarch64':
        cuda_lib_candidates.append(
            os.path.join(CUDA_HOME, 'lib', 'aarch64-linux-gnu'))
    cuda_lib_candidates += [
        os.path.join(CUDA_HOME, 'lib', 'x64'),   # Windows
        os.path.join(CUDA_HOME, 'lib'),           # fallback
    ]
    cuda_lib = next(
        (p for p in cuda_lib_candidates if os.path.isdir(p)),
        os.path.join(CUDA_HOME, 'lib64'),  # default even if missing
    )

    cuda_ops_src = '_cuda_ops.pyx' if USE_CYTHON else '_cuda_ops.c'

    # If Cython is not installed and the pre-generated .c file doesn't
    # exist, we cannot build.
    if not os.path.isfile(cuda_ops_src):
        print(f'[scaffolding] Skipping _cuda_ops — {cuda_ops_src} not found. '
              f'Install Cython (pip install cython) to compile from .pyx.')
        return None

    compile_args = ['-O3']
    if not IS_WINDOWS:
        compile_args.append('-ffast-math')

    link_args = [f'-L{cuda_lib}']
    libraries = ['cudart', 'cublas', 'curand']

    # On Linux, add rpath so the .so finds libcudart at runtime
    if not IS_WINDOWS and not IS_MACOS:
        link_args.append(f'-Wl,-rpath,{cuda_lib}')

    ext = Extension(
        name='scaffolding._cuda_ops',
        sources=[
            cuda_ops_src,
            '_cuda_kernels.cu',   # compiled by CudaBuildExt with nvcc
        ],
        include_dirs=[
            np.get_include(),
            cuda_include,
            '.',  # for _cuda_kernels.cuh
        ],
        library_dirs=[cuda_lib],
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c',
    )
    return ext


cuda_ext = _build_cuda_extension()
if cuda_ext is not None:
    extensions.append(cuda_ext)
    print('[scaffolding] CUDA extension will be built.')
else:
    if not _NO_CUDA:
        print('[scaffolding] CUDA toolkit not found — skipping CUDA extension.')

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
    # Filter out .cu files before cythonize (they confuse Cython)
    for ext in extensions:
        ext.sources = [s for s in ext.sources if not s.endswith('.cu')]

    extensions = cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generates .html annotation files
    )

    # Re-add .cu sources so CudaBuildExt can find them
    if cuda_ext is not None:
        for ext in extensions:
            if ext.name == 'scaffolding._cuda_ops':
                ext.sources.append('_cuda_kernels.cu')
                break

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
    author_email='engineering@pictofeed.io',
    description=(
        'A deep learning framework written entirely in Python and Cython — '
        'NumPy + Apple Accelerate + NVIDIA CUDA'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pictofeed/scaffolding',
    license='Proprietary',

    package_dir={
        'scaffolding': '.',
        'scaffolding.nn': 'nn',
        'scaffolding.optim': 'optim',
        'scaffolding.backends': 'backends',
        'scaffolding.cuda': 'cuda',
        'scaffolding.distributed': 'distributed',
        'scaffolding.utils': 'utils',
    },
    packages=[
        'scaffolding',
        'scaffolding.nn',
        'scaffolding.optim',
        'scaffolding.backends',
        'scaffolding.cuda',
        'scaffolding.distributed',
        'scaffolding.utils',
    ],
    ext_modules=extensions,
    cmdclass={'build_ext': CudaBuildExt},

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
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False,
)
