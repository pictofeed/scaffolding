# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Scaffolding — A deep learning framework written entirely in Python and Cython.

Implemented in Cython and Python with NumPy as the computational backend.
Uses NumPy as the computational backend with Cython-accelerated
hot-path operations (sigmoid, exp, matmul, AdamW, RMS norm) compiled
with ``nogil`` for maximum throughput.

Usage::

    import scaffolding as torch
    import scaffolding.nn as nn
    import scaffolding.nn.functional as F
    import scaffolding.optim as optim
    import scaffolding.distributed as dist
    from scaffolding.nn.parallel import DistributedDataParallel as DDP
    from scaffolding.utils.checkpoint import checkpoint as grad_checkpoint
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Pictofeed, LLC"

# ── Core tensor class & factory functions ──
from .tensor import (
    Tensor,
    tensor,
    zeros, zeros_like,
    ones, ones_like,
    full,
    randn, rand, randint,
    arange, linspace, empty,
    matmul, cat, stack, outer, where, cumsum,
    exp, log, sqrt, rsqrt, sigmoid, sin, cos,
    atan2, expm1, softmax, sort, multinomial,
    bernoulli,
    manual_seed, save, load, compile,
    set_float32_matmul_precision,
)

# ── Dtype constants ──
from .dtype import (
    dtype,
    float16, float32, float64,
    bfloat16, half,
    int8, int16, int32, int64, long,
    uint8,
)
# Override bool carefully (don't shadow builtin at module level for internal use)
from .dtype import bool as bool

# ── Device ──
from .device import device

# ── Autograd ──
from .autograd import (
    no_grad,
    inference_mode,
    is_grad_enabled,
    set_grad_enabled,
)

# ── Sub-packages ──
from . import nn
from . import optim
from . import distributed
from . import backends
from . import cuda
from . import utils

__all__ = [
    "__version__",
    "__author__",
    
    # Tensor
    'Tensor', 'tensor',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'full',
    'randn', 'rand', 'randint', 'arange', 'linspace', 'empty',
    'matmul', 'cat', 'stack', 'outer', 'where', 'cumsum',
    'exp', 'log', 'sqrt', 'rsqrt', 'sigmoid', 'sin', 'cos',
    'atan2', 'expm1', 'softmax', 'sort', 'multinomial', 'bernoulli',
    # Dtypes
    'dtype', 'float16', 'float32', 'float64', 'bfloat16', 'half',
    'int8', 'int16', 'int32', 'int64', 'long', 'uint8', 'bool',
    # Device
    'device',
    # Autograd
    'no_grad', 'inference_mode', 'is_grad_enabled', 'set_grad_enabled',
    # Utility
    'manual_seed', 'save', 'load', 'compile', 'set_float32_matmul_precision',
    # Sub-packages
    'nn', 'optim', 'distributed', 'backends', 'cuda', 'utils',
]
