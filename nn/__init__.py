# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.nn — Neural network module API (mirrors torch.nn)."""
from __future__ import annotations

# Module base class & containers
from .module import Module, ModuleList, Sequential

# Parameter
from .parameter import Parameter

# Layers
from .layers import (
    Linear,
    Embedding,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose2d,
    ConvTranspose3d,
    Dropout,
    SiLU,
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    LayerNorm,
    RMSNorm,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    AvgPool2d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Upsample,
    PixelShuffle,
    DataParallel,
    set_default_device,
)

# Functional API (accessible as nn.functional or F)
from . import functional

# Initialization routines
from . import init

# Parallel
from .parallel import DistributedDataParallel

# Utils (nn.utils.clip_grad_norm_)
from . import utils

# Auto-select CUDA as the default parameter device when available
try:
    from ..tensor import _USE_CUDA
    if _USE_CUDA:
        set_default_device('cuda:0')
except Exception:
    pass

__all__ = [
    'Module', 'ModuleList', 'Sequential',
    'Parameter',
    'Linear', 'Embedding',
    'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose2d', 'ConvTranspose3d',
    'Dropout',
    'SiLU', 'ReLU', 'GELU', 'Tanh', 'Sigmoid',
    'LayerNorm', 'RMSNorm', 'BatchNorm2d', 'BatchNorm3d', 'GroupNorm',
    'AvgPool2d', 'MaxPool2d', 'AdaptiveAvgPool2d',
    'Upsample', 'PixelShuffle',
    'DataParallel', 'DistributedDataParallel',
    'set_default_device',
    'functional', 'init', 'utils',
]
