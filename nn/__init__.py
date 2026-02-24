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
    Dropout,
    SiLU,
    ReLU,
    GELU,
    LayerNorm,
    RMSNorm,
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

__all__ = [
    'Module', 'ModuleList', 'Sequential',
    'Parameter',
    'Linear', 'Embedding', 'Conv1d', 'Dropout',
    'SiLU', 'ReLU', 'GELU', 'LayerNorm', 'RMSNorm',
    'DataParallel', 'DistributedDataParallel',
    'set_default_device',
    'functional', 'init', 'utils',
]
