# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.utils — gradient clipping and other utilities."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor


def clip_grad_norm_(parameters, max_norm: float,
                    norm_type: float = 2.0) -> float:
    """Clip gradient norm of an iterable of parameters.

    Returns the total norm before clipping.
    """
    params = list(parameters)
    if norm_type == float('inf'):
        total_norm = max(
            np.max(np.abs(p._grad)) if p._grad is not None else 0.0
            for p in params
        )
    else:
        total_norm = 0.0
        for p in params:
            if p._grad is not None:
                total_norm += np.sum(np.abs(p._grad) ** norm_type)
        total_norm = float(total_norm ** (1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p._grad is not None:
                p._grad *= clip_coef

    return total_norm
