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
    params = [p for p in parameters if p._grad is not None]
    if not params:
        return 0.0

    if norm_type == float('inf'):
        total_norm = max(np.max(np.abs(p._grad)) for p in params)
    elif norm_type == 2.0:
        # Optimized L2 norm: use dot product instead of abs**2
        total_norm_sq = 0.0
        for p in params:
            g = p._grad.ravel()
            total_norm_sq += np.dot(g, g)
        total_norm = math.sqrt(total_norm_sq)
    else:
        total_norm = 0.0
        for p in params:
            total_norm += np.sum(np.abs(p._grad) ** norm_type)
        total_norm = float(total_norm ** (1.0 / norm_type))

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p._grad *= clip_coef

    return total_norm
