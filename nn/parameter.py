# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.Parameter — learnable tensor wrapper."""
from __future__ import annotations

from ..tensor import Tensor
import numpy as np


class Parameter(Tensor):
    """A :class:`Tensor` that is automatically registered as a module parameter."""

    def __init__(self, data: Tensor | np.ndarray | None = None,
                 requires_grad: bool = True):
        if data is None:
            data = Tensor(np.empty(0, dtype=np.float32))
        if isinstance(data, Tensor):
            super().__init__(data._ensure_cpu().copy(), requires_grad=requires_grad)
        elif isinstance(data, np.ndarray):
            super().__init__(data.copy(), requires_grad=requires_grad)
        else:
            super().__init__(np.asarray(data), requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"
