# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Optimizer base class and AdamW implementation."""
from __future__ import annotations

import math
import numpy as np
from typing import Iterator

from ..tensor import Tensor
from ..nn.parameter import Parameter

# Try Accelerate-backed AdamW step (preferred on macOS)
try:
    from .._mps_ops import accelerate_adamw_step as _accel_adamw
    _USE_MPS_ADAM = True
except ImportError:
    _USE_MPS_ADAM = False

# Try Cython-accelerated AdamW step (fallback)
try:
    from .._tensor_ops import adamw_step_f32
    _USE_CYTHON_ADAM = True
except ImportError:
    _USE_CYTHON_ADAM = False

# Try CUDA-accelerated AdamW (fused GPU kernel)
try:
    from .._cuda_ops import cuda_adamw_step as _cuda_adamw
    _USE_CUDA_ADAM = True
except ImportError:
    _USE_CUDA_ADAM = False


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params, defaults: dict):
        self.defaults = defaults
        self.param_groups: list[dict] = []
        self.state: dict[int, dict] = {}

        if isinstance(params, (list, tuple)) and len(params) > 0:
            if isinstance(params[0], dict):
                for group in params:
                    pg = {**defaults, **group}
                    if 'params' in pg:
                        pg['params'] = list(pg['params'])
                    self.param_groups.append(pg)
            else:
                self.param_groups.append({**defaults, 'params': list(params)})
        else:
            self.param_groups.append({**defaults, 'params': list(params)})

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p._grad = None
                elif p._grad is not None:
                    p._grad.fill(0)

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {
            'state': self.state,
            'param_groups': [
                {k: v for k, v in g.items() if k != 'params'}
                for g in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: dict):
        self.state = state_dict.get('state', {})


class AdamW(Optimizer):
    """AdamW optimizer — decoupled weight decay regularization."""

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p._grad is None:
                    continue

                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {
                        'step': 0,
                        'm': np.zeros_like(p._data),
                        'v': np.zeros_like(p._data),
                    }
                st = self.state[pid]
                st['step'] += 1
                t = st['step']
                m, v = st['m'], st['v']
                grad = p._grad
                # Ensure grad dtype matches param dtype (autograd may
                # produce float64 gradients even for float32 params)
                if grad.dtype != p._data.dtype:
                    grad = grad.astype(p._data.dtype)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                if (_USE_CUDA_ADAM and p._data.dtype == np.float32
                        and hasattr(p, '_device') and p._device._type == 'cuda'):
                    # Fused CUDA AdamW kernel — single pass over all params
                    _cuda_adamw(p._data, grad, m, v,
                                lr, beta1, beta2, eps, wd, t)
                elif _USE_MPS_ADAM and p._data.dtype == np.float32:
                    # Use Accelerate vDSP vectorized AdamW (fastest)
                    _accel_adamw(p._data, grad, m, v,
                                lr, beta1, beta2, eps, wd, bc1, bc2)
                elif (_USE_CYTHON_ADAM and p._data.dtype == np.float32
                        and p._data.ndim <= 2):
                    # Use Cython nogil kernel
                    flat_p = np.ascontiguousarray(p._data, dtype=np.float32).ravel()
                    flat_g = np.ascontiguousarray(grad, dtype=np.float32).ravel()
                    flat_m = np.ascontiguousarray(m, dtype=np.float32).ravel()
                    flat_v = np.ascontiguousarray(v, dtype=np.float32).ravel()
                    adamw_step_f32(
                        flat_p, flat_g, flat_m, flat_v,
                        lr, beta1, beta2, eps, wd, bc1, bc2)
                    p._data = flat_p.reshape(p._data.shape)
                    st['m'] = flat_m.reshape(m.shape)
                    st['v'] = flat_v.reshape(v.shape)
                else:
                    # Pure Python path — fused for fewer temporaries
                    # Decoupled weight decay (in-place)
                    if wd != 0:
                        p._data *= (1.0 - lr * wd)
                    # Moment updates (in-place)
                    m *= beta1
                    m += (1.0 - beta1) * grad
                    v *= beta2
                    np.multiply(grad, grad, out=grad)  # reuse grad buffer
                    v += (1.0 - beta2) * grad
                    # Bias-corrected update (fused, minimal temporaries)
                    step_size = lr / bc1
                    # Compute sqrt(v/bc2) + eps in-place using a temp
                    denom = np.sqrt(v * (1.0 / bc2))
                    denom += eps
                    p._data -= step_size * (m / denom)

                p._version += 1
