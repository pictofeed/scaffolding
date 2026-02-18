# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
scaffolding.cuda.amp — Automatic Mixed Precision.

Provides ``autocast`` context manager and ``GradScaler`` for training
with FP16 / BF16 on NVIDIA GPUs.  When CUDA is not available the classes
degrade to passthrough / no-op behaviour identical to the previous stubs.
"""
from __future__ import annotations

import functools
import math
import threading
from contextlib import contextmanager
from typing import Any

import numpy as np

# ── CUDA availability flag ──
try:
    from .. import _cuda_ops as _cops  # type: ignore[attr-defined]
    _CUDA = True
except ImportError:
    _cops = None
    _CUDA = False


# ================================================================
#  autocast
# ================================================================

# Thread-local stack so nested autocasts compose correctly.
_tls = threading.local()


def _get_autocast_stack() -> list:
    if not hasattr(_tls, 'stack'):
        _tls.stack = []
    return _tls.stack


def is_autocast_enabled() -> bool:
    """Return True if we are inside an ``autocast`` region."""
    stack = _get_autocast_stack()
    return len(stack) > 0 and stack[-1].enabled


def get_autocast_dtype() -> np.dtype:
    """Return the target dtype of the innermost ``autocast``."""
    stack = _get_autocast_stack()
    if stack:
        return stack[-1]._target_dtype
    return np.dtype('float32')


# Ops that should be cast down to the reduced-precision dtype.
_FP16_ELIGIBLE_OPS = frozenset({
    'matmul', 'linear', 'conv1d', 'conv2d', 'conv3d',
    'bmm', 'baddbmm', 'addmm', 'addbmm',
    'layer_norm', 'rms_norm',
    'softmax', 'log_softmax',
    'gelu', 'silu', 'relu',
})

# Ops that MUST stay in FP32 for numerical stability.
_FP32_FORCED_OPS = frozenset({
    'cross_entropy', 'nll_loss', 'mse_loss',
    'batch_norm',  # running stats must be fp32
    'adamw',       # optimizer step must be fp32
    'sum', 'mean', 'var', 'norm',
})


class autocast:
    """Automatic Mixed Precision context manager / decorator.

    Usage::

        with scaffolding.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, target)
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: Any = None,
        cache_enabled: bool = True,
    ):
        self.enabled = enabled
        self._cache_enabled = cache_enabled
        # Default to float16; accept 'bfloat16' or np.float16 etc.
        if dtype is None:
            self._target_dtype = np.dtype('float16')
        elif isinstance(dtype, np.dtype):
            self._target_dtype = dtype
        elif isinstance(dtype, str):
            self._target_dtype = np.dtype(dtype)
        else:
            self._target_dtype = np.dtype(dtype)

    # ── context-manager protocol ──
    def __enter__(self):
        _get_autocast_stack().append(self)
        return self

    def __exit__(self, *args):
        _get_autocast_stack().pop()

    # ── decorator protocol ──
    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with self:
                return fn(*a, **kw)
        return wrapper

    # ── static helper used by kernels / functional ──
    @staticmethod
    def cast_inputs(*arrays: np.ndarray, op_name: str = '') -> tuple:
        """Conditionally cast inputs to the autocast dtype.

        * For eligible ops: cast down to fp16/bf16.
        * For FP32-forced ops: cast up to fp32.
        * Otherwise: passthrough unchanged.
        """
        if not is_autocast_enabled():
            return arrays

        target = get_autocast_dtype()

        if op_name in _FP32_FORCED_OPS:
            target = np.dtype('float32')
        elif op_name not in _FP16_ELIGIBLE_OPS:
            return arrays  # passthrough

        out = []
        for a in arrays:
            if isinstance(a, np.ndarray) and a.dtype.kind == 'f':
                out.append(a.astype(target, copy=False))
            else:
                out.append(a)
        return tuple(out)


# ================================================================
#  GradScaler
# ================================================================

_INITIAL_SCALE = 2.0 ** 16  # 65536
_GROWTH_FACTOR = 2.0
_BACKOFF_FACTOR = 0.5
_GROWTH_INTERVAL = 2000  # steps between scale increases
_MAX_SCALE = 2.0 ** 24
_MIN_SCALE = 1.0


class GradScaler:
    """Loss-scaling helper for mixed-precision training.

    Automatically adjusts the loss scale to avoid gradient underflow
    in FP16 while keeping parameters in FP32.

    Usage::

        scaler = GradScaler()
        with autocast():
            loss = model(x)
        scaled = scaler.scale(loss)
        scaled.backward()                # <- Tensor.backward
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(
        self,
        init_scale: float = _INITIAL_SCALE,
        growth_factor: float = _GROWTH_FACTOR,
        backoff_factor: float = _BACKOFF_FACTOR,
        growth_interval: int = _GROWTH_INTERVAL,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    # ── core API ──

    def scale(self, loss):
        """Multiply loss by the current scale factor.

        ``loss`` can be a Scaffolding ``Tensor`` or a plain float/ndarray.
        """
        if not self.enabled:
            return loss
        # Tensor (has .data)
        if hasattr(loss, 'data'):
            from .. import tensor as _t
            scaled_data = loss.data * self._scale
            out = _t.Tensor(
                scaled_data,
                requires_grad=loss.requires_grad,
                dtype=loss.dtype,
            )
            out._backward_fn = loss._backward_fn
            out._prev = loss._prev
            out._grad_fn_name = 'ScaleBackward'
            return out
        return loss * self._scale

    def unscale_(self, optimizer) -> None:
        """Divide all gradients by the scale and check for inf/nan."""
        if not self.enabled:
            return
        self._found_inf = False
        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad
                    if np.any(np.isinf(g)) or np.any(np.isnan(g)):
                        self._found_inf = True
                        return
                    p.grad = g * inv_scale

    def step(self, optimizer) -> None:
        """Call ``optimizer.step()`` unless inf/nan was detected."""
        if not self.enabled:
            optimizer.step()
            return
        if not self._found_inf:
            optimizer.step()

    def update(self) -> None:
        """Adjust the scale factor based on gradient health."""
        if not self.enabled:
            return
        if self._found_inf:
            self._scale = max(
                self._scale * self._backoff_factor, _MIN_SCALE
            )
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale = min(
                    self._scale * self._growth_factor, _MAX_SCALE
                )
                self._growth_tracker = 0

    # ── query ──

    def get_scale(self) -> float:
        return self._scale

    def is_enabled(self) -> bool:
        return self.enabled

    # ── state dict ──

    def state_dict(self) -> dict:
        return {
            'scale': self._scale,
            'growth_factor': self._growth_factor,
            'backoff_factor': self._backoff_factor,
            'growth_interval': self._growth_interval,
            'growth_tracker': self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._scale = state_dict['scale']
        self._growth_factor = state_dict.get(
            'growth_factor', _GROWTH_FACTOR
        )
        self._backoff_factor = state_dict.get(
            'backoff_factor', _BACKOFF_FACTOR
        )
        self._growth_interval = state_dict.get(
            'growth_interval', _GROWTH_INTERVAL
        )
        self._growth_tracker = state_dict.get('growth_tracker', 0)


__all__ = ['autocast', 'GradScaler', 'is_autocast_enabled',
           'get_autocast_dtype']
