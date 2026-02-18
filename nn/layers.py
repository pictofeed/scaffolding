# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Neural network layers — Linear, Embedding, Conv1d, Dropout, SiLU."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor
from .. import autograd as _ag
from .module import Module
from .parameter import Parameter


# ──────────────────────── Linear ──────────────────────────────────────

class Linear(Module):
    """Applies a linear transformation: y = xW^T + b."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1.0 / math.sqrt(in_features)
        w = np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        if bias:
            b = np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        result_data = np.matmul(x._data, self.weight._data.T)
        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            result_data = result_data + self.bias._data

        rg = x._requires_grad or self.weight._requires_grad
        rg = rg and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.LinearBackward()
            inputs = [x, self.weight]
            if has_bias:
                inputs.append(self.bias)
            grad_fn.inputs = inputs
            grad_fn.saved = {
                'input': x._data,
                'weight': self.weight._data,
                'has_bias': has_bias,
            }
        return Tensor._wrap(result_data, rg, grad_fn, x._device)

    def __repr__(self) -> str:
        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={has_bias})")


# ──────────────────────── Embedding ───────────────────────────────────

class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.weight = Parameter(Tensor(w))

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices._data.astype(np.intp)
        result_data = self.weight._data[idx]

        rg = self.weight._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.EmbeddingBackward()
            grad_fn.inputs = [self.weight]
            grad_fn.saved = {
                'indices': idx,
                'num_embeddings': self.num_embeddings,
            }
        return Tensor._wrap(result_data, rg, grad_fn, indices._device)

    def __repr__(self) -> str:
        return (f"Embedding({self.num_embeddings}, {self.embedding_dim})")


# ──────────────────────── Conv1d ──────────────────────────────────────

class Conv1d(Module):
    """1D convolution over temporal data."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0,
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        k = 1.0 / math.sqrt(in_channels * kernel_size / groups)
        w = np.random.uniform(
            -k, k,
            (out_channels, in_channels // groups, kernel_size)
        ).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, L) → (B, C_out, L_out)"""
        B, C_in, L = x._data.shape
        K = self.kernel_size
        P = self.padding
        L_out = L + 2 * P - K + 1

        # Pad input
        if P > 0:
            x_padded = np.pad(x._data, ((0, 0), (0, 0), (P, P)), mode='constant')
        else:
            x_padded = x._data

        out = np.zeros((B, self.out_channels, L_out), dtype=np.float32)
        c_in_per_group = C_in // self.groups
        c_out_per_group = self.out_channels // self.groups

        for grp in range(self.groups):
            co_start = grp * c_out_per_group
            ci_start = grp * c_in_per_group
            for co in range(c_out_per_group):
                for ci in range(c_in_per_group):
                    for k in range(K):
                        out[:, co_start + co, :] += (
                            self.weight._data[co_start + co, ci, k]
                            * x_padded[:, ci_start + ci, k:k + L_out]
                        )

        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            out += self.bias._data[np.newaxis, :, np.newaxis]

        rg = (x._requires_grad or self.weight._requires_grad) and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.Conv1dBackward()
            inputs = [x, self.weight]
            if has_bias:
                inputs.append(self.bias)
            grad_fn.inputs = inputs
            grad_fn.saved = {
                'input': x._data,
                'weight': self.weight._data,
                'padding': P,
                'groups': self.groups,
                'has_bias': has_bias,
            }
        return Tensor._wrap(out, rg, grad_fn, x._device)

    def __repr__(self) -> str:
        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, padding={self.padding}, "
                f"groups={self.groups}, bias={has_bias})")


# ──────────────────────── Dropout ─────────────────────────────────────

class Dropout(Module):
    """During training, randomly zeroes elements with probability p."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x._data.shape) > self.p).astype(x._data.dtype)
        scale = 1.0 / (1.0 - self.p)
        result_data = x._data * mask * scale

        rg = x._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.DropoutBackward()
            grad_fn.inputs = [x]
            grad_fn.saved = {'mask': mask * scale}
        return Tensor._wrap(result_data, rg, grad_fn, x._device)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ──────────────────────── Activations ─────────────────────────────────

class SiLU(Module):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    def forward(self, x: Tensor) -> Tensor:
        sig = 1.0 / (1.0 + np.exp(-x._data))
        result_data = x._data * sig
        rg = x._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.SiluBackward()
            grad_fn.inputs = [x]
            grad_fn.saved = {'x': x._data}
        return Tensor._wrap(result_data, rg, grad_fn, x._device)

    def __repr__(self) -> str:
        return "SiLU()"


class ReLU(Module):
    """ReLU activation."""

    def forward(self, x: Tensor) -> Tensor:
        result_data = np.maximum(x._data, 0)
        rg = x._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.ReluBackward()
            grad_fn.inputs = [x]
            grad_fn.saved = {'x': x._data}
        return Tensor._wrap(result_data, rg, grad_fn, x._device)


class GELU(Module):
    """Gaussian Error Linear Unit."""

    def forward(self, x: Tensor) -> Tensor:
        result_data = 0.5 * x._data * (1 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x._data + 0.044715 * x._data ** 3)))
        return Tensor._wrap(result_data, x._requires_grad, None, x._device)


# ──────────────────────── DataParallel stub ───────────────────────────

class DataParallel(Module):
    """Single-node DataParallel stub.

    In Scaffolding this is a no-op wrapper since we run on CPU/NumPy.
    """

    def __init__(self, module: Module, device_ids=None,
                 output_device=None):
        super().__init__()
        self._modules['module'] = module

    @property
    def module(self) -> Module:
        return self._modules['module']

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
