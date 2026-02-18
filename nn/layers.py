# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Neural network layers — Linear, Embedding, Conv1d, Dropout, SiLU."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor, _USE_MPS, _USE_CYTHON, _mops, _cops
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
        # Use BLAS sgemm_nt to avoid transposing weight
        w_data = self.weight._data
        if _USE_MPS and x._data.dtype == np.float32 and w_data.dtype == np.float32:
            result_data = _mops.accelerate_linear_forward(x._data, w_data)
        else:
            result_data = x._data @ w_data.T
        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            result_data = result_data + self.bias._data

        rg = (x._requires_grad or self.weight._requires_grad) and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.LinearBackward()
            inputs = [x, self.weight]
            if has_bias:
                inputs.append(self.bias)
            grad_fn.inputs = inputs
            grad_fn.saved = {
                'input': x._data,
                'weight': w_data,
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
        idx = indices._data
        if idx.dtype != np.intp:
            idx = idx.astype(np.intp)
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
        """x: (B, C_in, L) → (B, C_out, L_out)
        
        Uses vectorized im2col approach instead of Python loops.
        """
        B, C_in, L = x._data.shape
        K = self.kernel_size
        P = self.padding
        L_out = L + 2 * P - K + 1

        # Pad input
        if P > 0:
            x_padded = np.pad(x._data, ((0, 0), (0, 0), (P, P)), mode='constant')
        else:
            x_padded = x._data

        if self.groups == 1:
            # Standard convolution: fully vectorized with im2col
            # Build im2col matrix: (B, C_in*K, L_out) via stride tricks
            # Equivalent to unfolding the input
            strides = x_padded.strides
            col = np.lib.stride_tricks.as_strided(
                x_padded,
                shape=(B, C_in, K, L_out),
                strides=(strides[0], strides[1], strides[2], strides[2]),
            ).reshape(B, C_in * K, L_out)
            # weight: (C_out, C_in, K) → (C_out, C_in*K)
            w_2d = self.weight._data.reshape(self.out_channels, -1)
            # out: (B, C_out, L_out)  via batched matmul
            out = np.einsum('oi,bil->bol', w_2d, col)
        elif self.groups == C_in and self.groups == self.out_channels:
            # Depthwise convolution: vectorized per-channel
            strides = x_padded.strides
            # unfolded: (B, C_in, K, L_out)
            unfolded = np.lib.stride_tricks.as_strided(
                x_padded,
                shape=(B, C_in, K, L_out),
                strides=(strides[0], strides[1], strides[2], strides[2]),
            )
            # weight: (C_out, 1, K) → (1, C_out, K, 1)
            w = self.weight._data.reshape(1, self.out_channels, K, 1)
            out = np.sum(unfolded * w, axis=2)
        else:
            # General grouped convolution: vectorized per group
            c_in_per_group = C_in // self.groups
            c_out_per_group = self.out_channels // self.groups
            strides = x_padded.strides
            out = np.zeros((B, self.out_channels, L_out), dtype=np.float32)
            for grp in range(self.groups):
                ci_start = grp * c_in_per_group
                co_start = grp * c_out_per_group
                x_grp = x_padded[:, ci_start:ci_start + c_in_per_group, :]
                s = x_grp.strides
                col_grp = np.lib.stride_tricks.as_strided(
                    x_grp,
                    shape=(B, c_in_per_group, K, L_out),
                    strides=(s[0], s[1], s[2], s[2]),
                ).reshape(B, c_in_per_group * K, L_out)
                w_grp = self.weight._data[co_start:co_start + c_out_per_group].reshape(c_out_per_group, -1)
                out[:, co_start:co_start + c_out_per_group, :] = np.einsum('oi,bil->bol', w_grp, col_grp)

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
        if _USE_MPS:
            result_data = _mops.accelerate_silu(x._data)
        elif _USE_CYTHON:
            flat = np.ascontiguousarray(x._data).ravel()
            if flat.dtype == np.float32:
                result_data = _cops.silu_1d_f32(flat).reshape(x._data.shape)
            else:
                sig = 1.0 / (1.0 + np.exp(-x._data))
                result_data = x._data * sig
        else:
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
        if _USE_MPS:
            result_data = _mops.accelerate_gelu(x._data)
        else:
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
