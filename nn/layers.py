# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Neural network layers — Linear, Embedding, Conv1d, Dropout, SiLU."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor, _USE_MPS, _USE_CYTHON, _USE_CUDA, _mops, _cops, _cuops, _is_cuda
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
        # GPU-resident fast path
        if _USE_CUDA and _is_cuda(x._device) and x._gpu is not None:
            # Determine target device from input tensor
            _target_dev = x._device._index if x._device._index is not None else 0
            # Ensure weight is on the same GPU as input
            if self.weight._gpu is None:
                self.weight._ensure_cpu()
                self.weight._gpu = _cuops.gputensor_from_numpy(
                    np.ascontiguousarray(self.weight._data), _target_dev)
            elif self.weight._gpu is not None and self.weight._gpu.device_id != _target_dev:
                self.weight._gpu = _cuops.gputensor_to_device(self.weight._gpu, _target_dev)
            if self.weight._gpu is not None:
                gt_bias = None
                has_bias = self.bias is not None and isinstance(self.bias, Parameter)
                if has_bias:
                    if self.bias._gpu is None:
                        self.bias._ensure_cpu()
                        self.bias._gpu = _cuops.gputensor_from_numpy(
                            np.ascontiguousarray(self.bias._data), _target_dev)
                    elif self.bias._gpu is not None and self.bias._gpu.device_id != _target_dev:
                        self.bias._gpu = _cuops.gputensor_to_device(self.bias._gpu, _target_dev)
                    gt_bias = self.bias._gpu
                # Set device context for kernel launch
                _cuops.set_device(_target_dev)
                return Tensor._wrap_gpu(
                    _cuops.dev_linear_forward(x._gpu, self.weight._gpu, gt_bias),
                    device=x._device)

        # Use BLAS sgemm_nt to avoid transposing weight
        # Ensure weight (and bias) CPU data is available — it may be GPU-only
        self.weight._ensure_cpu()
        w_data = self.weight._data
        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            self.bias._ensure_cpu()
        if _USE_CUDA and _is_cuda(x._device):
            result_data = _cuops.cuda_linear_forward(
                x._ensure_cpu() if x._data is None else x._data, w_data,
                self.bias._data if has_bias else None
            )
        elif _USE_MPS and x._data.dtype == np.float32 and w_data.dtype == np.float32:
            result_data = _mops.accelerate_linear_forward(x._data, w_data)
            if has_bias:
                result_data = result_data + self.bias._data
        else:
            result_data = x._data @ w_data.T
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
        # GPU-resident fast path
        if _USE_CUDA and _is_cuda(indices._device) and indices._gpu is not None:
            _target_dev = indices._device._index if indices._device._index is not None else 0
            if self.weight._gpu is None:
                self.weight._ensure_cpu()
                self.weight._gpu = _cuops.gputensor_from_numpy(
                    np.ascontiguousarray(self.weight._data), _target_dev)
            elif self.weight._gpu is not None and self.weight._gpu.device_id != _target_dev:
                self.weight._gpu = _cuops.gputensor_to_device(self.weight._gpu, _target_dev)
            if self.weight._gpu is not None:
                _cuops.set_device(_target_dev)
                return Tensor._wrap_gpu(
                    _cuops.dev_embedding(self.weight._gpu, indices._gpu,
                                         self.embedding_dim),
                    device=indices._device)

        idx = indices._ensure_cpu() if indices._data is None else indices._data
        if idx.dtype != np.intp:
            idx = idx.astype(np.intp)
        self.weight._ensure_cpu()          # weight may be GPU-only
        if _USE_CUDA and _is_cuda(indices._device):
            result_data = _cuops.cuda_embedding(self.weight._data, idx.astype(np.int64))
        else:
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
        x._ensure_cpu()
        self.weight._ensure_cpu()          # weight may be GPU-only
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
            self.bias._ensure_cpu()        # bias may be GPU-only
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
        from . import functional as F
        return F.dropout(x, self.p, True)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


# ──────────────────────── Activations ─────────────────────────────────

class SiLU(Module):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    def forward(self, x: Tensor) -> Tensor:
        if _USE_CUDA and _is_cuda(x._device):
            if x._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_silu(x._gpu), device=x._device)
            x._ensure_cpu()
            result_data = _cuops.cuda_silu(x._data)
        elif _USE_MPS:
            x._ensure_cpu()
            result_data = _mops.accelerate_silu(x._data)
        elif _USE_CYTHON:
            x._ensure_cpu()
            flat = np.ascontiguousarray(x._data).ravel()
            if flat.dtype == np.float32:
                result_data = _cops.silu_1d_f32(flat).reshape(x._data.shape)
            else:
                sig = 1.0 / (1.0 + np.exp(-x._data))
                result_data = x._data * sig
        else:
            x._ensure_cpu()
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
        if _USE_CUDA and _is_cuda(x._device):
            if x._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_relu(x._gpu), device=x._device)
            x._ensure_cpu()
            result_data = _cuops.cuda_relu(x._data)
        else:
            x._ensure_cpu()
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
        if _USE_CUDA and _is_cuda(x._device):
            if x._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_gelu(x._gpu), device=x._device)
            x._ensure_cpu()
            result_data = _cuops.cuda_gelu(x._data)
        elif _USE_MPS:
            x._ensure_cpu()
            result_data = _mops.accelerate_gelu(x._data)
        else:
            x._ensure_cpu()
            result_data = 0.5 * x._data * (1 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x._data + 0.044715 * x._data ** 3)))
        rg = x._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.GeluBackward()
            grad_fn.inputs = [x]
            grad_fn.saved = {'x': x._data}
        return Tensor._wrap(result_data, rg, grad_fn, x._device)


# ──────────────────────── DataParallel stub ───────────────────────────


class LayerNorm(Module):
    """Layer Normalization (Ba et al., 2016).

    Normalizes over the last *normalized_shape* dimensions.
    Mirrors ``torch.nn.LayerNorm``.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(Tensor(np.ones(self.normalized_shape,
                                                  dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(self.normalized_shape,
                                                  dtype=np.float32)))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # CUDA fast-path (GPU-resident)
        if _USE_CUDA and _is_cuda(x._device) and x._gpu is not None:
            _target_dev = x._device._index if x._device._index is not None else 0
            if self.weight is not None and self.weight._gpu is None:
                self.weight._ensure_cpu()
                self.weight._gpu = _cuops.gputensor_from_numpy(
                    np.ascontiguousarray(self.weight._data), _target_dev)
            if self.bias is not None and self.bias._gpu is None:
                self.bias._ensure_cpu()
                self.bias._gpu = _cuops.gputensor_from_numpy(
                    np.ascontiguousarray(self.bias._data), _target_dev)
            # Reshape to 2D for kernel
            orig_shape = x._gpu.shape
            D = self.normalized_shape[-1]
            N = x._gpu.numel // D
            # Need to reshape the GpuTensor
            from scaffolding._cuda_ops import GpuTensor
            x_2d = GpuTensor(x._gpu.buffer, (N, D), x._gpu.dtype)
            gt_gamma = self.weight._gpu if self.weight is not None else None
            gt_beta = self.bias._gpu if self.bias is not None else None
            y_2d = _cuops.dev_layer_norm(x_2d, gt_gamma, gt_beta, self.eps)
            y_out = GpuTensor(y_2d.buffer, orig_shape, y_2d.dtype)
            return Tensor._wrap_gpu(y_out, x._requires_grad, None, x._device)

        # CUDA fallback (CPU arrays)
        if _USE_CUDA and _is_cuda(x._device):
            if self.weight is not None:
                self.weight._ensure_cpu()
            if self.bias is not None:
                self.bias._ensure_cpu()
            gamma = self.weight._data if self.weight is not None else None
            beta = self.bias._data if self.bias is not None else None
            xd = x._ensure_cpu() if x._data is None else x._data
            y = _cuops.cuda_layer_norm(
                xd.reshape(-1, self.normalized_shape[-1]),
                gamma, beta, self.eps,
            )
            return Tensor._wrap(y.reshape(xd.shape),
                                x._requires_grad, None, x._device)

        # Generic NumPy path
        x._ensure_cpu()
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean = np.mean(x._data, axis=axes, keepdims=True)
        var = np.var(x._data, axis=axes, keepdims=True)
        x_norm = (x._data - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            self.weight._ensure_cpu()
            self.bias._ensure_cpu()
            x_norm = x_norm * self.weight._data + self.bias._data
        return Tensor._wrap(x_norm.astype(np.float32),
                            x._requires_grad, None, x._device)

    def extra_repr(self) -> str:
        return (f'{self.normalized_shape}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Like LayerNorm but without re-centring (no bias subtraction).
    """

    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(self.normalized_shape,
                                               dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        # CUDA fast-path (GPU-resident)
        if _USE_CUDA and _is_cuda(x._device) and x._gpu is not None:
            _target_dev = x._device._index if x._device._index is not None else 0
            if self.weight._gpu is None:
                self.weight._ensure_cpu()
                self.weight._gpu = _cuops.gputensor_from_numpy(
                    np.ascontiguousarray(self.weight._data), _target_dev)
            orig_shape = x._gpu.shape
            D = self.normalized_shape[-1]
            N = x._gpu.numel // D
            from scaffolding._cuda_ops import GpuTensor
            x_2d = GpuTensor(x._gpu.buffer, (N, D), x._gpu.dtype)
            y_2d = _cuops.dev_rms_norm(x_2d, self.weight._gpu, self.eps)
            y_out = GpuTensor(y_2d.buffer, orig_shape, y_2d.dtype)
            return Tensor._wrap_gpu(y_out, x._requires_grad, None, x._device)

        # CUDA fallback
        if _USE_CUDA and _is_cuda(x._device):
            self.weight._ensure_cpu()
            gamma = self.weight._data
            xd = x._ensure_cpu() if x._data is None else x._data
            y = _cuops.cuda_rms_norm(
                xd.reshape(-1, self.normalized_shape[-1]),
                gamma, self.eps,
            )
            return Tensor._wrap(y.reshape(xd.shape),
                                x._requires_grad, None, x._device)

        # Generic NumPy path
        x._ensure_cpu()
        self.weight._ensure_cpu()
        axes = tuple(range(-len(self.normalized_shape), 0))
        rms = np.sqrt(np.mean(x._data ** 2, axis=axes, keepdims=True) + self.eps)
        x_norm = x._data / rms * self.weight._data
        return Tensor._wrap(x_norm.astype(np.float32),
                            x._requires_grad, None, x._device)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}'


class DataParallel(Module):
    """Compatibility alias — see nn.parallel.DataParallel for real impl.

    This stub delegates to the real DataParallel when CUDA is available,
    otherwise falls through as a no-op wrapper.
    """

    def __new__(cls, module: Module, device_ids=None,
                output_device=None, **kwargs):
        from .parallel import DataParallel as _RealDP
        return _RealDP(module, device_ids=device_ids,
                       output_device=output_device, **kwargs)
