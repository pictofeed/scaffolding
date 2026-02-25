# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Neural network layers — Linear, Embedding, Conv1d/2d/3d, Dropout, SiLU, and more."""
from __future__ import annotations

import gc
import math
import numpy as np

from ..tensor import Tensor, _USE_MPS, _USE_CYTHON, _USE_CUDA, _mops, _cops, _cuops, _is_cuda
from .. import autograd as _ag
from .module import Module, _release_host_memory
from .parameter import Parameter


def _param_to_device(p: Parameter, device) -> None:
    """Upload a freshly-created parameter to *device* in-place and free CPU.

    Called during layer __init__ when a target device is specified so
    that the numpy weight array never persists in host RAM.
    """
    if device is None:
        return
    from ..device import device as Device
    dev = Device(device) if isinstance(device, str) else device
    if not _USE_CUDA or not _is_cuda(dev):
        return
    target_dev = dev._index if dev._index is not None else 0
    if p._data is not None and p._data.dtype in (np.float32, np.int64):
        p._gpu = _cuops.gputensor_from_numpy(
            np.ascontiguousarray(p._data), target_dev)
        p._data = None
        p._device = dev


# Module-level default device for parameter creation
_default_param_device = None


def set_default_device(device):
    """Set the default device used for new parameters.

    Usage::

        sf.nn.set_default_device('cuda:0')
        model = MyLargeModel()          # all weights go directly to GPU
        sf.nn.set_default_device(None)  # reset

    This avoids materialising the entire model in host RAM first.
    """
    global _default_param_device
    _default_param_device = device


# ──────────────────────── Linear ──────────────────────────────────────

class Linear(Module):
    """Applies a linear transformation: y = xW^T + b."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        _dev = device or _default_param_device
        k = 1.0 / math.sqrt(in_features)
        w = np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w                        # free source array immediately
        if bias:
            b = np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
        else:
            self.bias = None  # type: ignore[assignment]
        if _dev is not None:
            gc.collect(0)

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

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        _dev = device or _default_param_device
        w = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w
        if _dev is not None:
            gc.collect(0)

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
                 groups: int = 1, bias: bool = True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        _dev = device or _default_param_device
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        k = 1.0 / math.sqrt(in_channels * kernel_size / groups)
        w = np.random.uniform(
            -k, k,
            (out_channels, in_channels // groups, kernel_size)
        ).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w
        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
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
                 elementwise_affine: bool = True, device=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        _dev = device or _default_param_device
        if elementwise_affine:
            self.weight = Parameter(Tensor(np.ones(self.normalized_shape,
                                                  dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(self.normalized_shape,
                                                  dtype=np.float32)))
            _param_to_device(self.weight, _dev)
            _param_to_device(self.bias, _dev)
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

    def __init__(self, normalized_shape, eps: float = 1e-6, device=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        _dev = device or _default_param_device
        self.weight = Parameter(Tensor(np.ones(self.normalized_shape,
                                               dtype=np.float32)))
        _param_to_device(self.weight, _dev)

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


# ──────────────────────── Conv2d ───────────────────────────────────────

class Conv2d(Module):
    """2D convolution over spatial data (images).

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        _dev = device or _default_param_device

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        kH, kW = self.kernel_size
        k = 1.0 / math.sqrt(in_channels * kH * kW / groups)
        w = np.random.uniform(-k, k,
            (out_channels, in_channels // groups, kH, kW)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w

        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
        else:
            self.bias = None

    def forward(self, x):
        """x: (B, C_in, H, W) → (B, C_out, H_out, W_out) via im2col."""
        x._ensure_cpu()
        self.weight._ensure_cpu()

        B, C_in, H, W = x._data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # Pad input
        if pH > 0 or pW > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                              mode='constant')
        else:
            x_padded = x._data

        # im2col using stride tricks
        col = _im2col_2d(x_padded, kH, kW, sH, sW, dH, dW, H_out, W_out)

        if self.groups == 1:
            w_2d = self.weight._data.reshape(self.out_channels, -1)
            out = np.einsum('oi,biN->boN', w_2d, col)
            out = out.reshape(B, self.out_channels, H_out, W_out)
        elif self.groups == C_in and self.groups == self.out_channels:
            # Depthwise convolution
            col_dw = col.reshape(B, C_in, kH * kW, H_out * W_out)
            w = self.weight._data.reshape(self.out_channels, kH * kW, 1)
            out = np.sum(col_dw * w[np.newaxis, :, :, :].reshape(1, self.out_channels, kH * kW, 1),
                         axis=2)
            out = out.reshape(B, self.out_channels, H_out, W_out)
        else:
            c_in_per_group = C_in // self.groups
            c_out_per_group = self.out_channels // self.groups
            out = np.zeros((B, self.out_channels, H_out * W_out), dtype=np.float32)
            for grp in range(self.groups):
                ci_s = grp * c_in_per_group
                co_s = grp * c_out_per_group
                col_grp = col[:, ci_s * kH * kW:(ci_s + c_in_per_group) * kH * kW, :]
                w_grp = self.weight._data[co_s:co_s + c_out_per_group].reshape(c_out_per_group, -1)
                out[:, co_s:co_s + c_out_per_group, :] = np.einsum('oi,biN->boN', w_grp, col_grp)
            out = out.reshape(B, self.out_channels, H_out, W_out)

        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            self.bias._ensure_cpu()
            out += self.bias._data[np.newaxis, :, np.newaxis, np.newaxis]

        rg = (x._requires_grad or self.weight._requires_grad) and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.Conv2dBackward()
            inputs = [x, self.weight]
            if has_bias:
                inputs.append(self.bias)
            grad_fn.inputs = inputs
            grad_fn.saved = {
                'input': x._data, 'weight': self.weight._data,
                'padding': self.padding, 'stride': self.stride,
                'dilation': self.dilation, 'groups': self.groups,
                'has_bias': has_bias,
            }
        return Tensor._wrap(out, rg, grad_fn, x._device)


def _im2col_2d(x_padded, kH, kW, sH, sW, dH, dW, H_out, W_out):
    """Extract image patches into columns for 2D convolution.

    x_padded: (B, C_in, H_padded, W_padded)
    Returns: (B, C_in*kH*kW, H_out*W_out)
    """
    B, C_in, Hp, Wp = x_padded.shape
    # Build column indices
    col = np.zeros((B, C_in * kH * kW, H_out * W_out), dtype=x_padded.dtype)

    for i in range(kH):
        for j in range(kW):
            h_start = i * dH
            w_start = j * dW
            patch = x_padded[:, :,
                             h_start:h_start + sH * H_out:sH,
                             w_start:w_start + sW * W_out:sW]
            col[:, (i * kW + j) * C_in:(i * kW + j + 1) * C_in, :] = \
                patch.reshape(B, C_in, H_out * W_out)

    # Reorder to group by channel: (B, C_in * kH * kW, N)
    # The current layout is (B, kH*kW*C_in, N), re-transpose so
    # channels are grouped: col[:, c*kH*kW + i*kW + j]
    col2 = np.zeros_like(col)
    for c in range(C_in):
        for i in range(kH):
            for j in range(kW):
                src_idx = (i * kW + j) * C_in + c
                dst_idx = c * kH * kW + i * kW + j
                col2[:, dst_idx, :] = col[:, src_idx, :]
    return col2


# ──────────────────────── Conv3d ───────────────────────────────────────

class Conv3d(Module):
    """3D convolution over volumetric/temporal data.

    Input:  (B, C_in, D, H, W)
    Output: (B, C_out, D_out, H_out, W_out)

    Essential for video processing where D is the temporal dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        _dev = device or _default_param_device

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        kD, kH, kW = self.kernel_size
        k = 1.0 / math.sqrt(in_channels * kD * kH * kW / groups)
        w = np.random.uniform(-k, k,
            (out_channels, in_channels // groups, kD, kH, kW)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w

        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
        else:
            self.bias = None

    def forward(self, x):
        """x: (B, C_in, D, H, W) → (B, C_out, D_out, H_out, W_out)."""
        x._ensure_cpu()
        self.weight._ensure_cpu()

        B, C_in, D, H, W = x._data.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        dD, dH, dW = self.dilation

        D_out = (D + 2 * pD - dD * (kD - 1) - 1) // sD + 1
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        # Pad input
        if pD > 0 or pH > 0 or pW > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                              mode='constant')
        else:
            x_padded = x._data

        # im2col for 3D
        col = _im2col_3d(x_padded, kD, kH, kW, sD, sH, sW,
                         dD, dH, dW, D_out, H_out, W_out)

        if self.groups == 1:
            w_2d = self.weight._data.reshape(self.out_channels, -1)
            out = np.einsum('oi,biN->boN', w_2d, col)
            out = out.reshape(B, self.out_channels, D_out, H_out, W_out)
        else:
            c_in_per_group = C_in // self.groups
            c_out_per_group = self.out_channels // self.groups
            N = D_out * H_out * W_out
            out = np.zeros((B, self.out_channels, N), dtype=np.float32)
            for grp in range(self.groups):
                ci_s = grp * c_in_per_group
                co_s = grp * c_out_per_group
                k_size = c_in_per_group * kD * kH * kW
                col_grp = col[:, ci_s * kD * kH * kW:ci_s * kD * kH * kW + k_size, :]
                w_grp = self.weight._data[co_s:co_s + c_out_per_group].reshape(c_out_per_group, -1)
                out[:, co_s:co_s + c_out_per_group, :] = np.einsum('oi,biN->boN', w_grp, col_grp)
            out = out.reshape(B, self.out_channels, D_out, H_out, W_out)

        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            self.bias._ensure_cpu()
            out += self.bias._data[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

        rg = (x._requires_grad or self.weight._requires_grad) and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.Conv3dBackward()
            inputs = [x, self.weight]
            if has_bias:
                inputs.append(self.bias)
            grad_fn.inputs = inputs
            grad_fn.saved = {
                'input': x._data, 'weight': self.weight._data,
                'padding': self.padding, 'stride': self.stride,
                'dilation': self.dilation, 'groups': self.groups,
                'has_bias': has_bias,
            }
        return Tensor._wrap(out, rg, grad_fn, x._device)


def _im2col_3d(x_padded, kD, kH, kW, sD, sH, sW, dD, dH, dW,
               D_out, H_out, W_out):
    """Extract volumetric patches into columns for 3D convolution.

    x_padded: (B, C_in, D_padded, H_padded, W_padded)
    Returns: (B, C_in*kD*kH*kW, D_out*H_out*W_out)
    """
    B, C_in = x_padded.shape[:2]
    N = D_out * H_out * W_out
    col = np.zeros((B, C_in * kD * kH * kW, N), dtype=x_padded.dtype)

    idx = 0
    for c in range(C_in):
        for d in range(kD):
            for i in range(kH):
                for j in range(kW):
                    d_start = d * dD
                    h_start = i * dH
                    w_start = j * dW
                    patch = x_padded[:, c,
                                     d_start:d_start + sD * D_out:sD,
                                     h_start:h_start + sH * H_out:sH,
                                     w_start:w_start + sW * W_out:sW]
                    col[:, idx, :] = patch.reshape(B, N)
                    idx += 1
    return col


# ──────────────────────── ConvTranspose2d ─────────────────────────────

class ConvTranspose2d(Module):
    """2D transposed convolution (deconvolution) for upsampling.

    Input:  (B, C_in, H, W)
    Output: (B, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        _dev = device or _default_param_device

        kH, kW = self.kernel_size
        k = 1.0 / math.sqrt(in_channels * kH * kW)
        w = np.random.uniform(-k, k,
            (in_channels, out_channels // groups, kH, kW)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w

        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
        else:
            self.bias = None

    def forward(self, x):
        x._ensure_cpu()
        self.weight._ensure_cpu()

        B, C_in, H_in, W_in = x._data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        opH, opW = self.output_padding

        H_out = (H_in - 1) * sH - 2 * pH + kH + opH
        W_out = (W_in - 1) * sW - 2 * pW + kW + opW

        # Insert zeros (dilate input by stride)
        H_dilated = (H_in - 1) * sH + 1
        W_dilated = (W_in - 1) * sW + 1
        x_dilated = np.zeros((B, C_in, H_dilated, W_dilated), dtype=np.float32)
        x_dilated[:, :, ::sH, ::sW] = x._data

        # Pad for full convolution
        pad_top = kH - 1 - pH
        pad_bottom = kH - 1 - pH + opH
        pad_left = kW - 1 - pW
        pad_right = kW - 1 - pW + opW

        x_padded = np.pad(x_dilated,
                          ((0, 0), (0, 0),
                           (max(pad_top, 0), max(pad_bottom, 0)),
                           (max(pad_left, 0), max(pad_right, 0))),
                          mode='constant')

        # Flip the weight (rotate 180°) and transpose in/out channels
        # weight shape: (C_in, C_out/groups, kH, kW)
        w_flipped = self.weight._data[:, :, ::-1, ::-1]
        # Reshape for grouped correlation
        C_out = self.out_channels

        out = np.zeros((B, C_out, H_out, W_out), dtype=np.float32)

        if self.groups == 1:
            for i in range(kH):
                for j in range(kW):
                    patch = x_padded[:, :,
                                     i:i + H_out,
                                     j:j + W_out]
                    # patch: (B, C_in, H_out, W_out)
                    # w_flipped[:, :, i, j]: (C_in, C_out)
                    out += np.einsum('bchw,co->bohw',
                                    patch,
                                    w_flipped[:, :, i, j])
        else:
            c_in_per_group = C_in // self.groups
            c_out_per_group = C_out // self.groups
            for grp in range(self.groups):
                ci_s = grp * c_in_per_group
                co_s = grp * c_out_per_group
                for i in range(kH):
                    for j in range(kW):
                        patch = x_padded[:, ci_s:ci_s + c_in_per_group,
                                         i:i + H_out,
                                         j:j + W_out]
                        out[:, co_s:co_s + c_out_per_group] += np.einsum(
                            'bchw,co->bohw',
                            patch,
                            w_flipped[ci_s:ci_s + c_in_per_group, :, i, j])

        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            self.bias._ensure_cpu()
            out += self.bias._data[np.newaxis, :, np.newaxis, np.newaxis]

        return Tensor._wrap(out.astype(np.float32),
                            x._requires_grad or self.weight._requires_grad,
                            None, x._device)


# ──────────────────────── ConvTranspose3d ─────────────────────────────

class ConvTranspose3d(Module):
    """3D transposed convolution for temporal/volumetric upsampling.

    Input:  (B, C_in, D, H, W)
    Output: (B, C_out, D_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        _dev = device or _default_param_device

        kD, kH, kW = self.kernel_size
        k = 1.0 / math.sqrt(in_channels * kD * kH * kW)
        w = np.random.uniform(-k, k,
            (in_channels, out_channels // groups, kD, kH, kW)).astype(np.float32)
        self.weight = Parameter(Tensor(w))
        _param_to_device(self.weight, _dev)
        del w

        if bias:
            b = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
            _param_to_device(self.bias, _dev)
            del b
        else:
            self.bias = None

    def forward(self, x):
        x._ensure_cpu()
        self.weight._ensure_cpu()

        B, C_in, D_in, H_in, W_in = x._data.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        opD, opH, opW = self.output_padding

        D_out = (D_in - 1) * sD - 2 * pD + kD + opD
        H_out = (H_in - 1) * sH - 2 * pH + kH + opH
        W_out = (W_in - 1) * sW - 2 * pW + kW + opW

        # Dilate input by stride
        D_dilated = (D_in - 1) * sD + 1
        H_dilated = (H_in - 1) * sH + 1
        W_dilated = (W_in - 1) * sW + 1
        x_dilated = np.zeros((B, C_in, D_dilated, H_dilated, W_dilated), dtype=np.float32)
        x_dilated[:, :, ::sD, ::sH, ::sW] = x._data

        pad_d0 = kD - 1 - pD
        pad_d1 = kD - 1 - pD + opD
        pad_h0 = kH - 1 - pH
        pad_h1 = kH - 1 - pH + opH
        pad_w0 = kW - 1 - pW
        pad_w1 = kW - 1 - pW + opW

        x_padded = np.pad(x_dilated,
                          ((0, 0), (0, 0),
                           (max(pad_d0, 0), max(pad_d1, 0)),
                           (max(pad_h0, 0), max(pad_h1, 0)),
                           (max(pad_w0, 0), max(pad_w1, 0))),
                          mode='constant')

        w_flipped = self.weight._data[:, :, ::-1, ::-1, ::-1]
        C_out = self.out_channels

        out = np.zeros((B, C_out, D_out, H_out, W_out), dtype=np.float32)

        if self.groups == 1:
            for d in range(kD):
                for i in range(kH):
                    for j in range(kW):
                        patch = x_padded[:, :,
                                         d:d + D_out,
                                         i:i + H_out,
                                         j:j + W_out]
                        out += np.einsum('bcdhw,co->bodhw',
                                        patch,
                                        w_flipped[:, :, d, i, j])

        has_bias = self.bias is not None and isinstance(self.bias, Parameter)
        if has_bias:
            self.bias._ensure_cpu()
            out += self.bias._data[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

        return Tensor._wrap(out.astype(np.float32),
                            x._requires_grad or self.weight._requires_grad,
                            None, x._device)


# ──────────────────────── BatchNorm2d ──────────────────────────────────

class BatchNorm2d(Module):
    """2D Batch Normalization over (B, C, H, W) tensors."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        _dev = device or _default_param_device

        if affine:
            self.weight = Parameter(Tensor(np.ones(num_features, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(num_features, dtype=np.float32)))
            _param_to_device(self.weight, _dev)
            _param_to_device(self.bias, _dev)
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer('running_mean',
                                 Tensor(np.zeros(num_features, dtype=np.float32)))
            self.register_buffer('running_var',
                                 Tensor(np.ones(num_features, dtype=np.float32)))
            self.register_buffer('num_batches_tracked',
                                 Tensor(np.array(0, dtype=np.int64)))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        x._ensure_cpu()
        B, C, H, W = x._data.shape

        if self._training:
            mean = np.mean(x._data, axis=(0, 2, 3))
            var = np.var(x._data, axis=(0, 2, 3))
            if self.track_running_stats:
                self.running_mean._ensure_cpu()
                self.running_var._ensure_cpu()
                m = self.momentum
                self.running_mean._data = (1 - m) * self.running_mean._data + m * mean
                self.running_var._data = (1 - m) * self.running_var._data + m * var
                self.num_batches_tracked._data += 1
        else:
            if self.track_running_stats:
                self.running_mean._ensure_cpu()
                self.running_var._ensure_cpu()
                mean = self.running_mean._data
                var = self.running_var._data
            else:
                mean = np.mean(x._data, axis=(0, 2, 3))
                var = np.var(x._data, axis=(0, 2, 3))

        x_norm = (x._data - mean[np.newaxis, :, np.newaxis, np.newaxis]) / \
                 np.sqrt(var[np.newaxis, :, np.newaxis, np.newaxis] + self.eps)

        if self.affine:
            self.weight._ensure_cpu()
            self.bias._ensure_cpu()
            x_norm = x_norm * self.weight._data[np.newaxis, :, np.newaxis, np.newaxis] + \
                     self.bias._data[np.newaxis, :, np.newaxis, np.newaxis]

        return Tensor._wrap(x_norm.astype(np.float32),
                            x._requires_grad, None, x._device)


# ──────────────────────── BatchNorm3d ──────────────────────────────────

class BatchNorm3d(Module):
    """3D Batch Normalization over (B, C, D, H, W) tensors."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        _dev = device or _default_param_device

        if affine:
            self.weight = Parameter(Tensor(np.ones(num_features, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(num_features, dtype=np.float32)))
            _param_to_device(self.weight, _dev)
            _param_to_device(self.bias, _dev)
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer('running_mean',
                                 Tensor(np.zeros(num_features, dtype=np.float32)))
            self.register_buffer('running_var',
                                 Tensor(np.ones(num_features, dtype=np.float32)))
            self.register_buffer('num_batches_tracked',
                                 Tensor(np.array(0, dtype=np.int64)))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        x._ensure_cpu()
        B, C, D, H, W = x._data.shape

        if self._training:
            mean = np.mean(x._data, axis=(0, 2, 3, 4))
            var = np.var(x._data, axis=(0, 2, 3, 4))
            if self.track_running_stats:
                self.running_mean._ensure_cpu()
                self.running_var._ensure_cpu()
                m = self.momentum
                self.running_mean._data = (1 - m) * self.running_mean._data + m * mean
                self.running_var._data = (1 - m) * self.running_var._data + m * var
                self.num_batches_tracked._data += 1
        else:
            if self.track_running_stats:
                self.running_mean._ensure_cpu()
                self.running_var._ensure_cpu()
                mean = self.running_mean._data
                var = self.running_var._data
            else:
                mean = np.mean(x._data, axis=(0, 2, 3, 4))
                var = np.var(x._data, axis=(0, 2, 3, 4))

        shape = [1, C, 1, 1, 1]
        x_norm = (x._data - mean.reshape(shape)) / \
                 np.sqrt(var.reshape(shape) + self.eps)

        if self.affine:
            self.weight._ensure_cpu()
            self.bias._ensure_cpu()
            x_norm = x_norm * self.weight._data.reshape(shape) + \
                     self.bias._data.reshape(shape)

        return Tensor._wrap(x_norm.astype(np.float32),
                            x._requires_grad, None, x._device)


# ──────────────────────── GroupNorm ────────────────────────────────────

class GroupNorm(Module):
    """Group Normalization (Wu & He, 2018)."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        _dev = device or _default_param_device

        assert num_channels % num_groups == 0

        if affine:
            self.weight = Parameter(Tensor(np.ones(num_channels, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(num_channels, dtype=np.float32)))
            _param_to_device(self.weight, _dev)
            _param_to_device(self.bias, _dev)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        x._ensure_cpu()
        shape = x._data.shape
        B, C = shape[0], shape[1]
        G = self.num_groups
        spatial = shape[2:]

        # Reshape to (B, G, C//G, *spatial)
        x_grouped = x._data.reshape(B, G, C // G, *spatial)
        axes = tuple(range(2, x_grouped.ndim))
        mean = np.mean(x_grouped, axis=axes, keepdims=True)
        var = np.var(x_grouped, axis=axes, keepdims=True)
        x_norm = (x_grouped - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(shape)

        if self.affine:
            self.weight._ensure_cpu()
            self.bias._ensure_cpu()
            # Broadcast weight/bias over spatial dims
            w_shape = [1, C] + [1] * len(spatial)
            x_norm = x_norm * self.weight._data.reshape(w_shape) + \
                     self.bias._data.reshape(w_shape)

        rg = x._requires_grad and _ag.is_grad_enabled()
        return Tensor._wrap(x_norm.astype(np.float32), rg, None, x._device)


# ──────────────────────── AvgPool2d / MaxPool2d ───────────────────────

class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        x._ensure_cpu()
        B, C, H, W = x._data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        if pH > 0 or pW > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                              mode='constant')
        else:
            x_padded = x._data

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        out = np.zeros((B, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * sH, j * sW
                out[:, :, i, j] = np.mean(
                    x_padded[:, :, h_s:h_s + kH, w_s:w_s + kW],
                    axis=(2, 3))

        return Tensor._wrap(out, x._requires_grad, None, x._device)


class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        x._ensure_cpu()
        B, C, H, W = x._data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding

        if pH > 0 or pW > 0:
            x_padded = np.pad(x._data,
                              ((0, 0), (0, 0), (pH, pH), (pW, pW)),
                              mode='constant', constant_values=-np.inf)
        else:
            x_padded = x._data

        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1

        out = np.zeros((B, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * sH, j * sW
                out[:, :, i, j] = np.max(
                    x_padded[:, :, h_s:h_s + kH, w_s:w_s + kW],
                    axis=(2, 3))

        return Tensor._wrap(out, x._requires_grad, None, x._device)


# ──────────────────────── AdaptiveAvgPool2d ───────────────────────────

class AdaptiveAvgPool2d(Module):
    """Adaptive 2D average pooling to a target output size."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x):
        x._ensure_cpu()
        B, C, H, W = x._data.shape
        oH, oW = self.output_size

        out = np.zeros((B, C, oH, oW), dtype=np.float32)
        for i in range(oH):
            h_start = int(np.floor(i * H / oH))
            h_end = int(np.ceil((i + 1) * H / oH))
            for j in range(oW):
                w_start = int(np.floor(j * W / oW))
                w_end = int(np.ceil((j + 1) * W / oW))
                out[:, :, i, j] = np.mean(
                    x._data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return Tensor._wrap(out, x._requires_grad, None, x._device)


# ──────────────────────── Upsample ────────────────────────────────────

class Upsample(Module):
    """Upsample spatial (or spatio-temporal) data via nearest or bilinear interpolation."""

    def __init__(self, scale_factor=None, size=None, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, x):
        x._ensure_cpu()
        data = x._data
        ndim = data.ndim

        if ndim == 4:  # (B, C, H, W)
            return self._upsample_2d(x)
        elif ndim == 5:  # (B, C, D, H, W)
            return self._upsample_3d(x)
        else:
            raise ValueError(f"Upsample expects 4D or 5D input, got {ndim}D")

    def _upsample_2d(self, x):
        B, C, H, W = x._data.shape
        if self.size is not None:
            oH, oW = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        else:
            sf = self.scale_factor
            if isinstance(sf, (tuple, list)):
                oH, oW = int(H * sf[0]), int(W * sf[1])
            else:
                oH, oW = int(H * sf), int(W * sf)

        if self.mode == 'nearest':
            row_idx = (np.arange(oH) * H / oH).astype(int)
            col_idx = (np.arange(oW) * W / oW).astype(int)
            out = x._data[:, :, row_idx[:, None], col_idx[None, :]]
        elif self.mode == 'bilinear':
            out = _bilinear_interp_2d(x._data, oH, oW)
        else:
            raise ValueError(f"Unsupported upsample mode: {self.mode}")

        return Tensor._wrap(out.astype(np.float32),
                            x._requires_grad, None, x._device)

    def _upsample_3d(self, x):
        B, C, D, H, W = x._data.shape
        if self.size is not None:
            if isinstance(self.size, (tuple, list)):
                oD, oH, oW = self.size
            else:
                oD = oH = oW = self.size
        else:
            sf = self.scale_factor
            if isinstance(sf, (tuple, list)):
                oD, oH, oW = int(D * sf[0]), int(H * sf[1]), int(W * sf[2])
            else:
                oD, oH, oW = int(D * sf), int(H * sf), int(W * sf)

        if self.mode == 'nearest':
            d_idx = (np.arange(oD) * D / oD).astype(int)
            h_idx = (np.arange(oH) * H / oH).astype(int)
            w_idx = (np.arange(oW) * W / oW).astype(int)
            out = x._data[:, :,
                          d_idx[:, None, None],
                          h_idx[None, :, None],
                          w_idx[None, None, :]]
        else:
            # Trilinear: upsample depth then bilinear
            # First upsample along depth axis
            d_idx = np.linspace(0, D - 1, oD)
            d_floor = np.floor(d_idx).astype(int)
            d_ceil = np.minimum(d_floor + 1, D - 1)
            d_frac = (d_idx - d_floor).reshape(1, 1, oD, 1, 1).astype(np.float32)

            temp = x._data[:, :, d_floor] * (1 - d_frac) + x._data[:, :, d_ceil] * d_frac
            # Now bilinear on each spatial slice
            result = np.zeros((B, C, oD, oH, oW), dtype=np.float32)
            for di in range(oD):
                result[:, :, di] = _bilinear_interp_2d(temp[:, :, di], oH, oW)
            out = result

        return Tensor._wrap(out.astype(np.float32),
                            x._requires_grad, None, x._device)


def _bilinear_interp_2d(data, oH, oW):
    """Bilinear interpolation for 4D (B, C, H, W) or 3D (B, C, HW_slice) data."""
    B, C, H, W = data.shape
    h_idx = np.linspace(0, H - 1, oH)
    w_idx = np.linspace(0, W - 1, oW)
    hg, wg = np.meshgrid(h_idx, w_idx, indexing='ij')

    h0 = np.floor(hg).astype(int)
    h1 = np.minimum(h0 + 1, H - 1)
    w0 = np.floor(wg).astype(int)
    w1 = np.minimum(w0 + 1, W - 1)

    ha = (hg - h0).astype(np.float32)
    wa = (wg - w0).astype(np.float32)

    out = (data[:, :, h0, w0] * (1 - ha) * (1 - wa) +
           data[:, :, h1, w0] * ha * (1 - wa) +
           data[:, :, h0, w1] * (1 - ha) * wa +
           data[:, :, h1, w1] * ha * wa)
    return out


# ──────────────────────── Tanh ────────────────────────────────────────

class Tanh(Module):
    """Hyperbolic tangent activation."""

    def forward(self, x):
        x._ensure_cpu()
        result_data = np.tanh(x._data)
        rg = x._requires_grad and _ag.is_grad_enabled()
        grad_fn = None
        if rg:
            grad_fn = _ag.TanhBackward()
            grad_fn.inputs = [x]
            grad_fn.saved = {'result': result_data}
        return Tensor._wrap(result_data, rg, grad_fn, x._device)


# ──────────────────────── Sigmoid (Module) ────────────────────────────

class Sigmoid(Module):
    """Sigmoid activation as a module."""

    def forward(self, x):
        return x.sigmoid()


# ──────────────────────── PixelShuffle ────────────────────────────────

class PixelShuffle(Module):
    """Rearrange (B, C*r^2, H, W) → (B, C, H*r, W*r)."""

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x._ensure_cpu()
        B, C, H, W = x._data.shape
        r = self.upscale_factor
        assert C % (r * r) == 0
        C_out = C // (r * r)
        out = x._data.reshape(B, C_out, r, r, H, W)
        out = out.transpose(0, 1, 4, 2, 5, 3)
        out = out.reshape(B, C_out, H * r, W * r)
        return Tensor._wrap(out.copy().astype(np.float32),
                            x._requires_grad, None, x._device)


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
