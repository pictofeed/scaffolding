# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Core Tensor class backed by NumPy with autograd support."""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Sequence

from . import autograd as _ag
from .device import device as Device
from .dtype import dtype as Dtype

# Try to import Cython-accelerated ops; fall back to pure-Python
try:
    from . import _tensor_ops as _cops
    _USE_CYTHON = True
except ImportError:
    _cops = None  # type: ignore[assignment]
    _USE_CYTHON = False

# Try to import MPS/Accelerate ops (macOS only)
try:
    from . import _mps_ops as _mops
    _USE_MPS = True
except ImportError:
    _mops = None  # type: ignore[assignment]
    _USE_MPS = False

# Try to import CUDA ops (NVIDIA GPU)
try:
    from . import _cuda_ops as _cuops
    _USE_CUDA = True
except ImportError:
    _cuops = None  # type: ignore[assignment]
    _USE_CUDA = False


def _is_mps(device) -> bool:
    """Check if device is MPS (Metal Performance Shaders)."""
    return device is not None and device._type == 'mps'


def _is_cuda(device) -> bool:
    """Check if device is CUDA."""
    return device is not None and device._type == 'cuda'


# Singleton CPU device to avoid repeated Device('cpu') allocation
_CPU_DEVICE = Device('cpu')


class Tensor:
    """N-dimensional tensor with automatic differentiation.

    Wraps a :class:`numpy.ndarray` and records operations for
    reverse-mode AD when ``requires_grad=True``.
    """

    __slots__ = (
        '_data', '_requires_grad', '_grad', '_grad_fn', '_device',
        '_version', '_grad_out', '_gpu',
    )

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        data: Any,
        dtype: Dtype | np.dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
    ):
        if isinstance(data, Tensor):
            arr = data._ensure_cpu().copy()
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)

        if dtype is not None:
            if isinstance(dtype, Dtype):
                arr = arr.astype(dtype.to_numpy())
            else:
                arr = arr.astype(dtype)

        self._data: np.ndarray = arr
        self._requires_grad: bool = requires_grad
        self._grad: np.ndarray | None = None
        self._grad_fn: _ag.GradFn | None = None
        self._device: Device = Device(device) if device is not None else _CPU_DEVICE
        self._version: int = 0
        self._gpu = None
        # Upload to GPU if CUDA device (float32 and int64)
        if _USE_CUDA and _is_cuda(self._device) and arr.dtype in (np.float32, np.int64):
            self._gpu = _cuops.gputensor_from_numpy(np.ascontiguousarray(arr))

    # ------------------------------------------------------------------ #
    #  Properties                                                        #
    # ------------------------------------------------------------------ #

    @property
    def data(self) -> 'Tensor':
        t = Tensor.__new__(Tensor)
        t._data = self._data
        t._gpu = self._gpu
        t._requires_grad = False
        t._grad = None
        t._grad_fn = None
        t._device = self._device
        t._version = self._version
        return t

    @data.setter
    def data(self, value: 'Tensor'):
        if isinstance(value, Tensor):
            self._data = value._data
            self._gpu = value._gpu
        elif isinstance(value, np.ndarray):
            self._data = value
            self._gpu = None
        else:
            self._data = np.asarray(value)
            self._gpu = None

    @property
    def grad(self) -> 'Tensor | None':
        if self._grad is None:
            return None
        t = Tensor.__new__(Tensor)
        t._data = self._grad
        t._requires_grad = False
        t._grad = None
        t._grad_fn = None
        t._device = self._device
        t._gpu = None
        t._version = 0
        return t

    @grad.setter
    def grad(self, value):
        if value is None:
            self._grad = None
        elif isinstance(value, Tensor):
            self._grad = value._ensure_cpu()
        else:
            self._grad = np.asarray(value)

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_grad = val

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def shape(self) -> tuple:
        if self._data is not None:
            return self._data.shape
        if self._gpu is not None:
            return self._gpu.shape
        return ()

    @property
    def ndim(self) -> int:
        if self._data is not None:
            return self._data.ndim
        if self._gpu is not None:
            return len(self._gpu.shape)
        return 0

    @property
    def dtype(self) -> Dtype:
        if self._data is not None:
            return Dtype.from_numpy(self._data.dtype)
        if self._gpu is not None:
            return Dtype.from_numpy(self._gpu.dtype)
        return Dtype.from_numpy(np.float32)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    # ------------------------------------------------------------------ #
    #  Basic info methods                                                 #
    # ------------------------------------------------------------------ #

    def size(self, dim: int | None = None):
        s = self.shape
        if dim is not None:
            return s[dim]
        return s

    def dim(self) -> int:
        return self.ndim

    def numel(self) -> int:
        if self._data is not None:
            return self._data.size
        if self._gpu is not None:
            return self._gpu.numel
        return 0

    def item(self) -> float | int:
        self._ensure_cpu()
        return self._data.item()

    def data_ptr(self) -> int:
        self._ensure_cpu()
        return self._data.ctypes.data

    def type(self, dtype_str: str | None = None):
        if dtype_str is None:
            return f"scaffolding.{self.dtype.name}Tensor"
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        self._ensure_cpu()
        data_str = repr(self._data)
        prefix = "tensor("
        suffix = ")"
        if self._requires_grad:
            suffix = f", requires_grad=True)"
        elif self._grad_fn is not None:
            suffix = f", grad_fn={self._grad_fn})"
        return prefix + data_str + suffix

    def __bool__(self) -> bool:
        self._ensure_cpu()
        return bool(self._data)

    def __int__(self) -> int:
        self._ensure_cpu()
        return int(self._data)

    def __float__(self) -> float:
        self._ensure_cpu()
        return float(self._data)

    def __reduce__(self):
        """Pickle support — download GPU data first."""
        self._ensure_cpu()
        return (Tensor, (self._data, None, str(self._device), self._requires_grad))

    # ------------------------------------------------------------------ #
    #  Helpers for building autograd graph                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap(data: np.ndarray, requires_grad: bool = False,
              grad_fn: _ag.GradFn | None = None,
              device: Device | None = None) -> 'Tensor':
        t = Tensor.__new__(Tensor)
        t._data = data
        t._requires_grad = requires_grad
        t._grad = None
        t._grad_fn = grad_fn
        t._device = device if device is not None else _CPU_DEVICE
        t._version = 0
        t._gpu = None
        return t

    @staticmethod
    def _wrap_gpu(gpu, requires_grad: bool = False,
                  grad_fn: _ag.GradFn | None = None,
                  device: Device | None = None) -> 'Tensor':
        """Create a Tensor backed by a GpuTensor (no CPU data until needed)."""
        t = Tensor.__new__(Tensor)
        t._data = None
        t._gpu = gpu
        t._requires_grad = requires_grad
        t._grad = None
        t._grad_fn = grad_fn
        t._device = device if device is not None else _CPU_DEVICE
        t._version = 0
        return t

    def _ensure_cpu(self):
        """Ensure _data is populated (download from GPU if needed)."""
        if self._data is None and self._gpu is not None:
            self._data = _cuops.gputensor_to_numpy(self._gpu)
        return self._data

    def _needs_grad(self, *others: 'Tensor') -> bool:
        if not _ag.is_grad_enabled():
            return False
        if self._requires_grad:
            return True
        for o in others:
            if isinstance(o, Tensor) and o._requires_grad:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Arithmetic operators                                               #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        # GPU-resident fast path
        if self._gpu is not None:
            if isinstance(other, (int, float)):
                return Tensor._wrap_gpu(_cuops.dev_adds(self._gpu, float(other)),
                                        device=self._device)
            b = other if isinstance(other, Tensor) else _ensure_tensor(other)
            if b._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_add(self._gpu, b._gpu),
                                        device=self._device)
        self._ensure_cpu()
        if isinstance(other, (int, float)):
            result_data = self._data + other
            rg = self._requires_grad and _ag.is_grad_enabled()
            if rg:
                grad_fn = _ag.AddBackward()
                b = _ensure_tensor(other)
                grad_fn.inputs = [self, b]
                return Tensor._wrap(result_data, True, grad_fn, self._device)
            return Tensor._wrap(result_data, False, None, self._device)
        b = other if isinstance(other, Tensor) else _ensure_tensor(other)
        b._ensure_cpu()
        result_data = self._data + b._data
        rg = self._needs_grad(b)
        grad_fn = None
        if rg:
            grad_fn = _ag.AddBackward()
            grad_fn.inputs = [self, b]
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # GPU-resident fast path
        if self._gpu is not None:
            if isinstance(other, (int, float)):
                return Tensor._wrap_gpu(_cuops.dev_adds(self._gpu, -float(other)),
                                        device=self._device)
            b = other if isinstance(other, Tensor) else _ensure_tensor(other)
            if b._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_sub(self._gpu, b._gpu),
                                        device=self._device)
        self._ensure_cpu()
        if isinstance(other, (int, float)):
            result_data = self._data - other
            rg = self._requires_grad and _ag.is_grad_enabled()
            if rg:
                grad_fn = _ag.SubBackward()
                b = _ensure_tensor(other)
                grad_fn.inputs = [self, b]
                return Tensor._wrap(result_data, True, grad_fn, self._device)
            return Tensor._wrap(result_data, False, None, self._device)
        b = other if isinstance(other, Tensor) else _ensure_tensor(other)
        b._ensure_cpu()
        result_data = self._data - b._data
        rg = self._needs_grad(b)
        grad_fn = None
        if rg:
            grad_fn = _ag.SubBackward()
            grad_fn.inputs = [self, b]
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __rsub__(self, other):
        b = _ensure_tensor(other)
        return b.__sub__(self)

    def __mul__(self, other):
        # GPU-resident fast path
        if self._gpu is not None:
            if isinstance(other, (int, float)):
                return Tensor._wrap_gpu(_cuops.dev_muls(self._gpu, float(other)),
                                        device=self._device)
            b = other if isinstance(other, Tensor) else _ensure_tensor(other)
            if b._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_mul(self._gpu, b._gpu),
                                        device=self._device)
        self._ensure_cpu()
        if isinstance(other, (int, float)):
            result_data = self._data * other
            rg = self._requires_grad and _ag.is_grad_enabled()
            if rg:
                b = _ensure_tensor(other)
                grad_fn = _ag.MulBackward()
                grad_fn.inputs = [self, b]
                grad_fn.saved = {'a': self._data, 'b': b._data}
                return Tensor._wrap(result_data, True, grad_fn, self._device)
            return Tensor._wrap(result_data, False, None, self._device)
        b = other if isinstance(other, Tensor) else _ensure_tensor(other)
        b._ensure_cpu()
        result_data = self._data * b._data
        rg = self._needs_grad(b)
        grad_fn = None
        if rg:
            grad_fn = _ag.MulBackward()
            grad_fn.inputs = [self, b]
            grad_fn.saved = {'a': self._data, 'b': b._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # GPU-resident fast path
        if self._gpu is not None:
            if isinstance(other, (int, float)):
                return Tensor._wrap_gpu(_cuops.dev_muls(self._gpu, 1.0 / float(other)),
                                        device=self._device)
            b = other if isinstance(other, Tensor) else _ensure_tensor(other)
            if b._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_div(self._gpu, b._gpu),
                                        device=self._device)
        self._ensure_cpu()
        if isinstance(other, (int, float)):
            result_data = self._data / other
            rg = self._requires_grad and _ag.is_grad_enabled()
            if rg:
                b = _ensure_tensor(other)
                grad_fn = _ag.DivBackward()
                grad_fn.inputs = [self, b]
                grad_fn.saved = {'a': self._data, 'b': b._data}
                return Tensor._wrap(result_data, True, grad_fn, self._device)
            return Tensor._wrap(result_data, False, None, self._device)
        b = other if isinstance(other, Tensor) else _ensure_tensor(other)
        b._ensure_cpu()
        result_data = self._data / b._data
        rg = self._needs_grad(b)
        grad_fn = None
        if rg:
            grad_fn = _ag.DivBackward()
            grad_fn.inputs = [self, b]
            grad_fn.saved = {'a': self._data, 'b': b._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __rtruediv__(self, other):
        b = _ensure_tensor(other)
        return b.__truediv__(self)

    def __neg__(self):
        # GPU-resident fast path
        if self._gpu is not None:
            return Tensor._wrap_gpu(_cuops.dev_neg(self._gpu), device=self._device)
        self._ensure_cpu()
        result_data = -self._data
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.NegBackward()
            grad_fn.inputs = [self]
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __pow__(self, other):
        self._ensure_cpu()
        if isinstance(other, Tensor):
            other._ensure_cpu()
            exp_val = other._data
        else:
            exp_val = other
        result_data = np.power(self._data, exp_val)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.PowBackward()
            grad_fn.inputs = [self, None]
            grad_fn.saved = {'base': self._data, 'exp': exp_val, 'result': result_data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __rpow__(self, other):
        b = _ensure_tensor(other)
        return b.__pow__(self)

    # Comparison operators
    def __gt__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data > b._data, device=self._device)

    def __ge__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data >= b._data, device=self._device)

    def __lt__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data < b._data, device=self._device)

    def __le__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data <= b._data, device=self._device)

    def __eq__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data == b._data, device=self._device)

    def __ne__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data != b._data, device=self._device)

    def __invert__(self):
        self._ensure_cpu()
        return Tensor._wrap(~self._data, device=self._device)

    def __and__(self, other):
        self._ensure_cpu()
        b = _ensure_tensor(other)
        b._ensure_cpu()
        return Tensor._wrap(self._data & b._data, device=self._device)

    # Indexing
    def __getitem__(self, key):
        self._ensure_cpu()
        if isinstance(key, Tensor):
            key._ensure_cpu()
            key = key._data
        elif isinstance(key, tuple):
            key = tuple((k._ensure_cpu() if isinstance(k, Tensor) else k) for k in key)
        result_data = self._data[key]
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SliceBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'input_shape': self._data.shape, 'key': key}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def __setitem__(self, key, value):
        self._ensure_cpu()
        if isinstance(key, Tensor):
            key._ensure_cpu()
            key = key._data
        if isinstance(value, Tensor):
            value._ensure_cpu()
            value = value._data
        self._data[key] = value
        self._version += 1

    # ------------------------------------------------------------------ #
    #  Tensor math methods                                                #
    # ------------------------------------------------------------------ #

    def matmul(self, other: 'Tensor') -> 'Tensor':
        return matmul(self, other)

    def pow(self, exp) -> 'Tensor':
        return self.__pow__(exp)

    def exp(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_exp(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_exp(self._data)
        elif _USE_MPS:
            self._ensure_cpu()
            result_data = _mops.accelerate_exp(self._data)
        elif _USE_CYTHON:
            self._ensure_cpu()
            result_data = _cops.exp_forward(self._data)
        else:
            self._ensure_cpu()
            result_data = np.exp(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.ExpBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'result': result_data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def log(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_log(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_log(self._data)
        elif _USE_MPS:
            self._ensure_cpu()
            result_data = _mops.accelerate_log(self._data)
        else:
            self._ensure_cpu()
            result_data = np.log(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.LogBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'x': self._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def sqrt(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_sqrt(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_sqrt(self._data)
        elif _USE_MPS:
            self._ensure_cpu()
            result_data = _mops.accelerate_sqrt(self._data)
        else:
            self._ensure_cpu()
            result_data = np.sqrt(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SqrtBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'result': result_data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def rsqrt(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_rsqrt(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_rsqrt(self._data)
        elif _USE_MPS:
            self._ensure_cpu()
            result_data = _mops.accelerate_rsqrt(self._data)
        else:
            self._ensure_cpu()
            result_data = 1.0 / np.sqrt(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.RsqrtBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'x': self._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def sigmoid(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_sigmoid(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_sigmoid(self._data)
        elif _USE_MPS:
            self._ensure_cpu()
            result_data = _mops.accelerate_sigmoid(self._data)
        elif _USE_CYTHON:
            self._ensure_cpu()
            result_data = _cops.sigmoid_forward(self._data)
        else:
            self._ensure_cpu()
            # Numerically stable sigmoid using np.clip to avoid overflow
            x = np.clip(self._data, -88.0, 88.0)
            result_data = 1.0 / (1.0 + np.exp(-x))
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SigmoidBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'result': result_data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def sin(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_sin(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_sin(self._data)
        else:
            self._ensure_cpu()
            result_data = np.sin(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SinBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'x': self._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def cos(self) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_cos(self._gpu), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_cos(self._data)
        else:
            self._ensure_cpu()
            result_data = np.cos(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.CosBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'x': self._data}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def clamp(self, min=None, max=None) -> 'Tensor':
        if _USE_CUDA and _is_cuda(self._device):
            lo = min if min is not None else -3.4e38
            hi = max if max is not None else 3.4e38
            if self._gpu is not None:
                return Tensor._wrap_gpu(_cuops.dev_clamp(self._gpu, lo, hi), device=self._device)
            self._ensure_cpu()
            result_data = _cuops.cuda_clamp(self._data, lo, hi)
        else:
            self._ensure_cpu()
            result_data = np.clip(self._data, min, max)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.ClampBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'x': self._data, 'min': min, 'max': max}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    # ---- Reduction methods ----

    def sum(self, dim=None, keepdim=False) -> 'Tensor':
        self._ensure_cpu()
        result_data = np.sum(self._data, axis=dim, keepdims=keepdim)
        if isinstance(result_data, np.generic):
            result_data = np.array(result_data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SumBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {
                'input_shape': self._data.shape,
                'axis': dim,
                'keepdims': keepdim,
            }
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def mean(self, dim=None, keepdim=False) -> 'Tensor':
        self._ensure_cpu()
        result_data = np.mean(self._data, axis=dim, keepdims=keepdim)
        if isinstance(result_data, np.generic):
            result_data = np.array(result_data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.MeanBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {
                'input_shape': self._data.shape,
                'axis': dim,
                'keepdims': keepdim,
            }
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def nonzero(self, as_tuple=False):
        self._ensure_cpu()
        indices = np.nonzero(self._data)
        if as_tuple:
            return tuple(Tensor._wrap(np.asarray(idx), device=self._device) for idx in indices)
        return Tensor._wrap(np.stack(indices, axis=-1), device=self._device)

    # ---- Shape methods ----

    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # GPU-resident fast path: metadata-only reshape (zero copy)
        if self._gpu is not None:
            # Resolve -1 dim
            known = 1
            neg_idx = -1
            shape = list(shape)
            for i, s in enumerate(shape):
                if s == -1:
                    neg_idx = i
                else:
                    known *= s
            if neg_idx >= 0:
                shape[neg_idx] = self._gpu.numel // known
            shape = tuple(shape)
            from _cuda_ops import GpuTensor
            gt = GpuTensor(self._gpu.buffer, shape, self._gpu.dtype)
            return Tensor._wrap_gpu(gt, device=self._device)
        result_data = self._data.reshape(shape)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.ReshapeBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'input_shape': self._data.shape}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def view(self, *shape) -> 'Tensor':
        return self.reshape(*shape)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        self._ensure_cpu()
        result_data = np.swapaxes(self._data, dim0, dim1)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.TransposeBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'dim0': dim0, 'dim1': dim1}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def permute(self, *dims) -> 'Tensor':
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        self._ensure_cpu()
        result_data = np.transpose(self._data, dims)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.PermuteBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'dims': dims}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def unsqueeze(self, dim: int) -> 'Tensor':
        self._ensure_cpu()
        result_data = np.expand_dims(self._data, axis=dim)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.UnsqueezeBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'dim': dim}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def squeeze(self, dim: int | None = None) -> 'Tensor':
        self._ensure_cpu()
        if dim is not None:
            result_data = np.squeeze(self._data, axis=dim)
        else:
            result_data = np.squeeze(self._data)
        grad_fn = None
        rg = self._requires_grad and _ag.is_grad_enabled()
        if rg:
            grad_fn = _ag.SqueezeBackward()
            grad_fn.inputs = [self]
            grad_fn.saved = {'input_shape': self._data.shape}
        return Tensor._wrap(result_data, rg, grad_fn, self._device)

    def contiguous(self) -> 'Tensor':
        self._ensure_cpu()
        if self._data.flags.c_contiguous:
            return self
        return Tensor._wrap(np.ascontiguousarray(self._data),
                            self._requires_grad, self._grad_fn, self._device)

    def chunk(self, chunks: int, dim: int = 0) -> tuple['Tensor', ...]:
        self._ensure_cpu()
        arrays = np.array_split(self._data, chunks, axis=dim)
        results = []
        for arr in arrays:
            rg = self._requires_grad and _ag.is_grad_enabled()
            results.append(Tensor._wrap(arr, rg, None, self._device))
        return tuple(results)

    def split(self, split_size: int, dim: int = 0) -> tuple['Tensor', ...]:
        n = self.shape[dim]
        chunks = math.ceil(n / split_size)
        return self.chunk(chunks, dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        shape = self.shape
        if end_dim < 0:
            end_dim += len(shape)
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim + 1:]
        return self.reshape(*new_shape)

    def expand(self, *sizes) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(np.broadcast_to(self._data, sizes).copy(),
                            self._requires_grad, None, self._device)

    def repeat(self, *sizes) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(np.tile(self._data, sizes),
                            self._requires_grad, None, self._device)

    # ---- Type / device conversion ----

    def float(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.astype(np.float32),
                            self._requires_grad, None, self._device)

    def half(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.astype(np.float16),
                            self._requires_grad, None, self._device)

    def long(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.astype(np.int64),
                            self._requires_grad, None, self._device)

    def int(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.astype(np.int32),
                            self._requires_grad, None, self._device)

    def bool(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.astype(np.bool_),
                            self._requires_grad, None, self._device)

    def to(self, *args, **kwargs) -> 'Tensor':
        """Move tensor to device and/or convert dtype.

        Supports: .to(device), .to(dtype), .to(device, dtype=...)
        """
        new_dtype = None
        new_device = self._device
        for a in args:
            if isinstance(a, Device):
                new_device = a
            elif isinstance(a, str):
                if a in ('cpu', 'cuda', 'mps') or ':' in a:
                    new_device = Device(a)
                else:
                    new_device = Device(a)
            elif isinstance(a, Dtype):
                new_dtype = a
        if 'dtype' in kwargs:
            new_dtype = kwargs['dtype']
        if 'device' in kwargs:
            new_device = Device(kwargs['device'])

        arr = self._ensure_cpu()
        if new_dtype is not None:
            if isinstance(new_dtype, Dtype):
                arr = arr.astype(new_dtype.to_numpy())
            else:
                arr = arr.astype(new_dtype)
        t = Tensor._wrap(arr.copy(), self._requires_grad, None, new_device)
        # Upload to GPU if moving to CUDA
        if _USE_CUDA and _is_cuda(new_device) and t._data.dtype in (np.float32, np.int64):
            t._gpu = _cuops.gputensor_from_numpy(np.ascontiguousarray(t._data))
        return t

    def cpu(self) -> 'Tensor':
        return self.to('cpu')

    def cuda(self, device_id: int | None = None) -> 'Tensor':
        # Scaffolding runs on CPU; simply mark device
        dev = f"cuda:{device_id}" if device_id is not None else "cuda"
        return self.to(dev)

    def mps(self) -> 'Tensor':
        """Move tensor to MPS (Apple Metal) device."""
        return self.to('mps')

    def numpy(self) -> np.ndarray:
        self._ensure_cpu()
        return self._data.copy()

    def tolist(self):
        self._ensure_cpu()
        return self._data.tolist()

    # ---- In-place operations ----

    def mul_(self, other) -> 'Tensor':
        self._ensure_cpu()
        if isinstance(other, Tensor):
            other._ensure_cpu()
            self._data *= other._data
        else:
            self._data *= other
        self._version += 1
        return self

    def add_(self, other, alpha: float = 1.0) -> 'Tensor':
        self._ensure_cpu()
        if isinstance(other, Tensor):
            other._ensure_cpu()
            self._data += alpha * other._data
        else:
            self._data += alpha * other
        self._version += 1
        return self

    def sub_(self, other) -> 'Tensor':
        self._ensure_cpu()
        if isinstance(other, Tensor):
            other._ensure_cpu()
            self._data -= other._data
        else:
            self._data -= other
        self._version += 1
        return self

    def div_(self, other) -> 'Tensor':
        self._ensure_cpu()
        if isinstance(other, Tensor):
            other._ensure_cpu()
            self._data /= other._data
        else:
            self._data /= other
        self._version += 1
        return self

    def copy_(self, src: 'Tensor') -> 'Tensor':
        self._ensure_cpu()
        if isinstance(src, Tensor):
            src._ensure_cpu()
            np.copyto(self._data, src._data)
        else:
            np.copyto(self._data, src)
        self._version += 1
        return self

    def zero_(self) -> 'Tensor':
        self._ensure_cpu()
        self._data.fill(0)
        self._version += 1
        return self

    def fill_(self, value) -> 'Tensor':
        self._ensure_cpu()
        self._data.fill(value)
        self._version += 1
        return self

    def clamp_(self, min=None, max=None) -> 'Tensor':
        self._ensure_cpu()
        if min is not None:
            np.maximum(self._data, min, out=self._data)
        if max is not None:
            np.minimum(self._data, max, out=self._data)
        self._version += 1
        return self

    # ---- Autograd methods ----

    def backward(self, gradient=None):
        if gradient is not None:
            if isinstance(gradient, Tensor):
                gradient = gradient._data
        _ag.backward(self, gradient)

    def detach(self) -> 'Tensor':
        self._ensure_cpu()
        t = Tensor._wrap(self._data, False, None, self._device)
        return t

    def requires_grad_(self, requires_grad: bool = True) -> 'Tensor':
        self._requires_grad = requires_grad
        return self

    def retain_grad(self):
        pass  # All leaf tensors retain grad by default

    # ---- Misc ----

    def clone(self) -> 'Tensor':
        self._ensure_cpu()
        return Tensor._wrap(self._data.copy(), self._requires_grad,
                            None, self._device)

    def astype(self, dtype) -> 'Tensor':
        self._ensure_cpu()
        if isinstance(dtype, Dtype):
            return Tensor._wrap(self._data.astype(dtype.to_numpy()),
                                self._requires_grad, None, self._device)
        return Tensor._wrap(self._data.astype(dtype),
                            self._requires_grad, None, self._device)


# ====================================================================
# Module-level factory functions (torch.zeros, torch.ones, etc.)
# ====================================================================

def _resolve_dtype(dtype) -> np.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, Dtype):
        return dtype.to_numpy()
    return np.dtype(dtype)


def _resolve_device(device) -> Device:
    if device is None:
        return _CPU_DEVICE
    if isinstance(device, Device):
        return device
    return Device(device)


def _make_tensor(arr: np.ndarray, requires_grad: bool, device: Device) -> Tensor:
    """Create a Tensor, auto-uploading to GPU if device is CUDA."""
    t = Tensor._wrap(arr, requires_grad, None, device)
    if _USE_CUDA and _is_cuda(device) and arr.dtype in (np.float32, np.int64):
        t._gpu = _cuops.gputensor_from_numpy(np.ascontiguousarray(arr))
    return t


def tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.zeros(size, dtype=dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def zeros_like(input: Tensor, dtype=None, device=None) -> Tensor:
    dt = _resolve_dtype(dtype) or input._ensure_cpu().dtype
    arr = np.zeros(input.shape, dtype=dt)
    dev = _resolve_device(device) if device else input._device
    return _make_tensor(arr, False, dev)


def ones(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.ones(size, dtype=dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def ones_like(input: Tensor, dtype=None, device=None) -> Tensor:
    dt = _resolve_dtype(dtype) or input._ensure_cpu().dtype
    arr = np.ones(input.shape, dtype=dt)
    dev = _resolve_device(device) if device else input._device
    return _make_tensor(arr, False, dev)


def full(size, fill_value, dtype=None, device=None, requires_grad=False) -> Tensor:
    if isinstance(size, (tuple, list)):
        size = tuple(size)
    else:
        size = (size,)
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.full(size, fill_value, dtype=dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def randn(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.random.randn(*size).astype(dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def rand(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.random.rand(*size).astype(dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def randint(low, high=None, size=None, dtype=None, device=None) -> Tensor:
    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()
    if isinstance(size, (tuple, list)):
        size = tuple(size)
    else:
        size = (size,)
    dt = _resolve_dtype(dtype) or np.int64
    arr = np.random.randint(low, high, size=size).astype(dt)
    return _make_tensor(arr, False, _resolve_device(device))


def arange(*args, dtype=None, device=None, requires_grad=False) -> Tensor:
    arr = np.arange(*args)
    if dtype is not None:
        arr = arr.astype(_resolve_dtype(dtype))
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False) -> Tensor:
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.linspace(start, end, steps, dtype=dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


def empty(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    dt = _resolve_dtype(dtype) or np.float32
    arr = np.empty(size, dtype=dt)
    return _make_tensor(arr, requires_grad, _resolve_device(device))


# ====================================================================
# Tensor operations (torch.matmul, torch.cat, torch.stack, etc.)
# ====================================================================

def matmul(a: Tensor, b: Tensor) -> Tensor:
    if _USE_CUDA and _is_cuda(a._device):
        # GPU-resident fast path
        if a._gpu is not None and b._gpu is not None:
            a_ndim = len(a._gpu.shape)
            b_ndim = len(b._gpu.shape)
            if a_ndim == 2 and b_ndim == 2:
                return Tensor._wrap_gpu(_cuops.dev_matmul(a._gpu, b._gpu), device=a._device)
            elif a_ndim >= 3 or b_ndim >= 3:
                return Tensor._wrap_gpu(_cuops.dev_batched_matmul(a._gpu, b._gpu), device=a._device)
        a_data = a._ensure_cpu()
        b_data = b._ensure_cpu()
        if a_data.ndim == 2 and b_data.ndim == 2:
            result_data = _cuops.cuda_matmul(a_data, b_data)
        elif a_data.ndim >= 3 or b_data.ndim >= 3:
            result_data = _cuops.cuda_batched_matmul(a_data, b_data)
        else:
            result_data = np.matmul(a_data, b_data)
    elif _USE_MPS:
        a._ensure_cpu()
        b._ensure_cpu()
        a_data = a._data
        b_data = b._data
        if a_data.ndim >= 2 and b_data.ndim >= 2 and a_data.dtype == np.float32:
            if a_data.ndim == 2 and b_data.ndim == 2:
                a_c = a_data if a_data.flags.c_contiguous else np.ascontiguousarray(a_data)
                b_c = b_data if b_data.flags.c_contiguous else np.ascontiguousarray(b_data)
                result_data = _mops.blas_sgemm(a_c, b_c)
            else:
                result_data = _mops.accelerate_batched_matmul(a_data, b_data)
        else:
            result_data = np.matmul(a_data, b_data)
    elif _USE_CYTHON:
        a._ensure_cpu()
        b._ensure_cpu()
        result_data = _cops.matmul_forward(a._data, b._data)
    else:
        a._ensure_cpu()
        b._ensure_cpu()
        result_data = np.matmul(a._data, b._data)
    grad_fn = None
    rg = a._needs_grad(b)
    if rg:
        grad_fn = _ag.MatMulBackward()
        grad_fn.inputs = [a, b]
        grad_fn.saved = {'a': a._data, 'b': b._data}
    return Tensor._wrap(result_data, rg, grad_fn, a._device)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    for t in tensors:
        t._ensure_cpu()
    arrays = [t._data for t in tensors]
    result_data = np.concatenate(arrays, axis=dim)
    rg = any(t._requires_grad for t in tensors) and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.CatBackward()
        grad_fn.inputs = list(tensors)
        grad_fn.saved = {
            'sizes': [a.shape[dim] for a in arrays],
            'dim': dim
        }
    return Tensor._wrap(result_data, rg, grad_fn, tensors[0]._device)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    for t in tensors:
        t._ensure_cpu()
    arrays = [t._data for t in tensors]
    result_data = np.stack(arrays, axis=dim)
    rg = any(t._requires_grad for t in tensors) and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.StackBackward()
        grad_fn.inputs = list(tensors)
        grad_fn.saved = {'dim': dim, 'n': len(tensors)}
    return Tensor._wrap(result_data, rg, grad_fn, tensors[0]._device)


def outer(a: Tensor, b: Tensor) -> Tensor:
    result_data = np.outer(a._data, b._data)
    return Tensor._wrap(result_data, a._needs_grad(b), None, a._device)


def where(condition, x, y):
    if isinstance(condition, Tensor):
        condition._ensure_cpu()
        condition_data = condition._data
    else:
        condition_data = np.asarray(condition)
    if isinstance(x, Tensor):
        x._ensure_cpu()
        x_data = x._data
    else:
        x_data = x
    if isinstance(y, Tensor):
        y._ensure_cpu()
        y_data = y._data
    else:
        y_data = y

    result_data = np.where(condition_data, x_data, y_data)
    rg = False
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        rg = (x._requires_grad or y._requires_grad) and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.WhereBackward()
        grad_fn.inputs = [None, x if isinstance(x, Tensor) else None,
                          y if isinstance(y, Tensor) else None]
        grad_fn.saved = {'condition': condition_data}
    return Tensor._wrap(result_data, rg, grad_fn,
                        x._device if isinstance(x, Tensor) else Device('cpu'))


def cumsum(input: Tensor, dim: int) -> Tensor:
    result_data = np.cumsum(input._data, axis=dim)
    grad_fn = None
    rg = input._requires_grad and _ag.is_grad_enabled()
    if rg:
        grad_fn = _ag.CumsumBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'axis': dim}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# Unary element-wise ops
def exp(input: Tensor) -> Tensor:
    return input.exp()


def log(input: Tensor) -> Tensor:
    return input.log()


def sqrt(input: Tensor) -> Tensor:
    return input.sqrt()


def rsqrt(input: Tensor) -> Tensor:
    return input.rsqrt()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def sin(input: Tensor) -> Tensor:
    return input.sin()


def cos(input: Tensor) -> Tensor:
    return input.cos()


def atan2(y: Tensor, x: Tensor) -> Tensor:
    result_data = np.arctan2(y._data, x._data)
    rg = y._needs_grad(x)
    grad_fn = None
    if rg:
        grad_fn = _ag.Atan2Backward()
        grad_fn.inputs = [y, x]
        grad_fn.saved = {'y': y._data, 'x': x._data}
    return Tensor._wrap(result_data, rg, grad_fn, y._device)


def expm1(input: Tensor) -> Tensor:
    result_data = np.expm1(input._data)
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.Expm1Backward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': input._data}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def softmax(input: Tensor, dim: int) -> Tensor:
    if _USE_CUDA and _is_cuda(input._device) and input._gpu is not None:
        return Tensor._wrap_gpu(_cuops.dev_softmax(input._gpu, dim), device=input._device)
    x = input._ensure_cpu() if input._data is None else input._data
    if _USE_CUDA and _is_cuda(input._device):
        s = _cuops.cuda_softmax(x, dim)
    elif _USE_MPS and x.ndim == 2 and dim in (-1, x.ndim - 1) and x.dtype == np.float32:
        s = _mops.accelerate_softmax(x, dim)
    else:
        x_max = x.max(axis=dim, keepdims=True)
        e = np.exp(x - x_max)
        e_sum = e.sum(axis=dim, keepdims=True)
        np.divide(e, e_sum, out=e)
        s = e
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.SoftmaxBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'result': s, 'dim': dim}
    return Tensor._wrap(s, rg, grad_fn, input._device)


def sort(input: Tensor, dim: int = -1, descending: bool = False):
    input._ensure_cpu()
    if descending:
        sorted_arr = np.sort(input._data, axis=dim)[..., ::-1].copy()
        indices = np.argsort(input._data, axis=dim)[..., ::-1].copy()
    else:
        sorted_arr = np.sort(input._data, axis=dim)
        indices = np.argsort(input._data, axis=dim)
    return (Tensor._wrap(sorted_arr, False, None, input._device),
            Tensor._wrap(indices.astype(np.int64), False, None, input._device))


def multinomial(input: Tensor, num_samples: int,
                replacement: bool = True) -> Tensor:
    input._ensure_cpu()
    probs = input._data.astype(np.float64)
    if probs.ndim == 1:
        probs = probs / probs.sum()
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        result = np.random.choice(len(probs), size=num_samples,
                                  replace=replacement, p=probs)
        return Tensor._wrap(result.astype(np.int64), False, None, input._device)
    # Batched
    results = []
    for i in range(probs.shape[0]):
        p = probs[i]
        p = p / p.sum()
        p = np.clip(p, 0, None)
        p = p / p.sum()
        r = np.random.choice(len(p), size=num_samples, replace=replacement, p=p)
        results.append(r)
    return Tensor._wrap(np.stack(results).astype(np.int64), False, None, input._device)


def bernoulli(input: Tensor) -> Tensor:
    input._ensure_cpu()
    arr = (np.random.rand(*input._data.shape) < input._data).astype(input._data.dtype)
    return Tensor._wrap(arr, False, None, input._device)


# ====================================================================
# Utility functions
# ====================================================================

def manual_seed(seed: int) -> None:
    np.random.seed(seed)


def save(obj, path: str) -> None:
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path: str, map_location=None) -> Any:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def compile(model, **kwargs):
    """No-op torch.compile equivalent."""
    return model


def set_float32_matmul_precision(precision: str) -> None:
    """No-op for compatibility."""
    pass


def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
