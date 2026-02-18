# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.init — parameter initialization routines."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor
from .parameter import Parameter


def normal_(tensor: Tensor | Parameter, mean: float = 0.0,
            std: float = 1.0) -> Tensor:
    """Fill tensor with values from N(mean, std^2) in-place."""
    tensor._ensure_cpu()
    tensor._data[:] = np.random.normal(mean, std, tensor._data.shape).astype(tensor._data.dtype)
    tensor._gpu = None
    return tensor


def uniform_(tensor: Tensor | Parameter, a: float = 0.0,
             b: float = 1.0) -> Tensor:
    tensor._ensure_cpu()
    tensor._data[:] = np.random.uniform(a, b, tensor._data.shape).astype(tensor._data.dtype)
    tensor._gpu = None
    return tensor


def zeros_(tensor: Tensor | Parameter) -> Tensor:
    tensor._ensure_cpu()
    tensor._data.fill(0)
    tensor._gpu = None
    return tensor


def ones_(tensor: Tensor | Parameter) -> Tensor:
    tensor._ensure_cpu()
    tensor._data.fill(1)
    tensor._gpu = None
    return tensor


def constant_(tensor: Tensor | Parameter, val: float) -> Tensor:
    tensor._ensure_cpu()
    tensor._data.fill(val)
    tensor._gpu = None
    return tensor


def xavier_uniform_(tensor: Tensor | Parameter, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor | Parameter, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor | Parameter, a: float = 0,
                     mode: str = 'fan_in',
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor | Parameter, a: float = 0,
                    mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def _calculate_fan_in_and_fan_out(tensor):
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Fan in/out requires at least 2D tensor")
    fan_in = shape[1]
    fan_out = shape[0]
    if len(shape) > 2:
        receptive = 1
        for s in shape[2:]:
            receptive *= s
        fan_in *= receptive
        fan_out *= receptive
    return fan_in, fan_out


def _calculate_gain(nonlinearity, param=None):
    gains = {
        'linear': 1,
        'sigmoid': 1,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
    }
    return gains.get(nonlinearity, 1)
