# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.functional — stateless neural network operations (F.*)."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor, _USE_MPS, _is_mps, _mops
from .. import autograd as _ag


# ──────────────────────── Activations ─────────────────────────────────

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    result_data = np.maximum(input._data, 0)
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.ReluBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': input._data}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def silu(input: Tensor) -> Tensor:
    if _USE_MPS and _is_mps(input._device):
        result_data = _mops.accelerate_silu(input._data)
    else:
        sig = 1.0 / (1.0 + np.exp(-input._data))
        result_data = input._data * sig
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.SiluBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': input._data}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def gelu(input: Tensor) -> Tensor:
    if _USE_MPS and _is_mps(input._device):
        result_data = _mops.accelerate_gelu(input._data)
    else:
        x = input._data
        result_data = 0.5 * x * (1 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    return Tensor._wrap(result_data, input._requires_grad, None, input._device)


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def softplus(input: Tensor, beta: float = 1.0,
             threshold: float = 20.0) -> Tensor:
    x = input._data
    bx = beta * x
    result_data = np.where(bx > threshold, x, np.log(1 + np.exp(bx)) / beta)
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.SoftplusBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': x, 'beta': beta}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# ──────────────────────── Softmax ─────────────────────────────────────

def softmax(input: Tensor, dim: int = -1, dtype=None) -> Tensor:
    x = input._data
    if _USE_MPS and _is_mps(input._device) and x.ndim == 2 and dim in (-1, x.ndim - 1):
        s = _mops.accelerate_softmax(x, dim)
    else:
        x_max = np.max(x, axis=dim, keepdims=True)
        e = np.exp(x - x_max)
        s = e / np.sum(e, axis=dim, keepdims=True)
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.SoftmaxBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'result': s, 'dim': dim}
    return Tensor._wrap(s, rg, grad_fn, input._device)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    x = input._data
    x_max = np.max(x, axis=dim, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x - x_max), axis=dim, keepdims=True)) + x_max
    result_data = x - logsumexp
    return Tensor._wrap(result_data, input._requires_grad, None, input._device)


# ──────────────────────── Loss functions ──────────────────────────────

def cross_entropy(input: Tensor, target: Tensor,
                  reduction: str = 'mean') -> Tensor:
    """Cross-entropy loss with log-softmax."""
    x = input._data
    t = target._data.astype(np.intp)

    # Reshape to 2D if needed
    if x.ndim == 1:
        x = x.reshape(1, -1)
        t = t.reshape(-1)

    N, C = x.shape
    # Stable softmax
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    probs = e / np.sum(e, axis=-1, keepdims=True)

    # NLL
    log_probs = np.log(probs + 1e-12)
    losses = -log_probs[np.arange(N), t]

    if reduction == 'mean':
        result_data = np.array(losses.mean(), dtype=np.float32)
    elif reduction == 'sum':
        result_data = np.array(losses.sum(), dtype=np.float32)
    else:
        result_data = losses

    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.CrossEntropyBackward()
        grad_fn.inputs = [input, None]
        grad_fn.saved = {'probs': probs, 'target': t}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# ──────────────────────── Dropout ─────────────────────────────────────

def dropout(input: Tensor, p: float = 0.5,
            training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return input
    mask = (np.random.rand(*input._data.shape) > p).astype(input._data.dtype)
    scale = 1.0 / (1.0 - p)
    result_data = input._data * mask * scale
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.DropoutBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'mask': mask * scale}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# ──────────────────────── Padding ─────────────────────────────────────

def pad(input: Tensor, pad_widths, mode: str = 'constant',
        value: float = 0.0) -> Tensor:
    """Pad tensor. pad_widths is PyTorch-style: (left, right, ...) from last dim."""
    ndim = input._data.ndim
    # Convert PyTorch padding format to numpy
    pairs = []
    for i in range(0, len(pad_widths), 2):
        pairs.append((pad_widths[i], pad_widths[i + 1]))
    # PyTorch pads from last dim; numpy from first
    np_pad = [(0, 0)] * (ndim - len(pairs)) + list(reversed(pairs))

    if mode == 'constant':
        result_data = np.pad(input._data, np_pad, mode='constant',
                             constant_values=value)
    else:
        result_data = np.pad(input._data, np_pad, mode=mode)

    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.PadBackward()
        grad_fn.inputs = [input]
        slices = []
        for (lo, hi), orig_size in zip(np_pad, input._data.shape):
            slices.append(slice(lo, lo + orig_size))
        grad_fn.saved = {'slices': slices}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# ──────────────────────── Attention ───────────────────────────────────

def scaled_dot_product_attention(
        q: Tensor, k: Tensor, v: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None) -> Tensor:
    """Scaled dot-product attention (manual)."""
    d = q._data.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scores = np.matmul(q._data, np.swapaxes(k._data, -2, -1)) * scale
    if attn_mask is not None:
        mask_data = attn_mask._data if isinstance(attn_mask, Tensor) else attn_mask
        scores = scores + mask_data
    # Softmax
    s_max = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - s_max)
    attn_weights = e / np.sum(e, axis=-1, keepdims=True)
    if dropout_p > 0.0 and q._requires_grad:
        mask = (np.random.rand(*attn_weights.shape) > dropout_p).astype(
            attn_weights.dtype) / (1.0 - dropout_p)
        attn_weights = attn_weights * mask
    result_data = np.matmul(attn_weights, v._data)

    rg = (q._requires_grad or k._requires_grad or
          v._requires_grad) and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.ScaledDotProductAttentionBackward()
        grad_fn.inputs = [q, k, v]
        grad_fn.saved = {
            'q': q._data, 'k': k._data, 'v': v._data,
            'attn_weights': attn_weights, 'scale': scale,
        }
    return Tensor._wrap(result_data, rg, grad_fn, q._device)


# ──────────────────────── Linear (functional) ─────────────────────────

def linear(input: Tensor, weight: Tensor,
           bias: Tensor | None = None) -> Tensor:
    result_data = np.matmul(input._data, weight._data.T)
    if bias is not None:
        result_data = result_data + bias._data
    rg = (input._requires_grad or weight._requires_grad) and _ag.is_grad_enabled()
    return Tensor._wrap(result_data, rg, None, input._device)
