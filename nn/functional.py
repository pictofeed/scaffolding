# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.functional — stateless neural network operations (F.*)."""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor, _USE_MPS, _USE_CYTHON, _USE_CUDA, _mops, _cops, _cuops, _is_cuda
from .. import autograd as _ag


# ──────────────────────── Activations ─────────────────────────────────

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if _USE_CUDA and _is_cuda(input._device):
        if input._gpu is not None:
            return Tensor._wrap_gpu(_cuops.dev_relu(input._gpu), device=input._device)
        input._ensure_cpu()
        result_data = _cuops.cuda_relu(input._data)
    elif _USE_MPS and input._ensure_cpu().dtype == np.float32:
        result_data = _mops.accelerate_relu(input._data)
    else:
        input._ensure_cpu()
        result_data = np.maximum(input._data, 0)
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.ReluBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'mask': (input._data > 0)}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def silu(input: Tensor) -> Tensor:
    if _USE_CUDA and _is_cuda(input._device):
        if input._gpu is not None:
            return Tensor._wrap_gpu(_cuops.dev_silu(input._gpu), device=input._device)
        input._ensure_cpu()
        result_data = _cuops.cuda_silu(input._data)
        sig_data = None
    elif _USE_MPS:
        input._ensure_cpu()
        result_data, sig_data = _mops.accelerate_silu_fwd(input._data)
    elif _USE_CYTHON:
        input._ensure_cpu()
        flat = np.ascontiguousarray(input._data).ravel()
        if flat.dtype == np.float32:
            result_data = _cops.silu_1d_f32(flat).reshape(input._data.shape)
            sig_data = None
        else:
            sig_data = 1.0 / (1.0 + np.exp(-input._data))
            result_data = input._data * sig_data
    else:
        input._ensure_cpu()
        sig_data = 1.0 / (1.0 + np.exp(-input._data))
        result_data = input._data * sig_data
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.SiluBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': input._data, 'sigmoid': sig_data}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def gelu(input: Tensor) -> Tensor:
    if _USE_CUDA and _is_cuda(input._device):
        if input._gpu is not None:
            return Tensor._wrap_gpu(_cuops.dev_gelu(input._gpu), device=input._device)
        input._ensure_cpu()
        result_data = _cuops.cuda_gelu(input._data)
    elif _USE_MPS:
        input._ensure_cpu()
        result_data = _mops.accelerate_gelu(input._data)
    else:
        input._ensure_cpu()
        x = input._data
        result_data = 0.5 * x * (1 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.GeluBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'x': input._data}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def softplus(input: Tensor, beta: float = 1.0,
             threshold: float = 20.0) -> Tensor:
    input._ensure_cpu()
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
    if _USE_CUDA and _is_cuda(input._device) and input._gpu is not None:
        return Tensor._wrap_gpu(_cuops.dev_softmax(input._gpu, dim), device=input._device)
    x = input._ensure_cpu() if input._data is None else input._data
    if _USE_CUDA and _is_cuda(input._device):
        s = _cuops.cuda_softmax(x, dim)
    elif _USE_MPS and x.dtype == np.float32:
        # Reshape to 2D for Accelerate path
        if x.ndim == 2 and dim in (-1, x.ndim - 1):
            s = _mops.accelerate_softmax(np.ascontiguousarray(x), -1)
        elif dim in (-1, x.ndim - 1) and x.ndim >= 2:
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            s = _mops.accelerate_softmax(np.ascontiguousarray(x_2d), -1).reshape(orig_shape)
        else:
            x_max = x.max(axis=dim, keepdims=True)
            e = np.exp(x - x_max)
            e_sum = e.sum(axis=dim, keepdims=True)
            np.divide(e, e_sum, out=e)
            s = e
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


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    if _USE_CUDA and _is_cuda(input._device) and input._gpu is not None:
        return Tensor._wrap_gpu(_cuops.dev_log_softmax(input._gpu, dim), device=input._device)
    x = input._ensure_cpu() if input._data is None else input._data
    if _USE_CUDA and _is_cuda(input._device):
        result_data = _cuops.cuda_log_softmax(x, dim)
    else:
        x_max = np.max(x, axis=dim, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(x - x_max), axis=dim, keepdims=True)) + x_max
        result_data = x - logsumexp
    return Tensor._wrap(result_data, input._requires_grad, None, input._device)


# ──────────────────────── Loss functions ──────────────────────────────

def cross_entropy(input: Tensor, target: Tensor,
                  reduction: str = 'mean') -> Tensor:
    """Cross-entropy loss with log-softmax."""
    # GPU-resident fast path
    if _USE_CUDA and _is_cuda(input._device) and input._gpu is not None and target._gpu is not None and reduction == 'mean':
        loss_val, probs_gpu = _cuops.dev_cross_entropy(input._gpu, target._gpu)
        return Tensor._wrap(np.float32(loss_val), False, None, input._device)

    x = input._ensure_cpu() if input._data is None else input._data
    t = (target._ensure_cpu() if target._data is None else target._data).astype(np.intp)

    # Reshape to 2D if needed
    if x.ndim == 1:
        x = x.reshape(1, -1)
        t = t.reshape(-1)

    N = x.shape[0]
    rg = input._requires_grad and _ag.is_grad_enabled()

    # Fast path: fully fused CUDA cross-entropy kernel
    if _USE_CUDA and _is_cuda(input._device) and x.dtype == np.float32 and reduction == 'mean':
        loss_val, probs = _cuops.cuda_cross_entropy(x, t.astype(np.int64))
        result_data = np.float32(loss_val)
        grad_fn = None
        if rg:
            grad_fn = _ag.CrossEntropyBackward()
            grad_fn.inputs = [input, None]
            grad_fn.saved = {'probs': probs, 'target': t}
        return Tensor._wrap(result_data, rg, grad_fn, input._device)

    # Fast path: fully fused Accelerate kernel (no grad needed)
    if _USE_MPS and x.dtype == np.float32 and reduction == 'mean' and not rg:
        loss_val = _mops.accelerate_cross_entropy(
            np.ascontiguousarray(x),
            np.ascontiguousarray(t).astype(np.int64))
        return Tensor._wrap(np.float32(loss_val), False, None, input._device)

    # Compute softmax (use Accelerate if available)
    if _USE_MPS and x.dtype == np.float32:
        probs = _mops.accelerate_softmax(np.ascontiguousarray(x), -1)
    else:
        x_max = x.max(axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        e_sum = e.sum(axis=-1, keepdims=True)
        np.divide(e, e_sum, out=e)
        probs = e

    # NLL - avoid full log computation, only need selected entries
    selected_probs = probs[np.arange(N), t]
    losses = -np.log(selected_probs + 1e-12)

    if reduction == 'mean':
        result_data = np.float32(losses.mean())
    elif reduction == 'sum':
        result_data = np.float32(losses.sum())
    else:
        result_data = losses

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
    if _USE_CUDA and _is_cuda(input._device):
        if input._gpu is not None:
            result_gpu, mask_gpu = _cuops.dev_dropout(input._gpu, p)
            return Tensor._wrap_gpu(result_gpu, device=input._device)
        result_data, mask = _cuops.cuda_dropout(input._ensure_cpu(), p)
        mask = mask.astype(input._data.dtype) / (1.0 - p)
    else:
        x = input._ensure_cpu()
        mask = (np.random.rand(*x.shape) > p).astype(x.dtype)
        scale = 1.0 / (1.0 - p)
        result_data = x * mask * scale
        mask = mask * scale
    rg = input._requires_grad and _ag.is_grad_enabled()
    grad_fn = None
    if rg:
        grad_fn = _ag.DropoutBackward()
        grad_fn.inputs = [input]
        grad_fn.saved = {'mask': mask}
    return Tensor._wrap(result_data, rg, grad_fn, input._device)


# ──────────────────────── Padding ─────────────────────────────────────

def pad(input: Tensor, pad_widths, mode: str = 'constant',
        value: float = 0.0) -> Tensor:
    """Pad tensor. pad_widths is PyTorch-style: (left, right, ...) from last dim."""
    input._ensure_cpu()
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
    q._ensure_cpu()
    k._ensure_cpu()
    v._ensure_cpu()
    d = q._data.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    # Fused scores computation
    k_T = np.swapaxes(k._data, -2, -1)
    if _USE_MPS and q._data.dtype == np.float32 and q._data.ndim >= 2:
        scores = _mops.accelerate_batched_matmul(q._data, k_T)
    else:
        scores = np.matmul(q._data, k_T)
    scores *= scale
    if attn_mask is not None:
        mask_data = attn_mask._data if isinstance(attn_mask, Tensor) else attn_mask
        scores += mask_data
    # In-place softmax
    if _USE_MPS and scores.dtype == np.float32 and scores.ndim >= 2:
        orig_shape = scores.shape
        scores_2d = scores.reshape(-1, scores.shape[-1])
        attn_weights = _mops.accelerate_softmax(
            np.ascontiguousarray(scores_2d), -1).reshape(orig_shape)
    else:
        s_max = scores.max(axis=-1, keepdims=True)
        scores -= s_max
        np.exp(scores, out=scores)
        s_sum = scores.sum(axis=-1, keepdims=True)
        np.divide(scores, s_sum, out=scores)
        attn_weights = scores
    if dropout_p > 0.0 and q._requires_grad:
        mask = (np.random.rand(*attn_weights.shape) > dropout_p).astype(
            attn_weights.dtype)
        inv_keep = 1.0 / (1.0 - dropout_p)
        attn_weights = attn_weights * mask * inv_keep
    if _USE_MPS and q._data.dtype == np.float32 and q._data.ndim >= 2:
        result_data = _mops.accelerate_batched_matmul(attn_weights, v._data)
    else:
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
    if _USE_CUDA and _is_cuda(input._device):
        if input._gpu is not None and weight._gpu is not None:
            gt_bias = bias._gpu if (bias is not None and bias._gpu is not None) else None
            return Tensor._wrap_gpu(
                _cuops.dev_linear_forward(input._gpu, weight._gpu, gt_bias),
                device=input._device)
        result_data = _cuops.cuda_linear_forward(
            input._ensure_cpu() if input._data is None else input._data,
            weight._ensure_cpu() if weight._data is None else weight._data,
            (bias._ensure_cpu() if bias._data is None else bias._data) if bias is not None else None
        )
    else:
        input._ensure_cpu()
        weight._ensure_cpu()
        result_data = np.matmul(input._data, weight._data.T)
        if bias is not None:
            bias._ensure_cpu()
            result_data = result_data + bias._data
    rg = (input._requires_grad or weight._requires_grad) and _ag.is_grad_enabled()
    return Tensor._wrap(result_data, rg, None, input._device)
