# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Autograd engine — reverse-mode automatic differentiation.

Builds a DAG of :class:`GradFn` nodes during the forward pass and
traverses it in topological order during :meth:`Tensor.backward`.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .tensor import Tensor

# Try to import Accelerate BLAS for backward passes
try:
    from . import _mps_ops as _mops
    _USE_MPS = True
except ImportError:
    _mops = None
    _USE_MPS = False


# ──────────────────────── Global grad mode ────────────────────────────

_grad_enabled: bool = True
_inference_mode: bool = False


def is_grad_enabled() -> bool:
    return _grad_enabled and not _inference_mode


def set_grad_enabled(mode: bool) -> None:
    global _grad_enabled
    _grad_enabled = mode


class no_grad:
    """Context manager / decorator that disables gradient computation."""

    def __enter__(self):
        self._prev = _grad_enabled
        set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        set_grad_enabled(self._prev)

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with self:
                return fn(*a, **kw)
        return wrapper


class inference_mode:
    """Context manager / decorator that disables grad + inference mode."""

    def __enter__(self):
        global _inference_mode
        self._prev_grad = _grad_enabled
        self._prev_inf = _inference_mode
        set_grad_enabled(False)
        _inference_mode = True
        return self

    def __exit__(self, *args):
        global _inference_mode
        set_grad_enabled(self._prev_grad)
        _inference_mode = self._prev_inf

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with self:
                return fn(*a, **kw)
        return wrapper


# ──────────────────────── GradFn base class ───────────────────────────

class GradFn:
    """Base class for all autograd functions."""
    __slots__ = ('inputs', 'saved', 'name')

    def __init__(self, name: str = 'GradFn'):
        self.inputs: list[Tensor | None] = []
        self.saved: dict = {}
        self.name = name

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray | None, ...]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


# ──────────────────────── Topological backward ────────────────────────

def _topo_sort(root: 'Tensor') -> list['Tensor']:
    """Return tensors in reverse topological order for backward.
    
    Uses iterative DFS to avoid Python recursion overhead and stack limits.
    """
    visited: set[int] = set()
    order: list['Tensor'] = []
    # Iterative DFS using explicit stack
    stack: list[tuple['Tensor', bool]] = [(root, False)]
    while stack:
        t, processed = stack[-1]
        tid = id(t)
        if processed:
            stack.pop()
            if tid not in visited:
                visited.add(tid)
                order.append(t)
            continue
        if tid in visited:
            stack.pop()
            continue
        stack[-1] = (t, True)
        if t._grad_fn is not None:
            for inp in t._grad_fn.inputs:
                if inp is not None and id(inp) not in visited:
                    stack.append((inp, False))
    order.reverse()
    return order


def backward(root: 'Tensor', grad: np.ndarray | None = None) -> None:
    """Run backward pass from *root* tensor."""
    if grad is None:
        if root._data.size == 1:
            grad = np.ones_like(root._data)
        else:
            raise RuntimeError(
                "grad must be specified for non-scalar tensors")

    root._grad_out = grad
    order = _topo_sort(root)

    for t in order:
        g = getattr(t, '_grad_out', None)
        if g is None:
            continue
        gfn = t._grad_fn
        if gfn is not None:
            grads = gfn.backward(g)
            inputs = gfn.inputs
            for i in range(len(inputs)):
                inp = inputs[i]
                if inp is None:
                    continue
                if not inp._requires_grad:
                    continue
                ig = grads[i]
                if ig is None:
                    continue
                # Reduce broadcast dims
                inp_shape = inp._data.shape
                if ig.shape != inp_shape:
                    ig = _unbroadcast(ig, inp_shape)
                if inp._grad is not None:
                    np.add(inp._grad, ig, out=inp._grad)
                else:
                    inp._grad = ig if ig.flags.owndata else ig.copy()
                # Propagate for downstream nodes
                prev = getattr(inp, '_grad_out', None)
                if prev is not None:
                    np.add(prev, ig, out=prev)
                else:
                    inp._grad_out = ig if ig.flags.owndata else ig.copy()
            # Free saved tensors early to reduce memory pressure
            gfn.saved = None

    # Cleanup
    for t in order:
        try:
            del t._grad_out
        except AttributeError:
            pass


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum out dimensions that were broadcast."""
    # Fast path: shapes already match
    if grad.shape == shape:
        return grad
    # Handle scalar case
    if shape == ():
        return grad.sum().reshape(())
    # Pad shape to match grad ndim
    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        # Sum over the extra leading dimensions
        grad = grad.sum(axis=tuple(range(ndim_diff)), keepdims=False)
    # Now grad.ndim == len(shape); sum where shape has 1
    reduce_dims = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if reduce_dims:
        grad = grad.sum(axis=reduce_dims, keepdims=True)
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad


# ──────────────────────── Concrete GradFn nodes ───────────────────────

class AddBackward(GradFn):
    def __init__(self):
        super().__init__('AddBackward')

    def backward(self, g):
        return (g, g)


class SubBackward(GradFn):
    def __init__(self):
        super().__init__('SubBackward')

    def backward(self, g):
        return (g, -g)


class MulBackward(GradFn):
    def __init__(self):
        super().__init__('MulBackward')

    def backward(self, g):
        a_data, b_data = self.saved['a'], self.saved['b']
        return (g * b_data, g * a_data)


class DivBackward(GradFn):
    def __init__(self):
        super().__init__('DivBackward')

    def backward(self, g):
        a_data, b_data = self.saved['a'], self.saved['b']
        return (g / b_data, -g * a_data / (b_data ** 2))


class MatMulBackward(GradFn):
    def __init__(self):
        super().__init__('MatMulBackward')

    def backward(self, g):
        a, b = self.saved['a'], self.saved['b']
        if a.ndim == 1 and b.ndim == 1:
            return (g * b, g * a)
        if a.ndim >= 2 and b.ndim >= 2:
            b_T = _swapaxes(b, -2, -1)
            a_T = _swapaxes(a, -2, -1)
            if _USE_MPS and g.dtype == np.float32:
                ga = _mops.accelerate_batched_matmul(g, b_T) if g.ndim > 2 else (
                    _mops.blas_sgemm(np.ascontiguousarray(g), np.ascontiguousarray(b_T))
                    if g.ndim == 2 else np.matmul(g, b_T))
                gb = _mops.accelerate_batched_matmul(a_T, g) if a_T.ndim > 2 else (
                    _mops.blas_sgemm(np.ascontiguousarray(a_T), np.ascontiguousarray(g))
                    if a_T.ndim == 2 else np.matmul(a_T, g))
            else:
                ga = np.matmul(g, b_T)
                gb = np.matmul(a_T, g)
            return (ga, gb)
        if a.ndim == 1:
            ga = np.matmul(g, _swapaxes(b, -2, -1))
            gb = np.outer(a, g) if g.ndim == 1 else np.expand_dims(a, -1) @ np.expand_dims(g, -2)
            return (ga, gb)
        # b.ndim == 1
        ga = np.outer(g, b) if g.ndim == 1 else np.expand_dims(g, -1) @ np.expand_dims(b, 0)[None]
        gb = np.matmul(_swapaxes(a, -2, -1), g)
        return (ga, gb)


def _swapaxes(a, ax1, ax2):
    return np.swapaxes(a, ax1, ax2)


class PowBackward(GradFn):
    def __init__(self):
        super().__init__('PowBackward')

    def backward(self, g):
        base, exp_val = self.saved['base'], self.saved['exp']
        result = self.saved['result']
        gb = g * exp_val * np.power(base, exp_val - 1)
        return (gb, None)


class ExpBackward(GradFn):
    def __init__(self):
        super().__init__('ExpBackward')

    def backward(self, g):
        return (g * self.saved['result'],)


class LogBackward(GradFn):
    def __init__(self):
        super().__init__('LogBackward')

    def backward(self, g):
        return (g / self.saved['x'],)


class SqrtBackward(GradFn):
    def __init__(self):
        super().__init__('SqrtBackward')

    def backward(self, g):
        return (g / (2.0 * self.saved['result']),)


class RsqrtBackward(GradFn):
    def __init__(self):
        super().__init__('RsqrtBackward')

    def backward(self, g):
        x = self.saved['x']
        return (-0.5 * g * np.power(x, -1.5),)


class NegBackward(GradFn):
    def __init__(self):
        super().__init__('NegBackward')

    def backward(self, g):
        return (-g,)


class SigmoidBackward(GradFn):
    def __init__(self):
        super().__init__('SigmoidBackward')

    def backward(self, g):
        s = self.saved['result']
        return (g * s * (1.0 - s),)


class SiluBackward(GradFn):
    def __init__(self):
        super().__init__('SiluBackward')

    def backward(self, g):
        x = self.saved['x']
        s = self.saved.get('sigmoid', None)
        if s is None:
            if _USE_MPS and x.dtype == np.float32:
                s = _mops.accelerate_sigmoid(x)
            else:
                s = 1.0 / (1.0 + np.exp(-x))
        return (g * (s + x * s * (1.0 - s)),)


class SoftplusBackward(GradFn):
    def __init__(self):
        super().__init__('SoftplusBackward')

    def backward(self, g):
        x = self.saved['x']
        beta = self.saved.get('beta', 1.0)
        s = 1.0 / (1.0 + np.exp(-beta * x))
        return (g * s,)


class ReluBackward(GradFn):
    def __init__(self):
        super().__init__('ReluBackward')

    def backward(self, g):
        mask = self.saved.get('mask', None)
        if mask is None:
            mask = self.saved['x'] > 0
        return (g * mask,)


class GeluBackward(GradFn):
    def __init__(self):
        super().__init__('GeluBackward')

    def backward(self, g):
        x = self.saved['x']
        s2pi = 0.7978845608  # sqrt(2/pi)
        inner = s2pi * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        dtanh = 1.0 - t * t
        dinner = s2pi * (1.0 + 3.0 * 0.044715 * x * x)
        grad_x = 0.5 * (1.0 + t) + 0.5 * x * dtanh * dinner
        return (g * grad_x,)


class SinBackward(GradFn):
    def __init__(self):
        super().__init__('SinBackward')

    def backward(self, g):
        return (g * np.cos(self.saved['x']),)


class CosBackward(GradFn):
    def __init__(self):
        super().__init__('CosBackward')

    def backward(self, g):
        return (g * (-np.sin(self.saved['x'])),)


class Atan2Backward(GradFn):
    def __init__(self):
        super().__init__('Atan2Backward')

    def backward(self, g):
        y, x = self.saved['y'], self.saved['x']
        denom = x ** 2 + y ** 2
        return (g * x / denom, -g * y / denom)


class Expm1Backward(GradFn):
    def __init__(self):
        super().__init__('Expm1Backward')

    def backward(self, g):
        return (g * np.exp(self.saved['x']),)


class ClampBackward(GradFn):
    def __init__(self):
        super().__init__('ClampBackward')

    def backward(self, g):
        x = self.saved['x']
        lo = self.saved.get('min', None)
        hi = self.saved.get('max', None)
        mask = np.ones_like(x, dtype=g.dtype)
        if lo is not None:
            mask = mask * (x >= lo)
        if hi is not None:
            mask = mask * (x <= hi)
        return (g * mask,)


class SumBackward(GradFn):
    def __init__(self):
        super().__init__('SumBackward')

    def backward(self, g):
        shape = self.saved['input_shape']
        axis = self.saved.get('axis', None)
        keepdims = self.saved.get('keepdims', False)
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis=axis)
        return (np.broadcast_to(g, shape).copy(),)


class MeanBackward(GradFn):
    def __init__(self):
        super().__init__('MeanBackward')

    def backward(self, g):
        shape = self.saved['input_shape']
        axis = self.saved.get('axis', None)
        keepdims = self.saved.get('keepdims', False)
        if axis is not None:
            if isinstance(axis, int):
                n = shape[axis]
            else:
                n = 1
                for a in axis:
                    n *= shape[a]
            if not keepdims:
                g = np.expand_dims(g, axis=axis)
        else:
            n = 1
            for s in shape:
                n *= s
        return (np.broadcast_to(g / n, shape).copy(),)


class CumsumBackward(GradFn):
    def __init__(self):
        super().__init__('CumsumBackward')

    def backward(self, g):
        axis = self.saved['axis']
        return (np.flip(np.cumsum(np.flip(g, axis=axis), axis=axis), axis=axis),)


class CatBackward(GradFn):
    def __init__(self):
        super().__init__('CatBackward')

    def backward(self, g):
        sizes = self.saved['sizes']
        dim = self.saved['dim']
        grads = []
        idx = 0
        for s in sizes:
            slices = [slice(None)] * g.ndim
            slices[dim] = slice(idx, idx + s)
            grads.append(g[tuple(slices)])
            idx += s
        return tuple(grads)


class StackBackward(GradFn):
    def __init__(self):
        super().__init__('StackBackward')

    def backward(self, g):
        dim = self.saved['dim']
        n = self.saved['n']
        grads = []
        for i in range(n):
            slices = [slice(None)] * g.ndim
            slices[dim] = i
            grads.append(g[tuple(slices)])
        return tuple(grads)


class ReshapeBackward(GradFn):
    def __init__(self):
        super().__init__('ReshapeBackward')

    def backward(self, g):
        return (g.reshape(self.saved['input_shape']),)


class TransposeBackward(GradFn):
    def __init__(self):
        super().__init__('TransposeBackward')

    def backward(self, g):
        d0, d1 = self.saved['dim0'], self.saved['dim1']
        return (np.swapaxes(g, d0, d1),)


class PermuteBackward(GradFn):
    def __init__(self):
        super().__init__('PermuteBackward')

    def backward(self, g):
        dims = self.saved['dims']
        inv = [0] * len(dims)
        for i, d in enumerate(dims):
            inv[d] = i
        return (np.transpose(g, inv),)


class UnsqueezeBackward(GradFn):
    def __init__(self):
        super().__init__('UnsqueezeBackward')

    def backward(self, g):
        return (np.squeeze(g, axis=self.saved['dim']),)


class SqueezeBackward(GradFn):
    def __init__(self):
        super().__init__('SqueezeBackward')

    def backward(self, g):
        return (g.reshape(self.saved['input_shape']),)


class ChunkBackward(GradFn):
    """Not directly used (chunk returns views)."""
    pass


class WhereBackward(GradFn):
    def __init__(self):
        super().__init__('WhereBackward')

    def backward(self, g):
        cond = self.saved['condition']
        return (None, np.where(cond, g, 0), np.where(~cond, g, 0))


class SliceBackward(GradFn):
    def __init__(self):
        super().__init__('SliceBackward')

    def backward(self, g):
        shape = self.saved['input_shape']
        key = self.saved['key']
        result = np.zeros(shape, dtype=g.dtype)
        result[key] = g
        return (result,)


class SoftmaxBackward(GradFn):
    def __init__(self):
        super().__init__('SoftmaxBackward')

    def backward(self, g):
        s = self.saved['result']
        dim = self.saved['dim']
        ds = s * (g - np.sum(g * s, axis=dim, keepdims=True))
        return (ds,)


class CrossEntropyBackward(GradFn):
    def __init__(self):
        super().__init__('CrossEntropyBackward')

    def backward(self, g):
        probs = self.saved['probs']
        target = self.saved['target']
        n = probs.shape[0]
        grad_input = probs.copy()
        grad_input[np.arange(n), target] -= 1.0
        grad_input /= n
        if np.isscalar(g) or g.size == 1:
            grad_input *= float(g)
        else:
            grad_input *= g.reshape(-1, 1)
        return (grad_input, None)


class DropoutBackward(GradFn):
    def __init__(self):
        super().__init__('DropoutBackward')

    def backward(self, g):
        return (g * self.saved['mask'],)


class EmbeddingBackward(GradFn):
    def __init__(self):
        super().__init__('EmbeddingBackward')

    def backward(self, g):
        indices = self.saved['indices']
        num_embeddings = self.saved['num_embeddings']
        grad_weight = np.zeros((num_embeddings, g.shape[-1]), dtype=g.dtype)
        flat_indices = indices.flatten()
        flat_g = g.reshape(-1, g.shape[-1])
        np.add.at(grad_weight, flat_indices, flat_g)
        return (grad_weight,)


class Conv1dBackward(GradFn):
    def __init__(self):
        super().__init__('Conv1dBackward')

    def backward(self, g):
        x = self.saved['input']
        weight = self.saved['weight']
        padding = self.saved['padding']
        groups = self.saved['groups']
        has_bias = self.saved['has_bias']

        B, C_out, L_out = g.shape
        B, C_in, L_in = x.shape
        C_out_g, C_in_g, K = weight.shape

        # Pad input for im2col
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        else:
            x_padded = x
        L_padded = x_padded.shape[2]

        grad_bias = np.sum(g, axis=(0, 2)) if has_bias else None

        if groups == 1:
            # Vectorized grad_weight via im2col
            s = x_padded.strides
            col = np.lib.stride_tricks.as_strided(
                x_padded,
                shape=(B, C_in, K, L_out),
                strides=(s[0], s[1], s[2], s[2]),
            ).reshape(B, C_in * K, L_out)
            # grad_weight: einsum('bol,bil->oi')
            grad_weight = np.einsum('bol,bil->oi', g, col).reshape(weight.shape)
            
            # Vectorized grad_input via transposed convolution
            # Pad g and correlate with flipped weight
            w_2d = weight.reshape(C_out, -1)  # (C_out, C_in*K)
            # col_grad: (B, C_in*K, L_out)
            col_grad = np.einsum('oi,bol->bil', w_2d, g)
            col_grad = col_grad.reshape(B, C_in, K, L_out)
            
            # Scatter col_grad back to grad_input
            grad_padded = np.zeros_like(x_padded)
            for k in range(K):
                grad_padded[:, :, k:k + L_out] += col_grad[:, :, k, :]
            
            if padding > 0:
                grad_input = grad_padded[:, :, padding:-padding]
            else:
                grad_input = grad_padded
        elif groups == C_in and groups == C_out:
            # Depthwise backward
            s = x_padded.strides
            unfolded = np.lib.stride_tricks.as_strided(
                x_padded,
                shape=(B, C_in, K, L_out),
                strides=(s[0], s[1], s[2], s[2]),
            )
            # grad_weight: (C_out, 1, K)
            grad_weight = np.sum(g[:, :, np.newaxis, :] * unfolded, axis=(0, 3)).reshape(weight.shape)
            
            # grad_input
            grad_padded = np.zeros_like(x_padded)
            w = weight.reshape(C_out, K)
            for k in range(K):
                grad_padded[:, :, k:k + L_out] += g * w[np.newaxis, :, k:k+1]
            
            if padding > 0:
                grad_input = grad_padded[:, :, padding:-padding]
            else:
                grad_input = grad_padded
        else:
            # General grouped backward
            c_in_per_group = C_in // groups
            c_out_per_group = C_out // groups
            grad_input = np.zeros_like(x)
            grad_weight = np.zeros_like(weight)
            
            for grp in range(groups):
                co_start = grp * c_out_per_group
                ci_start = grp * c_in_per_group
                g_grp = g[:, co_start:co_start + c_out_per_group, :]
                x_grp = x_padded[:, ci_start:ci_start + c_in_per_group, :]
                w_grp = weight[co_start:co_start + c_out_per_group]
                
                s = x_grp.strides
                col_grp = np.lib.stride_tricks.as_strided(
                    x_grp,
                    shape=(B, c_in_per_group, K, L_out),
                    strides=(s[0], s[1], s[2], s[2]),
                ).reshape(B, c_in_per_group * K, L_out)
                
                w_2d = w_grp.reshape(c_out_per_group, -1)
                grad_weight[co_start:co_start + c_out_per_group] = np.einsum(
                    'bol,bil->oi', g_grp, col_grp).reshape(c_out_per_group, c_in_per_group, K)
                
                col_grad = np.einsum('oi,bol->bil', w_2d, g_grp).reshape(B, c_in_per_group, K, L_out)
                grad_padded = np.zeros((B, c_in_per_group, L_padded), dtype=x.dtype)
                for k in range(K):
                    grad_padded[:, :, k:k + L_out] += col_grad[:, :, k, :]
                if padding > 0:
                    grad_input[:, ci_start:ci_start + c_in_per_group, :] = grad_padded[:, :, padding:-padding]
                else:
                    grad_input[:, ci_start:ci_start + c_in_per_group, :] = grad_padded

        results = [grad_input, grad_weight]
        if has_bias:
            results.append(grad_bias)
        return tuple(results)


class LinearBackward(GradFn):
    def __init__(self):
        super().__init__('LinearBackward')

    def backward(self, g):
        x = self.saved['input']
        weight = self.saved['weight']
        has_bias = self.saved['has_bias']
        # g: (..., out_features), weight: (out_features, in_features)
        # Flatten batch dims for BLAS
        g_2d = g.reshape(-1, g.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        if _USE_MPS and g.dtype == np.float32 and weight.dtype == np.float32:
            # grad_input = g_2d @ weight  (no transpose needed)
            grad_input_2d = _mops.blas_sgemm(
                np.ascontiguousarray(g_2d),
                np.ascontiguousarray(weight))
            grad_input = grad_input_2d.reshape(g.shape[:-1] + (weight.shape[1],))
            # grad_weight = g_2d.T @ x_2d
            grad_weight = _mops.blas_sgemm_tn(
                np.ascontiguousarray(g_2d),
                np.ascontiguousarray(x_2d))
        else:
            grad_input = np.matmul(g, weight)
            grad_weight = np.matmul(g_2d.T, x_2d)
        results = [grad_input, grad_weight]
        if has_bias:
            grad_bias = np.sum(g_2d, axis=0)
            results.append(grad_bias)
        return tuple(results)


class ScaledDotProductAttentionBackward(GradFn):
    """Backward for manual SDPA fallback."""
    def __init__(self):
        super().__init__('SDPABackward')

    def backward(self, g):
        # Approximate backward — enough for training
        q, k, v, attn_weights = (
            self.saved['q'], self.saved['k'],
            self.saved['v'], self.saved['attn_weights'])
        scale = self.saved['scale']

        if _USE_MPS and g.dtype == np.float32 and g.ndim >= 2:
            # dV = attn^T @ dO
            dv = _mops.accelerate_batched_matmul(
                np.swapaxes(attn_weights, -2, -1), g)
            # d_attn = dO @ V^T
            d_attn = _mops.accelerate_batched_matmul(
                g, np.swapaxes(v, -2, -1))
        else:
            dv = np.matmul(np.swapaxes(attn_weights, -2, -1), g)
            d_attn = np.matmul(g, np.swapaxes(v, -2, -1))

        # d_scores = attn * (d_attn - sum(d_attn * attn, dim=-1, keepdim=True))
        d_scores = attn_weights * (d_attn - np.sum(d_attn * attn_weights, axis=-1, keepdims=True))
        d_scores *= scale

        if _USE_MPS and g.dtype == np.float32 and g.ndim >= 2:
            dq = _mops.accelerate_batched_matmul(d_scores, k)
            dk = _mops.accelerate_batched_matmul(
                np.swapaxes(d_scores, -2, -1), q)
        else:
            dq = np.matmul(d_scores, k)
            dk = np.matmul(np.swapaxes(d_scores, -2, -1), q)

        return (dq, dk, dv)


class PadBackward(GradFn):
    def __init__(self):
        super().__init__('PadBackward')

    def backward(self, g):
        slices = self.saved['slices']
        return (g[tuple(slices)],)
