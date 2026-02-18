# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
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
    """Return tensors in reverse topological order for backward."""
    visited: set[int] = set()
    order: list['Tensor'] = []

    def _visit(t: 'Tensor'):
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        if t._grad_fn is not None:
            for inp in t._grad_fn.inputs:
                if inp is not None:
                    _visit(inp)
        order.append(t)

    _visit(root)
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
        if t._grad_fn is not None:
            grads = t._grad_fn.backward(g)
            for inp, ig in zip(t._grad_fn.inputs, grads):
                if inp is None or ig is None:
                    continue
                if not inp.requires_grad:
                    continue
                # Reduce broadcast dims
                ig = _unbroadcast(ig, inp._data.shape)
                if inp._grad is not None:
                    inp._grad = inp._grad + ig
                else:
                    inp._grad = ig.copy()
                # Propagate for downstream nodes
                if hasattr(inp, '_grad_out'):
                    inp._grad_out = inp._grad_out + ig
                else:
                    inp._grad_out = ig.copy()

    # Cleanup
    for t in order:
        if hasattr(t, '_grad_out'):
            del t._grad_out


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum out dimensions that were broadcast."""
    if grad.shape == shape:
        return grad
    # Handle scalar case
    if shape == ():
        return np.sum(grad).reshape(())
    # Pad shape to match grad ndim
    ndim_diff = grad.ndim - len(shape)
    padded = (1,) * ndim_diff + shape
    reduce_dims = []
    for i, (gs, ss) in enumerate(zip(grad.shape, padded)):
        if ss == 1 and gs != 1:
            reduce_dims.append(i)
    if reduce_dims:
        grad = np.sum(grad, axis=tuple(reduce_dims), keepdims=True)
    if ndim_diff > 0:
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
            ga = np.matmul(g, _swapaxes(b, -2, -1))
            gb = np.matmul(_swapaxes(a, -2, -1), g)
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
        return (g * (self.saved['x'] > 0).astype(g.dtype),)


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

        grad_input = np.zeros_like(x)
        grad_weight = np.zeros_like(weight)
        grad_bias = np.sum(g, axis=(0, 2)) if has_bias else None

        c_in_per_group = C_in // groups
        c_out_per_group = C_out // groups

        # Depthwise / grouped conv backward
        for b in range(B):
            for grp in range(groups):
                co_start = grp * c_out_per_group
                ci_start = grp * c_in_per_group
                for co in range(c_out_per_group):
                    for ci in range(c_in_per_group):
                        for k in range(K):
                            for l in range(L_out):
                                in_pos = l + k - padding
                                if 0 <= in_pos < L_in:
                                    grad_weight[co_start + co, ci, k] += (
                                        g[b, co_start + co, l]
                                        * x[b, ci_start + ci, in_pos]
                                    )
                                    grad_input[b, ci_start + ci, in_pos] += (
                                        g[b, co_start + co, l]
                                        * weight[co_start + co, ci, k]
                                    )

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
        grad_input = np.matmul(g, weight)
        # Flatten batch dims for grad_weight
        g_2d = g.reshape(-1, g.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
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

        # dV = attn^T @ dO
        dv = np.matmul(np.swapaxes(attn_weights, -2, -1), g)
        # d_attn = dO @ V^T
        d_attn = np.matmul(g, np.swapaxes(v, -2, -1))
        # d_scores = attn * (d_attn - sum(d_attn * attn, dim=-1, keepdim=True))
        d_scores = attn_weights * (d_attn - np.sum(d_attn * attn_weights, axis=-1, keepdims=True))
        d_scores *= scale
        # dQ = d_scores @ K, dK = d_scores^T @ Q
        dq = np.matmul(d_scores, k)
        dk = np.matmul(np.swapaxes(d_scores, -2, -1), q)

        return (dq, dk, dv)


class PadBackward(GradFn):
    def __init__(self):
        super().__init__('PadBackward')

    def backward(self, g):
        slices = self.saved['slices']
        return (g[tuple(slices)],)
