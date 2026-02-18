# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: nonecheck=False, initializedcheck=False
"""
Cython-accelerated tensor operations for Scaffolding.

All hot-path numerical kernels are implemented with ``nogil``
where possible, enabling true parallel execution when the GIL
is released by the caller.

Build with::

    python setup.py build_ext --inplace
"""
cimport cython
from libc.math cimport exp, log, sqrt, fabs, tanh, fmax, fmin
from libc.math cimport sin as c_sin, cos as c_cos, atan2 as c_atan2
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

np.import_array()

# ──────────────────────── Type aliases ────────────────────────────────

ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
ctypedef np.int64_t   INT64


# ──────────────────────── Element-wise ops (nogil) ────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] sigmoid_1d_f32(FLOAT32[::1] x):
    """Compute sigmoid element-wise on a contiguous float32 array."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(n * sizeof(FLOAT32))
    cdef Py_ssize_t i
    cdef FLOAT32 val
    if out == NULL:
        raise MemoryError("sigmoid_1d_f32: allocation failed")
    with nogil:
        for i in range(n):
            val = x[i]
            if val >= 0:
                out[i] = 1.0 / (1.0 + exp(-val))
            else:
                val = exp(val)
                out[i] = val / (1.0 + val)
    # Wrap into numpy (steals pointer via buffer protocol workaround)
    result = np.empty(n, dtype=np.float32)
    memcpy(&(<FLOAT32 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT32))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT64, ndim=1] sigmoid_1d_f64(FLOAT64[::1] x):
    """Compute sigmoid element-wise on a contiguous float64 array."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT64 *out = <FLOAT64 *>malloc(n * sizeof(FLOAT64))
    cdef Py_ssize_t i
    cdef FLOAT64 val
    if out == NULL:
        raise MemoryError("sigmoid_1d_f64: allocation failed")
    with nogil:
        for i in range(n):
            val = x[i]
            if val >= 0:
                out[i] = 1.0 / (1.0 + exp(-val))
            else:
                val = exp(val)
                out[i] = val / (1.0 + val)
    result = np.empty(n, dtype=np.float64)
    memcpy(&(<FLOAT64 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT64))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] silu_1d_f32(FLOAT32[::1] x):
    """SiLU (Swish): x * sigmoid(x), float32."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(n * sizeof(FLOAT32))
    cdef Py_ssize_t i
    cdef FLOAT32 s
    if out == NULL:
        raise MemoryError("silu_1d_f32: allocation failed")
    with nogil:
        for i in range(n):
            if x[i] >= 0:
                s = 1.0 / (1.0 + exp(-x[i]))
            else:
                s = exp(x[i]) / (1.0 + exp(x[i]))
            out[i] = x[i] * s
    result = np.empty(n, dtype=np.float32)
    memcpy(&(<FLOAT32 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT32))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] softplus_1d_f32(FLOAT32[::1] x,
                                                     FLOAT32 beta,
                                                     FLOAT32 threshold):
    """Softplus: (1/beta) * log(1 + exp(beta * x)), float32."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(n * sizeof(FLOAT32))
    cdef Py_ssize_t i
    cdef FLOAT32 bx
    if out == NULL:
        raise MemoryError("softplus_1d_f32: allocation failed")
    with nogil:
        for i in range(n):
            bx = beta * x[i]
            if bx > threshold:
                out[i] = x[i]
            else:
                out[i] = log(1.0 + exp(bx)) / beta
    result = np.empty(n, dtype=np.float32)
    memcpy(&(<FLOAT32 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT32))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] exp_1d_f32(FLOAT32[::1] x):
    """Element-wise exp, float32."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(n * sizeof(FLOAT32))
    cdef Py_ssize_t i
    if out == NULL:
        raise MemoryError("exp_1d_f32: allocation failed")
    with nogil:
        for i in range(n):
            out[i] = exp(x[i])
    result = np.empty(n, dtype=np.float32)
    memcpy(&(<FLOAT32 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT32))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] rsqrt_1d_f32(FLOAT32[::1] x):
    """Element-wise reciprocal square root, float32."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(n * sizeof(FLOAT32))
    cdef Py_ssize_t i
    if out == NULL:
        raise MemoryError("rsqrt_1d_f32: allocation failed")
    with nogil:
        for i in range(n):
            out[i] = 1.0 / sqrt(x[i]) if x[i] > 0 else 0.0
    result = np.empty(n, dtype=np.float32)
    memcpy(&(<FLOAT32 *>np.PyArray_DATA(result))[0], out, n * sizeof(FLOAT32))
    free(out)
    return result


# ──────────────────────── Reduction ops (nogil) ───────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT32 sum_f32(FLOAT32[::1] x) nogil:
    """Sum all elements, float32. Returns scalar."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 total = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        total = total + x[i]
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT32 mean_f32(FLOAT32[::1] x) nogil:
    """Mean of all elements, float32. Returns scalar."""
    cdef Py_ssize_t n = x.shape[0]
    cdef FLOAT32 total = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        total = total + x[i]
    return total / <FLOAT32>n


# ──────────────────────── Matrix multiply (nogil) ─────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] matmul_2d_f32(
        FLOAT32[:, ::1] a, FLOAT32[:, ::1] b):
    """Matrix multiply (M,K) @ (K,N) -> (M,N), float32.

    Uses naive triple-loop — sufficient for moderate sizes; for large
    matrices the NumPy / BLAS fallback in ``matmul_forward`` is faster.
    """
    cdef Py_ssize_t M = a.shape[0]
    cdef Py_ssize_t K = a.shape[1]
    cdef Py_ssize_t N = b.shape[1]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(M * N * sizeof(FLOAT32))
    cdef Py_ssize_t i, j, k
    cdef FLOAT32 acc
    if out == NULL:
        raise MemoryError("matmul_2d_f32: allocation failed")
    with nogil:
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    acc = acc + a[i, k] * b[k, j]
                out[i * N + j] = acc
    result = np.empty((M, N), dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, M * N * sizeof(FLOAT32))
    free(out)
    return result


# ──────────────────────── Cross entropy (nogil) ───────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT32 cross_entropy_f32(FLOAT32[:, ::1] logits,
                                  INT64[::1] targets) nogil:
    """Cross-entropy loss over (N, C) logits and (N,) targets.

    Computes log-softmax inline to avoid allocating a full (N, C) buffer.
    """
    cdef Py_ssize_t N = logits.shape[0]
    cdef Py_ssize_t C = logits.shape[1]
    cdef FLOAT32 total_loss = 0.0
    cdef Py_ssize_t i, j
    cdef FLOAT32 max_val, sum_exp, log_s
    cdef INT64 tgt

    for i in range(N):
        tgt = targets[i]
        # Numerically stable log-softmax
        max_val = logits[i, 0]
        for j in range(1, C):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        sum_exp = 0.0
        for j in range(C):
            sum_exp = sum_exp + exp(logits[i, j] - max_val)
        log_s = logits[i, tgt] - max_val - log(sum_exp)
        total_loss = total_loss - log_s
    return total_loss / <FLOAT32>N


# ──────────────────────── RMS norm (nogil) ────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] rms_norm_f32(
        FLOAT32[:, ::1] x, FLOAT32[::1] weight, FLOAT32 eps):
    """RMS normalization: x * rsqrt(mean(x^2) + eps) * weight.

    x: (N, D), weight: (D,) → result: (N, D).
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t D = x.shape[1]
    cdef FLOAT32 *out = <FLOAT32 *>malloc(N * D * sizeof(FLOAT32))
    cdef Py_ssize_t i, j
    cdef FLOAT32 sq_sum, rms
    if out == NULL:
        raise MemoryError("rms_norm_f32: allocation failed")
    with nogil:
        for i in range(N):
            sq_sum = 0.0
            for j in range(D):
                sq_sum = sq_sum + x[i, j] * x[i, j]
            rms = 1.0 / sqrt(sq_sum / <FLOAT32>D + eps)
            for j in range(D):
                out[i * D + j] = x[i, j] * rms * weight[j]
    result = np.empty((N, D), dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, N * D * sizeof(FLOAT32))
    free(out)
    return result


# ──────────────────────── AdamW step (nogil) ──────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void adamw_step_f32(
        FLOAT32[::1] param,
        FLOAT32[::1] grad,
        FLOAT32[::1] m,
        FLOAT32[::1] v,
        FLOAT32 lr,
        FLOAT32 beta1,
        FLOAT32 beta2,
        FLOAT32 eps,
        FLOAT32 weight_decay,
        FLOAT32 bc1,
        FLOAT32 bc2) nogil:
    """In-place AdamW update on contiguous float32 arrays.

    bc1 = 1 - beta1^t, bc2 = 1 - beta2^t  (bias-correction terms).
    """
    cdef Py_ssize_t n = param.shape[0]
    cdef Py_ssize_t i
    cdef FLOAT32 m_hat, v_hat
    for i in range(n):
        # Decoupled weight decay
        param[i] = param[i] * (1.0 - lr * weight_decay)
        # Moment updates
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i]
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i]
        # Bias-corrected moments
        m_hat = m[i] / bc1
        v_hat = v[i] / bc2
        # Parameter update
        param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + eps)


# ──────────────────────── Python-facing wrappers ──────────────────────

def sigmoid_forward(np.ndarray x):
    """Dispatch sigmoid to the correct typed kernel."""
    cdef tuple shape = (<object>x).shape
    cdef object out_flat
    flat = np.ascontiguousarray(x).ravel()
    if flat.dtype == np.float32:
        out_flat = sigmoid_1d_f32(flat)
    elif flat.dtype == np.float64:
        out_flat = sigmoid_1d_f64(flat)
    else:
        out_flat = sigmoid_1d_f32(flat.astype(np.float32))
    return out_flat.reshape(shape)


def exp_forward(np.ndarray x):
    """Dispatch exp to the correct typed kernel."""
    cdef tuple shape = (<object>x).shape
    cdef object out_flat
    flat = np.ascontiguousarray(x).ravel()
    if flat.dtype == np.float32:
        out_flat = exp_1d_f32(flat)
    else:
        return np.exp(x)  # fallback
    return out_flat.reshape(shape)


def matmul_forward(np.ndarray a, np.ndarray b):
    """Dispatch matmul — uses Cython for small 2-D; BLAS for the rest."""
    if a.ndim == 2 and b.ndim == 2 and a.dtype == np.float32:
        M = a.shape[0]
        K = a.shape[1]
        N = b.shape[1]
        # Only use our kernel for small matrices; BLAS is faster for large
        if M * N * K < 65536:
            return matmul_2d_f32(
                np.ascontiguousarray(a),
                np.ascontiguousarray(b))
    return np.matmul(a, b)
