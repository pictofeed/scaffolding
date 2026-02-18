# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: nonecheck=False, initializedcheck=False
"""
Cython MPS/Accelerate tensor operations for Scaffolding.

Links against Apple's **Accelerate.framework** to use:
  - ``cblas_sgemm``      — BLAS matrix multiply (float32)
  - ``cblas_dgemm``      — BLAS matrix multiply (float64)
  - ``vvexpf / vvexp``   — vectorised exp
  - ``vvlogf / vvlog``   — vectorised log
  - ``vvsqrtf / vvsqrt`` — vectorised sqrt
  - ``vvtanhf``          — vectorised tanh
  - ``vDSP_vsadd``       — vector scalar add
  - ``vDSP_vsmul``       — vector scalar multiply
  - ``vDSP_vdiv``        — vector divide
  - ``vDSP_vmul``        — vector element-wise multiply
  - ``vDSP_sve``         — vector sum
  - ``vDSP_meanv``       — vector mean
  - ``vDSP_vclip``       — vector clamp

All hot-path routines are implemented with ``nogil`` where possible,
enabling true parallel execution on Apple Silicon's multi-core fabric.

Build with::

    python setup.py build_ext --inplace
"""
cimport cython
from libc.math cimport exp as c_exp, log as c_log, sqrt as c_sqrt
from libc.math cimport tanh as c_tanh, fmax, fmin
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

np.import_array()

# ──────────────────────── Type aliases ────────────────────────────────

ctypedef np.float32_t  FLOAT32
ctypedef np.float64_t  FLOAT64
ctypedef np.int64_t    INT64
ctypedef np.int32_t    INT32

# ──────────────────────────────────────────────────────────────────────
#  Accelerate.framework C declarations
# ──────────────────────────────────────────────────────────────────────
# These are the CBLAS / vDSP / vecLib symbols linked via
#   -framework Accelerate
# at compile time.  Cython ``cdef extern`` lets us call them ``nogil``.
# ──────────────────────────────────────────────────────────────────────

cdef extern from "Accelerate/Accelerate.h" nogil:
    # ── CBLAS enums ──
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor = 101
        CblasColMajor = 102
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans   = 111
        CblasTrans     = 112
        CblasConjTrans = 113

    # ── BLAS gemm ──
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc) nogil

    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB,
                     int M, int N, int K,
                     double alpha,
                     const double *A, int lda,
                     const double *B, int ldb,
                     double beta,
                     double *C, int ldc) nogil

    # ── BLAS sgemv / dot ──
    float cblas_sdot(int N, const float *X, int incX,
                     const float *Y, int incY) nogil
    double cblas_ddot(int N, const double *X, int incX,
                      const double *Y, int incY) nogil
    float cblas_snrm2(int N, const float *X, int incX) nogil

    # ── vecLib vector transcendentals ──
    void vvexpf(float *result, const float *x, const int *n) nogil
    void vvexp(double *result, const double *x, const int *n) nogil
    void vvlogf(float *result, const float *x, const int *n) nogil
    void vvlog(double *result, const double *x, const int *n) nogil
    void vvsqrtf(float *result, const float *x, const int *n) nogil
    void vvsqrt(double *result, const double *x, const int *n) nogil
    void vvtanhf(float *result, const float *x, const int *n) nogil
    void vvtanh(double *result, const double *x, const int *n) nogil
    void vvrecf(float *result, const float *x, const int *n) nogil

    # ── vDSP vector arithmetic (float) ──
    void vDSP_vsadd(const float *A, int IA,
                    const float *B,
                    float *C, int IC,
                    unsigned long N) nogil

    void vDSP_vsmul(const float *A, int IA,
                    const float *B,
                    float *C, int IC,
                    unsigned long N) nogil

    void vDSP_vmul(const float *A, int IA,
                   const float *B, int IB,
                   float *C, int IC,
                   unsigned long N) nogil

    void vDSP_vdiv(const float *B, int IB,     # note: B is divisor
                   const float *A, int IA,
                   float *C, int IC,
                   unsigned long N) nogil

    void vDSP_vadd(const float *A, int IA,
                   const float *B, int IB,
                   float *C, int IC,
                   unsigned long N) nogil

    void vDSP_vsub(const float *B, int IB,
                   const float *A, int IA,
                   float *C, int IC,
                   unsigned long N) nogil

    void vDSP_vneg(const float *A, int IA,
                   float *C, int IC,
                   unsigned long N) nogil

    void vDSP_sve(const float *A, int IA,
                  float *C,
                  unsigned long N) nogil

    void vDSP_meanv(const float *A, int IA,
                    float *C,
                    unsigned long N) nogil

    void vDSP_maxv(const float *A, int IA,
                   float *C,
                   unsigned long N) nogil

    void vDSP_vclip(const float *A, int IA,
                    const float *Lo, const float *Hi,
                    float *C, int IC,
                    unsigned long N) nogil

    void vDSP_vsq(const float *A, int IA,
                  float *C, int IC,
                  unsigned long N) nogil

    # ── vDSP vector arithmetic (double) ──
    void vDSP_vsaddD(const double *A, int IA,
                     const double *B,
                     double *C, int IC,
                     unsigned long N) nogil

    void vDSP_vsmulD(const double *A, int IA,
                     const double *B,
                     double *C, int IC,
                     unsigned long N) nogil

    void vDSP_vmulD(const double *A, int IA,
                    const double *B, int IB,
                    double *C, int IC,
                    unsigned long N) nogil

    void vDSP_sveD(const double *A, int IA,
                   double *C,
                   unsigned long N) nogil

    void vDSP_meanvD(const double *A, int IA,
                     double *C,
                     unsigned long N) nogil


# ──────────────────────────────────────────────────────────────────────
#  BLAS Matrix Multiply  (sgemm / dgemm)
# ──────────────────────────────────────────────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] blas_sgemm(
        FLOAT32[:, ::1] a, FLOAT32[:, ::1] b):
    """C = A @ B using Apple Accelerate cblas_sgemm (float32).

    A: (M, K)  B: (K, N)  →  C: (M, N)
    """
    cdef int M = a.shape[0]
    cdef int K = a.shape[1]
    cdef int N = b.shape[1]
    cdef float alpha = 1.0
    cdef float beta  = 0.0
    cdef float *out = <float *>malloc(M * N * sizeof(float))
    if out == NULL:
        raise MemoryError("blas_sgemm: allocation failed")
    with nogil:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    &a[0, 0], K,
                    &b[0, 0], N,
                    beta, out, N)
    result = np.empty((M, N), dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, M * N * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT64, ndim=2] blas_dgemm(
        FLOAT64[:, ::1] a, FLOAT64[:, ::1] b):
    """C = A @ B using Apple Accelerate cblas_dgemm (float64).

    A: (M, K)  B: (K, N)  →  C: (M, N)
    """
    cdef int M = a.shape[0]
    cdef int K = a.shape[1]
    cdef int N = b.shape[1]
    cdef double alpha = 1.0
    cdef double beta  = 0.0
    cdef double *out = <double *>malloc(M * N * sizeof(double))
    if out == NULL:
        raise MemoryError("blas_dgemm: allocation failed")
    with nogil:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    &a[0, 0], K,
                    &b[0, 0], N,
                    beta, out, N)
    result = np.empty((M, N), dtype=np.float64)
    memcpy(np.PyArray_DATA(result), out, M * N * sizeof(double))
    free(out)
    return result


# ──────────────────────────────────────────────────────────────────────
#  Element-wise transcendentals via vecLib
# ──────────────────────────────────────────────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_expf(FLOAT32[::1] x):
    """Element-wise exp via vecLib vvexpf (float32)."""
    cdef int n = x.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("veclib_expf: allocation failed")
    with nogil:
        vvexpf(out, &x[0], &n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT64, ndim=1] veclib_exp(FLOAT64[::1] x):
    """Element-wise exp via vecLib vvexp (float64)."""
    cdef int n = x.shape[0]
    cdef double *out = <double *>malloc(n * sizeof(double))
    if out == NULL:
        raise MemoryError("veclib_exp: allocation failed")
    with nogil:
        vvexp(out, &x[0], &n)
    result = np.empty(n, dtype=np.float64)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(double))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_logf(FLOAT32[::1] x):
    """Element-wise log via vecLib vvlogf (float32)."""
    cdef int n = x.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("veclib_logf: allocation failed")
    with nogil:
        vvlogf(out, &x[0], &n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_sqrtf(FLOAT32[::1] x):
    """Element-wise sqrt via vecLib vvsqrtf (float32)."""
    cdef int n = x.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("veclib_sqrtf: allocation failed")
    with nogil:
        vvsqrtf(out, &x[0], &n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_tanhf(FLOAT32[::1] x):
    """Element-wise tanh via vecLib vvtanhf (float32)."""
    cdef int n = x.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("veclib_tanhf: allocation failed")
    with nogil:
        vvtanhf(out, &x[0], &n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


# ──────────────────────────────────────────────────────────────────────
#  vDSP vector arithmetic
# ──────────────────────────────────────────────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vmul_f32(
        FLOAT32[::1] a, FLOAT32[::1] b):
    """Element-wise multiply via vDSP_vmul (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("vdsp_vmul_f32: allocation failed")
    with nogil:
        vDSP_vmul(&a[0], 1, &b[0], 1, out, 1, n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vadd_f32(
        FLOAT32[::1] a, FLOAT32[::1] b):
    """Element-wise add via vDSP_vadd (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("vdsp_vadd_f32: allocation failed")
    with nogil:
        vDSP_vadd(&a[0], 1, &b[0], 1, out, 1, n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsmul_f32(
        FLOAT32[::1] a, float scalar):
    """Scalar multiply via vDSP_vsmul (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("vdsp_vsmul_f32: allocation failed")
    with nogil:
        vDSP_vsmul(&a[0], 1, &scalar, out, 1, n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsadd_f32(
        FLOAT32[::1] a, float scalar):
    """Scalar add via vDSP_vsadd (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("vdsp_vsadd_f32: allocation failed")
    with nogil:
        vDSP_vsadd(&a[0], 1, &scalar, out, 1, n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float vdsp_sum_f32(FLOAT32[::1] x) nogil:
    """Sum all elements via vDSP_sve (float32)."""
    cdef float result
    cdef unsigned long n = x.shape[0]
    vDSP_sve(&x[0], 1, &result, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float vdsp_mean_f32(FLOAT32[::1] x) nogil:
    """Mean of all elements via vDSP_meanv (float32)."""
    cdef float result
    cdef unsigned long n = x.shape[0]
    vDSP_meanv(&x[0], 1, &result, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float vdsp_max_f32(FLOAT32[::1] x) nogil:
    """Max element via vDSP_maxv (float32)."""
    cdef float result
    cdef unsigned long n = x.shape[0]
    vDSP_maxv(&x[0], 1, &result, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsq_f32(FLOAT32[::1] x):
    """Element-wise square via vDSP_vsq (float32)."""
    cdef unsigned long n = x.shape[0]
    cdef float *out = <float *>malloc(n * sizeof(float))
    if out == NULL:
        raise MemoryError("vdsp_vsq_f32: allocation failed")
    with nogil:
        vDSP_vsq(&x[0], 1, out, 1, n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float blas_snrm2(FLOAT32[::1] x) nogil:
    """L2 norm via BLAS cblas_snrm2 (float32)."""
    cdef int n = x.shape[0]
    return cblas_snrm2(n, &x[0], 1)


# ──────────────────────────────────────────────────────────────────────
#  Composite operations (Accelerate-accelerated)
# ──────────────────────────────────────────────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] accelerate_sigmoid_1d(
        FLOAT32[::1] x):
    """Sigmoid via Accelerate: 1 / (1 + exp(-x)).

    Steps: negate → vvexpf → add 1 → reciprocal.
    """
    cdef int n = x.shape[0]
    cdef float *neg = <float *>malloc(n * sizeof(float))
    cdef float *ex  = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    cdef float one = 1.0
    cdef unsigned long un = <unsigned long>n
    cdef Py_ssize_t i
    if neg == NULL or ex == NULL or out == NULL:
        free(neg); free(ex); free(out)
        raise MemoryError("accelerate_sigmoid_1d: allocation failed")
    with nogil:
        # negate
        vDSP_vneg(&x[0], 1, neg, 1, un)
        # exp(-x)
        vvexpf(ex, neg, &n)
        # 1 + exp(-x)
        vDSP_vsadd(ex, 1, &one, ex, 1, un)
        # 1 / (1 + exp(-x))
        vvrecf(out, ex, &n)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(neg); free(ex); free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] accelerate_silu_1d(
        FLOAT32[::1] x):
    """SiLU (Swish) via Accelerate: x * sigmoid(x)."""
    cdef int n = x.shape[0]
    cdef float *neg = <float *>malloc(n * sizeof(float))
    cdef float *ex  = <float *>malloc(n * sizeof(float))
    cdef float *sig = <float *>malloc(n * sizeof(float))
    cdef float *out = <float *>malloc(n * sizeof(float))
    cdef float one = 1.0
    cdef unsigned long un = <unsigned long>n
    if neg == NULL or ex == NULL or sig == NULL or out == NULL:
        free(neg); free(ex); free(sig); free(out)
        raise MemoryError("accelerate_silu_1d: allocation failed")
    with nogil:
        vDSP_vneg(&x[0], 1, neg, 1, un)
        vvexpf(ex, neg, &n)
        vDSP_vsadd(ex, 1, &one, ex, 1, un)
        vvrecf(sig, ex, &n)
        vDSP_vmul(&x[0], 1, sig, 1, out, 1, un)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(neg); free(ex); free(sig); free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] accelerate_gelu_1d(
        FLOAT32[::1] x):
    """GELU via Accelerate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    cdef int n = x.shape[0]
    cdef float *x3    = <float *>malloc(n * sizeof(float))
    cdef float *inner = <float *>malloc(n * sizeof(float))
    cdef float *th    = <float *>malloc(n * sizeof(float))
    cdef float *out   = <float *>malloc(n * sizeof(float))
    cdef unsigned long un = <unsigned long>n
    cdef float coeff = 0.044715
    cdef float scale = 0.7978845608  # sqrt(2/pi)
    cdef float one = 1.0
    cdef float half = 0.5
    if x3 == NULL or inner == NULL or th == NULL or out == NULL:
        free(x3); free(inner); free(th); free(out)
        raise MemoryError("accelerate_gelu_1d: allocation failed")
    with nogil:
        # x^3
        vDSP_vsq(&x[0], 1, x3, 1, un)        # x^2
        vDSP_vmul(x3, 1, &x[0], 1, x3, 1, un) # x^3
        # 0.044715 * x^3
        vDSP_vsmul(x3, 1, &coeff, x3, 1, un)
        # x + 0.044715 * x^3
        vDSP_vadd(&x[0], 1, x3, 1, inner, 1, un)
        # sqrt(2/pi) * (...)
        vDSP_vsmul(inner, 1, &scale, inner, 1, un)
        # tanh(...)
        vvtanhf(th, inner, &n)
        # 1 + tanh(...)
        vDSP_vsadd(th, 1, &one, th, 1, un)
        # 0.5 * x
        vDSP_vsmul(&x[0], 1, &half, out, 1, un)
        # 0.5 * x * (1 + tanh(...))
        vDSP_vmul(out, 1, th, 1, out, 1, un)
    result = np.empty(n, dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, n * sizeof(float))
    free(x3); free(inner); free(th); free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] accelerate_rms_norm_2d(
        FLOAT32[:, ::1] x, FLOAT32[::1] weight, float eps):
    """RMS normalisation via vDSP: x * rsqrt(mean(x^2) + eps) * weight.

    x: (N, D), weight: (D,) → out: (N, D).
    """
    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef float *out = <float *>malloc(N * D * sizeof(float))
    cdef float *row_sq = <float *>malloc(D * sizeof(float))
    cdef unsigned long uD = <unsigned long>D
    cdef Py_ssize_t i
    cdef float mean_sq, rms
    if out == NULL or row_sq == NULL:
        free(out); free(row_sq)
        raise MemoryError("accelerate_rms_norm_2d: allocation failed")
    with nogil:
        for i in range(N):
            # Square the row
            vDSP_vsq(&x[i, 0], 1, row_sq, 1, uD)
            # Mean of squares
            vDSP_meanv(row_sq, 1, &mean_sq, uD)
            # rsqrt(mean + eps)
            rms = 1.0 / c_sqrt(mean_sq + eps)
            # x[i] * rms
            vDSP_vsmul(&x[i, 0], 1, &rms, &out[i * D], 1, uD)
            # * weight
            vDSP_vmul(&out[i * D], 1, &weight[0], 1, &out[i * D], 1, uD)
    result = np.empty((N, D), dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, N * D * sizeof(float))
    free(out); free(row_sq)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] accelerate_softmax_2d(
        FLOAT32[:, ::1] x):
    """Row-wise softmax via Accelerate: exp(x - max) / sum(exp(x - max)).

    x: (N, C) → out: (N, C).
    """
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef float *out = <float *>malloc(N * C * sizeof(float))
    cdef unsigned long uC = <unsigned long>C
    cdef int cC = C
    cdef Py_ssize_t i
    cdef float row_max, row_sum, neg_max
    if out == NULL:
        raise MemoryError("accelerate_softmax_2d: allocation failed")
    with nogil:
        for i in range(N):
            # max for numerical stability
            vDSP_maxv(&x[i, 0], 1, &row_max, uC)
            # x - max
            neg_max = -row_max
            vDSP_vsadd(&x[i, 0], 1, &neg_max, &out[i * C], 1, uC)
            # exp(x - max)  — in-place on out row
            vvexpf(&out[i * C], &out[i * C], &cC)
            # sum
            vDSP_sve(&out[i * C], 1, &row_sum, uC)
            # divide by sum (vDSP_vsdiv = vsmul by 1/sum)
            row_sum = 1.0 / row_sum
            vDSP_vsmul(&out[i * C], 1, &row_sum, &out[i * C], 1, uC)
    result = np.empty((N, C), dtype=np.float32)
    memcpy(np.PyArray_DATA(result), out, N * C * sizeof(float))
    free(out)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float accelerate_cross_entropy(
        FLOAT32[:, ::1] logits, INT64[::1] targets) nogil:
    """Cross-entropy via Accelerate (log-softmax then pick target).

    logits: (N, C), targets: (N,) → scalar loss.
    """
    cdef int N = logits.shape[0]
    cdef int C = logits.shape[1]
    cdef float *buf = <float *>malloc(C * sizeof(float))
    cdef unsigned long uC = <unsigned long>C
    cdef int cC = C
    cdef Py_ssize_t i
    cdef float row_max, row_sum, neg_max, total_loss = 0.0
    cdef INT64 tgt
    if buf == NULL:
        with gil:
            raise MemoryError("accelerate_cross_entropy: allocation failed")

    for i in range(N):
        tgt = targets[i]
        vDSP_maxv(&logits[i, 0], 1, &row_max, uC)
        neg_max = -row_max
        vDSP_vsadd(&logits[i, 0], 1, &neg_max, buf, 1, uC)
        vvexpf(buf, buf, &cC)
        vDSP_sve(buf, 1, &row_sum, uC)
        # log-softmax at target index
        total_loss = total_loss - (logits[i, tgt] - row_max - c_log(row_sum))

    free(buf)
    return total_loss / <float>N


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accelerate_adamw_step_f32(
        FLOAT32[::1] param,
        FLOAT32[::1] grad,
        FLOAT32[::1] m,
        FLOAT32[::1] v,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bc1,
        float bc2) nogil:
    """In-place AdamW update using vDSP vector ops (float32).

    Decoupled weight decay → moment update → bias-corrected step.
    """
    cdef unsigned long n = param.shape[0]
    cdef float *tmp = <float *>malloc(n * sizeof(float))
    cdef float *m_hat = <float *>malloc(n * sizeof(float))
    cdef float *v_hat = <float *>malloc(n * sizeof(float))
    cdef float *denom = <float *>malloc(n * sizeof(float))
    cdef float wd_factor = 1.0 - lr * weight_decay
    cdef float one_minus_b1 = 1.0 - beta1
    cdef float one_minus_b2 = 1.0 - beta2
    cdef float inv_bc1 = 1.0 / bc1
    cdef float inv_bc2 = 1.0 / bc2
    cdef float neg_lr = -lr
    cdef int nn = <int>n

    if tmp == NULL or m_hat == NULL or v_hat == NULL or denom == NULL:
        free(tmp); free(m_hat); free(v_hat); free(denom)
        with gil:
            raise MemoryError("accelerate_adamw_step_f32: allocation failed")

    # Decoupled weight decay: param *= (1 - lr * wd)
    vDSP_vsmul(&param[0], 1, &wd_factor, &param[0], 1, n)

    # m = beta1 * m + (1 - beta1) * grad
    vDSP_vsmul(&m[0], 1, &beta1, &m[0], 1, n)
    vDSP_vsmul(&grad[0], 1, &one_minus_b1, tmp, 1, n)
    vDSP_vadd(&m[0], 1, tmp, 1, &m[0], 1, n)

    # v = beta2 * v + (1 - beta2) * grad^2
    vDSP_vsmul(&v[0], 1, &beta2, &v[0], 1, n)
    vDSP_vsq(&grad[0], 1, tmp, 1, n)
    vDSP_vsmul(tmp, 1, &one_minus_b2, tmp, 1, n)
    vDSP_vadd(&v[0], 1, tmp, 1, &v[0], 1, n)

    # m_hat = m / bc1,  v_hat = v / bc2
    vDSP_vsmul(&m[0], 1, &inv_bc1, m_hat, 1, n)
    vDSP_vsmul(&v[0], 1, &inv_bc2, v_hat, 1, n)

    # denom = sqrt(v_hat) + eps
    vvsqrtf(denom, v_hat, &nn)
    vDSP_vsadd(denom, 1, &eps, denom, 1, n)

    # step = m_hat / denom
    vDSP_vdiv(denom, 1, m_hat, 1, tmp, 1, n)

    # param -= lr * step
    vDSP_vsmul(tmp, 1, &neg_lr, tmp, 1, n)
    vDSP_vadd(&param[0], 1, tmp, 1, &param[0], 1, n)

    free(tmp); free(m_hat); free(v_hat); free(denom)


# ──────────────────────────────────────────────────────────────────────
#  Python-facing dispatch wrappers
# ──────────────────────────────────────────────────────────────────────

def accelerate_sgemm(np.ndarray a not None, np.ndarray b not None):
    """Matrix multiply dispatch — float32 uses BLAS, else NumPy."""
    if a.ndim == 2 and b.ndim == 2:
        if a.dtype == np.float32:
            return blas_sgemm(np.ascontiguousarray(a),
                              np.ascontiguousarray(b))
        elif a.dtype == np.float64:
            return blas_dgemm(np.ascontiguousarray(a),
                              np.ascontiguousarray(b))
    return np.matmul(a, b)


def accelerate_sigmoid(np.ndarray x not None):
    """Sigmoid dispatch — flat + call Accelerate kernel + reshape."""
    cdef np.ndarray flat = np.ascontiguousarray(x).ravel()
    cdef tuple shape = (<object>x).shape
    if flat.dtype == np.float32:
        return accelerate_sigmoid_1d(flat).reshape(shape)
    return 1.0 / (1.0 + np.exp(-x))


def accelerate_exp(np.ndarray x not None):
    """Exp dispatch."""
    cdef np.ndarray flat = np.ascontiguousarray(x).ravel()
    cdef tuple shape = (<object>x).shape
    if flat.dtype == np.float32:
        return veclib_expf(flat).reshape(shape)
    elif flat.dtype == np.float64:
        return veclib_exp(flat).reshape(shape)
    return np.exp(x)


def accelerate_silu(np.ndarray x not None):
    """SiLU dispatch."""
    cdef np.ndarray flat = np.ascontiguousarray(x).ravel()
    cdef tuple shape = (<object>x).shape
    if flat.dtype == np.float32:
        return accelerate_silu_1d(flat).reshape(shape)
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def accelerate_rms_norm(np.ndarray x not None, np.ndarray weight not None,
                         float eps):
    """RMS norm dispatch."""
    if x.ndim == 2 and x.dtype == np.float32 and weight.dtype == np.float32:
        return accelerate_rms_norm_2d(
            np.ascontiguousarray(x),
            np.ascontiguousarray(weight),
            eps)
    # Fallback
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def accelerate_softmax(np.ndarray x not None, int axis=-1):
    """Softmax dispatch."""
    if x.ndim == 2 and axis == -1 and x.dtype == np.float32:
        return accelerate_softmax_2d(np.ascontiguousarray(x))
    # Fallback
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def accelerate_gelu(np.ndarray x not None):
    """GELU dispatch."""
    cdef np.ndarray flat = np.ascontiguousarray(x).ravel()
    cdef tuple shape = (<object>x).shape
    if flat.dtype == np.float32:
        return accelerate_gelu_1d(flat).reshape(shape)
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def accelerate_adamw_step(np.ndarray param not None,
                           np.ndarray grad not None,
                           np.ndarray m not None,
                           np.ndarray v not None,
                           float lr, float beta1, float beta2,
                           float eps, float weight_decay,
                           float bc1, float bc2):
    """AdamW step dispatch — in-place on param/m/v arrays."""
    if param.dtype == np.float32:
        accelerate_adamw_step_f32(
            np.ascontiguousarray(param).ravel(),
            np.ascontiguousarray(grad).ravel(),
            np.ascontiguousarray(m).ravel(),
            np.ascontiguousarray(v).ravel(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2)
    else:
        # Pure-NumPy fallback for non-float32
        param *= (1.0 - lr * weight_decay)
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / bc1
        v_hat = v / bc2
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
