# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
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

cdef extern from "math.h" nogil:
    float expf(float x)
    float tanhf(float x)

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
    cdef np.ndarray result = np.empty((M, N), dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    &a[0, 0], K,
                    &b[0, 0], N,
                    beta, out, N)
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
    cdef np.ndarray result = np.empty((M, N), dtype=np.float64)
    cdef double *out = <double *>np.PyArray_DATA(result)
    with nogil:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    &a[0, 0], K,
                    &b[0, 0], N,
                    beta, out, N)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] blas_sgemm_nt(
        FLOAT32[:, ::1] a, FLOAT32[:, ::1] b):
    """C = A @ B^T using cblas_sgemm with B transposed (float32).

    A: (M, K)  B: (N, K)  →  C: (M, N)
    Avoids allocating a transposed copy of B.
    """
    cdef int M = a.shape[0]
    cdef int K = a.shape[1]
    cdef int N = b.shape[0]
    cdef float alpha = 1.0
    cdef float beta  = 0.0
    cdef np.ndarray result = np.empty((M, N), dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, alpha,
                    &a[0, 0], K,
                    &b[0, 0], K,
                    beta, out, N)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] blas_sgemm_tn(
        FLOAT32[:, ::1] a, FLOAT32[:, ::1] b):
    """C = A^T @ B using cblas_sgemm with A transposed (float32).

    A: (K, M)  B: (K, N)  →  C: (M, N)
    Avoids allocating a transposed copy of A.
    """
    cdef int K = a.shape[0]
    cdef int M = a.shape[1]
    cdef int N = b.shape[1]
    cdef float alpha = 1.0
    cdef float beta  = 0.0
    cdef np.ndarray result = np.empty((M, N), dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    M, N, K, alpha,
                    &a[0, 0], M,
                    &b[0, 0], N,
                    beta, out, N)
    return result


# ──────────────────────────────────────────────────────────────────────
#  Element-wise transcendentals via vecLib
# ──────────────────────────────────────────────────────────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_expf(FLOAT32[::1] x):
    """Element-wise exp via vecLib vvexpf (float32)."""
    cdef int n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vvexpf(out, &x[0], &n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT64, ndim=1] veclib_exp(FLOAT64[::1] x):
    """Element-wise exp via vecLib vvexp (float64)."""
    cdef int n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float64)
    cdef double *out = <double *>np.PyArray_DATA(result)
    with nogil:
        vvexp(out, &x[0], &n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_logf(FLOAT32[::1] x):
    """Element-wise log via vecLib vvlogf (float32)."""
    cdef int n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vvlogf(out, &x[0], &n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_sqrtf(FLOAT32[::1] x):
    """Element-wise sqrt via vecLib vvsqrtf (float32)."""
    cdef int n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vvsqrtf(out, &x[0], &n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] veclib_tanhf(FLOAT32[::1] x):
    """Element-wise tanh via vecLib vvtanhf (float32)."""
    cdef int n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vvtanhf(out, &x[0], &n)
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
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vmul(&a[0], 1, &b[0], 1, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vadd_f32(
        FLOAT32[::1] a, FLOAT32[::1] b):
    """Element-wise add via vDSP_vadd (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vadd(&a[0], 1, &b[0], 1, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsub_f32(
        FLOAT32[::1] a, FLOAT32[::1] b):
    """Element-wise subtract (a - b) via vDSP_vsub (float32).
    Note: vDSP_vsub(B, IB, A, IA, C, IC, N) computes C = A - B."""
    cdef unsigned long n = a.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vsub(&b[0], 1, &a[0], 1, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vdiv_f32(
        FLOAT32[::1] a, FLOAT32[::1] b):
    """Element-wise divide (a / b) via vDSP_vdiv (float32).
    Note: vDSP_vdiv(B, IB, A, IA, C, IC, N) computes C = A / B."""
    cdef unsigned long n = a.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vdiv(&b[0], 1, &a[0], 1, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsmul_f32(
        FLOAT32[::1] a, float scalar):
    """Scalar multiply via vDSP_vsmul (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vsmul(&a[0], 1, &scalar, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] vdsp_vsadd_f32(
        FLOAT32[::1] a, float scalar):
    """Scalar add via vDSP_vsadd (float32)."""
    cdef unsigned long n = a.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vsadd(&a[0], 1, &scalar, out, 1, n)
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
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vsq(&x[0], 1, out, 1, n)
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
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef float one = 1.0
    cdef unsigned long un = <unsigned long>n
    cdef Py_ssize_t i
    if neg == NULL or ex == NULL:
        free(neg); free(ex)
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
    free(neg); free(ex)
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
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef float one = 1.0
    cdef unsigned long un = <unsigned long>n
    if neg == NULL or ex == NULL or sig == NULL:
        free(neg); free(ex); free(sig)
        raise MemoryError("accelerate_silu_1d: allocation failed")
    with nogil:
        vDSP_vneg(&x[0], 1, neg, 1, un)
        vvexpf(ex, neg, &n)
        vDSP_vsadd(ex, 1, &one, ex, 1, un)
        vvrecf(sig, ex, &n)
        vDSP_vmul(&x[0], 1, sig, 1, out, 1, un)
    free(neg); free(ex); free(sig)
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
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef unsigned long un = <unsigned long>n
    cdef float coeff = 0.044715
    cdef float scale = 0.7978845608  # sqrt(2/pi)
    cdef float one = 1.0
    cdef float half = 0.5
    if x3 == NULL or inner == NULL or th == NULL:
        free(x3); free(inner); free(th)
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
    free(x3); free(inner); free(th)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] accelerate_relu_1d(FLOAT32[::1] x):
    """ReLU via vDSP_vclip: clamp to [0, FLT_MAX]."""
    cdef unsigned long n = x.shape[0]
    cdef float lo = 0.0
    cdef float hi = 3.4028235e+38
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    with nogil:
        vDSP_vclip(&x[0], 1, &lo, &hi, out, 1, n)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple accelerate_silu_1d_fwd(FLOAT32[::1] x):
    """SiLU forward returning (result, sigmoid) for backward reuse."""
    cdef int n = x.shape[0]
    cdef float *neg = <float *>malloc(n * sizeof(float))
    cdef float *ex  = <float *>malloc(n * sizeof(float))
    cdef np.ndarray sig_arr = np.empty(n, dtype=np.float32)
    cdef float *sig = <float *>np.PyArray_DATA(sig_arr)
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef float one = 1.0
    cdef unsigned long un = <unsigned long>n
    if neg == NULL or ex == NULL:
        free(neg); free(ex)
        raise MemoryError("accelerate_silu_1d_fwd: allocation failed")
    with nogil:
        vDSP_vneg(&x[0], 1, neg, 1, un)
        vvexpf(ex, neg, &n)
        vDSP_vsadd(ex, 1, &one, ex, 1, un)
        vvrecf(sig, ex, &n)
        vDSP_vmul(&x[0], 1, sig, 1, out, 1, un)
    free(neg); free(ex)
    return (result, sig_arr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] gelu_fused_1d(FLOAT32[::1] x):
    """GELU via single-pass fused C loop (auto-vectorised by Clang -O3).

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Single pass over data = minimal memory traffic.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef Py_ssize_t i
    cdef float xi, x3, inner, t
    cdef float coeff = 0.044715
    cdef float scale = 0.7978845608  # sqrt(2/pi)
    with nogil:
        for i in range(n):
            xi = x[i]
            x3 = xi * xi * xi
            inner = scale * (xi + coeff * x3)
            t = tanhf(inner)
            out[i] = 0.5 * xi * (1.0 + t)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] gelu_tiled_1d(FLOAT32[::1] x):
    """GELU with cache-tiled vectorized processing.

    Processes data in L2-sized chunks so all 9 vDSP/vecLib operations
    execute while data is hot in cache, minimising memory traffic.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef Py_ssize_t TILE = 16384  # 64KB working set fits in L2
    cdef Py_ssize_t start, chunk
    cdef float *tmp = <float *>malloc(TILE * sizeof(float))
    cdef float *th  = <float *>malloc(TILE * sizeof(float))
    cdef float coeff = 0.044715
    cdef float scale = 0.7978845608  # sqrt(2/pi)
    cdef float one = 1.0
    cdef float half = 0.5
    cdef int cn
    cdef unsigned long uchunk
    if tmp == NULL or th == NULL:
        free(tmp); free(th)
        raise MemoryError("gelu_tiled_1d: allocation failed")
    with nogil:
        start = 0
        while start < n:
            chunk = n - start
            if chunk > TILE:
                chunk = TILE
            uchunk = <unsigned long>chunk
            cn = <int>chunk
            # x^2
            vDSP_vsq(&x[start], 1, tmp, 1, uchunk)
            # x^3
            vDSP_vmul(tmp, 1, &x[start], 1, tmp, 1, uchunk)
            # 0.044715 * x^3
            vDSP_vsmul(tmp, 1, &coeff, tmp, 1, uchunk)
            # x + 0.044715 * x^3
            vDSP_vadd(&x[start], 1, tmp, 1, tmp, 1, uchunk)
            # sqrt(2/pi) * (...)
            vDSP_vsmul(tmp, 1, &scale, tmp, 1, uchunk)
            # tanh(...)
            vvtanhf(th, tmp, &cn)
            # 1 + tanh(...)
            vDSP_vsadd(th, 1, &one, th, 1, uchunk)
            # 0.5 * x
            vDSP_vsmul(&x[start], 1, &half, &out[start], 1, uchunk)
            # 0.5 * x * (1 + tanh(...))
            vDSP_vmul(&out[start], 1, th, 1, &out[start], 1, uchunk)
            start = start + TILE
    free(tmp); free(th)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] silu_fused_1d(FLOAT32[::1] x):
    """SiLU via single-pass fused C loop (auto-vectorised).

    x * sigmoid(x) = x / (1 + exp(-x))
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef Py_ssize_t i
    cdef float xi, sig
    with nogil:
        for i in range(n):
            xi = x[i]
            sig = 1.0 / (1.0 + expf(-xi))
            out[i] = xi * sig
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=1] silu_tiled_1d(FLOAT32[::1] x):
    """SiLU with cache-tiled vectorized processing."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef Py_ssize_t TILE = 16384
    cdef Py_ssize_t start, chunk
    cdef float *neg = <float *>malloc(TILE * sizeof(float))
    cdef float *sig = <float *>malloc(TILE * sizeof(float))
    cdef float one = 1.0
    cdef int cn
    cdef unsigned long uchunk
    if neg == NULL or sig == NULL:
        free(neg); free(sig)
        raise MemoryError("silu_tiled_1d: allocation failed")
    with nogil:
        start = 0
        while start < n:
            chunk = n - start
            if chunk > TILE:
                chunk = TILE
            uchunk = <unsigned long>chunk
            cn = <int>chunk
            # -x
            vDSP_vneg(&x[start], 1, neg, 1, uchunk)
            # exp(-x)
            vvexpf(sig, neg, &cn)
            # 1 + exp(-x)
            vDSP_vsadd(sig, 1, &one, sig, 1, uchunk)
            # 1 / (1 + exp(-x))
            vvrecf(sig, sig, &cn)
            # x * sigmoid(x)
            vDSP_vmul(&x[start], 1, sig, 1, &out[start], 1, uchunk)
            start = start + TILE
    free(neg); free(sig)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple silu_tiled_1d_fwd(FLOAT32[::1] x):
    """SiLU tiled forward returning (result, sigmoid)."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray result = np.empty(n, dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef np.ndarray sig_arr = np.empty(n, dtype=np.float32)
    cdef float *sig_final = <float *>np.PyArray_DATA(sig_arr)
    cdef Py_ssize_t TILE = 16384
    cdef Py_ssize_t start, chunk
    cdef float *neg = <float *>malloc(TILE * sizeof(float))
    cdef float one = 1.0
    cdef int cn
    cdef unsigned long uchunk
    if neg == NULL:
        raise MemoryError("silu_tiled_1d_fwd: allocation failed")
    with nogil:
        start = 0
        while start < n:
            chunk = n - start
            if chunk > TILE:
                chunk = TILE
            uchunk = <unsigned long>chunk
            cn = <int>chunk
            vDSP_vneg(&x[start], 1, neg, 1, uchunk)
            vvexpf(&sig_final[start], neg, &cn)
            vDSP_vsadd(&sig_final[start], 1, &one, &sig_final[start], 1, uchunk)
            vvrecf(&sig_final[start], &sig_final[start], &cn)
            vDSP_vmul(&x[start], 1, &sig_final[start], 1, &out[start], 1, uchunk)
            start = start + TILE
    free(neg)
    return (result, sig_arr)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[FLOAT32, ndim=2] accelerate_rms_norm_2d(
        FLOAT32[:, ::1] x, FLOAT32[::1] weight, float eps):
    """RMS normalisation via vDSP: x * rsqrt(mean(x^2) + eps) * weight.

    x: (N, D), weight: (D,) → out: (N, D).
    """
    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef np.ndarray result = np.empty((N, D), dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef float *row_sq = <float *>malloc(D * sizeof(float))
    cdef unsigned long uD = <unsigned long>D
    cdef Py_ssize_t i
    cdef float mean_sq, rms
    if row_sq == NULL:
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
    free(row_sq)
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
    cdef np.ndarray result = np.empty((N, C), dtype=np.float32)
    cdef float *out = <float *>np.PyArray_DATA(result)
    cdef unsigned long uC = <unsigned long>C
    cdef int cC = C
    cdef Py_ssize_t i
    cdef float row_max, row_sum, neg_max
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
    """Sigmoid dispatch."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return accelerate_sigmoid_1d(flat).reshape(shape)
    return 1.0 / (1.0 + np.exp(-x))


def accelerate_exp(np.ndarray x not None):
    """Exp dispatch."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return veclib_expf(flat).reshape(shape)
    elif x.dtype == np.float64:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return veclib_exp(flat).reshape(shape)
    return np.exp(x)


def accelerate_silu(np.ndarray x not None):
    """SiLU dispatch — uses cache-tiled vectorized kernel."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return silu_tiled_1d(flat).reshape(shape)
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def accelerate_silu_fwd(np.ndarray x not None):
    """SiLU forward returning (result, sigmoid) for backward reuse."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        res, sig = silu_tiled_1d_fwd(flat)
        return res.reshape(shape), sig.reshape(shape)
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig, sig


def accelerate_relu(np.ndarray x not None):
    """ReLU dispatch via vDSP_vclip."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return accelerate_relu_1d(flat).reshape(shape)
    return np.maximum(x, 0)


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
    """GELU dispatch — uses cache-tiled vectorized kernel."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return gelu_tiled_1d(flat).reshape(shape)
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def accelerate_log(np.ndarray x not None):
    """Log dispatch via vecLib."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return veclib_logf(flat).reshape(shape)
    return np.log(x)


def accelerate_sqrt(np.ndarray x not None):
    """Sqrt dispatch via vecLib."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return veclib_sqrtf(flat).reshape(shape)
    return np.sqrt(x)


def accelerate_tanh(np.ndarray x not None):
    """Tanh dispatch via vecLib."""
    cdef np.ndarray flat
    cdef tuple shape = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        return veclib_tanhf(flat).reshape(shape)
    return np.tanh(x)


def accelerate_rsqrt(np.ndarray x not None):
    """Rsqrt dispatch: 1/sqrt(x) via accelerate_sqrt."""
    return 1.0 / accelerate_sqrt(x)


def accelerate_linear_forward(np.ndarray x not None, np.ndarray w not None):
    """Optimized x @ w.T for Linear layers using BLAS sgemm_nt.
    
    x: (..., in_features), w: (out_features, in_features)
    Returns: (..., out_features)
    """
    cdef tuple orig_shape = (<object>x).shape
    cdef int in_f = w.shape[1]
    cdef int out_f = w.shape[0]
    if x.dtype == np.float32 and w.dtype == np.float32:
        x_2d = np.ascontiguousarray(x.reshape(-1, in_f))
        w_c = np.ascontiguousarray(w)
        return (<object>blas_sgemm_nt(x_2d, w_c)).reshape(orig_shape[:-1] + (out_f,))
    return np.matmul(x, w.T)


def accelerate_adamw_step(np.ndarray param not None,
                           np.ndarray grad not None,
                           np.ndarray m not None,
                           np.ndarray v not None,
                           float lr, float beta1, float beta2,
                           float eps, float weight_decay,
                           float bc1, float bc2):
    """AdamW step dispatch — in-place on param/m/v arrays (any shape)."""
    cdef tuple shape = (<object>param).shape
    if param.dtype == np.float32:
        flat_p = np.ascontiguousarray(param).ravel()
        flat_g = np.ascontiguousarray(grad).ravel()
        flat_m = np.ascontiguousarray(m).ravel()
        flat_v = np.ascontiguousarray(v).ravel()
        accelerate_adamw_step_f32(
            flat_p, flat_g, flat_m, flat_v,
            lr, beta1, beta2, eps, weight_decay, bc1, bc2)
        # Copy back in-place
        np.copyto(param.ravel(), flat_p)
        np.copyto(m.ravel(), flat_m)
        np.copyto(v.ravel(), flat_v)
    else:
        # Pure-NumPy fallback for non-float32
        param *= (1.0 - lr * weight_decay)
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / bc1
        v_hat = v / bc2
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def accelerate_batched_matmul(a, b):
    """Batched matmul using Accelerate BLAS for each 2D slice."""
    if a.dtype != np.float32 or b.dtype != np.float32:
        return np.matmul(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return <object>blas_sgemm(np.ascontiguousarray(a), np.ascontiguousarray(b))
    if a.ndim < 2 or b.ndim < 2:
        return np.matmul(a, b)

    a_shape = a.shape
    b_shape = b.shape
    a_nd = len(a_shape)
    b_nd = len(b_shape)
    M = int(a_shape[a_nd - 2])
    K = int(a_shape[a_nd - 1])
    N = int(b_shape[b_nd - 1])

    batch_a = a.reshape(-1, M, K)
    if b.ndim == a.ndim:
        batch_b = b.reshape(-1, K, N)
    else:
        batch_b = np.broadcast_to(b, a_shape[:a_nd - 2] + b_shape[b_nd - 2:]).reshape(-1, K, N)

    batch_size = int(batch_a.shape[0])
    result = np.empty((batch_size, M, N), dtype=np.float32)

    for i in range(batch_size):
        bi = i if i < batch_b.shape[0] else 0
        result[i] = <object>blas_sgemm(
            np.ascontiguousarray(batch_a[i]),
            np.ascontiguousarray(batch_b[bi]))

    return result.reshape(a_shape[:a_nd - 1] + (N,))


# ──────────────────────────────────────────────────────────────────────
#  Accelerate-backed arithmetic & reduction dispatch
# ──────────────────────────────────────────────────────────────────────

def accelerate_add(np.ndarray a not None, np.ndarray b not None):
    """Element-wise add via vDSP_vadd (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32 and b.dtype == np.float32 and shape_a == (<object>b).shape:
        flat_a = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        flat_b = b.ravel() if b.flags.c_contiguous else np.ascontiguousarray(b).ravel()
        return (<object>vdsp_vadd_f32(flat_a, flat_b)).reshape(shape_a)
    return np.add(a, b)


def accelerate_sub(np.ndarray a not None, np.ndarray b not None):
    """Element-wise subtract via vDSP (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32 and b.dtype == np.float32 and shape_a == (<object>b).shape:
        flat_a = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        flat_b = b.ravel() if b.flags.c_contiguous else np.ascontiguousarray(b).ravel()
        return (<object>vdsp_vsub_f32(flat_a, flat_b)).reshape(shape_a)
    return np.subtract(a, b)


def accelerate_mul(np.ndarray a not None, np.ndarray b not None):
    """Element-wise multiply via vDSP_vmul (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32 and b.dtype == np.float32 and shape_a == (<object>b).shape:
        flat_a = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        flat_b = b.ravel() if b.flags.c_contiguous else np.ascontiguousarray(b).ravel()
        return (<object>vdsp_vmul_f32(flat_a, flat_b)).reshape(shape_a)
    return np.multiply(a, b)


def accelerate_div(np.ndarray a not None, np.ndarray b not None):
    """Element-wise divide via vDSP_vdiv (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32 and b.dtype == np.float32 and shape_a == (<object>b).shape:
        flat_a = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        flat_b = b.ravel() if b.flags.c_contiguous else np.ascontiguousarray(b).ravel()
        return (<object>vdsp_vdiv_f32(flat_a, flat_b)).reshape(shape_a)
    return np.divide(a, b)


def accelerate_adds(np.ndarray a not None, float scalar):
    """Scalar add via vDSP_vsadd (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32:
        flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        return (<object>vdsp_vsadd_f32(flat, scalar)).reshape(shape_a)
    return a + scalar


def accelerate_muls(np.ndarray a not None, float scalar):
    """Scalar multiply via vDSP_vsmul (float32), else NumPy."""
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32:
        flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        return (<object>vdsp_vsmul_f32(flat, scalar)).reshape(shape_a)
    return a * scalar


def accelerate_neg(np.ndarray a not None):
    """Negate via vDSP_vneg (float32), else NumPy."""
    cdef unsigned long n
    cdef np.ndarray flat, result
    cdef float *out
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32:
        flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        n = flat.shape[0]
        result = np.empty(n, dtype=np.float32)
        out = <float *>np.PyArray_DATA(result)
        with nogil:
            vDSP_vneg(<float *>np.PyArray_DATA(flat), 1, out, 1, n)
        return result.reshape(shape_a)
    return np.negative(a)


def accelerate_abs(np.ndarray a not None):
    """Abs via vDSP — square then sqrt for float32, else NumPy.
    Uses vDSP_vsq + vvsqrtf which is SIMD-vectorised."""
    cdef unsigned long n
    cdef np.ndarray flat, result
    cdef float *sq_buf
    cdef float *out
    cdef int ni
    cdef tuple shape_a = (<object>a).shape
    if a.dtype == np.float32:
        flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
        n = flat.shape[0]
        ni = <int>n
        result = np.empty(n, dtype=np.float32)
        sq_buf = <float *>malloc(n * sizeof(float))
        out = <float *>np.PyArray_DATA(result)
        if sq_buf == NULL:
            raise MemoryError("accelerate_abs: allocation failed")
        with nogil:
            vDSP_vsq(<float *>np.PyArray_DATA(flat), 1, sq_buf, 1, n)
            vvsqrtf(out, sq_buf, &ni)
        free(sq_buf)
        return result.reshape(shape_a)
    return np.abs(a)


def accelerate_pow_scalar(np.ndarray x not None, float exponent):
    """Power with scalar exponent — special-cases x^2 via vDSP_vsq."""
    cdef np.ndarray flat, result
    cdef unsigned long n
    cdef float *out
    cdef float *tmp
    cdef int ni
    cdef tuple shape_x = (<object>x).shape
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        n = flat.shape[0]
        if exponent == 2.0:
            return (<object>vdsp_vsq_f32(flat)).reshape(shape_x)
        elif exponent == 0.5:
            return (<object>veclib_sqrtf(flat)).reshape(shape_x)
        elif exponent == -0.5:
            # rsqrt: 1/sqrt(x) = sqrt then reciprocal
            result = np.empty(n, dtype=np.float32)
            out = <float *>np.PyArray_DATA(result)
            tmp = <float *>malloc(n * sizeof(float))
            ni = <int>n
            if tmp == NULL:
                raise MemoryError("accelerate_pow_scalar: allocation failed")
            with nogil:
                vvsqrtf(tmp, <float *>np.PyArray_DATA(flat), &ni)
                vvrecf(out, tmp, &ni)
            free(tmp)
            return result.reshape(shape_x)
    return np.power(x, exponent)


def accelerate_clamp(np.ndarray x not None, float lo, float hi):
    """Clamp via vDSP_vclip (float32), else NumPy."""
    cdef unsigned long n
    cdef np.ndarray flat, result
    cdef float *out
    if x.dtype == np.float32:
        flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
        n = flat.shape[0]
        result = np.empty(n, dtype=np.float32)
        out = <float *>np.PyArray_DATA(result)
        with nogil:
            vDSP_vclip(<float *>np.PyArray_DATA(flat), 1, &lo, &hi, out, 1, n)
        return result.reshape((<object>x).shape)
    return np.clip(x, lo, hi)


def accelerate_sum(np.ndarray x not None, dim=None, bint keepdim=False):
    """Sum via vDSP_sve (float32), else NumPy.
    If dim is None, returns scalar sum.
    If dim is specified, reduces along that axis."""
    cdef unsigned long n, outer, inner, dim_size, row_stride
    cdef np.ndarray flat, result_flat
    cdef float *inp
    cdef float *outp
    cdef float val
    cdef Py_ssize_t i
    cdef int axis, ndim
    cdef list perm, res_shape
    cdef tuple new_shape
    if dim is None:
        if x.dtype == np.float32:
            flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
            return vdsp_sum_f32(flat)
        return float(np.sum(x))
    # dim-wise reduction
    axis = int(dim)
    ndim = x.ndim
    if axis < 0:
        axis = ndim + axis
    if x.dtype == np.float32:
        # Move target axis to last, make contiguous, then stride
        perm = list(range(ndim))
        perm.pop(axis)
        perm.append(axis)
        xt = np.ascontiguousarray(np.transpose(x, perm))
        new_shape = xt.shape
        dim_size = <unsigned long>new_shape[ndim - 1]
        outer = 1
        for i in range(ndim - 1):
            outer *= <unsigned long>new_shape[i]
        flat = xt.reshape(outer, dim_size)
        result_flat = np.empty(outer, dtype=np.float32)
        inp = <float *>np.PyArray_DATA(flat)
        outp = <float *>np.PyArray_DATA(result_flat)
        with nogil:
            for i in range(<Py_ssize_t>outer):
                vDSP_sve(&inp[i * dim_size], 1, &outp[i], dim_size)
        # Build result shape
        res_shape = []
        for i in range(ndim):
            if i == axis:
                if keepdim:
                    res_shape.append(1)
            else:
                res_shape.append(x.shape[i])
        return result_flat.reshape(tuple(res_shape))
    r = np.sum(x, axis=axis, keepdims=keepdim)
    return r


def accelerate_mean(np.ndarray x not None, dim=None, bint keepdim=False):
    """Mean via vDSP_meanv (float32), else NumPy.
    If dim is None, returns scalar mean.
    If dim is specified, reduces along that axis."""
    cdef unsigned long n, outer, dim_size
    cdef np.ndarray flat, result_flat
    cdef float *inp
    cdef float *outp
    cdef Py_ssize_t i
    cdef int axis, ndim
    cdef list perm, res_shape
    cdef tuple new_shape
    if dim is None:
        if x.dtype == np.float32:
            flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
            return vdsp_mean_f32(flat)
        return float(np.mean(x))
    # dim-wise reduction
    axis = int(dim)
    ndim = x.ndim
    if axis < 0:
        axis = ndim + axis
    if x.dtype == np.float32:
        perm = list(range(ndim))
        perm.pop(axis)
        perm.append(axis)
        xt = np.ascontiguousarray(np.transpose(x, perm))
        new_shape = xt.shape
        dim_size = <unsigned long>new_shape[ndim - 1]
        outer = 1
        for i in range(ndim - 1):
            outer *= <unsigned long>new_shape[i]
        flat = xt.reshape(outer, dim_size)
        result_flat = np.empty(outer, dtype=np.float32)
        inp = <float *>np.PyArray_DATA(flat)
        outp = <float *>np.PyArray_DATA(result_flat)
        with nogil:
            for i in range(<Py_ssize_t>outer):
                vDSP_meanv(&inp[i * dim_size], 1, &outp[i], dim_size)
        res_shape = []
        for i in range(ndim):
            if i == axis:
                if keepdim:
                    res_shape.append(1)
            else:
                res_shape.append(x.shape[i])
        return result_flat.reshape(tuple(res_shape))
    r = np.mean(x, axis=axis, keepdims=keepdim)
    return r


def accelerate_cumsum(np.ndarray x not None, int dim):
    """Cumulative sum — uses running vDSP_sve for float32 on last dim."""
    cdef int ndim = x.ndim
    cdef int axis = dim if dim >= 0 else ndim + dim
    cdef tuple shape
    cdef unsigned long outer, L
    cdef Py_ssize_t i
    cdef np.ndarray result
    cdef float *inp
    cdef float *outp
    cdef float running
    cdef unsigned long j
    if x.dtype != np.float32 or axis != ndim - 1:
        return np.cumsum(x, axis=axis)
    # Fast path for last-dim cumsum
    shape = (<object>x).shape
    outer = 1
    for i in range(ndim - 1):
        outer *= <unsigned long>shape[i]
    L = <unsigned long>shape[ndim - 1]
    flat = np.ascontiguousarray(x).reshape(outer, L)
    result = np.empty_like(flat)
    inp = <float *>np.PyArray_DATA(flat)
    outp = <float *>np.PyArray_DATA(result)
    with nogil:
        for i in range(<Py_ssize_t>outer):
            running = 0.0
            for j in range(L):
                running = running + inp[i * L + j]
                outp[i * L + j] = running
    return result.reshape(shape)
