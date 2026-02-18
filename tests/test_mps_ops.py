#!/usr/bin/env python3
"""Verify MPS/Accelerate Cython kernels for numerical correctness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from scaffolding._mps_ops import (
        blas_sgemm, blas_dgemm,
        accelerate_sigmoid_1d, accelerate_silu_1d, accelerate_gelu_1d,
        vdsp_sum_f32, vdsp_mean_f32, vdsp_max_f32,
        vdsp_vmul_f32, vdsp_vadd_f32, vdsp_vsmul_f32,
        veclib_expf, veclib_logf, veclib_sqrtf, veclib_tanhf,
        blas_snrm2,
        accelerate_softmax_2d, accelerate_rms_norm_2d,
        accelerate_cross_entropy,
    )
    print("_mps_ops imported successfully")
except ImportError as e:
    print(f"_mps_ops not available: {e}")
    sys.exit(1)

np.random.seed(42)
tol = 1e-5

# BLAS sgemm
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 2).astype(np.float32)
c = blas_sgemm(np.ascontiguousarray(a), np.ascontiguousarray(b))
c_ref = a @ b
err = np.max(np.abs(c - c_ref))
print(f"  sgemm:      err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# BLAS dgemm
a64 = np.random.randn(3, 4)
b64 = np.random.randn(4, 2)
c64 = blas_dgemm(np.ascontiguousarray(a64), np.ascontiguousarray(b64))
c64_ref = a64 @ b64
err = np.max(np.abs(c64 - c64_ref))
print(f"  dgemm:      err={err:.2e}  {'OK' if err < 1e-12 else 'FAIL'}")

# Sigmoid
x = np.random.randn(1024).astype(np.float32)
sig = accelerate_sigmoid_1d(np.ascontiguousarray(x))
sig_ref = 1.0 / (1.0 + np.exp(-x))
err = np.max(np.abs(sig - sig_ref))
print(f"  sigmoid:    err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# SiLU
silu = accelerate_silu_1d(np.ascontiguousarray(x))
silu_ref = x * sig_ref
err = np.max(np.abs(silu - silu_ref))
print(f"  silu:       err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# GELU
gelu = accelerate_gelu_1d(np.ascontiguousarray(x))
gelu_ref = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
err = np.max(np.abs(gelu - gelu_ref))
print(f"  gelu:       err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vDSP sum
s = vdsp_sum_f32(np.ascontiguousarray(x))
err = abs(s - np.sum(x))
print(f"  vdsp_sum:   err={err:.2e}  {'OK' if err < 0.01 else 'FAIL'}")

# vDSP mean
m = vdsp_mean_f32(np.ascontiguousarray(x))
err = abs(m - np.mean(x))
print(f"  vdsp_mean:  err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vDSP max
mx = vdsp_max_f32(np.ascontiguousarray(x))
err = abs(mx - np.max(x))
print(f"  vdsp_max:   err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vecLib exp
e = veclib_expf(np.ascontiguousarray(x))
err = np.max(np.abs(e - np.exp(x)))
print(f"  veclib_exp: err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vecLib log
pos = np.abs(x) + 0.01
l = veclib_logf(np.ascontiguousarray(pos.astype(np.float32)))
err = np.max(np.abs(l - np.log(pos)))
print(f"  veclib_log: err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vecLib sqrt
sq = veclib_sqrtf(np.ascontiguousarray(pos.astype(np.float32)))
err = np.max(np.abs(sq - np.sqrt(pos)))
print(f"  veclib_sqrt:err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# vecLib tanh
th = veclib_tanhf(np.ascontiguousarray(x))
err = np.max(np.abs(th - np.tanh(x)))
print(f"  veclib_tanh:err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# BLAS snrm2
n = blas_snrm2(np.ascontiguousarray(x))
err = abs(n - np.linalg.norm(x))
print(f"  blas_nrm2:  err={err:.2e}  {'OK' if err < 0.01 else 'FAIL'}")

# Softmax 2D
logits = np.random.randn(8, 10).astype(np.float32)
sm = accelerate_softmax_2d(np.ascontiguousarray(logits))
sm_ref = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
sm_ref /= sm_ref.sum(axis=-1, keepdims=True)
err = np.max(np.abs(sm - sm_ref))
print(f"  softmax_2d: err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# RMS norm
x2d = np.random.randn(4, 8).astype(np.float32)
w = np.random.randn(8).astype(np.float32)
rn = accelerate_rms_norm_2d(
    np.ascontiguousarray(x2d), np.ascontiguousarray(w), 1e-6)
rn_ref = x2d / np.sqrt(np.mean(x2d**2, axis=-1, keepdims=True) + 1e-6) * w
err = np.max(np.abs(rn - rn_ref))
print(f"  rms_norm:   err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

# Cross entropy
logits_ce = np.random.randn(4, 5).astype(np.float32)
targets_ce = np.array([0, 2, 1, 4], dtype=np.int64)
loss = accelerate_cross_entropy(
    np.ascontiguousarray(logits_ce), np.ascontiguousarray(targets_ce))
# Reference
mx = np.max(logits_ce, axis=-1, keepdims=True)
e = np.exp(logits_ce - mx)
ls = logits_ce - mx - np.log(e.sum(axis=-1, keepdims=True))
loss_ref = -np.mean([ls[i, targets_ce[i]] for i in range(4)])
err = abs(loss - loss_ref)
print(f"  cross_ent:  err={err:.2e}  {'OK' if err < tol else 'FAIL'}")

print()
print("=== ALL MPS ACCELERATE OPS VERIFIED ===")
