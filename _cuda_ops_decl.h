/* ╔══════════════════════════════════════════════════════════════════════╗
 * ║  Scaffolding — Deep Learning Framework                               ║
 * ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * _cuda_ops_decl.h — Pure C declarations of extern "C" kernel wrappers.
 *
 * This header is safe to include from C code (e.g. Cython-generated .c
 * files).  It does NOT include any CUDA device headers such as
 * <curand_kernel.h> or <cuda_fp16.h> — those contain C++ constructs
 * (templates, overloads) that break under a C compiler.
 *
 * The actual kernel implementations and the full CUDA header
 * (_cuda_kernels.cuh) are compiled by nvcc (C++) only.
 */

#ifndef SCAFFOLDING_CUDA_OPS_DECL_H
#define SCAFFOLDING_CUDA_OPS_DECL_H

#include <stdint.h>

/* Forward-declare opaque CUDA types so gcc is happy without
 * pulling in cuda_runtime.h (which is mostly C-safe but can
 * drag in device headers via curand).  The Cython .pyx file
 * already declares these via its own "cdef extern" blocks. */

/* cudaStream_t is typedef'd as a pointer in cuda_runtime.h.
 * We only need the pointer type here so the function prototypes
 * compile.  If cuda_runtime.h has already been included, skip. */
#ifndef __CUDA_RUNTIME_H__
typedef void* cudaStream_t;
#endif

/* cublasHandle_t is likewise an opaque pointer. */
#ifndef CUBLAS_API_H_
typedef void* cublasHandle_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- Element-wise unary --- */
int cuda_exp_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_log_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_sqrt_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_rsqrt_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_sigmoid_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_tanh_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_sin_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_cos_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_neg_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_abs_f32(const float* x, float* y, int64_t n, cudaStream_t stream);

/* --- Element-wise binary --- */
int cuda_add_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_sub_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_mul_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_div_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);

/* --- Scalar broadcast --- */
int cuda_adds_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s);
int cuda_muls_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s);

/* --- Activations --- */
int cuda_relu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_relu_backward_f32(const float* grad, const float* x, float* dx,
                           int64_t n, cudaStream_t stream);
int cuda_silu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_silu_fwd_f32(const float* x, float* y, float* sig, int64_t n,
                      cudaStream_t stream);
int cuda_silu_backward_f32(const float* grad, const float* x, const float* sig,
                           float* dx, int64_t n, cudaStream_t stream);
int cuda_gelu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_gelu_backward_f32(const float* grad, const float* x,
                           float* dx, int64_t n, cudaStream_t stream);

/* --- Clamp --- */
int cuda_clamp_f32(const float* x, float* y, float lo, float hi,
                   int64_t n, cudaStream_t stream);

/* --- Softmax --- */
int cuda_softmax_f32(const float* x, float* y, int rows, int cols,
                     cudaStream_t stream);
int cuda_log_softmax_f32(const float* x, float* y, int rows, int cols,
                         cudaStream_t stream);

/* --- Cross-entropy --- */
int cuda_cross_entropy_fwd_f32(const float* logits, const int64_t* targets,
                               float* losses, float* probs,
                               int N, int C, cudaStream_t stream);

/* --- Reductions --- */
int cuda_sum_f32(const float* x, float* out, int64_t n, cudaStream_t stream);
int cuda_sum_dim_f32(const float* x, float* out, int64_t outer,
                     int64_t dim_size, int64_t inner, cudaStream_t stream);

/* --- Normalization --- */
int cuda_layer_norm_f32(const float* x, const float* gamma, const float* beta,
                        float* y, float* mean, float* rstd,
                        int N, int D, float eps, cudaStream_t stream);
int cuda_rms_norm_f32(const float* x, const float* gamma,
                      float* y, float* rstd,
                      int N, int D, float eps, cudaStream_t stream);

/* --- GEMM via cuBLAS --- */
int cuda_sgemm(cublasHandle_t handle,
               int M, int N, int K,
               const float* A, const float* B, float* C,
               float alpha, float beta,
               int transA, int transB);
int cuda_sgemm_batched(cublasHandle_t handle,
                       int M, int N, int K,
                       const float* A, const float* B, float* C,
                       float alpha, float beta,
                       int batchCount, int64_t strideA, int64_t strideB,
                       int64_t strideC);

/* --- AdamW --- */
int cuda_adamw_step_f32(float* param, const float* grad,
                        float* m, float* v,
                        float lr, float beta1, float beta2,
                        float eps, float wd, float bc1, float bc2,
                        int64_t n, cudaStream_t stream);

/* --- Memory utilities --- */
int cuda_fill_f32(float* ptr, float value, int64_t n, cudaStream_t stream);
int cuda_copy_f32(const float* src, float* dst, int64_t n, cudaStream_t stream);

/* --- Dropout --- */
int cuda_dropout_f32(const float* x, float* y, float* mask,
                     float p, int64_t n,
                     unsigned long long seed, cudaStream_t stream);

/* --- Bias add (row broadcast) --- */
int cuda_bias_add_row_f32(float* C, const float* bias, int M, int N,
                          cudaStream_t stream);

/* --- Embedding --- */
int cuda_embedding_f32(const float* weight, const int64_t* indices,
                       float* output, int num_indices, int embed_dim,
                       cudaStream_t stream);

/* --- Where --- */
int cuda_where_f32(const int* cond, const float* x, const float* y,
                   float* out, int64_t n, cudaStream_t stream);

/* --- Transpose --- */
int cuda_transpose_2d_f32(const float* inp, float* out, int rows, int cols,
                          cudaStream_t stream);

/* --- TF32 control --- */
int cuda_set_tf32(cublasHandle_t handle, int enable);

/* --- Strided copy (slice / cat / stack helper) --- */
int cuda_strided_copy_f32(const float* src, float* dst,
                          int64_t n, int64_t src_offset,
                          int64_t dst_offset, cudaStream_t stream);

/* --- Power (element-wise, scalar exponent) --- */
int cuda_pow_scalar_f32(const float* x, float* y,
                        float exponent, int64_t n, cudaStream_t stream);

/* --- Cumulative sum --- */
int cuda_cumsum_f32(const float* x, float* y,
                    int64_t outer, int64_t dim_size, int64_t inner,
                    cudaStream_t stream);

/* --- Mean along dimension --- */
int cuda_mean_dim_f32(const float* x, float* out,
                      int64_t outer, int64_t dim_size, int64_t inner,
                      cudaStream_t stream);

/* --- Arange --- */
int cuda_arange_f32(float* out, int64_t n, cudaStream_t stream);
int cuda_arange_i64(int64_t* out, int64_t n, cudaStream_t stream);

/* --- Fill i64 --- */
int cuda_fill_i64(int64_t* out, int64_t value, int64_t n, cudaStream_t stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SCAFFOLDING_CUDA_OPS_DECL_H */
