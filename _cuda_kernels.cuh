/* ╔══════════════════════════════════════════════════════════════════════╗
 * ║  Scaffolding — Deep Learning Framework                               ║
 * ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * _cuda_kernels.cuh — CUDA kernel declarations, version-compat macros,
 *                      and common device utilities.
 *
 * Supports CUDA compute capability 3.5 (Kepler) through 9.0+ (Hopper).
 * Uses compile-time guards for architecture-specific optimizations:
 *   - sm_35+:  Basic CUDA, __ldg()
 *   - sm_53+:  FP16 arithmetic (__half intrinsics)
 *   - sm_60+:  Atomic add for double
 *   - sm_70+:  Tensor cores (WMMA), warp shuffle sync
 *   - sm_75+:  INT8 tensor cores
 *   - sm_80+:  BF16, TF32, async memcpy, reduce_add_sync
 *   - sm_86+:  Extended shared memory
 *   - sm_89+:  FP8 (Hopper preliminary)
 *   - sm_90+:  Hopper full: warp specialisation, TMA, FP8 tensor cores
 */

#ifndef SCAFFOLDING_CUDA_KERNELS_CUH
#define SCAFFOLDING_CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

/* curand_kernel.h contains C++ overloads / templates — only include
 * when compiling with nvcc (which is a C++ compiler). */
#if defined(__CUDACC__)
  #include <curand_kernel.h>
#endif

/* ---------- Half / BFloat16 headers (version-gated) ---------- */
#if defined(__CUDACC__)
  #include <cuda_fp16.h>
  #if CUDA_VERSION >= 11000
    #include <cuda_bf16.h>
  #endif
#endif

#include <stdint.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

/* Everything below this point uses CUDA device intrinsics (__device__,
 * __shared__, __syncthreads, etc.) and is only valid when compiled by
 * nvcc.  The plain-C declarations that the Cython bindings need live
 * in _cuda_ops_decl.h, which is safe for gcc/clang. */
#if defined(__CUDACC__)

/* ================================================================
 *  ARCHITECTURE DETECTION — Tesla K80 (GK210, sm_37)
 *
 *  The K80 has two GK210 GPUs, each with:
 *    - 13 SMX, 192 CUDA cores/SMX (2496 total)
 *    - 64 KB configurable shared memory / L1 per SMX
 *    - Max 2048 threads per SM, 16 resident blocks per SM
 *    - 65536 registers per SM, max 255 per thread
 *    - 240 GB/s GDDR5 bandwidth
 *    - __ldg() intrinsic (read-only texture cache)
 *    - No native FP16 compute, no tensor cores
 *    - Warp-synchronous execution (pre-Volta: lockstep warps)
 *
 *  Use SCAFFOLDING_K80 macro to gate K80-specific tuning.
 * ================================================================ */

#if defined(__CUDA_ARCH__)
  /* sm_37 = GK210 = Tesla K80 */
  #if __CUDA_ARCH__ == 370
    #define SCAFFOLDING_K80 1
  #else
    #define SCAFFOLDING_K80 0
  #endif

  /* Broad Kepler family: sm_35, sm_37 */
  #if __CUDA_ARCH__ >= 350 && __CUDA_ARCH__ < 500
    #define SCAFFOLDING_KEPLER 1
  #else
    #define SCAFFOLDING_KEPLER 0
  #endif

  /* Pre-Volta: warps execute in lockstep, no independent scheduling */
  #if __CUDA_ARCH__ < 700
    #define SCAFFOLDING_LOCKSTEP_WARPS 1
  #else
    #define SCAFFOLDING_LOCKSTEP_WARPS 0
  #endif
#else
  #define SCAFFOLDING_K80 0
  #define SCAFFOLDING_KEPLER 0
  #define SCAFFOLDING_LOCKSTEP_WARPS 0
#endif

/* ================================================================
 *  VERSION COMPATIBILITY MACROS
 * ================================================================ */

/* Warp size — constant across all NVIDIA architectures to date */
#define WARP_SIZE 32

/* Warp shuffle: CUDA 9.0+ requires _sync variants with a mask.
 * For older CUDA (< 9.0), fall back to non-sync versions. */
#if CUDA_VERSION >= 9000
  #define SHFL_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFF, (val), (offset))
  #define SHFL_XOR(val, mask)    __shfl_xor_sync(0xFFFFFFFF, (val), (mask))
  #define SHFL(val, src)         __shfl_sync(0xFFFFFFFF, (val), (src))
  #define BALLOT(pred)           __ballot_sync(0xFFFFFFFF, (pred))
  #define SYNCWARP()             __syncwarp()
#else
  #define SHFL_DOWN(val, offset) __shfl_down((val), (offset))
  #define SHFL_XOR(val, mask)    __shfl_xor((val), (mask))
  #define SHFL(val, src)         __shfl((val), (src))
  #define BALLOT(pred)           __ballot((pred))
  #define SYNCWARP()             do {} while(0)
#endif

/* __ldg (read-only cache load) — sm_35+ */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  #define LDG(ptr) __ldg(ptr)
#else
  #define LDG(ptr) (*(ptr))
#endif

/* atomicAdd for double — natively supported on sm_60+.
 * For sm_35/sm_50: software CAS emulation. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline double atomicAdd_double(double* addr, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

/* ================================================================
 *  BLOCK / GRID SIZING HELPERS
 *
 *  K80 (Kepler GK210) tuning notes:
 *    - 192 cores/SMX in 6 groups of 32 → 6 warp schedulers
 *    - Max 2048 threads/SM, 16 blocks/SM
 *    - 48 KB shared memory preferred (set via cudaFuncSetCacheConfig)
 *    - 128-thread blocks give 16 blocks/SM = 2048 threads → 100% occupancy
 *      while keeping register pressure low
 *    - 256-thread blocks give 8 blocks/SM → 100% occupancy, fewer blocks
 *      to schedule which can reduce overhead on simpler kernels
 *    - For bandwidth-bound element-wise ops, 128 threads often wins
 *      due to better L1/texture cache utilization per block
 * ================================================================ */

/* Default block size for 1D kernels — tuned per architecture */
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
  /* K80/Kepler: 128 threads × 16 blocks = 2048 threads = full SM occupancy.
   * Smaller blocks → more blocks → hide latency better on 6 warp schedulers.
   * Also reduces register pressure per block → higher occupancy. */
  #define BLOCK_1D 128
  /* Max grid blocks: K80 has 13 SMXs, cap to avoid over-subscription */
  #define K80_MAX_GRID 8192
#else
  #define BLOCK_1D 256
  #define K80_MAX_GRID 65535
#endif

/* Block size specifically for memory-bandwidth-bound kernels (AdamW, etc.) */
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
  #define BLOCK_BW 128   /* bandwidth-bound: more blocks, better latency hiding */
#else
  #define BLOCK_BW 256
#endif

/* Block size for compute-bound kernels (GELU, softmax, etc.) */
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
  #define BLOCK_COMPUTE 256  /* compute-bound: larger blocks amortize overhead */
#else
  #define BLOCK_COMPUTE 256
#endif

/* For 2D tile kernels (e.g. GEMM, softmax) */
#define TILE_DIM  16
#define TILE_DIM2 32

/* Grid-stride loop macro — handles arbitrary N with any grid size */
#define GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; \
         i < (int64_t)(n); \
         i += (int64_t)blockDim.x * gridDim.x)

/* Compute grid size from n elements and block size */
static inline int ceildiv(int64_t n, int block) {
    return (int)((n + block - 1) / block);
}

/* Cap grid size to avoid over-subscription */
static inline int _grid_size_impl(int64_t n, int block, int max_blocks) {
    int g = ceildiv(n, block);
    return g < max_blocks ? g : max_blocks;
}

/* Allow grid_size(n, block) and grid_size(n, block, max_blocks) */
#define _GS3(n, block, max) _grid_size_impl((n), (block), (max))
#define _GS2(n, block)      _grid_size_impl((n), (block), K80_MAX_GRID)
#define _GS_SEL(_1, _2, _3, NAME, ...) NAME
#define grid_size(...) _GS_SEL(__VA_ARGS__, _GS3, _GS2)(__VA_ARGS__)

/* ================================================================
 *  FAST MATH DEVICE FUNCTIONS
 *
 *  K80 notes: __expf() and __fdividef() are hardware-accelerated
 *  SFU (Special Function Unit) intrinsics on Kepler. Much faster
 *  than their full-precision counterparts.  Accuracy is ~23-bit
 *  mantissa, sufficient for DL inference/training.
 * ================================================================ */

__device__ __forceinline__ float fast_sigmoid(float x) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* Use SFU intrinsics: __expf for fast exp, __fdividef for fast division.
     * On K80 the SFU can execute these in ~8 cycles vs ~32 for full precision. */
    return __fdividef(1.0f, 1.0f + __expf(-x));
#else
    return 1.0f / (1.0f + expf(-x));
#endif
}

__device__ __forceinline__ float fast_tanh(float x) {
    /* Use CUDA intrinsic */
    return tanhf(x);
}

__device__ __forceinline__ float fast_gelu(float x) {
    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff = 0.044715f;
    float x3 = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

/* ================================================================
 *  WARP REDUCTION PRIMITIVES
 * ================================================================ */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += SHFL_DOWN(val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, SHFL_DOWN(val, offset));
    }
    return val;
}

/* Block-level reduction using shared memory + warp reduction */
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  /* one per warp */
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

/* ================================================================
 *  ERROR CHECKING
 * ================================================================ */

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        return -1;                                                     \
    }                                                                  \
} while(0)

#define CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t status = (call);                                    \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                \
                __FILE__, __LINE__, (int)status);                      \
        return -1;                                                     \
    }                                                                  \
} while(0)

/* ================================================================
 *  KERNEL LAUNCH CONFIGURATION
 * ================================================================ */

/* Optimal block size lookup based on the kernel's register usage.
 * Used via cudaOccupancyMaxPotentialBlockSize. */
template <typename KernelFunc>
static inline void optimal_launch_config(KernelFunc kernel,
                                         int64_t n,
                                         int* block_out,
                                         int* grid_out,
                                         int smem = 0) {
    int min_grid, block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel, smem, 0);
    *block_out = block;
    *grid_out = grid_size(n, block);
}

/* ================================================================
 *  FORWARD DECLARATIONS — All kernels defined in _cuda_kernels.cu
 * ================================================================ */

/* --- Element-wise unary kernels --- */
extern "C" {
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

/* --- Element-wise binary kernels --- */
int cuda_add_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_sub_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_mul_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);
int cuda_div_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s);

/* --- Scalar broadcast binary --- */
int cuda_adds_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s);
int cuda_muls_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s);

/* --- Activation kernels --- */
int cuda_relu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_relu_backward_f32(const float* grad, const float* x, float* dx,
                           int64_t n, cudaStream_t stream);
int cuda_silu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_silu_fwd_f32(const float* x, float* y, float* sig, int64_t n, cudaStream_t stream);
int cuda_silu_backward_f32(const float* grad, const float* x, const float* sig,
                           float* dx, int64_t n, cudaStream_t stream);
int cuda_gelu_f32(const float* x, float* y, int64_t n, cudaStream_t stream);
int cuda_gelu_backward_f32(const float* grad, const float* x,
                           float* dx, int64_t n, cudaStream_t stream);

/* --- Clamp --- */
int cuda_clamp_f32(const float* x, float* y, float lo, float hi,
                   int64_t n, cudaStream_t stream);

/* --- Softmax (per-row, 2D) --- */
int cuda_softmax_f32(const float* x, float* y, int rows, int cols, cudaStream_t stream);
int cuda_log_softmax_f32(const float* x, float* y, int rows, int cols, cudaStream_t stream);

/* --- Cross-entropy (fused softmax + NLL) --- */
int cuda_cross_entropy_fwd_f32(const float* logits, const int64_t* targets,
                               float* losses, float* probs,
                               int N, int C, cudaStream_t stream);

/* --- Reductions --- */
int cuda_sum_f32(const float* x, float* out, int64_t n, cudaStream_t stream);
int cuda_sum_dim_f32(const float* x, float* out, int64_t outer, int64_t dim_size,
                     int64_t inner, cudaStream_t stream);

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
               bool transA, bool transB);
int cuda_sgemm_batched(cublasHandle_t handle,
                       int M, int N, int K,
                       const float* A, const float* B, float* C,
                       float alpha, float beta,
                       int batchCount, int64_t strideA, int64_t strideB, int64_t strideC);

/* --- AdamW fused optimizer step --- */
int cuda_adamw_step_f32(float* param, const float* grad,
                        float* m, float* v,
                        float lr, float beta1, float beta2,
                        float eps, float wd, float bc1, float bc2,
                        int64_t n, cudaStream_t stream);

/* --- Memory utilities --- */
int cuda_fill_f32(float* ptr, float value, int64_t n, cudaStream_t stream);
int cuda_copy_f32(const float* src, float* dst, int64_t n, cudaStream_t stream);

/* --- Dropout (with cuRAND states) --- */
int cuda_dropout_f32(const float* x, float* y, float* mask,
                     float p, int64_t n,
                     unsigned long long seed, cudaStream_t stream);

/* --- Embedding forward --- */
int cuda_embedding_f32(const float* weight, const int64_t* indices,
                       float* output, int num_indices, int embed_dim,
                       cudaStream_t stream);

/* --- Where (conditional select) --- */
int cuda_where_f32(const bool* cond, const float* x, const float* y,
                   float* out, int64_t n, cudaStream_t stream);

/* --- Transpose --- */
int cuda_transpose_2d_f32(const float* in, float* out, int rows, int cols,
                          cudaStream_t stream);

}  /* extern "C" */

#endif /* __CUDACC__ */

#endif /* SCAFFOLDING_CUDA_KERNELS_CUH */
