/* ╔══════════════════════════════════════════════════════════════════════╗
 * ║  Scaffolding — Deep Learning Framework                               ║
 * ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * _cuda_kernels.cu — Production CUDA kernels for all tensor operations.
 *
 * Architecture support:
 *   sm_35  (Kepler)   — baseline, grid-stride loops, __ldg
 *   sm_50  (Maxwell)  — improved scheduling
 *   sm_53  (Maxwell)  — FP16 intrinsics
 *   sm_60  (Pascal)   — native atomicAdd(double), unified memory
 *   sm_70  (Volta)    — tensor cores (WMMA), independent thread scheduling
 *   sm_75  (Turing)   — INT8 tensor cores, faster FP16
 *   sm_80  (Ampere)   — BF16, TF32, async memcpy, reduced precision atomics
 *   sm_86  (GA10x)    — extended shared memory
 *   sm_89  (Ada)      — FP8 (via transform engine)
 *   sm_90  (Hopper)   — TMA, warp specialisation, FP8 tensor cores
 *
 * All kernels use grid-stride loops for portability across all architectures.
 * Block sizes are auto-tuned via cudaOccupancyMaxPotentialBlockSize where
 * beneficial , or use empirically optimal defaults (256 for elementwise,
 * 128/256 for reductions).
 *
 * Compile with:
 *   nvcc -O3 --use_fast_math -std=c++17 \
 *        -gencode arch=compute_35,code=sm_35 \
 *        -gencode arch=compute_50,code=sm_50 \
 *        -gencode arch=compute_60,code=sm_60 \
 *        -gencode arch=compute_70,code=sm_70 \
 *        -gencode arch=compute_75,code=sm_75 \
 *        -gencode arch=compute_80,code=sm_80 \
 *        -gencode arch=compute_86,code=sm_86 \
 *        -gencode arch=compute_89,code=sm_89 \
 *        -gencode arch=compute_90,code=sm_90 \
 *        -c _cuda_kernels.cu -o _cuda_kernels.o
 */

#include "_cuda_kernels.cuh"

/* ================================================================
 *  ELEMENT-WISE UNARY KERNELS
 *
 *  All use grid-stride loops + __ldg for read-only cache loads.
 *  Vectorised float4 loads where alignment permits (4x throughput).
 * ================================================================ */

/* ---- Vectorised float4 kernel template ---- */
template <typename Func>
__global__ void elementwise_unary_f32_kernel(const float* __restrict__ x,
                                             float* __restrict__ y,
                                             int64_t n,
                                             Func fn) {
    /* float4 vectorised path — 4x memory throughput */
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 in = x4[i];
        float4 out;
        out.x = fn(in.x);
        out.y = fn(in.y);
        out.z = fn(in.z);
        out.w = fn(in.w);
        y4[i] = out;
    }

    /* Handle tail elements (n % 4 != 0) */
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        y[i] = fn(LDG(&x[i]));
    }
}

/* Simpler non-vectorised kernel for guaranteed correctness */
template <typename Func>
__global__ void unary_f32_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int64_t n,
                                 Func fn) {
    GRID_STRIDE_LOOP(i, n) {
        y[i] = fn(LDG(&x[i]));
    }
}

/* ---- Concrete unary ops ----
 *
 * K80/Kepler optimization: Use SFU (Special Function Unit) intrinsics
 * (__expf, __logf, __fdividef) for ~4x faster transcendentals at the
 * cost of ~1 ULP accuracy loss.  For DL this is more than sufficient.
 */
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
  struct ExpOp   { __device__ __forceinline__ float operator()(float x) const { return __expf(x); } };
  struct LogOp   { __device__ __forceinline__ float operator()(float x) const { return __logf(x); } };
  struct SqrtOp  { __device__ __forceinline__ float operator()(float x) const { return sqrtf(x); } };
  struct RsqrtOp { __device__ __forceinline__ float operator()(float x) const { return rsqrtf(x); } };
  struct SigOp   { __device__ __forceinline__ float operator()(float x) const { return __fdividef(1.0f, 1.0f + __expf(-x)); } };
  struct TanhOp  { __device__ __forceinline__ float operator()(float x) const { return tanhf(x); } };
  struct SinOp   { __device__ __forceinline__ float operator()(float x) const { return __sinf(x); } };
  struct CosOp   { __device__ __forceinline__ float operator()(float x) const { return __cosf(x); } };
  struct NegOp   { __device__ __forceinline__ float operator()(float x) const { return -x; } };
  struct AbsOp   { __device__ __forceinline__ float operator()(float x) const { return fabsf(x); } };
#else
  struct ExpOp   { __device__ __forceinline__ float operator()(float x) const { return expf(x); } };
  struct LogOp   { __device__ __forceinline__ float operator()(float x) const { return logf(x); } };
  struct SqrtOp  { __device__ __forceinline__ float operator()(float x) const { return sqrtf(x); } };
  struct RsqrtOp { __device__ __forceinline__ float operator()(float x) const { return rsqrtf(x); } };
  struct SigOp   { __device__ __forceinline__ float operator()(float x) const { return 1.0f / (1.0f + expf(-x)); } };
  struct TanhOp  { __device__ __forceinline__ float operator()(float x) const { return tanhf(x); } };
  struct SinOp   { __device__ __forceinline__ float operator()(float x) const { return sinf(x); } };
  struct CosOp   { __device__ __forceinline__ float operator()(float x) const { return cosf(x); } };
  struct NegOp   { __device__ __forceinline__ float operator()(float x) const { return -x; } };
  struct AbsOp   { __device__ __forceinline__ float operator()(float x) const { return fabsf(x); } };
#endif

/* Macro to define the extern "C" launcher for each unary op
 * Uses float4-vectorised kernel for 4x memory throughput */
#define DEFINE_UNARY_LAUNCHER(name, OpStruct)                                 \
extern "C" int cuda_##name##_f32(const float* x, float* y,                    \
                                 int64_t n, cudaStream_t stream) {            \
    if (n == 0) return 0;                                                     \
    int block = BLOCK_1D;                                                     \
    int64_t work = (n + 3) / 4;  /* ceil(n/4) threads for float4 + tail */    \
    int grid = grid_size(work, block);                                        \
    elementwise_unary_f32_kernel<<<grid, block, 0, stream>>>(x, y, n,         \
                                                              OpStruct());    \
    CUDA_CHECK(cudaGetLastError());                                           \
    return 0;                                                                 \
}

DEFINE_UNARY_LAUNCHER(exp, ExpOp)
DEFINE_UNARY_LAUNCHER(log, LogOp)
DEFINE_UNARY_LAUNCHER(sqrt, SqrtOp)
DEFINE_UNARY_LAUNCHER(rsqrt, RsqrtOp)
DEFINE_UNARY_LAUNCHER(sigmoid, SigOp)
DEFINE_UNARY_LAUNCHER(tanh, TanhOp)
DEFINE_UNARY_LAUNCHER(sin, SinOp)
DEFINE_UNARY_LAUNCHER(cos, CosOp)
DEFINE_UNARY_LAUNCHER(neg, NegOp)
DEFINE_UNARY_LAUNCHER(abs, AbsOp)


/* ================================================================
 *  ELEMENT-WISE BINARY KERNELS
 *
 *  Float4-vectorised for 4x memory throughput, matching unary pattern.
 * ================================================================ */

/* ---- Vectorised float4 binary kernel template ---- */
template <typename Func>
__global__ void elementwise_binary_f32_kernel(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              float* __restrict__ c,
                                              int64_t n, Func fn) {
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];
        float4 vc;
        vc.x = fn(va.x, vb.x);
        vc.y = fn(va.y, vb.y);
        vc.z = fn(va.z, vb.z);
        vc.w = fn(va.w, vb.w);
        c4[i] = vc;
    }

    /* Handle tail elements (n % 4 != 0) */
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        c[i] = fn(LDG(&a[i]), LDG(&b[i]));
    }
}

struct AddOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };
struct SubOp { __device__ __forceinline__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ __forceinline__ float operator()(float a, float b) const { return a * b; } };
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
struct DivOp { __device__ __forceinline__ float operator()(float a, float b) const { return __fdividef(a, b); } };
#else
struct DivOp { __device__ __forceinline__ float operator()(float a, float b) const { return a / b; } };
#endif

#define DEFINE_BINARY_LAUNCHER(name, OpStruct)                                \
extern "C" int cuda_##name##_f32(const float* a, const float* b, float* c,    \
                                 int64_t n, cudaStream_t stream) {            \
    if (n == 0) return 0;                                                     \
    int block = BLOCK_1D;                                                     \
    int64_t work = (n + 3) / 4;                                               \
    int grid = grid_size(work, block);                                        \
    elementwise_binary_f32_kernel<<<grid, block, 0, stream>>>(                \
        a, b, c, n, OpStruct());                                              \
    CUDA_CHECK(cudaGetLastError());                                           \
    return 0;                                                                 \
}

DEFINE_BINARY_LAUNCHER(add, AddOp)
DEFINE_BINARY_LAUNCHER(sub, SubOp)
DEFINE_BINARY_LAUNCHER(mul, MulOp)
DEFINE_BINARY_LAUNCHER(div, DivOp)


/* ================================================================
 *  SCALAR BROADCAST BINARY KERNELS (float4-vectorised)
 * ================================================================ */

__global__ void adds_f32_kernel(const float* __restrict__ a, float s,
                                float* __restrict__ c, int64_t n) {
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (int64_t i = tid; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vc;
        vc.x = va.x + s; vc.y = va.y + s;
        vc.z = va.z + s; vc.w = va.w + s;
        c4[i] = vc;
    }
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        c[i] = LDG(&a[i]) + s;
    }
}

__global__ void muls_f32_kernel(const float* __restrict__ a, float s,
                                float* __restrict__ c, int64_t n) {
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (int64_t i = tid; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vc;
        vc.x = va.x * s; vc.y = va.y * s;
        vc.z = va.z * s; vc.w = va.w * s;
        c4[i] = vc;
    }
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        c[i] = LDG(&a[i]) * s;
    }
}

extern "C" int cuda_adds_f32(const float* a, float scalar, float* c,
                             int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    int block = BLOCK_1D;
    int64_t work = (n + 3) / 4;
    int grid_s = grid_size(work, block);
    adds_f32_kernel<<<grid_s, block, 0, stream>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_muls_f32(const float* a, float scalar, float* c,
                             int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    int block = BLOCK_1D;
    int64_t work = (n + 3) / 4;
    int grid_s = grid_size(work, block);
    muls_f32_kernel<<<grid_s, block, 0, stream>>>(a, scalar, c, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  ACTIVATION KERNELS (Fused forward / backward)
 *
 *  Forward paths use float4-vectorised template for throughput.
 *  Backward paths remain scalar (2-input, 1-output pattern).
 * ================================================================ */

/* ---- Activation functors (used by vectorised template) ---- */
struct ReluOp  { __device__ __forceinline__ float operator()(float x) const { return x > 0.0f ? x : 0.0f; } };
struct SiluOp  { __device__ __forceinline__ float operator()(float x) const { return fast_silu(x); } };
struct GeluOp  { __device__ __forceinline__ float operator()(float x) const { return fast_gelu(x); } };

/* ---- ReLU forward (vectorised via DEFINE_UNARY_LAUNCHER below) ---- */

/* Float4-vectorised backward kernel template for 2-input patterns */
template <typename BackwardFunc>
__global__ void elementwise_backward2_f32_kernel(const float* __restrict__ grad,
                                                  const float* __restrict__ x,
                                                  float* __restrict__ dx,
                                                  int64_t n,
                                                  BackwardFunc fn) {
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const float4* g4 = reinterpret_cast<const float4*>(grad);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* d4 = reinterpret_cast<float4*>(dx);
    for (int64_t i = tid; i < n4; i += stride) {
        float4 vg = g4[i];
        float4 vx = x4[i];
        float4 vd;
        vd.x = fn(vg.x, vx.x);
        vd.y = fn(vg.y, vx.y);
        vd.z = fn(vg.z, vx.z);
        vd.w = fn(vg.w, vx.w);
        d4[i] = vd;
    }
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        dx[i] = fn(LDG(&grad[i]), LDG(&x[i]));
    }
}

struct ReluBackOp {
    __device__ __forceinline__ float operator()(float g, float x) const {
        return x > 0.0f ? g : 0.0f;
    }
};

__global__ void relu_backward_f32_kernel(const float* __restrict__ grad,
                                         const float* __restrict__ x,
                                         float* __restrict__ dx, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        dx[i] = LDG(&x[i]) > 0.0f ? LDG(&grad[i]) : 0.0f;
    }
}

/* ReLU forward uses vectorised template */
DEFINE_UNARY_LAUNCHER(relu, ReluOp)

extern "C" int cuda_relu_backward_f32(const float* grad, const float* x,
                                      float* dx, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    int block = BLOCK_1D;
    int64_t work = (n + 3) / 4;
    int grid = grid_size(work, block);
    elementwise_backward2_f32_kernel<<<grid, block, 0, stream>>>(
        grad, x, dx, n, ReluBackOp());
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/* ---- SiLU (x * sigmoid(x)) ---- */
/* SiLU simple forward uses vectorised functor */

__global__ void silu_fwd_f32_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    float* __restrict__ sig, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        float v = LDG(&x[i]);
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        float s = __fdividef(1.0f, 1.0f + __expf(-v));
#else
        float s = 1.0f / (1.0f + expf(-v));
#endif
        sig[i] = s;
        y[i] = v * s;
    }
}

__global__ void silu_backward_f32_kernel(const float* __restrict__ grad,
                                         const float* __restrict__ x,
                                         const float* __restrict__ sig,
                                         float* __restrict__ dx, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        float g = LDG(&grad[i]);
        float xi = LDG(&x[i]);
        float si = LDG(&sig[i]);
        /* d(silu)/dx = sig + x * sig * (1 - sig) = sig * (1 + x * (1-sig)) */
        dx[i] = g * si * (1.0f + xi * (1.0f - si));
    }
}

/* SiLU forward uses vectorised template */
DEFINE_UNARY_LAUNCHER(silu, SiluOp)

extern "C" int cuda_silu_fwd_f32(const float* x, float* y, float* sig,
                                 int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    silu_fwd_f32_kernel<<<grid_size(n, BLOCK_1D), BLOCK_1D, 0, stream>>>(x, y, sig, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int cuda_silu_backward_f32(const float* grad, const float* x,
                                      const float* sig, float* dx,
                                      int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    silu_backward_f32_kernel<<<grid_size(n, BLOCK_1D), BLOCK_1D, 0, stream>>>(
        grad, x, sig, dx, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/* GELU forward uses vectorised functor */

__global__ void gelu_backward_f32_kernel(const float* __restrict__ grad,
                                         const float* __restrict__ x,
                                         float* __restrict__ dx, int64_t n) {
    const float kS2Pi = 0.7978845608028654f;
    const float kC = 0.044715f;
    GRID_STRIDE_LOOP(i, n) {
        float g = LDG(&grad[i]);
        float xi = LDG(&x[i]);
        float x2 = xi * xi;
        float inner = kS2Pi * (xi + kC * xi * x2);
        float t = tanhf(inner);
        float dtanh = 1.0f - t * t;
        float dinner = kS2Pi * (1.0f + 3.0f * kC * x2);
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        dx[i] = g * (0.5f * (1.0f + t) + 0.5f * xi * dtanh * dinner);
#else
        dx[i] = g * (0.5f * (1.0f + t) + 0.5f * xi * dtanh * dinner);
#endif
    }
}

/* GELU forward uses vectorised template */
DEFINE_UNARY_LAUNCHER(gelu, GeluOp)

extern "C" int cuda_gelu_backward_f32(const float* grad, const float* x,
                                      float* dx, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    gelu_backward_f32_kernel<<<grid_size(n, BLOCK_1D), BLOCK_1D, 0, stream>>>(
        grad, x, dx, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/* ---- Clamp ---- */
__global__ void clamp_f32_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 float lo, float hi, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        float v = LDG(&x[i]);
        y[i] = fminf(fmaxf(v, lo), hi);
    }
}

extern "C" int cuda_clamp_f32(const float* x, float* y, float lo, float hi,
                              int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    clamp_f32_kernel<<<grid_size(n, BLOCK_1D), BLOCK_1D, 0, stream>>>(x, y, lo, hi, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  SOFTMAX (numerically stable, per-row for 2D input)
 *
 *  Strategy:
 *    - Small cols (≤1024): one warp/block per row, shared memory reduction
 *    - Large cols: multi-pass with global memory
 *
 *  Three passes: (1) find max, (2) exp & sum, (3) divide
 *  Fused into 2 passes with online softmax algorithm for sm_70+.
 * ================================================================ */

/* Standard 3-pass softmax: one block per row
 *
 * K80 optimization: Use __expf SFU intrinsic and __fdividef for
 * the hot-loop exp/divide operations. Also prefer L1 cache
 * (48 KB L1 / 16 KB shared) for the streaming access pattern. */
__global__ void softmax_f32_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int rows, int cols) {
    /* Each block handles one row */
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + (int64_t)row * cols;
    float* y_row = y + (int64_t)row * cols;

    /* Pass 1: find max (warp reduction) */
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, LDG(&x_row[j]));
    }
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float row_max = s_max;

    /* Pass 2: exp(x - max) and sum */
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        float e = __expf(LDG(&x_row[j]) - row_max);
#else
        float e = expf(LDG(&x_row[j]) - row_max);
#endif
        y_row[j] = e;
        local_sum += e;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    float inv_sum = __fdividef(1.0f, s_sum);
#else
    float inv_sum = 1.0f / s_sum;
#endif

    /* Pass 3: normalise */
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y_row[j] *= inv_sum;
    }
}

/* Online softmax: 2 passes, better for sm_70+ where warp ops are fast.
 * Computes max and sum in a single pass using the "online" trick. */
__global__ void softmax_online_f32_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + (int64_t)row * cols;
    float* y_row = y + (int64_t)row * cols;

    /* Online max + sum computation */
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = LDG(&x_row[j]);
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }
    }

    /* Warp-level reduction of (max, sum) pairs */
    /* This is tricky: we need to combine (max_a, sum_a) with (max_b, sum_b) */
    __shared__ float s_max_arr[32];
    __shared__ float s_sum_arr[32];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid = threadIdx.x / WARP_SIZE;

    /* Intra-warp reduction */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = SHFL_DOWN(local_max, offset);
        float other_sum = SHFL_DOWN(local_sum, offset);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum += other_sum * expf(other_max - local_max);
        }
    }

    if (lane == 0) {
        s_max_arr[wid] = local_max;
        s_sum_arr[wid] = local_sum;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (wid == 0) {
        local_max = (threadIdx.x < num_warps) ? s_max_arr[threadIdx.x] : -FLT_MAX;
        local_sum = (threadIdx.x < num_warps) ? s_sum_arr[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_max = SHFL_DOWN(local_max, offset);
            float other_sum = SHFL_DOWN(local_sum, offset);
            if (other_max > local_max) {
                local_sum = local_sum * expf(local_max - other_max) + other_sum;
                local_max = other_max;
            } else {
                local_sum += other_sum * expf(other_max - local_max);
            }
        }
    }

    __shared__ float s_final_max, s_final_sum;
    if (threadIdx.x == 0) {
        s_final_max = local_max;
        s_final_sum = local_sum;
    }
    __syncthreads();

    float row_max = s_final_max;
    float inv_sum = 1.0f / s_final_sum;

    /* Second pass: compute exp(x - max) / sum */
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y_row[j] = expf(LDG(&x_row[j]) - row_max) * inv_sum;
    }
}

extern "C" int cuda_softmax_f32(const float* x, float* y,
                                int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return 0;

#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* K80: prefer 3-pass kernel even for large cols.
     * Online softmax uses significantly more registers (max/sum state per
     * thread + shared memory for inter-warp reduction) which kills
     * occupancy on Kepler's 65536 registers/SM.
     * Also cap block size to 512 — K80 gets better occupancy. */
    int block = 256;
    if (cols <= 32) block = 32;
    else if (cols <= 64) block = 64;
    else if (cols <= 128) block = 128;
    else if (cols <= 256) block = 256;
    else block = 512;
    softmax_f32_kernel<<<rows, block, 0, stream>>>(x, y, rows, cols);
#else
    /* Choose block size based on cols */
    int block = 256;
    if (cols <= 32) block = 32;
    else if (cols <= 64) block = 64;
    else if (cols <= 128) block = 128;
    else if (cols <= 256) block = 256;
    else if (cols <= 512) block = 512;
    else block = 1024;

    /* Use online softmax for large vocabularies, standard for small */
    if (cols > 1024) {
        softmax_online_f32_kernel<<<rows, block, 0, stream>>>(x, y, rows, cols);
    } else {
        softmax_f32_kernel<<<rows, block, 0, stream>>>(x, y, rows, cols);
    }
#endif
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  LOG SOFTMAX
 * ================================================================ */

__global__ void log_softmax_f32_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + (int64_t)row * cols;
    float* y_row = y + (int64_t)row * cols;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        local_max = fmaxf(local_max, LDG(&x_row[j]));
    }
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float row_max = s_max;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        local_sum += __expf(LDG(&x_row[j]) - row_max);
#else
        local_sum += expf(LDG(&x_row[j]) - row_max);
#endif
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_lse;
    if (threadIdx.x == 0) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        s_lse = __logf(local_sum) + row_max;
#else
        s_lse = logf(local_sum) + row_max;
#endif
    }
    __syncthreads();
    float lse = s_lse;

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y_row[j] = LDG(&x_row[j]) - lse;
    }
}

extern "C" int cuda_log_softmax_f32(const float* x, float* y,
                                    int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return 0;
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    int block = 256;
    if (cols <= 32) block = 32;
    else if (cols <= 64) block = 64;
    else if (cols <= 128) block = 128;
    else if (cols <= 256) block = 256;
    else block = 512;
#else
    int block = (cols <= 1024) ? ((cols + 31) / 32 * 32) : 1024;
    if (block < 32) block = 32;
    if (block > 1024) block = 1024;
#endif
    log_softmax_f32_kernel<<<rows, block, 0, stream>>>(x, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  CROSS-ENTROPY LOSS (Fused softmax + NLL)
 *
 *  Each block processes one sample.
 *  Outputs per-sample losses and softmax probabilities (for backward).
 * ================================================================ */

__global__ void cross_entropy_fwd_f32_kernel(const float* __restrict__ logits,
                                             const int64_t* __restrict__ targets,
                                             float* __restrict__ losses,
                                             float* __restrict__ probs,
                                             int N, int C) {
    int sample = blockIdx.x;
    if (sample >= N) return;

    const float* row = logits + (int64_t)sample * C;
    float* prob_row = probs + (int64_t)sample * C;
    int64_t target = targets[sample];

    /* Find max */
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        local_max = fmaxf(local_max, LDG(&row[j]));
    }
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float row_max = s_max;

    /* exp & sum */
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        float e = __expf(LDG(&row[j]) - row_max);
#else
        float e = expf(LDG(&row[j]) - row_max);
#endif
        prob_row[j] = e;
        local_sum += e;
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    float inv_sum = __fdividef(1.0f, s_sum);
#else
    float inv_sum = 1.0f / s_sum;
#endif

    /* Normalise and compute loss for target class */
    float target_prob = 0.0f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        prob_row[j] *= inv_sum;
        if (j == target) {
            target_prob = prob_row[j];
        }
    }

    /* Reduce target_prob across threads (only one thread has it) */
    target_prob = block_reduce_sum(target_prob);
    if (threadIdx.x == 0) {
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        losses[sample] = -__logf(target_prob + 1e-12f);
#else
        losses[sample] = -logf(target_prob + 1e-12f);
#endif
    }
}

extern "C" int cuda_cross_entropy_fwd_f32(const float* logits,
                                          const int64_t* targets,
                                          float* losses, float* probs,
                                          int N, int C, cudaStream_t stream) {
    if (N == 0) return 0;
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* K80: cap at 512 threads to keep register pressure manageable */
    int block = 256;
    if (C <= 64) block = 64;
    else if (C <= 128) block = 128;
    else block = 512;
#else
    int block = 256;
    if (C <= 64) block = 64;
    else if (C <= 128) block = 128;
    else if (C <= 512) block = 512;
    else block = 1024;
#endif
    cross_entropy_fwd_f32_kernel<<<N, block, 0, stream>>>(
        logits, targets, losses, probs, N, C);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  REDUCTION KERNELS
 * ================================================================ */

/* Global sum reduction — two-phase: per-block partial sums → final sum.
 * Float4-vectorised for 4x memory throughput on bandwidth-bound reduction. */
__global__ void sum_f32_kernel(const float* __restrict__ x, float* __restrict__ out,
                               int64_t n) {
    float sum = 0.0f;
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        sum += LDG(&x[i]);
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

extern "C" int cuda_sum_f32(const float* x, float* out, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    /* Zero the output first */
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* K80: 13 SMXs, cap blocks to avoid over-subscription with atomicAdd */
    int block = BLOCK_1D;
    int64_t work = (n + 3) / 4;
    int grid_s = grid_size(work, block, 256);
#else
    int block = 256;
    int64_t work = (n + 3) / 4;
    int grid_s = grid_size(work, block, 1024);  /* cap at 1024 blocks for atomicAdd */
#endif
    sum_f32_kernel<<<grid_s, block, 0, stream>>>(x, out, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

/* Sum along a specific dimension.
 * Input shape is factored as [outer, dim_size, inner].
 * Output shape is [outer, inner].
 * Each thread computes one output element. */
__global__ void sum_dim_f32_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int64_t outer, int64_t dim_size, int64_t inner) {
    int64_t total = outer * inner;
    GRID_STRIDE_LOOP(idx, total) {
        int64_t o = idx / inner;
        int64_t i = idx % inner;
        float sum = 0.0f;
        for (int64_t d = 0; d < dim_size; d++) {
            sum += LDG(&x[(o * dim_size + d) * inner + i]);
        }
        out[idx] = sum;
    }
}

extern "C" int cuda_sum_dim_f32(const float* x, float* out,
                                int64_t outer, int64_t dim_size, int64_t inner,
                                cudaStream_t stream) {
    int64_t total = outer * inner;
    if (total == 0) return 0;
    int block = BLOCK_1D;
    int grid_s = grid_size(total, block);
    sum_dim_f32_kernel<<<grid_s, block, 0, stream>>>(x, out, outer, dim_size, inner);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  LAYER NORMALIZATION
 *
 *  Input: (N, D) — normalise each row.
 *  Outputs: y, mean, rstd (reciprocal standard deviation).
 *  One block per row.
 *
 *  K80 optimisation: Welford's online algorithm computes mean and
 *  variance in a SINGLE PASS, halving global memory reads (one scan
 *  instead of two). This is critical on K80's 240 GB/s GDDR5 where
 *  bandwidth is the bottleneck.  The warp-level merge uses Welford's
 *  parallel combine formula.
 * ================================================================ */

#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
/* Welford single-pass kernel: reads data ONCE, computes mean+var together */
__global__ void layer_norm_f32_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ y,
                                     float* __restrict__ mean_out,
                                     float* __restrict__ rstd_out,
                                     int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + (int64_t)row * D;
    float* y_row = y + (int64_t)row * D;

    /* Welford's online algorithm — each thread accumulates its portion */
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float val = LDG(&x_row[j]);
        count++;
        float delta = val - mean;
        mean += __fdividef(delta, (float)count);
        float delta2 = val - mean;
        m2 += delta * delta2;
    }

    /* Parallel Welford reduction across the block */
    /* First, intra-warp reduction using shuffles */
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid = threadIdx.x / WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float o_mean = SHFL_DOWN(mean, offset);
        float o_m2 = SHFL_DOWN(m2, offset);
        int o_count = __float_as_int(SHFL_DOWN(__int_as_float(count), offset));
        if (o_count > 0) {
            int new_count = count + o_count;
            float delta = o_mean - mean;
            float new_mean = mean + __fdividef(delta * (float)o_count, (float)new_count);
            float new_m2 = m2 + o_m2 + delta * delta * __fdividef((float)count * (float)o_count, (float)new_count);
            mean = new_mean;
            m2 = new_m2;
            count = new_count;
        }
    }

    /* Inter-warp reduction via shared memory */
    __shared__ float s_mean[32];
    __shared__ float s_m2[32];
    __shared__ int s_count[32];

    if (lane == 0) {
        s_mean[wid] = mean;
        s_m2[wid] = m2;
        s_count[wid] = count;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (wid == 0) {
        mean = (threadIdx.x < num_warps) ? s_mean[threadIdx.x] : 0.0f;
        m2 = (threadIdx.x < num_warps) ? s_m2[threadIdx.x] : 0.0f;
        count = (threadIdx.x < num_warps) ? s_count[threadIdx.x] : 0;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float o_mean = SHFL_DOWN(mean, offset);
            float o_m2 = SHFL_DOWN(m2, offset);
            int o_count = __float_as_int(SHFL_DOWN(__int_as_float(count), offset));
            if (o_count > 0) {
                int new_count = count + o_count;
                float delta = o_mean - mean;
                float new_mean = mean + __fdividef(delta * (float)o_count, (float)new_count);
                float new_m2 = m2 + o_m2 + delta * delta * __fdividef((float)count * (float)o_count, (float)new_count);
                mean = new_mean;
                m2 = new_m2;
                count = new_count;
            }
        }
    }

    __shared__ float s_mu, s_rstd;
    if (threadIdx.x == 0) {
        s_mu = mean;
        float var = __fdividef(m2, (float)D);
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mu = s_mu;
    float rstd = s_rstd;

    /* Normalise + affine — second pass (unavoidable, need mean/rstd first) */
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = (LDG(&x_row[j]) - mu) * rstd;
        if (gamma != nullptr && beta != nullptr) {
            y_row[j] = xhat * LDG(&gamma[j]) + LDG(&beta[j]);
        } else {
            y_row[j] = xhat;
        }
    }

    if (threadIdx.x == 0) {
        if (mean_out != nullptr) mean_out[row] = mu;
        if (rstd_out != nullptr) rstd_out[row] = rstd;
    }
}

#else  /* !K80/Kepler: standard 2-pass */

__global__ void layer_norm_f32_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ y,
                                     float* __restrict__ mean_out,
                                     float* __restrict__ rstd_out,
                                     int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + (int64_t)row * D;
    float* y_row = y + (int64_t)row * D;

    /* Compute mean */
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        local_sum += LDG(&x_row[j]);
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = local_sum / (float)D;
    __syncthreads();
    float mu = s_mean;

    /* Compute variance */
    float local_var = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float diff = LDG(&x_row[j]) - mu;
        local_var += diff * diff;
    }
    local_var = block_reduce_sum(local_var);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        float var = local_var / (float)D;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    /* Normalise + affine */
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = (LDG(&x_row[j]) - mu) * rstd;
        if (gamma != nullptr && beta != nullptr) {
            y_row[j] = xhat * LDG(&gamma[j]) + LDG(&beta[j]);
        } else {
            y_row[j] = xhat;
        }
    }

    if (threadIdx.x == 0) {
        if (mean_out != nullptr) mean_out[row] = mu;
        if (rstd_out != nullptr) rstd_out[row] = rstd;
    }
}

#endif  /* SCAFFOLDING_K80 || SCAFFOLDING_KEPLER */

extern "C" int cuda_layer_norm_f32(const float* x, const float* gamma,
                                   const float* beta, float* y,
                                   float* mean, float* rstd,
                                   int N, int D, float eps, cudaStream_t stream) {
    if (N == 0 || D == 0) return 0;
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* K80: cap at 512 to keep register pressure low.
     * For D=1024 benchmark, 256 threads is natural — each does 4 elements. */
    int block = 256;
    if (D <= 64) block = 64;
    else if (D <= 128) block = 128;
    else block = 256;
#else
    int block = 256;
    if (D <= 64) block = 64;
    else if (D <= 128) block = 128;
    else if (D <= 512) block = 512;
    else block = 1024;
#endif
    layer_norm_f32_kernel<<<N, block, 0, stream>>>(
        x, gamma, beta, y, mean, rstd, N, D, eps);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  RMS NORMALIZATION
 *
 *  RMS norm: y = x / rms(x) * gamma, where rms = sqrt(mean(x^2) + eps)
 * ================================================================ */

__global__ void rms_norm_f32_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ y,
                                   float* __restrict__ rstd_out,
                                   int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + (int64_t)row * D;
    float* y_row = y + (int64_t)row * D;

    /* Compute mean of squares */
    float local_sq = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = LDG(&x_row[j]);
        local_sq += v * v;
    }
    local_sq = block_reduce_sum(local_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(local_sq / (float)D + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    /* Normalise + scale */
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xhat = LDG(&x_row[j]) * rstd;
        if (gamma != nullptr) {
            y_row[j] = xhat * LDG(&gamma[j]);
        } else {
            y_row[j] = xhat;
        }
    }

    if (threadIdx.x == 0 && rstd_out != nullptr) {
        rstd_out[row] = rstd;
    }
}

extern "C" int cuda_rms_norm_f32(const float* x, const float* gamma,
                                 float* y, float* rstd,
                                 int N, int D, float eps, cudaStream_t stream) {
    if (N == 0 || D == 0) return 0;
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    int block = 256;
    if (D <= 64) block = 64;
    else if (D <= 128) block = 128;
    else block = 256;
#else
    int block = 256;
    if (D <= 64) block = 64;
    else if (D <= 128) block = 128;
    else if (D <= 512) block = 512;
    else block = 1024;
#endif
    rms_norm_f32_kernel<<<N, block, 0, stream>>>(x, gamma, y, rstd, N, D, eps);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  GEMM via cuBLAS
 *
 *  cuBLAS uses column-major by default. We store row-major.
 *  Trick: C_row = A_row @ B_row  ←→  C_col^T = B_col^T @ A_col^T
 *  So we call cublasSgemm with (B^T, A^T) and swap M/N.
 *
 *  cuBLAS automatically uses Tensor Cores on sm_70+ when available
 *  and matrix dimensions are multiples of 8 / 16.
 * ================================================================ */

extern "C" int cuda_sgemm(cublasHandle_t handle,
                           int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha, float beta,
                           bool transA, bool transB) {
    /* Row-major to column-major mapping:
     * C = A @ B  (row-major MxN = MxK @ KxN)
     * In col-major:  C^T = B^T @ A^T  (NxM = NxK @ KxM)
     * So cublasSgemm(N, M, K, B, A, C) */

    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    /* Leading dimensions for row-major storage */
    int lda, ldb, ldc;
    if (!transA && !transB) {
        /* C = A @ B, both row-major: call cublas as C^T = B^T @ A^T */
        lda = N;  /* B's leading dim (row-major cols) */
        ldb = K;  /* A's leading dim */
        ldc = N;  /* C's leading dim */
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, lda, A, ldb, &beta, C, ldc));
    } else if (transA && !transB) {
        /* C = A^T @ B */
        lda = N;
        ldb = M;
        ldc = N;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 N, M, K, &alpha, B, lda, A, ldb, &beta, C, ldc));
    } else if (!transA && transB) {
        /* C = A @ B^T */
        lda = K;
        ldb = K;
        ldc = N;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, lda, A, ldb, &beta, C, ldc));
    } else {
        /* C = A^T @ B^T */
        lda = K;
        ldb = M;
        ldc = N;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                 N, M, K, &alpha, B, lda, A, ldb, &beta, C, ldc));
    }
    return 0;
}

/* Strided batched GEMM */
extern "C" int cuda_sgemm_batched(cublasHandle_t handle,
                                  int M, int N, int K,
                                  const float* A, const float* B, float* C,
                                  float alpha, float beta,
                                  int batchCount,
                                  int64_t strideA, int64_t strideB, int64_t strideC) {
    /* Row-major → col-major trick: swap A/B, swap M/N */
    int lda = N;
    int ldb = K;
    int ldc = N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha,
        B, lda, strideB,
        A, ldb, strideA,
        &beta,
        C, ldc, strideC,
        batchCount));
    return 0;
}


/* ================================================================
 *  ADAMW FUSED OPTIMIZER STEP
 *
 *  Single kernel: weight decay + moment updates + bias correction
 *  + parameter update. All in one pass for maximum memory bandwidth.
 * ================================================================ */

/* ================================================================
 *  ADAMW FUSED OPTIMIZER STEP
 *
 *  Single kernel: weight decay + moment updates + bias correction
 *  + parameter update. All in one pass for maximum memory bandwidth.
 *
 *  K80 tuning: AdamW is purely bandwidth-bound (reads param+grad+m+v,
 *  writes param+m+v = 7 float arrays). On K80's 240 GB/s GDDR5:
 *    - __launch_bounds__ to hint register allocation
 *    - Use __fdividef and rsqrtf for fast division and inverse sqrt
 *    - 128-thread blocks for maximum block-level parallelism
 *    - Float4 vectorisation for 4x memory throughput
 * ================================================================ */

#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
__launch_bounds__(128, 16)  /* 128 threads × 16 blocks/SM = 2048 = 100% occupancy */
#endif
__global__ void adamw_step_f32_kernel(float* __restrict__ param,
                                      const float* __restrict__ grad,
                                      float* __restrict__ m,
                                      float* __restrict__ v,
                                      float lr, float beta1, float beta2,
                                      float eps, float wd,
                                      float bc1, float bc2,
                                      int64_t n) {
    /* Float4 vectorised path for 4x memory throughput */
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float4* p4 = reinterpret_cast<float4*>(param);
    const float4* g4 = reinterpret_cast<const float4*>(grad);
    float4* m4 = reinterpret_cast<float4*>(m);
    float4* v4 = reinterpret_cast<float4*>(v);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 pp = p4[i];
        float4 gg = g4[i];
        float4 mm = m4[i];
        float4 vv = v4[i];

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float p_k = (&pp.x)[k];
            float g_k = (&gg.x)[k];
            float m_k = (&mm.x)[k];
            float v_k = (&vv.x)[k];

            p_k -= lr * wd * p_k;
            m_k = beta1 * m_k + (1.0f - beta1) * g_k;
            v_k = beta2 * v_k + (1.0f - beta2) * g_k * g_k;

#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
            float m_hat = __fdividef(m_k, bc1);
            float v_hat = __fdividef(v_k, bc2);
            p_k -= lr * __fdividef(m_hat, sqrtf(v_hat) + eps);
#else
            float m_hat = m_k / bc1;
            float v_hat = v_k / bc2;
            p_k -= lr * m_hat / (sqrtf(v_hat) + eps);
#endif
            (&pp.x)[k] = p_k;
            (&mm.x)[k] = m_k;
            (&vv.x)[k] = v_k;
        }

        p4[i] = pp;
        m4[i] = mm;
        v4[i] = vv;
    }

    /* Tail elements */
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        float p = param[i];
        float g = LDG(&grad[i]);
        float mi = m[i];
        float vi = v[i];

        p -= lr * wd * p;
        mi = beta1 * mi + (1.0f - beta1) * g;
        vi = beta2 * vi + (1.0f - beta2) * g * g;

#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
        float m_hat = __fdividef(mi, bc1);
        float v_hat = __fdividef(vi, bc2);
        p -= lr * __fdividef(m_hat, sqrtf(v_hat) + eps);
#else
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        p -= lr * m_hat / (sqrtf(v_hat) + eps);
#endif
        param[i] = p;
        m[i] = mi;
        v[i] = vi;
    }
}

extern "C" int cuda_adamw_step_f32(float* param, const float* grad,
                                   float* m, float* v,
                                   float lr, float beta1, float beta2,
                                   float eps, float wd, float bc1, float bc2,
                                   int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
#if SCAFFOLDING_K80 || SCAFFOLDING_KEPLER
    /* K80: AdamW is bandwidth-bound (7 arrays × N elements).
     * Prefer 48 KB L1 / 16 KB shared (streaming access, no shared mem usage).
     * Use BLOCK_BW=128 for max occupancy. */
    static bool cache_set = false;
    if (!cache_set) {
        cudaFuncSetCacheConfig(adamw_step_f32_kernel, cudaFuncCachePreferL1);
        cache_set = true;
    }
    int block = BLOCK_BW;
#else
    int block = BLOCK_1D;
#endif
    int64_t work = (n + 3) / 4;
    int grid_s = grid_size(work, block);
    adamw_step_f32_kernel<<<grid_s, block, 0, stream>>>(
        param, grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  MEMORY UTILITY KERNELS (float4-vectorised)
 * ================================================================ */

__global__ void fill_f32_kernel(float* __restrict__ ptr, float value, int64_t n) {
    int64_t n4 = n / 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    float4* p4 = reinterpret_cast<float4*>(ptr);
    float4 val4;
    val4.x = value; val4.y = value; val4.z = value; val4.w = value;
    for (int64_t i = tid; i < n4; i += stride) {
        p4[i] = val4;
    }
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        ptr[i] = value;
    }
}

extern "C" int cuda_fill_f32(float* ptr, float value, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    int64_t work = (n + 3) / 4;
    fill_f32_kernel<<<grid_size(work, BLOCK_1D), BLOCK_1D, 0, stream>>>(ptr, value, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void copy_f32_kernel(const float* __restrict__ src,
                                float* __restrict__ dst, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        dst[i] = LDG(&src[i]);
    }
}

extern "C" int cuda_copy_f32(const float* src, float* dst, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    /* Prefer cudaMemcpyAsync for device-to-device copies */
    CUDA_CHECK(cudaMemcpyAsync(dst, src, n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    return 0;
}


/* ================================================================
 *  DROPOUT WITH cuRAND
 *
 *  Note: No __launch_bounds__ here — cuRAND Philox state is very
 *  register-heavy (~40+ regs), so forcing min blocks would cause
 *  excessive spilling or launch failure.
 * ================================================================ */

__global__ void dropout_f32_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   float* __restrict__ mask,
                                   float p, float scale, int64_t n,
                                   unsigned long long seed) {
    /* One RNG state per thread, generating 4 randoms at a time
       via native Philox-4x32 output — much faster than per-element init */
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    /* Each thread handles elements tid*4, tid*4+1, ..., striding by stride*4 */
    for (int64_t base = tid * 4; base < n; base += stride * 4) {
        float4 r = curand_uniform4(&state);
        float rv[4] = {r.x, r.y, r.z, r.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int64_t i = base + k;
            if (i < n) {
                float m = (rv[k] >= p) ? scale : 0.0f;
                mask[i] = m;
                y[i] = LDG(&x[i]) * m;
            }
        }
    }
}

extern "C" int cuda_dropout_f32(const float* x, float* y, float* mask,
                                float p, int64_t n,
                                unsigned long long seed, cudaStream_t stream) {
    if (n == 0) return 0;
    float scale = 1.0f / (1.0f - p);
    /* Each thread handles 4 elements, so divide grid by 4 */
    int64_t n_threads = (n + 3) / 4;
    dropout_f32_kernel<<<grid_size(n_threads, BLOCK_1D), BLOCK_1D, 0, stream>>>(
        x, y, mask, p, scale, n, seed);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  ROW-WISE BIAS ADD — C[i,j] += bias[j]  for i in [0, M)
 *
 *  K80 optimisation: float4-vectorised when N is divisible by 4.
 *  Falls back to scalar for non-aligned N.
 * ================================================================ */

__global__ void bias_add_row_f32_kernel(float* __restrict__ C,
                                        const float* __restrict__ bias,
                                        int M, int N) {
    /* Check if N is float4-aligned for vectorised path */
    if ((N & 3) == 0) {
        int N4 = N / 4;
        int64_t total4 = (int64_t)M * N4;
        int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        float4* C4 = reinterpret_cast<float4*>(C);
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        for (int64_t i = tid; i < total4; i += stride) {
            int col4 = (int)(i % N4);
            float4 cv = C4[i];
            float4 bv = b4[col4];
            cv.x += bv.x; cv.y += bv.y;
            cv.z += bv.z; cv.w += bv.w;
            C4[i] = cv;
        }
    } else {
        int64_t total = (int64_t)M * N;
        GRID_STRIDE_LOOP(idx, total) {
            C[idx] += LDG(&bias[idx % N]);
        }
    }
}

extern "C" int cuda_bias_add_row_f32(float* C, const float* bias,
                                     int M, int N, cudaStream_t stream) {
    int64_t total = (int64_t)M * N;
    if (total == 0) return 0;
    int block = BLOCK_1D;
    int64_t work = ((N & 3) == 0) ? ((int64_t)M * (N / 4)) : total;
    int grid = grid_size(work, block);
    bias_add_row_f32_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  EMBEDDING FORWARD (float4-vectorised)
 * ================================================================ */

__global__ void embedding_f32_kernel(const float* __restrict__ weight,
                                     const int64_t* __restrict__ indices,
                                     float* __restrict__ output,
                                     int num_indices, int embed_dim) {
    int idx = blockIdx.x;
    if (idx >= num_indices) return;

    int64_t row = indices[idx];
    const float* src = weight + row * embed_dim;
    float* dst = output + (int64_t)idx * embed_dim;

    /* Float4 vectorised copy when embed_dim is aligned */
    if ((embed_dim & 3) == 0) {
        int ed4 = embed_dim / 4;
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (int j = threadIdx.x; j < ed4; j += blockDim.x) {
            dst4[j] = src4[j];
        }
    } else {
        for (int j = threadIdx.x; j < embed_dim; j += blockDim.x) {
            dst[j] = LDG(&src[j]);
        }
    }
}

extern "C" int cuda_embedding_f32(const float* weight, const int64_t* indices,
                                  float* output, int num_indices, int embed_dim,
                                  cudaStream_t stream) {
    if (num_indices == 0) return 0;
    int block;
    if ((embed_dim & 3) == 0) {
        /* With float4, each thread handles 4 elements, so fewer threads needed */
        int ed4 = embed_dim / 4;
        block = 64;
        if (ed4 > 64) block = 128;
        if (ed4 > 128) block = 256;
    } else {
        block = 256;
        if (embed_dim <= 64) block = 64;
        else if (embed_dim <= 128) block = 128;
    }
    embedding_f32_kernel<<<num_indices, block, 0, stream>>>(
        weight, indices, output, num_indices, embed_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  WHERE (conditional select)
 * ================================================================ */

__global__ void where_f32_kernel(const bool* __restrict__ cond,
                                 const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 float* __restrict__ out, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        out[i] = cond[i] ? LDG(&x[i]) : LDG(&y[i]);
    }
}

extern "C" int cuda_where_f32(const bool* cond, const float* x, const float* y,
                              float* out, int64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    where_f32_kernel<<<grid_size(n, BLOCK_1D), BLOCK_1D, 0, stream>>>(cond, x, y, out, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  TRANSPOSE 2D (tiled, shared memory for coalesced writes)
 *
 *  Uses shared memory tiles to convert row-reads to column-writes
 *  with proper bank-conflict avoidance (+1 padding).
 * ================================================================ */

__global__ void transpose_2d_f32_kernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int rows, int cols) {
    /* Tile with +1 padding to avoid bank conflicts */
    __shared__ float tile[TILE_DIM2][TILE_DIM2 + 1];

    int bx = blockIdx.x * TILE_DIM2;
    int by = blockIdx.y * TILE_DIM2;

    /* Load tile from input (coalesced reads) */
    for (int j = 0; j < TILE_DIM2; j += blockDim.y) {
        int row = by + threadIdx.y + j;
        int col = bx + threadIdx.x;
        if (row < rows && col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = LDG(&in[(int64_t)row * cols + col]);
        }
    }
    __syncthreads();

    /* Write transposed tile (coalesced writes) */
    int out_bx = by;
    int out_by = bx;
    for (int j = 0; j < TILE_DIM2; j += blockDim.y) {
        int row = out_by + threadIdx.y + j;
        int col = out_bx + threadIdx.x;
        if (row < cols && col < rows) {
            out[(int64_t)row * rows + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" int cuda_transpose_2d_f32(const float* in, float* out,
                                     int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return 0;
    dim3 block(TILE_DIM2, 8);  /* 32x8 = 256 threads */
    dim3 grid((cols + TILE_DIM2 - 1) / TILE_DIM2,
              (rows + TILE_DIM2 - 1) / TILE_DIM2);
    transpose_2d_f32_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}


/* ================================================================
 *  HALF PRECISION (FP16) KERNELS — sm_53+
 *
 *  These use __half / half2 intrinsics for 2x throughput on
 *  Pascal+ GPUs. Gated by CUDA_VERSION and __CUDA_ARCH__.
 * ================================================================ */

#if defined(__CUDACC__) && CUDA_VERSION >= 9000

/* FP16 element-wise using half2 vectorisation */
__global__ void relu_f16_kernel(const __half* __restrict__ x,
                                __half* __restrict__ y, int64_t n) {
    int64_t n2 = n / 2;
    const __half2* x2 = reinterpret_cast<const __half2*>(x);
    __half2* y2 = reinterpret_cast<__half2*>(y);
    __half2 zero2 = __float2half2_rn(0.0f);

    GRID_STRIDE_LOOP(i, n2) {
        __half2 v = x2[i];
    #if CUDA_VERSION >= 11000 && __CUDA_ARCH__ >= 530
        y2[i] = __hmax2(v, zero2);
    #else
        float2 f;
        f.x = fmaxf(__half2float(__low2half(v)), 0.0f);
        f.y = fmaxf(__half2float(__high2half(v)), 0.0f);
        y2[i] = __float22half2_rn(f);
    #endif
    }
    /* Handle odd element */
    if (n % 2 == 1) {
        int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0) {
            float v = __half2float(x[n - 1]);
            y[n - 1] = __float2half(fmaxf(v, 0.0f));
        }
    }
}

__global__ void sigmoid_f16_kernel(const __half* __restrict__ x,
                                   __half* __restrict__ y, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        float v = __half2float(x[i]);
        y[i] = __float2half(1.0f / (1.0f + expf(-v)));
    }
}

#endif /* CUDA_VERSION >= 9000 */


/* ================================================================
 *  BFLOAT16 KERNELS — sm_80+ (Ampere)
 * ================================================================ */

#if defined(__CUDACC__) && CUDA_VERSION >= 11000

__global__ void relu_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y, int64_t n) {
    GRID_STRIDE_LOOP(i, n) {
        float v = __bfloat162float(x[i]);
        y[i] = __float2bfloat16(fmaxf(v, 0.0f));
    }
}

#endif /* CUDA_VERSION >= 11000 */


/* ================================================================
 *  TF32 SUPPORT (Ampere+)
 *
 *  TF32 is handled by cuBLAS automatically when:
 *    1. CUBLAS_TF32_TENSOR_OP_MATH is set (CUDA 11+)
 *    2. Input matrices are float32
 *    3. Compute capability >= 8.0
 *
 *  We provide a function to enable/disable it on the cuBLAS handle.
 * ================================================================ */

extern "C" int cuda_set_tf32(cublasHandle_t handle, int enable) {
#if CUDA_VERSION >= 11000
    cublasSetMathMode(handle,
        enable ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
#else
    (void)handle; (void)enable;
#endif
    return 0;
}
