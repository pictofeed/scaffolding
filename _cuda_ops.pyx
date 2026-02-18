# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: nonecheck=False, initializedcheck=False
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
_cuda_ops.pyx — Cython bindings for CUDA kernels.

Provides Python-callable wrappers around the extern "C" functions
defined in _cuda_kernels.cu.  Each wrapper:
  1. Accepts NumPy arrays (CPU) or raw device pointers (int)
  2. Manages GPU memory (H2D / D2H copies when needed)
  3. Calls the CUDA kernel on the correct stream
  4. Returns results as NumPy arrays or device buffers

All device pointer manipulation uses <uintptr_t> casts — zero overhead.
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

np.import_array()

# ================================================================
#  CUDA Runtime extern declarations
# ================================================================

cdef extern from "cuda_runtime.h" nogil:
    ctypedef void* cudaStream_t
    ctypedef int cudaError_t

    # Device management
    cudaError_t cudaGetDeviceCount(int* count)
    cudaError_t cudaSetDevice(int device)
    cudaError_t cudaGetDevice(int* device)
    cudaError_t cudaDeviceSynchronize()
    cudaError_t cudaStreamSynchronize(cudaStream_t stream)

    # Memory management
    cudaError_t cudaMalloc(void** devPtr, size_t size)
    cudaError_t cudaFree(void* devPtr)
    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream)
    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream)
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                int kind, cudaStream_t stream)
    cudaError_t cudaMemset(void* devPtr, int value, size_t count)
    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                                cudaStream_t stream)
    cudaError_t cudaMemGetInfo(size_t* free_mem, size_t* total_mem)

    # Stream management
    cudaError_t cudaStreamCreate(cudaStream_t* pStream)
    cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
    cudaError_t cudaStreamDestroy(cudaStream_t stream)

    # Event management
    ctypedef void* cudaEvent_t
    cudaError_t cudaEventCreate(cudaEvent_t* event)
    cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
    cudaError_t cudaEventDestroy(cudaEvent_t event)
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    cudaError_t cudaEventSynchronize(cudaEvent_t event)
    cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                    unsigned int flags)

    # Device properties
    struct cudaDeviceProp:
        char name[256]
        size_t totalGlobalMem
        int major
        int minor
        int multiProcessorCount
        size_t sharedMemPerBlock
        int maxThreadsPerBlock
        int warpSize
        size_t totalConstMem
        int maxThreadsPerMultiProcessor

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)

    # Peer access
    cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

    # Error handling
    const char* cudaGetErrorString(cudaError_t error)

    # Memcpy kinds
    enum:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2
        cudaMemcpyDeviceToDevice = 3

    # Stream flags
    enum:
        cudaStreamDefault = 0
        cudaStreamNonBlocking = 1

    # Event flags
    enum:
        cudaEventDefault = 0
        cudaEventDisableTiming = 2

    # Success
    enum:
        cudaSuccess = 0


# ================================================================
#  cuBLAS extern declarations
# ================================================================

cdef extern from "cublas_v2.h" nogil:
    ctypedef void* cublasHandle_t
    ctypedef int cublasStatus_t

    cublasStatus_t cublasCreate(cublasHandle_t* handle)
    cublasStatus_t cublasDestroy(cublasHandle_t handle)
    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
    cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId)
    # Math mode for TF32
    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode)

    enum:
        CUBLAS_STATUS_SUCCESS = 0


# ================================================================
#  CUDA kernel function declarations (from _cuda_kernels.cu)
# ================================================================

cdef extern from "_cuda_ops_decl.h" nogil:
    # --- Element-wise unary ---
    int cuda_exp_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_log_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_sqrt_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_rsqrt_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_sigmoid_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_tanh_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_sin_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_cos_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_neg_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_abs_f32(const float* x, float* y, int64_t n, cudaStream_t stream)

    # --- Element-wise binary ---
    int cuda_add_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s)
    int cuda_sub_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s)
    int cuda_mul_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s)
    int cuda_div_f32(const float* a, const float* b, float* c, int64_t n, cudaStream_t s)

    # --- Scalar broadcast ---
    int cuda_adds_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s)
    int cuda_muls_f32(const float* a, float scalar, float* c, int64_t n, cudaStream_t s)

    # --- Activations ---
    int cuda_relu_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_relu_backward_f32(const float* grad, const float* x, float* dx,
                               int64_t n, cudaStream_t stream)
    int cuda_silu_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_silu_fwd_f32(const float* x, float* y, float* sig, int64_t n,
                          cudaStream_t stream)
    int cuda_silu_backward_f32(const float* grad, const float* x, const float* sig,
                               float* dx, int64_t n, cudaStream_t stream)
    int cuda_gelu_f32(const float* x, float* y, int64_t n, cudaStream_t stream)
    int cuda_gelu_backward_f32(const float* grad, const float* x,
                               float* dx, int64_t n, cudaStream_t stream)

    # --- Clamp ---
    int cuda_clamp_f32(const float* x, float* y, float lo, float hi,
                       int64_t n, cudaStream_t stream)

    # --- Softmax ---
    int cuda_softmax_f32(const float* x, float* y, int rows, int cols,
                         cudaStream_t stream)
    int cuda_log_softmax_f32(const float* x, float* y, int rows, int cols,
                             cudaStream_t stream)

    # --- Cross-entropy ---
    int cuda_cross_entropy_fwd_f32(const float* logits, const int64_t* targets,
                                   float* losses, float* probs,
                                   int N, int C, cudaStream_t stream)

    # --- Reductions ---
    int cuda_sum_f32(const float* x, float* out, int64_t n, cudaStream_t stream)
    int cuda_sum_dim_f32(const float* x, float* out, int64_t outer,
                         int64_t dim_size, int64_t inner, cudaStream_t stream)

    # --- Normalization ---
    int cuda_layer_norm_f32(const float* x, const float* gamma, const float* beta,
                            float* y, float* mean, float* rstd,
                            int N, int D, float eps, cudaStream_t stream)
    int cuda_rms_norm_f32(const float* x, const float* gamma,
                          float* y, float* rstd,
                          int N, int D, float eps, cudaStream_t stream)

    # --- GEMM ---
    int cuda_sgemm(cublasHandle_t handle,
                   int M, int N, int K,
                   const float* A, const float* B, float* C,
                   float alpha, float beta,
                   bint transA, bint transB)
    int cuda_sgemm_batched(cublasHandle_t handle,
                           int M, int N, int K,
                           const float* A, const float* B, float* C,
                           float alpha, float beta,
                           int batchCount, int64_t strideA, int64_t strideB,
                           int64_t strideC)

    # --- AdamW ---
    int cuda_adamw_step_f32(float* param, const float* grad,
                            float* m, float* v,
                            float lr, float beta1, float beta2,
                            float eps, float wd, float bc1, float bc2,
                            int64_t n, cudaStream_t stream)

    # --- Memory utilities ---
    int cuda_fill_f32(float* ptr, float value, int64_t n, cudaStream_t stream)
    int cuda_copy_f32(const float* src, float* dst, int64_t n, cudaStream_t stream)

    # --- Dropout ---
    int cuda_dropout_f32(const float* x, float* y, float* mask,
                         float p, int64_t n,
                         unsigned long long seed, cudaStream_t stream)

    # --- Embedding ---
    int cuda_embedding_f32(const float* weight, const int64_t* indices,
                           float* output, int num_indices, int embed_dim,
                           cudaStream_t stream)

    # --- Where ---
    int cuda_where_f32(const bint* cond, const float* x, const float* y,
                       float* out, int64_t n, cudaStream_t stream)

    # --- Transpose ---
    int cuda_transpose_2d_f32(const float* inp, float* out, int rows, int cols,
                              cudaStream_t stream)

    # --- TF32 control ---
    int cuda_set_tf32(cublasHandle_t handle, int enable)


# ================================================================
#  HELPER: Check CUDA errors
# ================================================================

cdef inline void _check_cuda(cudaError_t err) except *:
    cdef const char* msg
    if err != cudaSuccess:
        msg = cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error: {msg.decode('utf-8')}")

cdef inline void _check_kernel(int ret) except *:
    if ret != 0:
        raise RuntimeError("CUDA kernel launch failed")


# ================================================================
#  GPU MEMORY BUFFER — wraps a device pointer with ref-counting
# ================================================================

cdef class CudaBuffer:
    """Holds a device pointer + metadata. Python-ref-counted (freed on __dealloc__)."""
    cdef void* ptr
    cdef size_t nbytes
    cdef int device_id
    cdef bint owns_memory

    def __cinit__(self):
        self.ptr = NULL
        self.nbytes = 0
        self.device_id = 0
        self.owns_memory = False

    def __dealloc__(self):
        if self.owns_memory and self.ptr != NULL:
            cudaFree(self.ptr)
            self.ptr = NULL

    @property
    def data_ptr(self):
        return <uintptr_t>self.ptr

    @property
    def size(self):
        return self.nbytes


cdef CudaBuffer _make_buffer(size_t nbytes, int device_id):
    """Allocate a new device buffer."""
    cdef CudaBuffer buf = CudaBuffer.__new__(CudaBuffer)
    cdef cudaError_t err
    err = cudaMalloc(&buf.ptr, nbytes)
    _check_cuda(err)
    buf.nbytes = nbytes
    buf.device_id = device_id
    buf.owns_memory = True
    return buf

cdef CudaBuffer _wrap_ptr(void* ptr, size_t nbytes, int device_id):
    """Wrap an existing device pointer (no ownership)."""
    cdef CudaBuffer buf = CudaBuffer.__new__(CudaBuffer)
    buf.ptr = ptr
    buf.nbytes = nbytes
    buf.device_id = device_id
    buf.owns_memory = False
    return buf


# ================================================================
#  DEVICE MANAGEMENT
# ================================================================

def get_device_count():
    """Return number of CUDA devices."""
    cdef int count = 0
    _check_cuda(cudaGetDeviceCount(&count))
    return count

def set_device(int device):
    """Set the current CUDA device."""
    _check_cuda(cudaSetDevice(device))

def get_current_device():
    """Get the current CUDA device index."""
    cdef int device = 0
    _check_cuda(cudaGetDevice(&device))
    return device

def device_synchronize():
    """Synchronize the current device."""
    _check_cuda(cudaDeviceSynchronize())

def get_device_properties(int device):
    """Get device properties as a dict."""
    cdef cudaDeviceProp prop
    _check_cuda(cudaGetDeviceProperties(&prop, device))
    return {
        'name': prop.name.decode('utf-8'),
        'total_memory': prop.totalGlobalMem,
        'major': prop.major,
        'minor': prop.minor,
        'multi_processor_count': prop.multiProcessorCount,
        'shared_mem_per_block': prop.sharedMemPerBlock,
        'max_threads_per_block': prop.maxThreadsPerBlock,
        'warp_size': prop.warpSize,
        'max_threads_per_mp': prop.maxThreadsPerMultiProcessor,
    }

def get_device_capability(int device=0):
    """Return (major, minor) compute capability."""
    cdef cudaDeviceProp prop
    _check_cuda(cudaGetDeviceProperties(&prop, device))
    return (prop.major, prop.minor)

def get_device_name(int device=0):
    """Return device name string."""
    cdef cudaDeviceProp prop
    _check_cuda(cudaGetDeviceProperties(&prop, device))
    return prop.name.decode('utf-8')

def mem_info():
    """Return (free, total) memory in bytes."""
    cdef size_t free_mem = 0, total_mem = 0
    _check_cuda(cudaMemGetInfo(&free_mem, &total_mem))
    return (free_mem, total_mem)


# ================================================================
#  STREAM MANAGEMENT
# ================================================================

cdef class CudaStream:
    """Wraps a cudaStream_t with Python lifetime management."""
    cdef cudaStream_t _stream
    cdef bint _owns
    cdef int _device_id

    def __cinit__(self, int device_id=0, bint non_blocking=True):
        self._device_id = device_id
        self._owns = True
        cdef unsigned int flags = cudaStreamNonBlocking if non_blocking else cudaStreamDefault
        _check_cuda(cudaStreamCreateWithFlags(&self._stream, flags))

    def __dealloc__(self):
        if self._owns and self._stream != NULL:
            cudaStreamDestroy(self._stream)

    def synchronize(self):
        _check_cuda(cudaStreamSynchronize(self._stream))

    def wait_event(self, CudaEvent event):
        _check_cuda(cudaStreamWaitEvent(self._stream, event._event, 0))

    @property
    def cuda_stream(self):
        return <uintptr_t>self._stream

    @property
    def device_id(self):
        return self._device_id


cdef class CudaEvent:
    """Wraps a cudaEvent_t for timing and synchronisation."""
    cdef cudaEvent_t _event
    cdef bint _owns

    def __cinit__(self, bint enable_timing=True):
        self._owns = True
        cdef unsigned int flags = cudaEventDefault if enable_timing else cudaEventDisableTiming
        _check_cuda(cudaEventCreateWithFlags(&self._event, flags))

    def __dealloc__(self):
        if self._owns and self._event != NULL:
            cudaEventDestroy(self._event)

    def record(self, CudaStream stream=None):
        cdef cudaStream_t s = NULL
        if stream is not None:
            s = stream._stream
        _check_cuda(cudaEventRecord(self._event, s))

    def synchronize(self):
        _check_cuda(cudaEventSynchronize(self._event))

    def elapsed_time(self, CudaEvent end):
        """Return elapsed time in milliseconds between self (start) and end."""
        cdef float ms = 0.0
        _check_cuda(cudaEventElapsedTime(&ms, self._event, end._event))
        return ms


# Default stream (NULL)
cdef cudaStream_t _default_stream = NULL

cdef cudaStream_t _get_stream(stream) except *:
    """Extract cudaStream_t from a CudaStream or None."""
    if stream is None:
        return _default_stream
    if isinstance(stream, CudaStream):
        return (<CudaStream>stream)._stream
    raise TypeError(f"Expected CudaStream or None, got {type(stream)}")


# ================================================================
#  cuBLAS HANDLE MANAGEMENT
# ================================================================

cdef cublasHandle_t _cublas_handle = NULL
cdef bint _cublas_init = False

cdef cublasHandle_t _get_cublas() except *:
    global _cublas_handle, _cublas_init
    cdef cublasStatus_t status
    if not _cublas_init:
        status = cublasCreate(&_cublas_handle)
        if status != CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"cuBLAS init failed: {status}")
        _cublas_init = True
    return _cublas_handle

def cublas_set_stream(stream=None):
    """Set the cuBLAS handle stream."""
    cdef cublasHandle_t h = _get_cublas()
    cdef cudaStream_t s = _get_stream(stream)
    cublasSetStream(h, s)


# ================================================================
#  GPU MEMORY ALLOCATION (Python API)
# ================================================================

def gpu_malloc(size_t nbytes):
    """Allocate GPU memory, return CudaBuffer."""
    cdef int device = 0
    cudaGetDevice(&device)
    return _make_buffer(nbytes, device)

def gpu_free(CudaBuffer buf):
    """Free a CudaBuffer (also happens automatically on GC)."""
    if buf.owns_memory and buf.ptr != NULL:
        _check_cuda(cudaFree(buf.ptr))
        buf.ptr = NULL
        buf.owns_memory = False

def gpu_memcpy_h2d(CudaBuffer dst, np.ndarray src not None, stream=None):
    """Copy from host (NumPy) to device."""
    cdef cudaStream_t s = _get_stream(stream)
    cdef np.ndarray flat = np.ascontiguousarray(src).ravel()
    cdef size_t nbytes = flat.nbytes
    _check_cuda(cudaMemcpyAsync(dst.ptr, <void*>np.PyArray_DATA(flat),
                                nbytes, cudaMemcpyHostToDevice, s))

def gpu_memcpy_d2h(np.ndarray dst not None, CudaBuffer src, stream=None):
    """Copy from device to host (NumPy) ."""
    cdef cudaStream_t s = _get_stream(stream)
    cdef np.ndarray flat = np.ascontiguousarray(dst).ravel()
    cdef size_t nbytes = flat.nbytes
    _check_cuda(cudaMemcpyAsync(<void*>np.PyArray_DATA(flat), src.ptr,
                                nbytes, cudaMemcpyDeviceToHost, s))

def gpu_memcpy_d2d(CudaBuffer dst, CudaBuffer src, size_t nbytes, stream=None):
    """Copy device to device."""
    cdef cudaStream_t s = _get_stream(stream)
    _check_cuda(cudaMemcpyAsync(dst.ptr, src.ptr, nbytes,
                                cudaMemcpyDeviceToDevice, s))


# ================================================================
#  NUMPY ↔ GPU CONVENIENCE
# ================================================================

def numpy_to_gpu(np.ndarray arr not None, stream=None):
    """Upload a NumPy array to GPU. Returns (CudaBuffer, shape, dtype)."""
    cdef np.ndarray flat = np.ascontiguousarray(arr).ravel()
    cdef size_t nbytes = flat.nbytes
    cdef int device = 0
    cudaGetDevice(&device)
    cdef CudaBuffer buf = _make_buffer(nbytes, device)
    cdef cudaStream_t s = _get_stream(stream)
    _check_cuda(cudaMemcpyAsync(buf.ptr, <void*>np.PyArray_DATA(flat),
                                nbytes, cudaMemcpyHostToDevice, s))
    return buf, (<object>arr).shape, arr.dtype

def gpu_to_numpy(CudaBuffer buf, tuple shape, dtype, stream=None):
    """Download GPU buffer to a NumPy array."""
    cdef np.ndarray result = np.empty(shape, dtype=dtype)
    cdef cudaStream_t s = _get_stream(stream)
    cdef size_t nbytes = result.nbytes
    _check_cuda(cudaMemcpyAsync(<void*>np.PyArray_DATA(result), buf.ptr,
                                nbytes, cudaMemcpyDeviceToHost, s))
    # Must sync before Python sees the data
    _check_cuda(cudaStreamSynchronize(s))
    return result


# ================================================================
#  GpuTensor — GPU-RESIDENT TENSOR (CudaBuffer + shape + dtype)
#
#  Keeps data on device. Eliminates H2D/D2H per-op overhead.
#  Upload once, run many ops, download once.
# ================================================================

class GpuTensor:
    """GPU-resident tensor wrapping a CudaBuffer with shape/dtype metadata."""
    __slots__ = ('buffer', 'shape', 'dtype', 'numel')

    def __init__(self, CudaBuffer buffer, tuple shape, dtype):
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
        cdef int64_t n = 1
        for s in shape:
            n *= s
        self.numel = n


def gputensor_from_numpy(np.ndarray arr not None):
    """Upload NumPy array to GPU, return GpuTensor."""
    cdef np.ndarray flat = np.ascontiguousarray(arr).ravel()
    cdef size_t nbytes = flat.nbytes
    cdef int device = 0
    cudaGetDevice(&device)
    cdef CudaBuffer buf = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(buf.ptr, <void*>np.PyArray_DATA(flat),
                           nbytes, cudaMemcpyHostToDevice))
    return GpuTensor(buf, (<object>arr).shape, arr.dtype)


def gputensor_to_numpy(gt):
    """Download GpuTensor to NumPy array (sync)."""
    cdef CudaBuffer buf = <CudaBuffer>(gt.buffer)
    cdef np.ndarray result = np.empty(gt.shape, dtype=gt.dtype)
    cdef size_t nbytes = result.nbytes
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), buf.ptr,
                           nbytes, cudaMemcpyDeviceToHost))
    return result


cdef CudaBuffer _gt_alloc(int64_t numel):
    """Alloc an f32 device buffer for numel elements."""
    cdef int device = 0
    cudaGetDevice(&device)
    return _make_buffer(numel * sizeof(float), device)


cdef CudaBuffer _gt_alloc_bytes(size_t nbytes):
    """Alloc a device buffer for given bytes."""
    cdef int device = 0
    cudaGetDevice(&device)
    return _make_buffer(nbytes, device)


# ────────────────────────────────────────────────────────────────
#  DEVICE-RESIDENT OPS  (GpuTensor → GpuTensor, zero transfer)
# ────────────────────────────────────────────────────────────────

# --- Unary ops ---

cdef _dev_unary(gt, int (*kernel)(const float*, float*, int64_t, cudaStream_t) noexcept nogil):
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(n)
    cdef int ret
    with nogil:
        ret = kernel(<float*>src.ptr, <float*>dst.ptr, n, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, gt.shape, gt.dtype)

def dev_exp(gt):
    return _dev_unary(gt, cuda_exp_f32)

def dev_log(gt):
    return _dev_unary(gt, cuda_log_f32)

def dev_sqrt(gt):
    return _dev_unary(gt, cuda_sqrt_f32)

def dev_rsqrt(gt):
    return _dev_unary(gt, cuda_rsqrt_f32)

def dev_sigmoid(gt):
    return _dev_unary(gt, cuda_sigmoid_f32)

def dev_tanh(gt):
    return _dev_unary(gt, cuda_tanh_f32)

def dev_sin(gt):
    return _dev_unary(gt, cuda_sin_f32)

def dev_cos(gt):
    return _dev_unary(gt, cuda_cos_f32)

def dev_neg(gt):
    return _dev_unary(gt, cuda_neg_f32)

def dev_abs(gt):
    return _dev_unary(gt, cuda_abs_f32)

def dev_relu(gt):
    return _dev_unary(gt, cuda_relu_f32)

def dev_silu(gt):
    return _dev_unary(gt, cuda_silu_f32)

def dev_gelu(gt):
    return _dev_unary(gt, cuda_gelu_f32)


# --- Binary ops ---

cdef _dev_binary(gt_a, gt_b, int (*kernel)(const float*, const float*, float*, int64_t, cudaStream_t) noexcept nogil):
    cdef CudaBuffer a = <CudaBuffer>(gt_a.buffer)
    cdef CudaBuffer b = <CudaBuffer>(gt_b.buffer)
    cdef int64_t n = gt_a.numel
    cdef CudaBuffer c = _gt_alloc(n)
    cdef int ret
    with nogil:
        ret = kernel(<float*>a.ptr, <float*>b.ptr, <float*>c.ptr, n, _default_stream)
    _check_kernel(ret)
    return GpuTensor(c, gt_a.shape, gt_a.dtype)

def dev_add(gt_a, gt_b):
    return _dev_binary(gt_a, gt_b, cuda_add_f32)

def dev_sub(gt_a, gt_b):
    return _dev_binary(gt_a, gt_b, cuda_sub_f32)

def dev_mul(gt_a, gt_b):
    return _dev_binary(gt_a, gt_b, cuda_mul_f32)

def dev_div(gt_a, gt_b):
    return _dev_binary(gt_a, gt_b, cuda_div_f32)


# --- Scalar broadcast ops ---

def dev_adds(gt, float scalar):
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(n)
    cdef int ret
    with nogil:
        ret = cuda_adds_f32(<float*>src.ptr, scalar, <float*>dst.ptr, n, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, gt.shape, gt.dtype)

def dev_muls(gt, float scalar):
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(n)
    cdef int ret
    with nogil:
        ret = cuda_muls_f32(<float*>src.ptr, scalar, <float*>dst.ptr, n, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, gt.shape, gt.dtype)


# --- Clamp ---

def dev_clamp(gt, float lo, float hi):
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(n)
    cdef int ret
    with nogil:
        ret = cuda_clamp_f32(<float*>src.ptr, <float*>dst.ptr, lo, hi, n, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, gt.shape, gt.dtype)


# --- Softmax ---

def dev_softmax(gt, int dim=-1):
    """Softmax along last dim, device-resident."""
    cdef tuple orig_shape = gt.shape
    cdef int ndim = len(orig_shape)
    cdef int rows, cols
    if ndim >= 2 and (dim == -1 or dim == ndim - 1):
        cols = orig_shape[ndim - 1]
        rows = gt.numel // cols
    else:
        cols = orig_shape[ndim - 1]
        rows = gt.numel // cols

    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef CudaBuffer dst = _gt_alloc(gt.numel)
    cdef int ret
    with nogil:
        ret = cuda_softmax_f32(<float*>src.ptr, <float*>dst.ptr, rows, cols, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, orig_shape, gt.dtype)


def dev_log_softmax(gt, int dim=-1):
    """Log-softmax along last dim, device-resident."""
    cdef tuple orig_shape = gt.shape
    cdef int ndim = len(orig_shape)
    cdef int rows, cols
    cols = orig_shape[ndim - 1]
    rows = gt.numel // cols

    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef CudaBuffer dst = _gt_alloc(gt.numel)
    cdef int ret
    with nogil:
        ret = cuda_log_softmax_f32(<float*>src.ptr, <float*>dst.ptr, rows, cols, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, orig_shape, gt.dtype)


# --- Cross-entropy ---

def dev_cross_entropy(gt_logits, gt_targets):
    """Fused cross-entropy on device. Returns (float_loss, GpuTensor_probs)."""
    cdef CudaBuffer logits_buf = <CudaBuffer>(gt_logits.buffer)
    cdef CudaBuffer targets_buf = <CudaBuffer>(gt_targets.buffer)
    cdef int N = gt_logits.shape[0]
    cdef int C = gt_logits.shape[1]
    cdef size_t loss_bytes = N * sizeof(float)
    cdef size_t logit_bytes = N * C * sizeof(float)

    cdef CudaBuffer d_losses = _gt_alloc(N)
    cdef CudaBuffer d_probs = _gt_alloc(N * C)

    cdef int ret
    with nogil:
        ret = cuda_cross_entropy_fwd_f32(
            <float*>logits_buf.ptr, <int64_t*>targets_buf.ptr,
            <float*>d_losses.ptr, <float*>d_probs.ptr,
            N, C, _default_stream)
    _check_kernel(ret)

    # Download losses to compute mean (small array)
    cdef np.ndarray losses = np.empty(N, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(losses), d_losses.ptr,
                           loss_bytes, cudaMemcpyDeviceToHost))
    cdef float mean_loss = float(np.mean(losses))
    cdef GpuTensor probs_gt = GpuTensor(d_probs, (N, C), np.float32)
    return mean_loss, probs_gt


# --- GEMM (cuBLAS) ---

def dev_matmul(gt_a, gt_b):
    """C = A @ B via cuBLAS, device-resident. A (M,K), B (K,N)."""
    cdef CudaBuffer a = <CudaBuffer>(gt_a.buffer)
    cdef CudaBuffer b = <CudaBuffer>(gt_b.buffer)
    cdef int M = gt_a.shape[0]
    cdef int K = gt_a.shape[1]
    cdef int N = gt_b.shape[1]
    cdef CudaBuffer c = _gt_alloc(M * N)

    cdef cublasHandle_t handle = _get_cublas()
    cdef int ret = cuda_sgemm(handle, M, N, K,
                              <float*>a.ptr, <float*>b.ptr, <float*>c.ptr,
                              1.0, 0.0, False, False)
    _check_kernel(ret)
    return GpuTensor(c, (M, N), np.float32)


def dev_matmul_nt(gt_a, gt_b):
    """C = A @ B^T via cuBLAS, device-resident. A (M,K), B (N,K)."""
    cdef CudaBuffer a = <CudaBuffer>(gt_a.buffer)
    cdef CudaBuffer b = <CudaBuffer>(gt_b.buffer)
    cdef int M = gt_a.shape[0]
    cdef int K = gt_a.shape[1]
    cdef int N = gt_b.shape[0]
    cdef CudaBuffer c = _gt_alloc(M * N)

    cdef cublasHandle_t handle = _get_cublas()
    cdef int ret = cuda_sgemm(handle, M, N, K,
                              <float*>a.ptr, <float*>b.ptr, <float*>c.ptr,
                              1.0, 0.0, False, True)
    _check_kernel(ret)
    return GpuTensor(c, (M, N), np.float32)


def dev_batched_matmul(gt_a, gt_b):
    """Batched matmul on device. a: (..., M, K), b: (..., K, N)."""
    cdef tuple a_shape = gt_a.shape
    cdef tuple b_shape = gt_b.shape
    cdef int ndim_a = len(a_shape)
    cdef int ndim_b = len(b_shape)
    cdef int M = a_shape[ndim_a - 2]
    cdef int K = a_shape[ndim_a - 1]
    cdef int N_ = b_shape[ndim_b - 1]
    cdef int batch = 1
    cdef int i
    for i in range(ndim_a - 2):
        batch *= a_shape[i]

    cdef CudaBuffer a = <CudaBuffer>(gt_a.buffer)
    cdef CudaBuffer b = <CudaBuffer>(gt_b.buffer)
    cdef CudaBuffer c = _gt_alloc(batch * M * N_)

    cdef cublasHandle_t handle = _get_cublas()
    cdef int64_t strideA = M * K
    cdef int64_t strideB = K * N_
    cdef int64_t strideC = M * N_
    cdef int ret = cuda_sgemm_batched(handle, M, N_, K,
                                      <float*>a.ptr, <float*>b.ptr, <float*>c.ptr,
                                      1.0, 0.0, batch, strideA, strideB, strideC)
    _check_kernel(ret)
    cdef tuple out_shape = a_shape[:ndim_a-2] + (M, N_)
    return GpuTensor(c, out_shape, np.float32)


# --- Layer norm ---

def dev_layer_norm(gt_x, gt_gamma, gt_beta, float eps=1e-5):
    """Layer norm on device. Input (N, D)."""
    cdef CudaBuffer x_buf = <CudaBuffer>(gt_x.buffer)
    cdef int N = gt_x.shape[0]
    cdef int D = gt_x.shape[1]
    cdef CudaBuffer y_buf = _gt_alloc(N * D)
    cdef CudaBuffer mean_buf = _gt_alloc(N)
    cdef CudaBuffer rstd_buf = _gt_alloc(N)
    cdef float* gptr = NULL
    cdef float* bptr = NULL
    if gt_gamma is not None:
        gptr = <float*>(<CudaBuffer>(gt_gamma.buffer)).ptr
    if gt_beta is not None:
        bptr = <float*>(<CudaBuffer>(gt_beta.buffer)).ptr

    cdef int ret
    with nogil:
        ret = cuda_layer_norm_f32(<float*>x_buf.ptr, gptr, bptr,
                                  <float*>y_buf.ptr, <float*>mean_buf.ptr, <float*>rstd_buf.ptr,
                                  N, D, eps, _default_stream)
    _check_kernel(ret)
    return GpuTensor(y_buf, gt_x.shape, np.float32)


# --- RMS norm ---

def dev_rms_norm(gt_x, gt_gamma, float eps=1e-5):
    """RMS norm on device. Input (N, D)."""
    cdef CudaBuffer x_buf = <CudaBuffer>(gt_x.buffer)
    cdef int N = gt_x.shape[0]
    cdef int D = gt_x.shape[1]
    cdef CudaBuffer y_buf = _gt_alloc(N * D)
    cdef CudaBuffer rstd_buf = _gt_alloc(N)
    cdef float* gptr = NULL
    if gt_gamma is not None:
        gptr = <float*>(<CudaBuffer>(gt_gamma.buffer)).ptr

    cdef int ret
    with nogil:
        ret = cuda_rms_norm_f32(<float*>x_buf.ptr, gptr,
                                <float*>y_buf.ptr, <float*>rstd_buf.ptr,
                                N, D, eps, _default_stream)
    _check_kernel(ret)
    return GpuTensor(y_buf, gt_x.shape, np.float32)


# --- Dropout ---

def dev_dropout(gt, float p, unsigned long long seed=0):
    """Dropout on device. Returns (output_GpuTensor, mask_GpuTensor)."""
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(n)
    cdef CudaBuffer mask_buf = _gt_alloc(n)

    cdef int ret
    with nogil:
        ret = cuda_dropout_f32(<float*>src.ptr, <float*>dst.ptr, <float*>mask_buf.ptr,
                               p, n, seed, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, gt.shape, gt.dtype), GpuTensor(mask_buf, gt.shape, gt.dtype)


# --- Embedding ---

def dev_embedding(gt_weight, gt_indices, int embed_dim):
    """Embedding lookup on device."""
    cdef CudaBuffer w_buf = <CudaBuffer>(gt_weight.buffer)
    cdef CudaBuffer idx_buf = <CudaBuffer>(gt_indices.buffer)
    cdef int num_idx = gt_indices.numel
    cdef CudaBuffer out_buf = _gt_alloc(num_idx * embed_dim)

    cdef int ret
    with nogil:
        ret = cuda_embedding_f32(<float*>w_buf.ptr, <int64_t*>idx_buf.ptr,
                                 <float*>out_buf.ptr, num_idx, embed_dim, _default_stream)
    _check_kernel(ret)
    cdef tuple out_shape = gt_indices.shape + (embed_dim,)
    return GpuTensor(out_buf, out_shape, np.float32)


# --- Linear forward ---

def dev_linear_forward(gt_x, gt_weight, gt_bias=None):
    """y = x @ W^T + bias on device."""
    cdef int M = gt_x.numel // gt_x.shape[len(gt_x.shape) - 1]
    cdef int K = gt_x.shape[len(gt_x.shape) - 1]
    cdef int N = gt_weight.shape[0]
    cdef CudaBuffer x_buf = <CudaBuffer>(gt_x.buffer)
    cdef CudaBuffer w_buf = <CudaBuffer>(gt_weight.buffer)
    cdef CudaBuffer c_buf = _gt_alloc(M * N)
    cdef CudaBuffer b_buf
    cdef np.ndarray c_np
    cdef np.ndarray b_np
    cdef tuple out_shape

    cdef cublasHandle_t handle = _get_cublas()
    cdef int ret = cuda_sgemm(handle, M, N, K,
                              <float*>x_buf.ptr, <float*>w_buf.ptr, <float*>c_buf.ptr,
                              1.0, 0.0, False, True)
    _check_kernel(ret)

    # Add bias if present
    if gt_bias is not None:
        b_buf = <CudaBuffer>(gt_bias.buffer)
        c_np = np.empty((M, N), dtype=np.float32)
        _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(c_np), c_buf.ptr,
                               M * N * sizeof(float), cudaMemcpyDeviceToHost))
        b_np = np.empty(N, dtype=np.float32)
        _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(b_np), b_buf.ptr,
                               N * sizeof(float), cudaMemcpyDeviceToHost))
        c_np += b_np
        _check_cuda(cudaMemcpy(c_buf.ptr, <void*>np.PyArray_DATA(c_np),
                               M * N * sizeof(float), cudaMemcpyHostToDevice))

    out_shape = gt_x.shape[:-1] + (N,)
    return GpuTensor(c_buf, out_shape, np.float32)


# --- AdamW fused step ---

def dev_adamw_step(gt_param, gt_grad, gt_m, gt_v,
                   float lr, float beta1, float beta2,
                   float eps, float wd, float bc1, float bc2):
    """Fused AdamW step on device. Modifies param/m/v in-place (their buffers)."""
    cdef CudaBuffer p = <CudaBuffer>(gt_param.buffer)
    cdef CudaBuffer g = <CudaBuffer>(gt_grad.buffer)
    cdef CudaBuffer m = <CudaBuffer>(gt_m.buffer)
    cdef CudaBuffer v = <CudaBuffer>(gt_v.buffer)
    cdef int64_t n = gt_param.numel

    cdef int ret
    with nogil:
        ret = cuda_adamw_step_f32(<float*>p.ptr, <float*>g.ptr,
                                  <float*>m.ptr, <float*>v.ptr,
                                  lr, beta1, beta2, eps, wd, bc1, bc2,
                                  n, _default_stream)
    _check_kernel(ret)
    # In-place: buffers modified. Caller keeps same GpuTensors.


# --- Sum reduction ---

def dev_sum(gt):
    """Global sum on device, returns Python float."""
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef CudaBuffer dst = _gt_alloc(1)
    cdef int ret
    with nogil:
        ret = cuda_sum_f32(<float*>src.ptr, <float*>dst.ptr, n, _default_stream)
    _check_kernel(ret)
    cdef float result = 0.0
    _check_cuda(cudaMemcpy(&result, dst.ptr, sizeof(float), cudaMemcpyDeviceToHost))
    return float(result)


# --- Fill / Copy utilities ---

def dev_fill(gt, float value):
    """Fill GpuTensor with scalar value (in-place)."""
    cdef CudaBuffer buf = <CudaBuffer>(gt.buffer)
    cdef int64_t n = gt.numel
    cdef int ret
    with nogil:
        ret = cuda_fill_f32(<float*>buf.ptr, value, n, _default_stream)
    _check_kernel(ret)

def dev_copy(gt_src, gt_dst):
    """Copy device → device."""
    cdef CudaBuffer src = <CudaBuffer>(gt_src.buffer)
    cdef CudaBuffer dst = <CudaBuffer>(gt_dst.buffer)
    cdef int64_t n = gt_src.numel
    cdef int ret
    with nogil:
        ret = cuda_copy_f32(<float*>src.ptr, <float*>dst.ptr, n, _default_stream)
    _check_kernel(ret)


# --- Transpose ---

def dev_transpose_2d(gt):
    """2D transpose on device."""
    cdef CudaBuffer src = <CudaBuffer>(gt.buffer)
    cdef int rows = gt.shape[0]
    cdef int cols = gt.shape[1]
    cdef CudaBuffer dst = _gt_alloc(rows * cols)
    cdef int ret
    with nogil:
        ret = cuda_transpose_2d_f32(<float*>src.ptr, <float*>dst.ptr, rows, cols, _default_stream)
    _check_kernel(ret)
    return GpuTensor(dst, (cols, rows), gt.dtype)


# ================================================================
#  HIGH-LEVEL KERNEL WRAPPERS — accept/return NumPy arrays
#
#  Each wrapper:
#    1. Uploads input(s) to GPU
#    2. Allocates output buffer on GPU
#    3. Launches kernel
#    4. Downloads result to NumPy
#    5. Frees GPU buffers (via CudaBuffer ref-counting)
#
#  For on-device tensor workflows, callers can use the dev_* variants
#  which accept/return GpuTensor directly.
# ================================================================

# ----------------------------------------------------------------
#  UNARY OPS (host NumPy in → host NumPy out)
# ----------------------------------------------------------------

cdef np.ndarray _unary_op_host(np.ndarray x, int (*kernel)(const float*, float*, int64_t, cudaStream_t) noexcept nogil):
    """Generic wrapper for unary f32 kernel: NumPy → GPU → kernel → NumPy."""
    cdef np.ndarray flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    # Allocate device buffers
    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)

    # H2D
    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(flat),
                           nbytes, cudaMemcpyHostToDevice))

    # Launch
    cdef int ret
    with nogil:
        ret = kernel(<float*>d_x.ptr, <float*>d_y.ptr, n, _default_stream)
    _check_kernel(ret)

    # D2H
    cdef np.ndarray result = np.empty((<object>x).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr,
                           nbytes, cudaMemcpyDeviceToHost))
    return result


def cuda_exp(np.ndarray x not None):
    return _unary_op_host(x, cuda_exp_f32)

def cuda_log(np.ndarray x not None):
    return _unary_op_host(x, cuda_log_f32)

def cuda_sqrt(np.ndarray x not None):
    return _unary_op_host(x, cuda_sqrt_f32)

def cuda_rsqrt(np.ndarray x not None):
    return _unary_op_host(x, cuda_rsqrt_f32)

def cuda_sigmoid(np.ndarray x not None):
    return _unary_op_host(x, cuda_sigmoid_f32)

def cuda_tanh(np.ndarray x not None):
    return _unary_op_host(x, cuda_tanh_f32)

def cuda_sin(np.ndarray x not None):
    return _unary_op_host(x, cuda_sin_f32)

def cuda_cos(np.ndarray x not None):
    return _unary_op_host(x, cuda_cos_f32)

def cuda_neg(np.ndarray x not None):
    return _unary_op_host(x, cuda_neg_f32)

def cuda_abs(np.ndarray x not None):
    return _unary_op_host(x, cuda_abs_f32)


# ----------------------------------------------------------------
#  BINARY OPS
# ----------------------------------------------------------------

cdef np.ndarray _binary_op_host(np.ndarray a, np.ndarray b,
                                int (*kernel)(const float*, const float*, float*, int64_t, cudaStream_t) noexcept nogil):
    cdef np.ndarray fa = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
    cdef np.ndarray fb = b.ravel() if b.flags.c_contiguous else np.ascontiguousarray(b).ravel()
    cdef int64_t n = fa.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(nbytes, device)
    cdef CudaBuffer d_b = _make_buffer(nbytes, device)
    cdef CudaBuffer d_c = _make_buffer(nbytes, device)

    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(fa), nbytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_b.ptr, <void*>np.PyArray_DATA(fb), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = kernel(<float*>d_a.ptr, <float*>d_b.ptr, <float*>d_c.ptr, n, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>a).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result


def cuda_add(np.ndarray a not None, np.ndarray b not None):
    return _binary_op_host(a, b, cuda_add_f32)

def cuda_sub(np.ndarray a not None, np.ndarray b not None):
    return _binary_op_host(a, b, cuda_sub_f32)

def cuda_mul(np.ndarray a not None, np.ndarray b not None):
    return _binary_op_host(a, b, cuda_mul_f32)

def cuda_div(np.ndarray a not None, np.ndarray b not None):
    return _binary_op_host(a, b, cuda_div_f32)


# ----------------------------------------------------------------
#  SCALAR BROADCAST OPS
# ----------------------------------------------------------------

def cuda_adds(np.ndarray a not None, float scalar):
    cdef np.ndarray flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(nbytes, device)
    cdef CudaBuffer d_c = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_adds_f32(<float*>d_a.ptr, scalar, <float*>d_c.ptr, n, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>a).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result

def cuda_muls(np.ndarray a not None, float scalar):
    cdef np.ndarray flat = a.ravel() if a.flags.c_contiguous else np.ascontiguousarray(a).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(nbytes, device)
    cdef CudaBuffer d_c = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_muls_f32(<float*>d_a.ptr, scalar, <float*>d_c.ptr, n, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>a).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  ACTIVATION OPS
# ----------------------------------------------------------------

def cuda_relu(np.ndarray x not None):
    return _unary_op_host(x, cuda_relu_f32)

def cuda_silu(np.ndarray x not None):
    return _unary_op_host(x, cuda_silu_f32)

def cuda_gelu(np.ndarray x not None):
    return _unary_op_host(x, cuda_gelu_f32)

def cuda_silu_fwd(np.ndarray x not None):
    """SiLU forward returning (result, sigmoid) for backward caching."""
    cdef np.ndarray flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)
    cdef CudaBuffer d_sig = _make_buffer(nbytes, device)

    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_silu_fwd_f32(<float*>d_x.ptr, <float*>d_y.ptr,
                                <float*>d_sig.ptr, n, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>x).shape, dtype=np.float32)
    cdef np.ndarray sig = np.empty((<object>x).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, nbytes, cudaMemcpyDeviceToHost))
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(sig), d_sig.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result, sig

def cuda_clamp(np.ndarray x not None, float lo, float hi):
    cdef np.ndarray flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_clamp_f32(<float*>d_x.ptr, <float*>d_y.ptr, lo, hi, n, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>x).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  SOFTMAX
# ----------------------------------------------------------------

def cuda_softmax(np.ndarray x not None, int dim=-1):
    """Softmax along last dimension. Input must be 2D."""
    cdef np.ndarray x2d
    cdef tuple orig_shape = (<object>x).shape
    if x.ndim == 2:
        x2d = np.ascontiguousarray(x)
    elif x.ndim >= 2 and (dim == -1 or dim == x.ndim - 1):
        x2d = np.ascontiguousarray(x.reshape(-1, x.shape[x.ndim - 1]))
    else:
        # Fallback
        x2d = np.ascontiguousarray(x.reshape(-1, x.shape[x.ndim - 1]))

    cdef int rows = x2d.shape[0]
    cdef int cols = x2d.shape[1]
    cdef size_t nbytes = rows * cols * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(x2d), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_softmax_f32(<float*>d_x.ptr, <float*>d_y.ptr, rows, cols, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((rows, cols), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result.reshape(orig_shape)

def cuda_log_softmax(np.ndarray x not None, int dim=-1):
    """Log-softmax along last dimension."""
    cdef np.ndarray x2d
    cdef tuple orig_shape = (<object>x).shape
    if x.ndim >= 2 and (dim == -1 or dim == x.ndim - 1):
        x2d = np.ascontiguousarray(x.reshape(-1, x.shape[x.ndim - 1]))
    else:
        x2d = np.ascontiguousarray(x.reshape(-1, x.shape[x.ndim - 1]))

    cdef int rows = x2d.shape[0]
    cdef int cols = x2d.shape[1]
    cdef size_t nbytes = rows * cols * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)
    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(x2d), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_log_softmax_f32(<float*>d_x.ptr, <float*>d_y.ptr, rows, cols, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((rows, cols), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result.reshape(orig_shape)


# ----------------------------------------------------------------
#  CROSS-ENTROPY (fused)
# ----------------------------------------------------------------

def cuda_cross_entropy(np.ndarray logits not None, np.ndarray targets not None):
    """Fused cross-entropy: softmax + NLL. Returns (mean_loss, probs)."""
    cdef np.ndarray x2d = np.ascontiguousarray(logits) if logits.ndim == 2 else \
                          np.ascontiguousarray(logits.reshape(1, -1))
    cdef np.ndarray tgt = np.ascontiguousarray(targets).astype(np.int64).ravel()

    cdef int N = x2d.shape[0]
    cdef int C = x2d.shape[1]
    cdef size_t logit_bytes = N * C * sizeof(float)
    cdef size_t target_bytes = N * sizeof(int64_t)
    cdef size_t loss_bytes = N * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_logits = _make_buffer(logit_bytes, device)
    cdef CudaBuffer d_targets = _make_buffer(target_bytes, device)
    cdef CudaBuffer d_losses = _make_buffer(loss_bytes, device)
    cdef CudaBuffer d_probs = _make_buffer(logit_bytes, device)

    _check_cuda(cudaMemcpy(d_logits.ptr, <void*>np.PyArray_DATA(x2d),
                           logit_bytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_targets.ptr, <void*>np.PyArray_DATA(tgt),
                           target_bytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_cross_entropy_fwd_f32(
            <float*>d_logits.ptr, <int64_t*>d_targets.ptr,
            <float*>d_losses.ptr, <float*>d_probs.ptr,
            N, C, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray losses = np.empty(N, dtype=np.float32)
    cdef np.ndarray probs = np.empty((N, C), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(losses), d_losses.ptr,
                           loss_bytes, cudaMemcpyDeviceToHost))
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(probs), d_probs.ptr,
                           logit_bytes, cudaMemcpyDeviceToHost))
    return float(np.mean(losses)), probs


# ----------------------------------------------------------------
#  GEMM (cuBLAS)
# ----------------------------------------------------------------

def cuda_matmul(np.ndarray a not None, np.ndarray b not None):
    """Matrix multiply C = A @ B via cuBLAS. Both must be f32, 2D."""
    cdef np.ndarray ac = np.ascontiguousarray(a)
    cdef np.ndarray bc = np.ascontiguousarray(b)
    cdef int M = ac.shape[0]
    cdef int K = ac.shape[1]
    cdef int N = bc.shape[1]
    cdef size_t a_bytes = M * K * sizeof(float)
    cdef size_t b_bytes = K * N * sizeof(float)
    cdef size_t c_bytes = M * N * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(a_bytes, device)
    cdef CudaBuffer d_b = _make_buffer(b_bytes, device)
    cdef CudaBuffer d_c = _make_buffer(c_bytes, device)

    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(ac), a_bytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_b.ptr, <void*>np.PyArray_DATA(bc), b_bytes, cudaMemcpyHostToDevice))

    cdef cublasHandle_t handle = _get_cublas()
    cdef int ret = cuda_sgemm(handle, M, N, K,
                              <float*>d_a.ptr, <float*>d_b.ptr, <float*>d_c.ptr,
                              1.0, 0.0, False, False)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((M, N), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, c_bytes, cudaMemcpyDeviceToHost))
    return result

def cuda_matmul_nt(np.ndarray a not None, np.ndarray b not None):
    """C = A @ B^T via cuBLAS (for linear layers)."""
    cdef np.ndarray ac = np.ascontiguousarray(a)
    cdef np.ndarray bc = np.ascontiguousarray(b)
    cdef int M = ac.shape[0]
    cdef int K = ac.shape[1]
    cdef int N = bc.shape[0]  # B is (N, K), transposed to (K, N)
    cdef size_t a_bytes = M * K * sizeof(float)
    cdef size_t b_bytes = N * K * sizeof(float)
    cdef size_t c_bytes = M * N * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(a_bytes, device)
    cdef CudaBuffer d_b = _make_buffer(b_bytes, device)
    cdef CudaBuffer d_c = _make_buffer(c_bytes, device)

    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(ac), a_bytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_b.ptr, <void*>np.PyArray_DATA(bc), b_bytes, cudaMemcpyHostToDevice))

    cdef cublasHandle_t handle = _get_cublas()
    cdef int ret = cuda_sgemm(handle, M, N, K,
                              <float*>d_a.ptr, <float*>d_b.ptr, <float*>d_c.ptr,
                              1.0, 0.0, False, True)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((M, N), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, c_bytes, cudaMemcpyDeviceToHost))
    return result

def cuda_batched_matmul(np.ndarray a not None, np.ndarray b not None):
    """Batched matmul: (..., M, K) @ (..., K, N) → (..., M, N)."""
    cdef tuple a_shape = (<object>a).shape
    cdef tuple b_shape = (<object>b).shape
    cdef int ndim_a = a.ndim
    cdef int ndim_b = b.ndim

    # Flatten batch dims
    cdef int M = a_shape[ndim_a - 2]
    cdef int K = a_shape[ndim_a - 1]
    cdef int N_ = b_shape[ndim_b - 1]
    cdef int batch = 1
    cdef int i
    for i in range(ndim_a - 2):
        batch *= a_shape[i]

    cdef np.ndarray ac = np.ascontiguousarray(a.reshape(batch, M, K))
    cdef np.ndarray bc = np.ascontiguousarray(b.reshape(batch, K, N_))

    cdef size_t a_bytes = batch * M * K * sizeof(float)
    cdef size_t b_bytes = batch * K * N_ * sizeof(float)
    cdef size_t c_bytes = batch * M * N_ * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_a = _make_buffer(a_bytes, device)
    cdef CudaBuffer d_b = _make_buffer(b_bytes, device)
    cdef CudaBuffer d_c = _make_buffer(c_bytes, device)

    _check_cuda(cudaMemcpy(d_a.ptr, <void*>np.PyArray_DATA(ac), a_bytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_b.ptr, <void*>np.PyArray_DATA(bc), b_bytes, cudaMemcpyHostToDevice))

    cdef cublasHandle_t handle = _get_cublas()
    cdef int64_t strideA = M * K
    cdef int64_t strideB = K * N_
    cdef int64_t strideC = M * N_
    cdef int ret = cuda_sgemm_batched(handle, M, N_, K,
                                      <float*>d_a.ptr, <float*>d_b.ptr, <float*>d_c.ptr,
                                      1.0, 0.0, batch, strideA, strideB, strideC)
    _check_kernel(ret)

    cdef tuple out_shape = a_shape[:ndim_a-2] + (M, N_)
    cdef np.ndarray result = np.empty(out_shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_c.ptr, c_bytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  LAYER NORM / RMS NORM
# ----------------------------------------------------------------

def cuda_layer_norm(np.ndarray x not None, np.ndarray gamma, np.ndarray beta,
                    float eps=1e-5):
    """Layer norm on (N, D) input."""
    cdef np.ndarray xc = np.ascontiguousarray(x)
    cdef int N = xc.shape[0]
    cdef int D = xc.shape[1]
    cdef size_t x_bytes = N * D * sizeof(float)
    cdef size_t d_bytes = D * sizeof(float)
    cdef size_t n_bytes = N * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(x_bytes, device)
    cdef CudaBuffer d_y = _make_buffer(x_bytes, device)
    cdef CudaBuffer d_gamma = _make_buffer(d_bytes, device) if gamma is not None else None
    cdef CudaBuffer d_beta = _make_buffer(d_bytes, device) if beta is not None else None
    cdef CudaBuffer d_mean = _make_buffer(n_bytes, device)
    cdef CudaBuffer d_rstd = _make_buffer(n_bytes, device)

    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(xc), x_bytes, cudaMemcpyHostToDevice))
    if gamma is not None:
        _check_cuda(cudaMemcpy(d_gamma.ptr, <void*>np.PyArray_DATA(np.ascontiguousarray(gamma)),
                               d_bytes, cudaMemcpyHostToDevice))
    if beta is not None:
        _check_cuda(cudaMemcpy(d_beta.ptr, <void*>np.PyArray_DATA(np.ascontiguousarray(beta)),
                               d_bytes, cudaMemcpyHostToDevice))

    cdef float* gptr = <float*>d_gamma.ptr if d_gamma is not None else NULL
    cdef float* bptr = <float*>d_beta.ptr if d_beta is not None else NULL

    cdef int ret
    with nogil:
        ret = cuda_layer_norm_f32(<float*>d_x.ptr, gptr, bptr,
                                  <float*>d_y.ptr, <float*>d_mean.ptr, <float*>d_rstd.ptr,
                                  N, D, eps, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((N, D), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, x_bytes, cudaMemcpyDeviceToHost))
    return result

def cuda_rms_norm(np.ndarray x not None, np.ndarray gamma, float eps=1e-5):
    """RMS norm on (N, D) input."""
    cdef np.ndarray xc = np.ascontiguousarray(x)
    cdef int N = xc.shape[0]
    cdef int D = xc.shape[1]
    cdef size_t x_bytes = N * D * sizeof(float)
    cdef size_t d_bytes = D * sizeof(float)
    cdef size_t n_bytes = N * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(x_bytes, device)
    cdef CudaBuffer d_y = _make_buffer(x_bytes, device)
    cdef CudaBuffer d_gamma = _make_buffer(d_bytes, device) if gamma is not None else None
    cdef CudaBuffer d_rstd = _make_buffer(n_bytes, device)

    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(xc), x_bytes, cudaMemcpyHostToDevice))
    if gamma is not None:
        _check_cuda(cudaMemcpy(d_gamma.ptr, <void*>np.PyArray_DATA(np.ascontiguousarray(gamma)),
                               d_bytes, cudaMemcpyHostToDevice))

    cdef float* gptr = <float*>d_gamma.ptr if d_gamma is not None else NULL

    cdef int ret
    with nogil:
        ret = cuda_rms_norm_f32(<float*>d_x.ptr, gptr,
                                <float*>d_y.ptr, <float*>d_rstd.ptr,
                                N, D, eps, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((N, D), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, x_bytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  ADAMW FUSED OPTIMIZER STEP
# ----------------------------------------------------------------

def cuda_adamw_step(np.ndarray param not None, np.ndarray grad not None,
                    np.ndarray m not None, np.ndarray v not None,
                    float lr, float beta1, float beta2,
                    float eps, float wd, float bc1, float bc2):
    """Fused AdamW step on GPU. Modifies param, m, v in-place."""
    cdef np.ndarray p_flat = np.ascontiguousarray(param).ravel()
    cdef np.ndarray g_flat = np.ascontiguousarray(grad).ravel()
    cdef np.ndarray m_flat = np.ascontiguousarray(m).ravel()
    cdef np.ndarray v_flat = np.ascontiguousarray(v).ravel()
    cdef int64_t n = p_flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_p = _make_buffer(nbytes, device)
    cdef CudaBuffer d_g = _make_buffer(nbytes, device)
    cdef CudaBuffer d_m = _make_buffer(nbytes, device)
    cdef CudaBuffer d_v = _make_buffer(nbytes, device)

    _check_cuda(cudaMemcpy(d_p.ptr, <void*>np.PyArray_DATA(p_flat), nbytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_g.ptr, <void*>np.PyArray_DATA(g_flat), nbytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_m.ptr, <void*>np.PyArray_DATA(m_flat), nbytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_v.ptr, <void*>np.PyArray_DATA(v_flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_adamw_step_f32(<float*>d_p.ptr, <float*>d_g.ptr,
                                  <float*>d_m.ptr, <float*>d_v.ptr,
                                  lr, beta1, beta2, eps, wd, bc1, bc2,
                                  n, _default_stream)
    _check_kernel(ret)

    # Copy back (in-place update)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(p_flat), d_p.ptr, nbytes, cudaMemcpyDeviceToHost))
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(m_flat), d_m.ptr, nbytes, cudaMemcpyDeviceToHost))
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(v_flat), d_v.ptr, nbytes, cudaMemcpyDeviceToHost))

    # Write back to original arrays
    np.copyto(param.ravel(), p_flat)
    np.copyto(m.ravel(), m_flat)
    np.copyto(v.ravel(), v_flat)


# ----------------------------------------------------------------
#  DROPOUT
# ----------------------------------------------------------------

def cuda_dropout(np.ndarray x not None, float p, unsigned long long seed=0):
    """Dropout with cuRAND. Returns (output, mask*scale)."""
    cdef np.ndarray flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t nbytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(nbytes, device)
    cdef CudaBuffer d_y = _make_buffer(nbytes, device)
    cdef CudaBuffer d_mask = _make_buffer(nbytes, device)

    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(flat), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_dropout_f32(<float*>d_x.ptr, <float*>d_y.ptr, <float*>d_mask.ptr,
                               p, n, seed, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((<object>x).shape, dtype=np.float32)
    cdef np.ndarray mask = np.empty((<object>x).shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_y.ptr, nbytes, cudaMemcpyDeviceToHost))
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(mask), d_mask.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result, mask


# ----------------------------------------------------------------
#  EMBEDDING
# ----------------------------------------------------------------

def cuda_embedding(np.ndarray weight not None, np.ndarray indices not None):
    """Embedding lookup on GPU."""
    cdef np.ndarray wc = np.ascontiguousarray(weight)
    cdef np.ndarray idx = np.ascontiguousarray(indices).astype(np.int64).ravel()
    cdef int num_idx = idx.shape[0]
    cdef int embed_dim = wc.shape[1]
    cdef size_t w_bytes = wc.shape[0] * embed_dim * sizeof(float)
    cdef size_t idx_bytes = num_idx * sizeof(int64_t)
    cdef size_t out_bytes = num_idx * embed_dim * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_w = _make_buffer(w_bytes, device)
    cdef CudaBuffer d_idx = _make_buffer(idx_bytes, device)
    cdef CudaBuffer d_out = _make_buffer(out_bytes, device)

    _check_cuda(cudaMemcpy(d_w.ptr, <void*>np.PyArray_DATA(wc), w_bytes, cudaMemcpyHostToDevice))
    _check_cuda(cudaMemcpy(d_idx.ptr, <void*>np.PyArray_DATA(idx), idx_bytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_embedding_f32(<float*>d_w.ptr, <int64_t*>d_idx.ptr,
                                 <float*>d_out.ptr, num_idx, embed_dim, _default_stream)
    _check_kernel(ret)

    cdef tuple out_shape = (<object>indices).shape + (embed_dim,)
    cdef np.ndarray result = np.empty(out_shape, dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_out.ptr, out_bytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  TRANSPOSE
# ----------------------------------------------------------------

def cuda_transpose_2d(np.ndarray x not None):
    """Transpose 2D matrix on GPU with shared-memory tiling."""
    cdef np.ndarray xc = np.ascontiguousarray(x)
    cdef int rows = xc.shape[0]
    cdef int cols = xc.shape[1]
    cdef size_t nbytes = rows * cols * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_in = _make_buffer(nbytes, device)
    cdef CudaBuffer d_out = _make_buffer(nbytes, device)

    _check_cuda(cudaMemcpy(d_in.ptr, <void*>np.PyArray_DATA(xc), nbytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_transpose_2d_f32(<float*>d_in.ptr, <float*>d_out.ptr,
                                    rows, cols, _default_stream)
    _check_kernel(ret)

    cdef np.ndarray result = np.empty((cols, rows), dtype=np.float32)
    _check_cuda(cudaMemcpy(<void*>np.PyArray_DATA(result), d_out.ptr, nbytes, cudaMemcpyDeviceToHost))
    return result


# ----------------------------------------------------------------
#  LINEAR FORWARD (fused matmul + bias via cuBLAS)
# ----------------------------------------------------------------

def cuda_linear_forward(np.ndarray x not None, np.ndarray weight not None,
                        np.ndarray bias=None):
    """y = x @ W^T + bias, using cuBLAS sgemm_nt."""
    cdef np.ndarray xc = np.ascontiguousarray(x.reshape(-1, x.shape[x.ndim - 1]))
    cdef np.ndarray wc = np.ascontiguousarray(weight)
    cdef int M = xc.shape[0]
    cdef int K = xc.shape[1]
    cdef int N = wc.shape[0]  # weight is (out_features, in_features)

    cdef np.ndarray result = cuda_matmul_nt(xc, wc)

    if bias is not None:
        result = result + np.ascontiguousarray(bias).ravel()

    return result.reshape((<object>x).shape[:-1] + (N,))


# ----------------------------------------------------------------
#  TF32 CONTROL
# ----------------------------------------------------------------

def set_tf32_enabled(bint enable):
    """Enable/disable TF32 tensor core math for cuBLAS (Ampere+)."""
    cdef cublasHandle_t handle = _get_cublas()
    cuda_set_tf32(handle, 1 if enable else 0)


# ----------------------------------------------------------------
#  SUM REDUCTION
# ----------------------------------------------------------------

def cuda_sum(np.ndarray x not None):
    """Global sum reduction on GPU."""
    cdef np.ndarray flat = x.ravel() if x.flags.c_contiguous else np.ascontiguousarray(x).ravel()
    cdef int64_t n = flat.shape[0]
    cdef size_t in_bytes = n * sizeof(float)
    cdef int device = 0
    cudaGetDevice(&device)

    cdef CudaBuffer d_x = _make_buffer(in_bytes, device)
    cdef CudaBuffer d_out = _make_buffer(sizeof(float), device)

    _check_cuda(cudaMemcpy(d_x.ptr, <void*>np.PyArray_DATA(flat), in_bytes, cudaMemcpyHostToDevice))

    cdef int ret
    with nogil:
        ret = cuda_sum_f32(<float*>d_x.ptr, <float*>d_out.ptr, n, _default_stream)
    _check_kernel(ret)

    cdef float result = 0.0
    _check_cuda(cudaMemcpy(&result, d_out.ptr, sizeof(float), cudaMemcpyDeviceToHost))
    return float(result)


# ----------------------------------------------------------------
#  FILL / COPY UTILITIES
# ----------------------------------------------------------------

def cuda_fill(CudaBuffer buf, float value, int64_t n):
    """Fill a device buffer with a scalar value."""
    cdef int ret
    with nogil:
        ret = cuda_fill_f32(<float*>buf.ptr, value, n, _default_stream)
    _check_kernel(ret)
