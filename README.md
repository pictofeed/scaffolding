<div align="center">
    <img src="img/scaffolding-logo.png" alt="Scaffolding Logo" width="500"/>
</div>

# Scaffolding

**A deep learning framework written entirely in Python, Cython, and CUDA C.**

Scaffolding is a lightweight, production-ready deep learning framework built from the ground up using NumPy as its computational backend, Cython-accelerated hot-path operations, and native CUDA kernels for NVIDIA GPU execution. On macOS, Scaffolding leverages Apple's Accelerate framework (BLAS, vDSP, vecLib) through native Cython bindings for near-hardware-level performance on both CPU and Apple Silicon. On NVIDIA GPUs, tensors live entirely in VRAM with no PyTorch dependency — ideal for RAM-constrained GPU servers.

---

## Features

- **Pure Python + Cython + CUDA C** — no PyTorch or external framework dependencies; every operation is implemented in Python with Cython kernels and CUDA C for the performance-critical GPU paths
- **Automatic Differentiation** — full reverse-mode autograd engine with gradient accumulation, `no_grad` / `inference_mode` context managers, and `retain_graph` support
- **Neural Network Modules** — `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Conv1d`/`Conv2d`/`Conv3d`, `nn.LayerNorm`, `nn.RMSNorm`, `nn.BatchNorm2d`, `nn.GroupNorm`, pooling layers, and more
- **Functional API** — `nn.functional` with `softmax`, `cross_entropy`, `silu`, `gelu`, `relu`, `conv2d`, `conv3d`, `group_norm`, `interpolate`, and others
- **Text-to-Video Generation** — complete diffusion-based pipeline with 3D UNet, DDPM/DDIM schedulers, Tora (CogVideoX DiT), text encoder, and optional VAE latent diffusion
- **NVIDIA CUDA Backend** — GPU-resident tensors (`GpuTensor`) with zero-copy VRAM storage, automatic upload for all numeric dtypes (`float16`/`float32`/`float64`/`uint8`/`int8`–`int64`), device-to-device copy, and fused CUDA kernels for elementwise ops, matmul, linear forward, SiLU, and embedding lookup
- **Streaming Memory Management** — designed for RAM-constrained GPU hosts (e.g. 2.5 GB system RAM + 24 GB VRAM): `Module._release_cpu_shadows()` drops CPU weight copies after forward passes, mini-batched UNet inference (`_UNET_FRAME_BATCH`), frame-by-frame VAE decode, and `malloc_trim` integration on Linux
- **Optimizers** — `AdamW` with weight decay, bias correction, and learning rate scheduling (`CosineAnnealingLR`, `LambdaLR`, `StepLR`)
- **Apple MPS Backend** — Cython-wrapped Apple Accelerate.framework (cblas, vDSP, vecLib) for hardware-accelerated matmul, elementwise ops, softmax, RMS norm, and cross-entropy on macOS
- **Gradient Checkpointing** — memory-efficient training via `utils.checkpoint`
- **Device Abstraction** — `cpu`, `cuda`, and `mps` device types with `.to()`, `.cpu()`, `.cuda()`, `.mps()` tensor methods
- **Dtype System** — `float16`, `float32`, `float64`, `bfloat16`, `int8`–`int64`, `uint8`, `bool` — all uploadable to CUDA
- **Two Build Systems** — setuptools + Cython *and* CMake, with a Makefile convenience wrapper

---

## Architecture

```
scaffolding/
├── __init__.py              # Public API surface
├── tensor.py                # Tensor class, factory functions, math ops, GPU-resident storage
├── autograd.py              # Reverse-mode autograd engine
├── device.py                # Device abstraction (cpu, cuda, mps)
├── dtype.py                 # Data type definitions
│
├── _tensor_ops.pyx          # Cython: generic CPU hot-path kernels
├── _mps_ops.pyx             # Cython: Apple Accelerate-backed kernels
├── _cuda_ops.pyx            # Cython: NVIDIA CUDA backend (GpuTensor, device ops)
├── _cuda_kernels.cu         # CUDA C: fused GPU kernels (elementwise, matmul, linear, SiLU, …)
├── _cuda_kernels.cuh        # CUDA C: kernel declarations
├── _cuda_ops_decl.h         # C header: Cython ↔ CUDA bridge declarations
│
├── nn/
│   ├── module.py            # nn.Module base class + _release_cpu_shadows()
│   ├── parameter.py         # nn.Parameter
│   ├── layers.py            # Linear, Conv1d/2d/3d, ConvTranspose2d/3d, BatchNorm, GroupNorm, pooling, Upsample, etc.
│   ├── functional.py        # Functional API (softmax, cross_entropy, conv2d, conv3d, interpolate, etc.)
│   ├── video.py             # Text-to-video diffusion pipeline (3D UNet, DDPM/DDIM, text encoder)
│   ├── init.py              # Weight initialization (xavier, kaiming, normal, etc.)
│   ├── parallel.py          # DistributedDataParallel stub
│   └── utils.py             # clip_grad_norm_
│
├── optim/
│   ├── optimizer.py         # AdamW optimizer
│   └── lr_scheduler.py      # Learning rate schedulers
│
├── diffusion/
│   ├── models.py            # UNet2DConditionModel, DiTModel, AutoencoderKL
│   ├── schedulers.py        # DDPM, DDIM, DPM-Solver++, CogVideoXDPM, Euler, PNDM, FlowMatch
│   ├── utils.py             # classifier_free_guidance, randn_tensor, get_beta_schedule
│   └── pipelines/
│       ├── _base.py          # DiffusionPipeline, StableDiffusionPipeline, CogVideoXPipeline
│       └── tora.py           # ToraPipeline (CogVideoX DiT + trajectory control)
│
├── backends/
│   ├── mps.py               # MPS/Accelerate backend detection & dispatch
│   ├── cuda.py              # CUDA backend — device properties, cache, TF32 control
│   └── cudnn.py             # cuDNN backend configuration
│
├── cuda/
│   └── amp.py               # Automatic mixed precision
│
├── distributed/             # Distributed training stubs
│   └── __init__.py
│
├── utils/
│   └── checkpoint.py        # Gradient checkpointing
│
├── tests/
│   ├── smoke_test.py        # End-to-end framework smoke test
│   └── test_mps_ops.py      # Numerical correctness tests for MPS ops
│
├── setup.py                 # Setuptools + Cython build
├── CMakeLists.txt           # CMake build system
├── Makefile                 # Convenience wrapper
├── pyproject.toml           # PEP 517 build metadata
└── requirements.txt         # Runtime dependencies
```
### Cython Extensions

| Extension | File | Description |
|---|---|---|
| `_tensor_ops` | `_tensor_ops.pyx` | Generic CPU kernels — sigmoid, exp, log, tanh, matmul, softmax, RMS norm, AdamW step. NumPy-backed with `nogil` computation blocks. |
| `_mps_ops` | `_mps_ops.pyx` | Apple Accelerate kernels — cblas_sgemm/dgemm (BLAS), vDSP vector ops, vecLib transcendentals, fused softmax, RMS norm, cross-entropy, AdamW. Linked against `-framework Accelerate`. |
| `_cuda_ops` | `_cuda_ops.pyx` + `_cuda_kernels.cu` | NVIDIA CUDA backend — `GpuTensor` GPU-resident storage, `CudaBuffer` memory management, fused elementwise kernels (float4-vectorised add/sub/mul/div/scalar), `cuda_linear_forward`, `cuda_silu`, `cuda_embedding`, device-to-device copy, `empty_cache`. Supports all numeric dtypes. |

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| NumPy | ≥ 1.24 |
| Cython | ≥ 3.0 |
| setuptools | ≥ 68.0 |

**macOS MPS Backend** (optional):
- macOS with Accelerate.framework (included in Xcode Command Line Tools)
- Apple Silicon (arm64) or Intel Mac with Metal support

**NVIDIA CUDA Backend** (optional):
- NVIDIA GPU with CUDA compute capability ≥ 3.5 (Kepler K80 or newer)
- CUDA Toolkit ≥ 11.0
- No PyTorch required — scaffolding ships its own CUDA kernels

---

## Installation

### From Source (Recommended)

```bash
# Clone and enter the directory
cd scaffolding

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy cython setuptools wheel

# Build Cython extensions in-place
make build
```

### Editable Install

```bash
pip install -e ".[dev]"
```

### Build a Wheel

```bash
make wheel
# Output: dist/scaffolding-0.1.0-*.whl
```

---

## Build Systems

Scaffolding provides two independent build systems. Both produce the same compiled `.so` extensions.


## Diffusion Models & Pipelines

Scaffolding provides a full-featured, modular diffusion package in `scaffolding.diffusion` for both image and video generation, including:

- **Noise Schedulers:**
    - `DDPMScheduler`, `DDIMScheduler`, `DPMSolverMultistepScheduler`, `CogVideoXDPMScheduler`, `EulerDiscreteScheduler`, `PNDMScheduler`, `FlowMatchEulerDiscreteScheduler`
- **Model Architectures:**
    - `UNet2DConditionModel` (Stable Diffusion-style), `DiTModel` (Diffusion Transformer), `AutoencoderKL` (VAE for latent diffusion)
- **Pipelines:**
    - `DiffusionPipeline` (base), `StableDiffusionPipeline`, `CogVideoXPipeline`, `ToraPipeline`
- **Utilities:**
    - `classifier_free_guidance`, `rescale_noise_cfg`, `randn_tensor`, `get_beta_schedule`

### Example: Stable Diffusion Pipeline

```python
from scaffolding.diffusion import (
        StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
)

# Instantiate components
unet = UNet2DConditionModel()
vae = AutoencoderKL()
sched = DDPMScheduler()

pipe = StableDiffusionPipeline(
        unet=unet,
        scheduler=sched,
        vae=vae,
)

# Generate image from prompt embeddings (see text encoder integration)
prompt_embeds = ...  # (B, S, D) text embeddings
image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=50)
print(image.shape)  # (B, 3, H, W)
```

### Example: Tora Text-to-Video (GPU-Resident Output)

```python
from scaffolding.diffusion.pipelines.tora import ToraPipeline
from scaffolding.diffusion.schedulers import CogVideoXDPMScheduler
import scaffolding as sf

pipe = ToraPipeline.from_pretrained("./models/Tora_T2V", dtype=sf.float16)
pipe.scheduler = CogVideoXDPMScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate with output_type="sf_tensor" → frames stay on GPU as uint8 tensors
# No PIL Images or numpy arrays touch CPU RAM
with sf.inference_mode():
    output = pipe(
        prompt="a drone flying over a mountain lake at sunset",
        num_frames=49,
        num_inference_steps=50,
        height=480,
        width=720,
        guidance_scale=6.0,
        output_type="sf_tensor",  # GPU-resident uint8 frames (H, W, C)
        generator=sf.Generator(device="cuda").manual_seed(42),
    )

# output.frames[0] is a list of GPU-resident sf.Tensor (H, W, C) uint8
print(len(output.frames[0]))  # 49 frames
print(output.frames[0][0].shape)  # (480, 720, 3)
print(output.frames[0][0].device)  # cuda:0
```

### Example: CogVideoX Video Diffusion

```python
from scaffolding.diffusion import (
        CogVideoXPipeline, CogVideoXDPMScheduler, DiTModel
)

transformer = DiTModel(patch_size=2, in_channels=16, hidden_size=1152, depth=28)
sched = CogVideoXDPMScheduler(
        num_train_timesteps=1000,
        snr_shift_scale=3.0,
        prediction_type='v_prediction',
)

pipe = CogVideoXPipeline(
        transformer=transformer,
        scheduler=sched,
        num_frames=49,
        latent_channels=16,
)

# Generate video from prompt embeddings
prompt_embeds = ...  # (B, S, D) text embeddings
video = pipe(prompt_embeds=prompt_embeds, num_inference_steps=50)
print(video.shape)  # (B, 16, 49, H, W)
```

### Schedulers Overview

| Scheduler | Description |
|---|---|
| `DDPMScheduler` | Classic DDPM (Ho et al. 2020), linear/cosine beta schedules |
| `DDIMScheduler` | Deterministic DDIM (Song et al. 2020), fast sampling |
| `DPMSolverMultistepScheduler` | DPM-Solver++ (Lu et al. 2022), high-quality ODE solver |
| `CogVideoXDPMScheduler` | SNR-shifted DPM-Solver++ for video, temporal coherence (CogVideoX) |
| `EulerDiscreteScheduler` | Euler ancestral sampler (Karras et al. 2022) |
| `PNDMScheduler` | Pseudo Numerical Diffusion (Liu et al. 2022), 4th-order multistep |
| `FlowMatchEulerDiscreteScheduler` | Flow Matching (Lipman et al. 2023), SD3/FLUX |

#### CogVideoXDPMScheduler Highlights

- SNR-shifted schedule for temporal coherence in video diffusion
- Configurable timestep spacing (`linspace`, `leading`, `trailing`)
- Dynamic thresholding (Imagen-style) for high guidance scales
- Zero-terminal-SNR rescaling option
- DPM-Solver++ first/second-order solvers
- Supports `v_prediction`, `epsilon`, and `sample` prediction types

### Utilities

- `classifier_free_guidance(model, latents, timesteps, prompt_embeds, negative_prompt_embeds, guidance_scale)` — Run a model with classifier-free guidance in one call
- `rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=0.7)` — Imagen-style guidance rescaling
- `randn_tensor(shape, seed=None)` — Generate random noise as a Tensor
- `get_beta_schedule(schedule, num_timesteps, beta_start, beta_end)` — Build beta schedules for custom schedulers

---

```bash
# Build extensions in-place for development
python setup.py build_ext --inplace

# Or use the Makefile shortcut
make build
```

This compiles `_tensor_ops.pyx`, (on macOS) `_mps_ops.pyx`, and (when CUDA is available) `_cuda_ops.pyx` + `_cuda_kernels.cu` into shared libraries using Cython's optimized compiler directives:

- `boundscheck=False`
- `wraparound=False`
- `cdivision=True`
- `nonecheck=False`

### 2. CMake

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Or use the Makefile shortcut
make build-cmake
```

The CMake build automatically detects:
- Python interpreter and NumPy include paths
- Cython compiler
- Accelerate.framework and Metal.framework (macOS)
- CUDA Toolkit and `nvcc` compiler (Linux/Windows)

MPS extensions are conditionally compiled only when Accelerate is found.
CUDA extensions (`_cuda_ops` + `_cuda_kernels.cu`) are compiled only when `nvcc` is on `$PATH`.

### Makefile Targets

| Target | Description |
|---|---|
| `make build` | Setuptools + Cython in-place build |
| `make build-cmake` | CMake configure + build |
| `make install` | Editable pip install |
| `make wheel` | Build wheel distribution |
| `make test` | Run test suite |
| `make clean` | Remove all build artefacts |

---

## Quick Start

```python
import scaffolding as sf
import scaffolding.nn as nn
import scaffolding.nn.functional as F
import scaffolding.optim as optim

# Create tensors
x = sf.randn(4, 3, requires_grad=True)
w = sf.randn(3, 2, requires_grad=True)

# Forward pass
y = sf.matmul(x, w)
loss = y.sum()

# Backward pass
loss.backward()
print(x.grad.shape)  # (4, 3)
```

### Defining a Model

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        return self.fc2(x)

model = MLP(256, 512, 10)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training step
x = sf.randn(32, 256)
target = sf.randint(0, 10, (32,))

logits = model(x)
loss = F.cross_entropy(logits, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Using the CUDA Backend (NVIDIA GPUs)

```python
import scaffolding as sf
import scaffolding.backends.cuda as cuda

if cuda.is_available():
    print(f"GPUs: {cuda.device_count()}")
    props = cuda.get_device_properties(0)
    print(f"  {props['name']}: {props['total_memory'] / 1e9:.1f} GB VRAM")

    x = sf.randn(64, 64).cuda()  # uploaded to GPU
    y = sf.randn(64, 64).cuda()

    # Matmul dispatches to CUDA kernels
    z = sf.matmul(x, y)

    # Elementwise ops use fused float4-vectorised CUDA kernels
    a = sf.sigmoid(x)
    b = sf.exp(x)

    # Move model weights to GPU (streams one parameter at a time)
    model = MLP(256, 512, 10)
    model.to("cuda")  # parameters & buffers uploaded to VRAM

    # After forward pass, drop CPU weight shadows to save host RAM
    output = model(x)
    model._release_cpu_shadows()
```

### Using the MPS Backend (macOS)

```python
import scaffolding.backends.mps as mps

if mps.is_available():
    x = sf.randn(64, 64).mps()
    y = sf.randn(64, 64).mps()

    # Matmul dispatches to Accelerate cblas_sgemm
    z = sf.matmul(x, y)

    # Elementwise ops dispatch to vecLib / vDSP
    a = sf.sigmoid(x)
    b = sf.exp(x)
    c = sf.softmax(x, dim=-1)
```

When a tensor is on the `cuda` device, operations dispatch to fused CUDA C kernels:

| Operation | CUDA Kernel |
|---|---|
| Elementwise add/sub/mul/div | `float4`-vectorised `cuda_add`, `cuda_sub`, `cuda_mul`, `cuda_div` |
| Scalar multiply / add | `cuda_scalar_mul`, `cuda_scalar_add` |
| Linear forward | `cuda_linear_forward` (fused matmul + bias) |
| SiLU | `cuda_silu` |
| Embedding lookup | `cuda_embedding` |
| Memory transfer | `cuda_memcpy_dtod` (device-to-device) |
| Cache management | `empty_cache` (VRAM cleanup) |

When a tensor is on the `mps` device, operations automatically dispatch to Apple Accelerate's optimized implementations:

| Operation | Accelerate Function |
|---|---|
| Matrix multiply (f32) | `cblas_sgemm` |
| Matrix multiply (f64) | `cblas_dgemm` |
| Sigmoid | `vvrecf` + `vvexpf` + `vDSP_vneg` |
| SiLU | sigmoid × input via `vDSP_vmul` |
| GELU | vecLib `vvtanhf` approximation |
| Softmax | `vDSP_maxv` + `vvexpf` + `vDSP_sve` |
| RMS Norm | `vDSP_vsq` + `vDSP_meanv` + `vvsqrtf` |
| Exp / Log / Sqrt / Tanh | `vvexpf` / `vvlogf` / `vvsqrtf` / `vvtanhf` |
| L2 Norm | `cblas_snrm2` |
| Cross-Entropy | fused log-softmax + NLL |
| AdamW Step | fused first/second moment + weight decay |

---

## Testing

### Smoke Test

Runs an end-to-end check of all major subsystems — tensors, autograd, modules, optimizers, functional ops, MPS detection, and CUDA device support:

```bash
PYTHONPATH=. python tests/smoke_test.py
```

### MPS Numerical Accuracy

Verifies every Accelerate-backed kernel against a NumPy reference implementation with tolerance < 1e-5:

```bash
PYTHONPATH=. python tests/test_mps_ops.py
```

### With pytest

```bash
make test
# or
python -m pytest tests/ -v
```

---

## Autograd

Scaffolding implements a tape-based reverse-mode automatic differentiation engine. Every `Tensor` operation records itself onto a computation graph when `requires_grad=True`.

```python
x = sf.tensor([2.0, 3.0], requires_grad=True)
y = x * x + x
y.sum().backward()
print(x.grad)  # [5.0, 7.0]  (dy/dx = 2x + 1)
```

**Context Managers:**

```python
with sf.no_grad():
    # Disables gradient tracking
    z = model(x)

with sf.inference_mode():
    # Like no_grad, for inference
    z = model(x)
```

---

## Modules & Layers

All modules inherit from `nn.Module` and support:

- `parameters()` — iterate all learnable parameters
- `named_parameters()` — iterate with names
- `state_dict()` / `load_state_dict()` — serialization
- `train()` / `eval()` — mode switching
- `to(device)` — device transfer (streams parameters one-by-one to avoid peak RSS spikes)
- `_release_cpu_shadows()` — drop CPU copies of GPU-resident parameters to free host RAM

### Available Layers

| Layer | Description |
|---|---|
| `nn.Linear(in, out)` | Fully-connected layer |
| `nn.Embedding(num, dim)` | Embedding lookup table |
| `nn.Conv1d(in, out, k)` | 1D convolution |
| `nn.Conv2d(in, out, k)` | 2D convolution (im2col-based, groups/stride/dilation) |
| `nn.Conv3d(in, out, k)` | 3D convolution for volumetric/temporal data |
| `nn.ConvTranspose2d(in, out, k)` | 2D transposed convolution (upsampling) |
| `nn.ConvTranspose3d(in, out, k)` | 3D transposed convolution |
| `nn.BatchNorm2d(features)` | 2D batch normalization with running stats |
| `nn.BatchNorm3d(features)` | 3D batch normalization |
| `nn.LayerNorm(dim)` | Layer normalization |
| `nn.RMSNorm(dim)` | Root mean square normalization |
| `nn.GroupNorm(groups, channels)` | Group normalization |
| `nn.AvgPool2d(k)` | Average pooling |
| `nn.MaxPool2d(k)` | Max pooling |
| `nn.AdaptiveAvgPool2d(size)` | Adaptive average pooling |
| `nn.Upsample(scale)` | Nearest/bilinear/trilinear upsampling |
| `nn.PixelShuffle(r)` | Sub-pixel convolution shuffle |
| `nn.Dropout(p)` | Dropout regularization |
| `nn.SiLU()` | SiLU / Swish activation |
| `nn.ReLU()` | ReLU activation |
| `nn.GELU()` | GELU activation |
| `nn.Tanh()` | Tanh activation |
| `nn.Sigmoid()` | Sigmoid activation |

### Weight Initialization

```python
from scaffolding.nn.init import xavier_uniform_, kaiming_normal_, normal_, zeros_

xavier_uniform_(layer.weight)
zeros_(layer.bias)
```

---

## Optimizers

### AdamW

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

optimizer.step()
optimizer.zero_grad()
```

On MPS devices, the AdamW step dispatches to a fused Accelerate kernel that computes moment updates, bias correction, and weight decay in a single vectorized pass.

### Learning Rate Schedulers

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(...)
    scheduler.step()
```

| Scheduler | Description |
|---|---|
| `CosineAnnealingLR` | Cosine annealing with warm restarts |
| `LambdaLR` | Custom lambda-based schedule |
| `StepLR` | Step decay at fixed intervals |

---

## Gradient Checkpointing

Trade compute for memory during training:

```python
from scaffolding.utils.checkpoint import checkpoint

class DeepModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.block1, x)
        x = checkpoint(self.block2, x)
        return x
```

---

## GPU Memory Management

Scaffolding is designed to run large models on GPU servers with very limited host RAM (e.g. 2.5 GB system RAM + 24 GB VRAM). Several mechanisms keep CPU memory near zero during inference:

### CPU Shadow Release

When a model is moved to GPU with `.to("cuda")`, parameters are streamed one-by-one to VRAM. After the upload, `_release_cpu_shadows()` drops the CPU-side NumPy arrays, freeing host memory:

```python
model.to("cuda")
output = model(x)
model._release_cpu_shadows()  # free CPU copies of all GPU-resident parameters
```

This is called automatically by `ToraPipeline` after each UNet mini-batch and VAE decode step.

### Mini-Batched UNet Inference

The `ToraPipeline` denoising loop processes frames in mini-batches of `_UNET_FRAME_BATCH` (default: 4) instead of all frames at once, reducing peak CPU activation memory from hundreds of MB to ~10-15 MB per step.

### Frame-by-Frame VAE Decode

After denoising, the VAE decodes one frame at a time from the latent tensor — never materializing the full `z_data` on the CPU. Each frame is immediately returned as a GPU-resident `sf.Tensor` (`uint8`, shape `(H, W, 3)`).

### Output Types

`ToraPipeline.__call__()` accepts `output_type`:
- `"pil"` (default) — returns PIL Images (requires CPU RAM for pixel buffers)
- `"sf_tensor"` — returns GPU-resident `sf.Tensor` objects, keeping all data in VRAM

### Host Memory Cleanup

On Linux, `_release_cpu_shadows()` calls `ctypes.CDLL("libc.so.6").malloc_trim(0)` to return freed pages to the OS immediately, preventing RSS bloat from glibc's `mmap`/`brk` arena retention.

---

## API Reference

### Tensor Factory Functions

| Function | Description |
|---|---|
| `sf.tensor(data)` | Create tensor from list/array |
| `sf.zeros(*shape)` | Tensor of zeros |
| `sf.ones(*shape)` | Tensor of ones |
| `sf.randn(*shape)` | Standard normal random tensor |
| `sf.rand(*shape)` | Uniform [0, 1) random tensor |
| `sf.randint(lo, hi, shape)` | Random integer tensor |
| `sf.arange(start, end, step)` | Range tensor |
| `sf.linspace(start, end, steps)` | Linearly spaced tensor |
| `sf.full(shape, value)` | Tensor filled with value |
| `sf.empty(*shape)` | Uninitialized tensor |

### Tensor Operations

| Operation | Description |
|---|---|
| `+`, `-`, `*`, `/`, `**` | Elementwise arithmetic |
| `@`, `sf.matmul(a, b)` | Matrix multiplication |
| `sf.cat(tensors, dim)` | Concatenation |
| `sf.stack(tensors, dim)` | Stacking |
| `sf.exp(x)`, `sf.log(x)` | Elementwise transcendentals |
| `sf.sigmoid(x)` | Sigmoid activation |
| `sf.softmax(x, dim)` | Softmax |
| `x.sum()`, `x.mean()` | Reductions |
| `x.reshape(shape)` | View/reshape |
| `x.transpose(d0, d1)` | Transpose |
| `x.contiguous()` | Ensure contiguous layout |

---

## Convolutions

Scaffolding implements 1D, 2D, and 3D convolutions with full support for stride, padding, dilation, groups, and bias. The forward pass uses an im2col approach with `np.einsum` for efficient batched matrix multiplication. Backward passes (autograd) are fully implemented.

```python
import scaffolding as sf
import scaffolding.nn as nn
import scaffolding.nn.functional as F

# 2D Convolution
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
x = sf.randn(4, 3, 32, 32)  # (batch, channels, height, width)
out = conv(x)               # (4, 64, 32, 32)

# 3D Convolution (video / volumetric)
conv3d = nn.Conv3d(3, 64, kernel_size=3, padding=1)
vid = sf.randn(1, 3, 16, 32, 32)  # (batch, channels, depth, height, width)
out3d = conv3d(vid)                # (1, 64, 16, 32, 32)

# Functional API
weight = sf.randn(64, 3, 3, 3)
out = F.conv2d(x, weight, padding=1)

# Transposed convolution (upsampling)
up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
out_up = up(out)  # spatial dimensions doubled
```

---

## Text-to-Video Generation

Scaffolding includes a complete text-to-video diffusion pipeline in `nn.video`. The system uses a 3D UNet denoising architecture with spatial-temporal attention and DDIM sampling.

### Architecture

| Component | Class | Description |
|---|---|---|
| Tokenizer | `TextTokenizer` | Hash-based word tokenizer (vocab 8192, max length 77) |
| Text Encoder | `TextEncoder` | Multi-layer transformer with self-attention + positional embeddings |
| Timestep Embedding | `TimestepEmbedding` | Sinusoidal encoding + MLP for diffusion timestep conditioning |
| Denoising UNet | `SimpleUNet3D` / `UNet3D` | Encoder-bottleneck-decoder with skip connections and cross-attention |
| Residual Block | `ResBlock3D` | Conv3d + GroupNorm + SiLU with timestep conditioning |
| Attention | `SpatialTemporalAttention` | Self-attention + cross-attention with text context |
| Noise Scheduler | `DDPMScheduler` / `DDIMScheduler` | Linear/cosine beta schedules, forward noise, reverse denoising |
| VAE (optional) | `VideoEncoder` / `VideoDecoder` | Encode video to latent space for latent diffusion |
| Pipeline | `TextToVideoPipeline` | End-to-end generation with classifier-free guidance |

### Usage

```python
from scaffolding.nn.video import TextToVideoPipeline

# Create pipeline
pipeline = TextToVideoPipeline(
    num_frames=16,
    frame_height=64,
    frame_width=64,
    text_embed_dim=256,
    model_channels=64,
    num_heads=4,
)

# Generate video from text prompt
video = pipeline.generate(
    prompt="a cat walking on grass",
    num_steps=50,
    guidance_scale=7.5,
)
print(video.shape)  # (1, 3, 16, 64, 64) — (B, C, T, H, W)

# Save frames to disk
pipeline.save_video_frames(video, output_dir="./output_frames")
```

### Training

```python
import scaffolding as sf
import scaffolding.optim as optim

pipeline = TextToVideoPipeline(num_frames=16, frame_height=64, frame_width=64)
optimizer = optim.AdamW(pipeline.parameters(), lr=1e-4)

# Training loop
for video_batch, prompts in dataloader:
    loss = pipeline.training_step(video_batch, prompts)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## License

Copyright © 2026 Pictofeed, LLC. All rights reserved.
