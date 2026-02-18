<div align="center">
    <img src="img/scaffolding-logo.png" alt="Scaffolding Logo" width="500"/>
</div>

# Scaffolding

**A deep learning framework written entirely in Python and Cython.**

Scaffolding is a lightweight, production-ready deep learning framework built from the ground up using NumPy as its computational backend and Cython-accelerated hot-path operations for maximum throughput. On macOS, Scaffolding leverages Apple's Accelerate framework (BLAS, vDSP, vecLib) through native Cython bindings for near-hardware-level performance on both CPU and Apple Silicon.

---

## Features

- **Pure Python + Cython** — no external C++ dependencies; every operation is implemented in Python with Cython kernels for the performance-critical paths
- **Automatic Differentiation** — full reverse-mode autograd engine with gradient accumulation, `no_grad` / `inference_mode` context managers, and `retain_graph` support
- **Neural Network Modules** — `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`, `nn.Dropout`, and more
- **Functional API** — `nn.functional` with `softmax`, `cross_entropy`, `silu`, `gelu`, `relu`, `dropout`, `embedding`, and others
- **Optimizers** — `AdamW` with weight decay, bias correction, and learning rate scheduling (`CosineAnnealingLR`, `LambdaLR`, `StepLR`)
- **Apple MPS Backend** — Cython-wrapped Apple Accelerate.framework (cblas, vDSP, vecLib) for hardware-accelerated matmul, elementwise ops, softmax, RMS norm, and cross-entropy on macOS
- **Gradient Checkpointing** — memory-efficient training via `utils.checkpoint`
- **Device Abstraction** — `cpu` and `mps` device types with `.to()`, `.cpu()`, `.mps()` tensor methods
- **Dtype System** — `float16`, `float32`, `float64`, `bfloat16`, `int8`–`int64`, `uint8`, `bool`
- **Two Build Systems** — setuptools + Cython *and* CMake, with a Makefile convenience wrapper

---

## Architecture

```
scaffolding/
├── __init__.py              # Public API surface
├── tensor.py                # Tensor class, factory functions, math ops
├── autograd.py              # Reverse-mode autograd engine
├── device.py                # Device abstraction (cpu, mps)
├── dtype.py                 # Data type definitions
│
├── _tensor_ops.pyx          # Cython: generic CPU hot-path kernels
├── _mps_ops.pyx             # Cython: Apple Accelerate-backed kernels
│
├── nn/
│   ├── module.py            # nn.Module base class
│   ├── parameter.py         # nn.Parameter
│   ├── layers.py            # Linear, Embedding, LayerNorm, RMSNorm, Dropout
│   ├── functional.py        # Functional API (softmax, cross_entropy, silu, etc.)
│   ├── init.py              # Weight initialization (xavier, kaiming, normal, etc.)
│   ├── parallel.py          # DistributedDataParallel stub
│   └── utils.py             # clip_grad_norm_
│
├── optim/
│   ├── optimizer.py         # AdamW optimizer
│   └── lr_scheduler.py      # Learning rate schedulers
│
├── backends/
│   ├── mps.py               # MPS/Accelerate backend detection & dispatch
│   ├── cuda.py              # CUDA backend stub
│   └── cudnn.py             # cuDNN backend stub
│
├── cuda/
│   └── amp.py               # Automatic mixed precision stub
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

### 1. Setuptools + Cython (Default)

```bash
# Build extensions in-place for development
python setup.py build_ext --inplace

# Or use the Makefile shortcut
make build
```

This compiles `_tensor_ops.pyx` and (on macOS) `_mps_ops.pyx` into shared libraries using Cython's optimized compiler directives:

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

MPS extensions are conditionally compiled only when Accelerate is found.

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

Runs an end-to-end check of all major subsystems — tensors, autograd, modules, optimizers, functional ops, and MPS detection:

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
- `to(device)` — device transfer

### Available Layers

| Layer | Description |
|---|---|
| `nn.Linear(in, out)` | Fully-connected layer |
| `nn.Embedding(num, dim)` | Embedding lookup table |
| `nn.LayerNorm(dim)` | Layer normalization |
| `nn.RMSNorm(dim)` | Root mean square normalization |
| `nn.Dropout(p)` | Dropout regularization |

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

## License

Copyright © 2026 Pictofeed, LLC. All rights reserved.
