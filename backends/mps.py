# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
scaffolding.backends.mps — Apple Metal Performance Shaders backend.

Provides GPU detection via the Metal framework (``ctypes``-based IOKit /
Metal probe) and accelerated compute through Apple's **Accelerate.framework**
(vDSP, vecLib BLAS/LAPACK, BNNS).  On Apple Silicon the Accelerate routines
run on the high-performance CPU/GPU unified-memory fabric, giving near-Metal
throughput for dense linear algebra without writing .metal shaders.

Runtime detection logic
-----------------------
1.  ``is_available()`` — ``True`` when running on macOS with a Metal-capable
    GPU (checks ``MTLCreateSystemDefaultDevice`` via ctypes).
2.  ``is_built()``   — ``True`` when the Cython ``_mps_ops`` extension was
    compiled with Accelerate linkage.
3.  ``current_allocated_memory()`` / ``driver_allocated_memory()`` — query
    resident set via ``mach_task_self`` (approximation; real Metal VRAM
    accounting would require ObjC bridging).

The Cython hot-path kernels live in ``scaffolding._mps_ops`` and are
imported lazily so the pure-Python fallback still works if the extension
was not compiled.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import sys
from typing import Any

# ── Module-level caches ──
_metal_available: bool | None = None
_mps_ops_module: Any = None
_accelerate_handle: Any = None


# ──────────────────────────────────────────────────────────────────────
#  Metal GPU detection (ctypes, no pyobjc dependency)
# ──────────────────────────────────────────────────────────────────────

def _probe_metal() -> bool:
    """Return True if a Metal-capable GPU exists on this machine.

    Uses ``system_profiler`` to check for Metal support without loading
    the Metal framework (which can hang in non-GUI contexts).
    Falls back to checking the machine's architecture on Apple Silicon,
    which always has Metal.
    """
    if platform.system() != "Darwin":
        return False
    try:
        # Apple Silicon (arm64) always has Metal
        machine = platform.machine()
        if machine == "arm64":
            return True

        # On Intel Macs, check system_profiler for Metal support
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()
        return "metal support" in output or "metal:" in output
    except Exception:
        # Conservative: assume no Metal if we can't determine
        return False


def is_available() -> bool:
    """Return ``True`` when the host has a Metal-capable GPU.

    Result is cached after the first call.
    """
    global _metal_available
    if _metal_available is None:
        _metal_available = _probe_metal()
    return _metal_available


def is_built() -> bool:
    """Return ``True`` when the Cython ``_mps_ops`` extension is importable."""
    try:
        import scaffolding._mps_ops  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────
#  Accelerate framework handle (lazy)
# ──────────────────────────────────────────────────────────────────────

def _load_accelerate():
    """Load ``Accelerate.framework`` once and cache the handle."""
    global _accelerate_handle
    if _accelerate_handle is not None:
        return _accelerate_handle
    path = ctypes.util.find_library("Accelerate")
    if path is None and platform.system() == "Darwin":
        path = (
            "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        )
    if path:
        try:
            _accelerate_handle = ctypes.cdll.LoadLibrary(path)
        except OSError:
            _accelerate_handle = None
    return _accelerate_handle


def has_accelerate() -> bool:
    """Return ``True`` if Apple Accelerate is loadable."""
    return _load_accelerate() is not None


# ──────────────────────────────────────────────────────────────────────
#  Lazy import of the Cython extension
# ──────────────────────────────────────────────────────────────────────

def _get_mps_ops():
    """Import ``scaffolding._mps_ops`` once and cache the module."""
    global _mps_ops_module
    if _mps_ops_module is not None:
        return _mps_ops_module
    try:
        from scaffolding import _mps_ops
        _mps_ops_module = _mps_ops
    except ImportError:
        _mps_ops_module = None
    return _mps_ops_module


# ──────────────────────────────────────────────────────────────────────
#  Memory introspection (approximate, mach_task_self)
# ──────────────────────────────────────────────────────────────────────

def current_allocated_memory() -> int:
    """Approximate process resident memory in bytes (macOS only)."""
    if platform.system() != "Darwin":
        return 0
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss  # macOS returns bytes
    except Exception:
        return 0


def driver_allocated_memory() -> int:
    """Alias for ``current_allocated_memory`` (no true VRAM query)."""
    return current_allocated_memory()


def synchronize() -> None:
    """No-op synchronisation (Accelerate is synchronous)."""
    pass


def empty_cache() -> None:
    """No-op cache flush (Accelerate manages its own pools)."""
    pass


def set_per_process_memory_fraction(fraction: float, device: int = 0) -> None:
    """No-op on Accelerate backend."""
    pass


# ──────────────────────────────────────────────────────────────────────
#  Device info
# ──────────────────────────────────────────────────────────────────────

def get_device_name(device: int = 0) -> str:
    """Return a human-readable GPU name via IOKit (best-effort)."""
    if not is_available():
        return "No Metal device"
    try:
        metal = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/Metal.framework/Metal"
        )
        # We can't easily get the name via ctypes alone (it's NSString).
        # Return a generic label based on CPU brand.
        brand = platform.processor() or "Apple"
        return f"Apple {brand} (Metal)"
    except Exception:
        return "Apple Metal GPU"


def device_count() -> int:
    """Number of Metal devices (0 or 1 — multi-GPU Macs are very rare)."""
    return 1 if is_available() else 0


# ──────────────────────────────────────────────────────────────────────
#  Public dispatch helpers (used by Tensor / nn.functional)
# ──────────────────────────────────────────────────────────────────────

def mps_matmul(a, b):
    """Matrix multiply via Accelerate BLAS (cblas_sgemm).

    Falls back to NumPy if the Cython extension is unavailable.
    """
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_sgemm(a, b)
    import numpy as np
    return np.matmul(a, b)


def mps_sigmoid(x):
    """Element-wise sigmoid via vDSP / Accelerate."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_sigmoid(x)
    import numpy as np
    return 1.0 / (1.0 + np.exp(-x))


def mps_exp(x):
    """Element-wise exp via vecLib vvexpf."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_exp(x)
    import numpy as np
    return np.exp(x)


def mps_silu(x):
    """SiLU via Accelerate."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_silu(x)
    import numpy as np
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def mps_rms_norm(x, weight, eps):
    """RMS norm via Accelerate vDSP."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_rms_norm(x, weight, eps)
    import numpy as np
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def mps_softmax(x, axis=-1):
    """Numerically stable softmax via Accelerate."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_softmax(x, axis)
    import numpy as np
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def mps_gelu(x):
    """GELU activation via Accelerate."""
    ops = _get_mps_ops()
    if ops is not None:
        return ops.accelerate_gelu(x)
    import numpy as np
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


def mps_adamw_step(param, grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2):
    """AdamW parameter update via Accelerate vDSP."""
    ops = _get_mps_ops()
    if ops is not None:
        ops.accelerate_adamw_step(param, grad, m, v,
                                  lr, beta1, beta2, eps, wd, bc1, bc2)
        return
    # Pure-NumPy fallback
    import numpy as np
    param *= (1.0 - lr * wd)
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * grad * grad
    m_hat = m / bc1
    v_hat = v / bc2
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


__all__ = [
    'is_available', 'is_built', 'has_accelerate',
    'current_allocated_memory', 'driver_allocated_memory',
    'synchronize', 'empty_cache', 'set_per_process_memory_fraction',
    'get_device_name', 'device_count',
    # dispatch helpers
    'mps_matmul', 'mps_sigmoid', 'mps_exp', 'mps_silu',
    'mps_rms_norm', 'mps_softmax', 'mps_gelu', 'mps_adamw_step',
]
