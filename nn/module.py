# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.Module — base class for all neural network modules."""
from __future__ import annotations

import gc
import sys
import numpy as np
from collections import OrderedDict
from typing import Iterator

from ..tensor import Tensor
from .parameter import Parameter

# Pre-load malloc_trim on Linux for releasing freed pages to the OS.
_malloc_trim = None
if sys.platform == 'linux':
    try:
        import ctypes as _ctypes
        _malloc_trim = _ctypes.CDLL('libc.so.6').malloc_trim
    except Exception:
        pass

def _release_host_memory():
    """Full GC pass + malloc_trim to claw back host RSS."""
    gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


class Module:
    """Base class for all neural network modules.

    Mirrors :class:`torch.nn.Module` API.
    """

    _training: bool
    _modules: OrderedDict
    _parameters: OrderedDict
    _buffers: OrderedDict

    def __init__(self):
        object.__setattr__(self, '_training', True)
        object.__setattr__(self, '_modules', OrderedDict())
        object.__setattr__(self, '_parameters', OrderedDict())
        object.__setattr__(self, '_buffers', OrderedDict())

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---- Attribute management ----

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, (ModuleList,)):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        params = object.__getattribute__(self, '_parameters')
        if name in params:
            return params[name]
        modules = object.__getattribute__(self, '_modules')
        if name in modules:
            return modules[name]
        buffers = object.__getattribute__(self, '_buffers')
        if name in buffers:
            return buffers[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __delattr__(self, name: str):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        elif name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    # ---- Register buffer ----

    def register_buffer(self, name: str, tensor: Tensor | None,
                        persistent: bool = True):
        self._buffers[name] = tensor

    # ---- Parameter access ----

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                if isinstance(m, Module):
                    yield from m.parameters(recurse=True)
                elif isinstance(m, ModuleList):
                    for sub in m:
                        yield from sub.parameters(recurse=True)

    def named_parameters(self, prefix: str = '',
                         recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, p
        if recurse:
            for mname, m in self._modules.items():
                full_prefix = f"{prefix}.{mname}" if prefix else mname
                if isinstance(m, Module):
                    yield from m.named_parameters(full_prefix, recurse=True)
                elif isinstance(m, ModuleList):
                    for i, sub in enumerate(m):
                        sub_prefix = f"{full_prefix}.{i}"
                        yield from sub.named_parameters(sub_prefix, recurse=True)

    def named_modules(self, prefix: str = '') -> Iterator[tuple[str, 'Module']]:
        yield prefix, self
        for name, m in self._modules.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(m, Module):
                yield from m.named_modules(full_name)
            elif isinstance(m, ModuleList):
                for i, sub in enumerate(m):
                    sub_name = f"{full_name}.{i}"
                    yield from sub.named_modules(sub_name)

    # ---- State dict ----

    def state_dict(self) -> dict:
        sd = OrderedDict()
        for name, p in self._parameters.items():
            sd[name] = p
        for name, buf in self._buffers.items():
            if buf is not None:
                sd[name] = buf
        for mname, m in self._modules.items():
            if isinstance(m, Module):
                child_sd = m.state_dict()
                for k, v in child_sd.items():
                    sd[f"{mname}.{k}"] = v
            elif isinstance(m, ModuleList):
                for i, sub in enumerate(m):
                    child_sd = sub.state_dict()
                    for k, v in child_sd.items():
                        sd[f"{mname}.{i}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        own_sd = self.state_dict()
        for key, val in state_dict.items():
            if key in own_sd:
                target = own_sd[key]
                if isinstance(val, Tensor):
                    # If the target parameter lives on GPU, upload the
                    # incoming weight directly without keeping a CPU copy.
                    if target._gpu is not None:
                        arr = val._ensure_cpu()
                        from ..tensor import _USE_CUDA, _cuops
                        dev_id = (target._device._index
                                  if target._device._index is not None else 0)
                        target._gpu = _cuops.gputensor_from_numpy(
                            np.ascontiguousarray(arr), dev_id)
                        target._data = None
                    else:
                        target._data = val._ensure_cpu().copy()
                        target._gpu = None
                elif isinstance(val, np.ndarray):
                    if target._gpu is not None:
                        from ..tensor import _USE_CUDA, _cuops
                        dev_id = (target._device._index
                                  if target._device._index is not None else 0)
                        target._gpu = _cuops.gputensor_from_numpy(
                            np.ascontiguousarray(val), dev_id)
                        target._data = None
                    else:
                        target._data = val.copy()
                        target._gpu = None

    # ---- Training mode ----

    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> 'Module':
        self._training = mode
        for m in self._modules.values():
            if isinstance(m, Module):
                m.train(mode)
            elif isinstance(m, ModuleList):
                for sub in m:
                    sub.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    # ---- Device / dtype ----

    def to(self, *args, **kwargs) -> 'Module':
        # Stream parameters to device one-by-one so that the CPU copy
        # of each weight can be freed before the next one is uploaded.
        # This keeps peak host RSS ≈ size-of-largest-parameter instead
        # of ≈ total-model-size, which is critical on RAM-limited
        # machines with large GPU VRAM (e.g. 4 GB RAM + 12 GB K80).
        for name, p in self._parameters.items():
            new_p = Tensor.to(p, *args, **kwargs)
            # Update parameter in-place
            p._data = new_p._data
            p._gpu = new_p._gpu
            p._device = new_p._device
            # Immediately drop the CPU shadow so RSS can shrink
            if p._gpu is not None:
                p._data = None
            del new_p
        # Release freed numpy arrays back to the OS
        _release_host_memory()
        for name, buf in self._buffers.items():
            if buf is not None:
                self._buffers[name] = buf.to(*args, **kwargs)
        for m in self._modules.values():
            if isinstance(m, Module):
                m.to(*args, **kwargs)
            elif isinstance(m, ModuleList):
                for sub in m:
                    sub.to(*args, **kwargs)
        return self

    def cuda(self, device=None) -> 'Module':
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda:0')

    def cpu(self) -> 'Module':
        return self.to('cpu')

    def _release_cpu_shadows(self) -> None:
        """Drop CPU copies of parameters that are backed by GPU memory.

        During a forward pass, ``_ensure_cpu()`` downloads every accessed
        parameter to CPU while keeping the GPU copy alive (because
        ``_requires_grad=True``).  On a RAM-limited host this can double
        peak RSS because *every* parameter ends up resident on both
        CPU and GPU simultaneously.

        Call this after a forward pass completes to free those CPU
        shadows — the GPU copy remains canonical and will be re-
        downloaded on the next forward pass.  The cost is one extra
        device→host copy per step, but it prevents the OOM-killer
        from terminating the process on machines with ≤ 2.5 GB RAM.
        """
        for p in self.parameters():
            if p._gpu is not None and p._data is not None:
                p._data = None       # drop CPU shadow; GPU copy stays
        _release_host_memory()

    # ---- Gradient management ----

    def zero_grad(self, set_to_none: bool = False):
        for p in self.parameters():
            if set_to_none:
                p._grad = None
            elif p._grad is not None:
                p._grad = np.zeros_like(p._grad)

    # ---- Utilities ----

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, m in self._modules.items():
            child_repr = repr(m).replace('\n', '\n  ')
            lines.append(f"  ({name}): {child_repr}")
        for name, p in self._parameters.items():
            lines.append(f"  ({name}): Parameter({p.shape})")
        lines.append(")")
        return '\n'.join(lines)

    @property
    def module(self):
        """For DDP/DataParallel compatibility — return self."""
        return self

    def _apply(self, fn):
        for p in self._parameters.values():
            p._ensure_cpu()
            p._data = fn(p._data)
            p._gpu = None
        for name, buf in self._buffers.items():
            if buf is not None:
                buf._ensure_cpu()
                self._buffers[name]._data = fn(buf._data)
                self._buffers[name]._gpu = None
        for m in self._modules.values():
            if isinstance(m, Module):
                m._apply(fn)
            elif isinstance(m, ModuleList):
                for sub in m:
                    sub._apply(fn)
        return self


# ---- Container modules ----

class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules=None):
        super().__init__()
        self._module_list: list[Module] = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module: Module) -> 'ModuleList':
        idx = len(self._module_list)
        self._module_list.append(module)
        self._modules[str(idx)] = module
        return self

    def __getitem__(self, idx):
        return self._module_list[idx]

    def __len__(self) -> int:
        return len(self._module_list)

    def __iter__(self):
        return iter(self._module_list)

    def __repr__(self) -> str:
        lines = ["ModuleList("]
        for i, m in enumerate(self._module_list):
            lines.append(f"  ({i}): {repr(m)}")
        lines.append(")")
        return '\n'.join(lines)


class Sequential(Module):
    """A sequential container."""

    def __init__(self, *args):
        super().__init__()
        self._seq_modules: list[Module] = []
        for i, m in enumerate(args):
            self._seq_modules.append(m)
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._seq_modules:
            x = m(x)
        return x

    def __len__(self) -> int:
        return len(self._seq_modules)

    def __getitem__(self, idx):
        return self._seq_modules[idx]

    def __iter__(self):
        return iter(self._seq_modules)
