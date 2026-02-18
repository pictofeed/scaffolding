# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.Module — base class for all neural network modules."""
from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Iterator

from ..tensor import Tensor
from .parameter import Parameter


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
                    target._data = val._ensure_cpu().copy()
                    target._gpu = None
                elif isinstance(val, np.ndarray):
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
        for p in self._parameters.values():
            p.to(*args, **kwargs)
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
        return self.to('cuda')

    def cpu(self) -> 'Module':
        return self.to('cpu')

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
