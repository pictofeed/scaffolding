# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.parallel — DataParallel & DistributedDataParallel.

DataParallel splits input batches across multiple GPUs, runs the forward
pass on each replica in parallel (via CUDA streams), and gathers outputs
on the primary device.  Optimised for the K80 (2 × GK210 on one board).
"""
from __future__ import annotations

import copy
import numpy as np
from typing import List

from .module import Module
from ..tensor import Tensor, _USE_CUDA, _is_cuda
from ..device import device as Device

# Try importing CUDA ops for multi-GPU support
try:
    from .. import _cuda_ops as _cuops
    _HAS_CUDA = True
except ImportError:
    _cuops = None
    _HAS_CUDA = False


def _get_available_devices() -> list[int]:
    """Return list of available CUDA device indices."""
    if not _HAS_CUDA:
        return []
    try:
        n = _cuops.get_device_count()
        return list(range(n))
    except Exception:
        return []


# ================================================================
#  DataParallel — real multi-GPU batch splitting
# ================================================================

class DataParallel(Module):
    """Split-batch data parallelism across multiple GPUs.

    Usage::

        model = DataParallel(model, device_ids=[0, 1])
        output = model(input)   # input is split across GPUs automatically

    For the K80 with 2 GK210 dies, this effectively doubles throughput
    for large enough batches.

    Architecture:
      1. Scatter — split batch dim across devices
      2. Replicate — copy model params to each device
      3. Parallel forward — run forward on each device chunk
      4. Gather — concatenate outputs on the primary device
    """

    def __init__(self, module: Module, device_ids: list[int] | None = None,
                 output_device: int | None = None, dim: int = 0):
        super().__init__()
        if not _HAS_CUDA:
            self._modules['module'] = module
            self._device_ids = [0]
            self._output_device = 0
            self._dim = dim
            return

        available = _get_available_devices()
        if device_ids is None:
            device_ids = available if available else [0]
        self._device_ids = device_ids
        self._output_device = output_device if output_device is not None else device_ids[0]
        self._dim = dim

        # Store the module on the primary device
        self._modules['module'] = module

        # Cache for replicated parameter GpuTensors per non-primary device
        # Maps (param_id, device_id) → GpuTensor
        self._param_cache: dict = {}

        # Initialize P2P and all devices
        if len(device_ids) > 1:
            _cuops.init_all_devices()

    @property
    def module(self) -> Module:
        return self._modules['module']

    def forward(self, *args, **kwargs):
        # Single-GPU fast path — no overhead
        if len(self._device_ids) <= 1 or not _HAS_CUDA:
            return self.module(*args, **kwargs)

        # Multi-GPU path
        inputs = args
        # Split first positional argument along batch dim
        scattered = self._scatter(inputs, self._device_ids, self._dim)
        # Replicate model parameters to each device
        replicas = self._replicate(self.module, self._device_ids)
        # Run forward on each device
        outputs = self._parallel_apply(replicas, scattered, self._device_ids, kwargs)
        # Gather outputs back to primary device
        return self._gather(outputs, self._output_device, self._dim)

    def _scatter(self, inputs, device_ids, dim=0):
        """Split input tensors across devices along dim."""
        n_devices = len(device_ids)
        # inputs is a tuple of args. We split each Tensor arg.
        result = []
        for dev_idx, dev_id in enumerate(device_ids):
            chunk_args = []
            for inp in inputs:
                if isinstance(inp, Tensor) and inp._gpu is not None:
                    # Split along batch dim
                    batch_size = inp.shape[dim]
                    chunk_size = (batch_size + n_devices - 1) // n_devices
                    start = dev_idx * chunk_size
                    end = min(start + chunk_size, batch_size)
                    if start >= batch_size:
                        # This device gets no data
                        chunk_args.append(None)
                        continue
                    # Slice the tensor
                    cpu_data = inp._ensure_cpu()
                    chunk_data = cpu_data[start:end].copy()
                    chunk_t = Tensor(chunk_data, device=Device(f'cuda:{dev_id}'))
                    chunk_args.append(chunk_t)
                elif isinstance(inp, Tensor):
                    # CPU tensor — just split the numpy data
                    cpu_data = inp._ensure_cpu()
                    batch_size = cpu_data.shape[dim]
                    chunk_size = (batch_size + n_devices - 1) // n_devices
                    start = dev_idx * chunk_size
                    end = min(start + chunk_size, batch_size)
                    chunk_data = cpu_data[start:end].copy()
                    chunk_t = Tensor(chunk_data, device=inp._device)
                    chunk_args.append(chunk_t)
                else:
                    chunk_args.append(inp)
            result.append(tuple(chunk_args))
        return result

    def _replicate(self, module, device_ids):
        """Create module replicas on each device.

        The primary device (device_ids[0]) uses the original module.
        Other devices get parameter copies via gputensor_to_device.
        """
        replicas = [module]  # Primary uses original

        for dev_id in device_ids[1:]:
            # Create a shallow copy of the module structure
            replica = self._copy_module_to_device(module, dev_id)
            replicas.append(replica)

        return replicas

    def _copy_module_to_device(self, module, target_dev):
        """Deep-copy a module's parameters to target device."""
        import copy as _copy
        # Use a simple approach: copy the module, then move params
        replica = _copy.copy(module)
        # Copy internal dicts (don't share with original)
        replica._parameters = type(module._parameters)(module._parameters)
        replica._modules = type(module._modules)(module._modules)
        replica._buffers = type(module._buffers)(module._buffers)

        # Move each parameter to target device
        from .parameter import Parameter as Param
        for name, p in list(replica._parameters.items()):
            if p is not None:
                cache_key = (id(p), target_dev)
                if cache_key in self._param_cache:
                    # Reuse cached GPU copy
                    new_p = Param.__new__(Param)
                    new_p._data = p._data
                    new_p._gpu = self._param_cache[cache_key]
                    new_p._device = Device(f'cuda:{target_dev}')
                    new_p._requires_grad = p._requires_grad
                    new_p._grad = None
                    new_p._grad_fn = None
                    new_p._version = 0
                else:
                    # Create GPU copy on target device
                    new_p = p.to(f'cuda:{target_dev}')
                    self._param_cache[cache_key] = new_p._gpu
                replica._parameters[name] = new_p

        # Recursively handle submodules
        for name, child in list(replica._modules.items()):
            if isinstance(child, Module):
                replica._modules[name] = self._copy_module_to_device(child, target_dev)

        return replica

    def _parallel_apply(self, replicas, scattered, device_ids, kwargs):
        """Run forward on each replica.

        For K80 (2 GPUs), this runs sequentially but each on its own
        device/stream.  CUDA kernels are asynchronous, so both GPUs
        work in parallel after the launches.
        """
        outputs = []
        for replica, chunk_args, dev_id in zip(replicas, scattered, device_ids):
            # Check if any arg is None (no data for this device)
            if any(a is None for a in chunk_args if isinstance(a, (Tensor, type(None)))):
                outputs.append(None)
                continue
            # Set device context
            _cuops.set_device(dev_id)
            out = replica(*chunk_args, **kwargs)
            outputs.append(out)

        # Sync all devices
        for dev_id in device_ids:
            _cuops.set_device(dev_id)
            _cuops.device_synchronize()

        # Restore primary device
        _cuops.set_device(device_ids[0])
        return [o for o in outputs if o is not None]

    def _gather(self, outputs, output_device, dim=0):
        """Gather outputs from all devices to output_device."""
        if len(outputs) == 0:
            return None
        if len(outputs) == 1:
            return outputs[0]

        # Check if outputs are tuples (e.g., cross-entropy returns (loss, probs))
        if isinstance(outputs[0], (tuple, list)):
            gathered_parts = []
            for i in range(len(outputs[0])):
                elements = [o[i] for o in outputs]
                gathered_parts.append(
                    self._gather(elements, output_device, dim))
            return type(outputs[0])(gathered_parts)

        # Download all to CPU
        cpu_arrays = []
        for out in outputs:
            if isinstance(out, Tensor):
                arr = out._ensure_cpu()
                if arr is None:
                    arr = out._data
                cpu_arrays.append(arr)
            elif isinstance(out, np.ndarray):
                cpu_arrays.append(out)
            else:
                cpu_arrays.append(np.asarray(out))

        if cpu_arrays:
            # Scalar (0-d) outputs: average them (e.g., loss values)
            if cpu_arrays[0].ndim == 0:
                gathered = np.mean(cpu_arrays).astype(cpu_arrays[0].dtype)
            else:
                gathered = np.concatenate(cpu_arrays, axis=dim)
            result = Tensor(gathered, device=Device(f'cuda:{output_device}'))
            return result

        return outputs[0]

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix, recurse=recurse)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.module.load_state_dict(sd, strict=strict)

    def train(self, mode=True):
        self.module.train(mode)
        self._training = mode
        # Invalidate param cache on mode change
        if hasattr(self, '_param_cache'):
            self._param_cache.clear()
        return self

    def eval(self):
        return self.train(False)


class DistributedDataParallel(Module):
    """DDP wrapper — pass-through on CPU/Scaffolding.

    In a real distributed setting each rank would all-reduce gradients
    after the backward pass.  Here we simply forward to the wrapped
    module (single-process execution).
    """

    def __init__(self, module: Module, device_ids=None,
                 output_device=None, **kwargs):
        super().__init__()
        self._modules['module'] = module

    @property
    def module(self) -> Module:
        return self._modules['module']

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix, recurse=recurse)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.module.load_state_dict(sd, strict=strict)

    def train(self, mode=True):
        self.module.train(mode)
        self._training = mode
        return self

    def eval(self):
        return self.train(False)
