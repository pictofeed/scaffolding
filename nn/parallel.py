# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""nn.parallel — DistributedDataParallel & DataParallel."""
from __future__ import annotations

from .module import Module


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
