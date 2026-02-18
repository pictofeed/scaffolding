#!/usr/bin/env python3
"""Test multi-GPU / DataParallel support."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scaffolding as torch
import scaffolding.nn as nn
from scaffolding.nn.parallel import DataParallel


# ── 1. Device parsing ──
d0 = torch.device('cuda:0')
d1 = torch.device('cuda:1')
assert d0.type == 'cuda' and d0.index == 0, f"Bad d0: {d0}"
assert d1.type == 'cuda' and d1.index == 1, f"Bad d1: {d1}"
print("[PASS] Device parsing: cuda:0 / cuda:1")

# ── 2. DataParallel construction ──
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
dp = DataParallel(model, device_ids=[0, 1])
assert dp._device_ids == [0, 1]
assert dp._output_device == 0
print(f"[PASS] DataParallel created: devices={dp._device_ids}")

# ── 3. Single-GPU / CPU fallback ──
t = torch.Tensor(np.random.randn(8, 8).astype(np.float32))
out = dp(t)
assert out.shape == (8, 4), f"Bad shape: {out.shape}"
print(f"[PASS] Forward (CPU fallback): {out.shape}")

# ── 4. nn.DataParallel from layers ──
dp2 = nn.DataParallel(model, device_ids=[0])
out2 = dp2(t)
assert out2.shape == (8, 4)
print(f"[PASS] nn.DataParallel alias: {out2.shape}")

# ── 5. cuda.init_multi_gpu (no-op on macOS) ──
n = torch.cuda.init_multi_gpu()
print(f"[PASS] cuda.init_multi_gpu() -> {n} devices")

# ── 6. Module.cuda(device=...) ──
m2 = SimpleModel()
m2_cuda = m2.cuda(0)
assert m2_cuda is m2  # in-place
print("[PASS] Module.cuda(0)")

# ── 7. Tensor.cuda(device_id=...) ──
t_cuda = t.cuda(0)
assert str(t_cuda.device) == 'cuda:0'
print("[PASS] Tensor.cuda(0)")

t_cuda1 = t.cuda(1)
assert str(t_cuda1.device) == 'cuda:1'
print("[PASS] Tensor.cuda(1)")

print("\n=== ALL MULTI-GPU TESTS PASSED ===")
