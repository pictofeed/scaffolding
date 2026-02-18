#!/usr/bin/env python3
"""Quick smoke test for Scaffolding with MPS backend."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scaffolding as torch
import scaffolding.nn as nn
import scaffolding.nn.functional as F
import scaffolding.optim as optim
import scaffolding.distributed as dist
from scaffolding.backends import mps, cudnn, cuda as cuda_be

print("=== Scaffolding Smoke Test ===")
#print(f"Version:       {torch.__version__}")
print(f"MPS available: {mps.is_available()}")
print(f"MPS built:     {mps.is_built()}")
print(f"Accelerate:    {mps.has_accelerate()}")
print(f"Device name:   {mps.get_device_name()}")
print(f"Device count:  {mps.device_count()}")
print()

# Tensor creation
t = torch.tensor([1.0, 2.0, 3.0])
print(f"CPU tensor: {t}")
t_mps = t.to('mps')
print(f"MPS tensor: {t_mps}, device={t_mps.device}")
t_back = t_mps.cpu()
print(f"Back to CPU: {t_back}, device={t_back.device}")

# Factory functions
z = torch.zeros(2, 3)
o = torch.ones(2, 3)
r = torch.randn(2, 3)
print(f"zeros: {z.shape}, ones: {o.shape}, randn: {r.shape}")

# Autograd
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(f"Autograd: x={x}, x.grad={x.grad}")

# nn.Module
linear = nn.Linear(4, 2)
inp = torch.randn(1, 4)
out = linear(inp)
print(f"nn.Linear: in={inp.shape} -> out={out.shape}")

# nn.Embedding
emb = nn.Embedding(10, 8)
idx = torch.tensor([0, 3, 7], dtype=torch.long)
e_out = emb(idx)
print(f"nn.Embedding: {idx.shape} -> {e_out.shape}")

# F.softmax
logits = torch.randn(2, 5)
probs = F.softmax(logits, dim=-1)
print(f"F.softmax: {probs.shape}, sum={probs.sum().item():.4f}")

# F.cross_entropy
targets = torch.tensor([1, 3])
loss = F.cross_entropy(logits, targets)
print(f"F.cross_entropy: {loss.item():.4f}")

# F.silu
silu_out = F.silu(torch.randn(3))
print(f"F.silu: {silu_out.shape}")

# F.gelu
gelu_out = F.gelu(torch.randn(3))
print(f"F.gelu: {gelu_out.shape}")

# Optimizer
model = nn.Linear(4, 2)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
optimizer.zero_grad()
out = model(torch.randn(1, 4))
loss = out.sum()
loss.backward()
optimizer.step()
print(f"AdamW step: OK")

# no_grad context
with torch.no_grad():
    t2 = torch.tensor([1.0], requires_grad=True)
    t3 = t2 * 2
    print(f"no_grad: requires_grad={t3.requires_grad}")

# MPS dispatch helpers
print()
print("=== MPS dispatch helpers ===")
print(f"mps_matmul: {type(mps.mps_matmul.__name__)}")
print(f"mps_sigmoid: {type(mps.mps_sigmoid.__name__)}")
print(f"mps_exp: {type(mps.mps_exp.__name__)}")
print(f"mps_softmax: {type(mps.mps_softmax.__name__)}")
print(f"mps_gelu: {type(mps.mps_gelu.__name__)}")

# Backends
print()
print(f"cudnn.benchmark: {cudnn.benchmark}")
print(f"cuda.matmul.allow_tf32: {cuda_be.matmul.allow_tf32}")
print(f"dist.is_initialized: {dist.is_initialized()}")

print()
print("=== ALL TESTS PASSED ===")
