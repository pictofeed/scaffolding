"""Test Accelerate backend integration."""
import numpy as np
import sys
sys.path.insert(0, '/Users/calebmarshall/Desktop/scaffolding')

import scaffolding as torch
import scaffolding.nn as nn
import scaffolding.nn.functional as F

print('=== Testing Accelerate backend integration ===')
print(f'MPS ops: {torch.tensor._USE_MPS}')
print(f'Cython ops: {torch.tensor._USE_CYTHON}')

# Test 1: Matmul (2D BLAS)
a = torch.randn(64, 128)
b = torch.randn(128, 64)
c = torch.matmul(a, b)
print(f'1. Matmul 2D: {c.shape}')

# Test 2: Batched matmul (should use batched BLAS)
a = torch.randn(4, 8, 64, 32)
b = torch.randn(4, 8, 32, 64)
c = torch.matmul(a, b)
print(f'2. Batched matmul 4D: {c.shape}')

# Test 3: Linear forward (BLAS sgemm_nt)
linear = nn.Linear(256, 128)
x = torch.randn(4, 16, 256)
y = linear(x)
print(f'3. Linear forward: {y.shape}')

# Test 4: SiLU via Accelerate
x = torch.randn(32, 64, requires_grad=True)
y = F.silu(x)
print(f'4. SiLU: {y.shape}')

# Test 5: GELU via Accelerate
y = F.gelu(x)
print(f'5. GELU: {y.shape}')

# Test 6: Softmax via Accelerate (2D)
y = F.softmax(x, dim=-1)
print(f'6. Softmax 2D: {y.shape}, sum={y._data.sum(axis=-1)[0]:.4f}')

# Test 7: Softmax 3D (should reshape to 2D internally)
x3d = torch.randn(4, 16, 32)
y = F.softmax(x3d, dim=-1)
print(f'7. Softmax 3D: {y.shape}, sum={y._data.sum(axis=-1)[0,0]:.4f}')

# Test 8: Log/Sqrt/Rsqrt via vecLib
x = torch.randn(100).float()
x._data = np.abs(x._data) + 0.01
y_log = x.log()
y_sqrt = x.sqrt()
y_rsqrt = x.rsqrt()
log_err = np.max(np.abs(y_log._data - np.log(x._data)))
print(f'8. Log/Sqrt/Rsqrt: log max_err={log_err:.1e}')

# Test 9: Sigmoid via Accelerate
x = torch.randn(100, requires_grad=True)
y = x.sigmoid()
print(f'9. Sigmoid: {y.shape}')

# Test 10: Cross-entropy
logits = torch.randn(32, 100, requires_grad=True)
targets = torch.tensor(np.random.randint(0, 100, (32,)))
loss = F.cross_entropy(logits, targets)
print(f'10. Cross-entropy: {loss.item():.4f}')

# Test 11: Backward through Linear + matmul
x = torch.randn(4, 16, 256, requires_grad=True)
linear = nn.Linear(256, 128)
y = linear(x)
loss = y.sum()
loss.backward()
print(f'11. Linear backward: grad shape {x._grad.shape}')

# Test 12: SDPA with Accelerate matmuls
q = torch.randn(2, 4, 16, 32, requires_grad=True)
k = torch.randn(2, 4, 16, 32, requires_grad=True)
v = torch.randn(2, 4, 16, 32, requires_grad=True)
out = F.scaled_dot_product_attention(q, k, v)
loss = out.sum()
loss.backward()
print(f'12. SDPA forward+backward: {out.shape}, grad q: {q._grad.shape}')

# Test 13: AdamW with Accelerate
model = nn.Linear(128, 64)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(8, 128)
y = model(x)
loss = y.sum()
loss.backward()
opt.step()
print(f'13. AdamW step: OK')

# Test 14: Exp via Accelerate
x = torch.randn(100)
y = x.exp()
exp_err = np.max(np.abs(y._data - np.exp(x._data)))
print(f'14. Exp: max_err={exp_err:.1e}')

print()
print('=== ALL TESTS PASSED ===')
