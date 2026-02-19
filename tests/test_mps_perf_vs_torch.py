#!/usr/bin/env python3
"""
Comprehensive MPS performance benchmark: scaffolding vs PyTorch.
Tests every operation used in helix_v9.py with timing comparisons.

Usage:
    python tests/test_mps_perf_vs_torch.py
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Import both frameworks ──────────────────────────────────────────
import scaffolding as sf
import scaffolding.nn as sf_nn
import scaffolding.nn.functional as sf_F
import scaffolding.optim as sf_optim

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_F

# ── Config ──────────────────────────────────────────────────────────
WARMUP   = 3        # warmup iterations (not timed)
REPEATS  = 20       # timed iterations
B, S, D  = 2, 256, 512   # batch, seq_len, dim (helix-scale)
H        = 8        # attention heads
HD       = D // H   # head dim
VOCAB    = 4096

sf_device  = sf.device("mps")
pt_device  = torch.device("cpu")   # PyTorch CPU for fair comparison
# (scaffolding "MPS" = Accelerate CPU paths, not Metal GPU)

np.random.seed(42)

# ── Helpers ─────────────────────────────────────────────────────────
class Timer:
    """Measures median wall-clock time over REPEATS iterations."""
    def __init__(self, label, warmup=WARMUP, repeats=REPEATS):
        self.label = label
        self.warmup = warmup
        self.repeats = repeats

    def time_fn(self, fn_sf, fn_pt):
        # warmup
        for _ in range(self.warmup):
            fn_sf()
            fn_pt()
        # timed
        times_sf = []
        times_pt = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            fn_sf()
            times_sf.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            fn_pt()
            times_pt.append(time.perf_counter() - t0)

        med_sf = sorted(times_sf)[len(times_sf) // 2]
        med_pt = sorted(times_pt)[len(times_pt) // 2]
        ratio = med_sf / med_pt if med_pt > 0 else float('inf')
        return med_sf, med_pt, ratio


def fmt(label, med_sf, med_pt, ratio):
    status = "OK" if ratio < 2.0 else "SLOW" if ratio < 5.0 else "VERY SLOW"
    return (f"  {label:<45s}  sf={med_sf*1e3:8.3f}ms  "
            f"pt={med_pt*1e3:8.3f}ms  "
            f"ratio={ratio:6.2f}x  [{status}]")


def make_sf(arr):
    """numpy array -> scaffolding Tensor on MPS device."""
    t = sf.tensor(arr)
    return t.to(sf_device)


_NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int64: torch.int64,
    np.int32: torch.int32,
    np.bool_: torch.bool,
}

def make_pt(arr):
    """numpy array -> PyTorch Tensor on CPU via buffer protocol."""
    a = np.ascontiguousarray(arr)
    shape = a.shape
    dt = _NP_TO_TORCH_DTYPE.get(a.dtype.type, torch.float32)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype=dt)
    return t.reshape(shape).to(pt_device)


def np_to_pt(arr, requires_grad=False, device=None):
    """numpy array -> PyTorch tensor with optional grad/device."""
    a = np.ascontiguousarray(arr)
    shape = a.shape
    dt = _NP_TO_TORCH_DTYPE.get(a.dtype.type, torch.float32)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype=dt).reshape(shape)
    if device is not None:
        t = t.to(device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


results = []

def bench(label, fn_sf, fn_pt, warmup=WARMUP, repeats=REPEATS):
    t = Timer(label, warmup, repeats)
    med_sf, med_pt, ratio = t.time_fn(fn_sf, fn_pt)
    line = fmt(label, med_sf, med_pt, ratio)
    results.append((label, med_sf, med_pt, ratio))
    print(line)


# ── Shared data arrays ─────────────────────────────────────────────
x_np    = np.random.randn(B, S, D).astype(np.float32)
y_np    = np.random.randn(B, S, D).astype(np.float32)
w_np    = np.random.randn(D, D).astype(np.float32)
v_np    = np.random.randn(D).astype(np.float32)
idx_np  = np.random.randint(0, VOCAB, (B, S)).astype(np.int64)
emb_np  = np.random.randn(VOCAB, D).astype(np.float32)
logits_np = np.random.randn(B * S, VOCAB).astype(np.float32)
targets_np = np.random.randint(0, VOCAB, (B * S,)).astype(np.int64)
data_np = np.random.randint(0, VOCAB, (10000,)).astype(np.int64)

# ═══════════════════════════════════════════════════════════════════
print("=" * 100)
print(f"  MPS Performance Benchmark: scaffolding vs PyTorch")
print(f"  Shapes: B={B}, S={S}, D={D}, H={H}, Vocab={VOCAB}")
print(f"  Warmup={WARMUP}, Repeats={REPEATS}")
print("=" * 100)

# ───────────────────────────────────────────────────────────────────
# SECTION 1: Tensor Creation
# ───────────────────────────────────────────────────────────────────
print("\n── Tensor Creation ──")

bench("torch.zeros(B,S,D)",
      lambda: sf.zeros(B, S, D, device=sf_device),
      lambda: torch.zeros(B, S, D, device=pt_device))

bench("torch.ones(B,S,D)",
      lambda: sf.ones(B, S, D, device=sf_device),
      lambda: torch.ones(B, S, D, device=pt_device))

bench("torch.randn(B,S,D)",
      lambda: sf.randn(B, S, D, device=sf_device),
      lambda: torch.randn(B, S, D, device=pt_device))

bench("torch.rand(B,S,D)",
      lambda: sf.rand(B, S, D, device=sf_device),
      lambda: torch.rand(B, S, D, device=pt_device))

bench("torch.randint(0,VOCAB,(B,S))",
      lambda: sf.randint(0, VOCAB, (B, S), device=sf_device),
      lambda: torch.randint(0, VOCAB, (B, S), device=pt_device))

bench("torch.arange(S)",
      lambda: sf.arange(S, device=sf_device),
      lambda: torch.arange(S, device=pt_device))

bench("torch.full((B,S),3.14)",
      lambda: sf.full((B, S), 3.14, device=sf_device),
      lambda: torch.full((B, S), 3.14, device=pt_device))

bench("torch.linspace(0,1,S)",
      lambda: sf.linspace(0, 1, S, device=sf_device),
      lambda: torch.linspace(0, 1, S, device=pt_device))

_sf_zl = make_sf(x_np[0, 0])
_pt_zl = make_pt(x_np[0, 0])
bench("torch.zeros_like(vec_D)",
      lambda: sf.zeros_like(_sf_zl),
      lambda: torch.zeros_like(_pt_zl))

bench("torch.tensor(list,dtype=long)",
      lambda: sf.tensor(idx_np, dtype=sf.long),
      lambda: np_to_pt(idx_np))

# ───────────────────────────────────────────────────────────────────
# SECTION 2: Element-wise Arithmetic
# ───────────────────────────────────────────────────────────────────
print("\n── Element-wise Arithmetic ──")

sf_x = make_sf(x_np); sf_y = make_sf(y_np)
pt_x = make_pt(x_np); pt_y = make_pt(y_np)

bench("add(tensor+tensor) [B,S,D]",
      lambda: sf_x + sf_y,
      lambda: pt_x + pt_y)

bench("add(tensor+scalar) [B,S,D]",
      lambda: sf_x + 1.0,
      lambda: pt_x + 1.0)

bench("sub(tensor-tensor) [B,S,D]",
      lambda: sf_x - sf_y,
      lambda: pt_x - pt_y)

bench("mul(tensor*tensor) [B,S,D]",
      lambda: sf_x * sf_y,
      lambda: pt_x * pt_y)

bench("mul(tensor*scalar) [B,S,D]",
      lambda: sf_x * 0.5,
      lambda: pt_x * 0.5)

bench("div(tensor/tensor) [B,S,D]",
      lambda: sf_x / (sf_y + sf.tensor(np.float32(2.0)).to(sf_device)),
      lambda: pt_x / (pt_y + 2.0))

bench("div(tensor/scalar) [B,S,D]",
      lambda: sf_x / 2.0,
      lambda: pt_x / 2.0)

bench("pow(tensor**2) [B,S,D]",
      lambda: sf_x ** 2,
      lambda: pt_x ** 2)

bench("neg(-tensor) [B,S,D]",
      lambda: -sf_x,
      lambda: -pt_x)

# ───────────────────────────────────────────────────────────────────
# SECTION 3: Math Functions
# ───────────────────────────────────────────────────────────────────
print("\n── Math Functions ──")

sf_xp = make_sf(np.abs(x_np) + 0.01)  # positive for log/sqrt
pt_xp = make_pt(np.abs(x_np) + 0.01)

bench("torch.exp [B,S,D]",
      lambda: sf.exp(sf_x),
      lambda: torch.exp(pt_x))

bench("torch.log [B,S,D]",
      lambda: sf.log(sf_xp),
      lambda: torch.log(pt_xp))

bench("torch.sqrt [B,S,D]",
      lambda: sf.sqrt(sf_xp),
      lambda: torch.sqrt(pt_xp))

bench("torch.rsqrt [B,S,D]",
      lambda: sf.rsqrt(sf_xp),
      lambda: torch.rsqrt(pt_xp))

bench("torch.sigmoid [B,S,D]",
      lambda: sf.sigmoid(sf_x),
      lambda: torch.sigmoid(pt_x))

bench("torch.sin [B,S,D]",
      lambda: sf.sin(sf_x),
      lambda: torch.sin(pt_x))

bench("torch.cos [B,S,D]",
      lambda: sf.cos(sf_x),
      lambda: torch.cos(pt_x))

bench("torch.atan2 [B,S,D]",
      lambda: sf.atan2(sf_x, sf_y),
      lambda: torch.atan2(pt_x, pt_y))

# ───────────────────────────────────────────────────────────────────
# SECTION 4: Reductions
# ───────────────────────────────────────────────────────────────────
print("\n── Reductions ──")

bench("sum(global) [B,S,D]",
      lambda: sf_x.sum(),
      lambda: pt_x.sum())

bench("sum(dim=-1) [B,S,D]",
      lambda: sf_x.sum(dim=-1),
      lambda: pt_x.sum(dim=-1))

bench("sum(dim=-1,keepdim) [B,S,D]",
      lambda: sf_x.sum(dim=-1, keepdim=True),
      lambda: pt_x.sum(dim=-1, keepdim=True))

bench("mean(dim=-1,keepdim) [B,S,D]",
      lambda: sf_x.mean(dim=-1, keepdim=True),
      lambda: pt_x.mean(dim=-1, keepdim=True))

bench("pow(2).mean(dim=-1,keepdim)",
      lambda: sf_x.pow(2).mean(dim=-1, keepdim=True),
      lambda: pt_x.pow(2).mean(dim=-1, keepdim=True))

# ───────────────────────────────────────────────────────────────────
# SECTION 5: Shape Operations
# ───────────────────────────────────────────────────────────────────
print("\n── Shape Operations ──")

bench("reshape(B*S,D)",
      lambda: sf_x.reshape(B * S, D),
      lambda: pt_x.reshape(B * S, D))

bench("unsqueeze(1)",
      lambda: sf_x.unsqueeze(1),
      lambda: pt_x.unsqueeze(1))

bench("squeeze(1) on (B,1,S,D)",
      lambda: sf_x.unsqueeze(1).squeeze(1),
      lambda: pt_x.unsqueeze(1).squeeze(1))

bench("transpose(-2,-1) [B,S,D]",
      lambda: sf_x.transpose(-2, -1),
      lambda: pt_x.transpose(-2, -1))

# permute for attention: (B,S,H,HD) -> (B,H,S,HD)
sf_4d = make_sf(x_np.reshape(B, S, H, HD))
pt_4d = make_pt(x_np.reshape(B, S, H, HD))
bench("permute(0,2,1,3) [B,S,H,HD]",
      lambda: sf_4d.permute(0, 2, 1, 3),
      lambda: pt_4d.permute(0, 2, 1, 3))

bench("contiguous [B,H,S,HD]",
      lambda: sf_4d.permute(0, 2, 1, 3).contiguous(),
      lambda: pt_4d.permute(0, 2, 1, 3).contiguous())

bench("chunk(3,dim=-1) [B,S,D]",
      lambda: sf_x.chunk(3, dim=-1),
      lambda: pt_x.chunk(3, dim=-1))

# ───────────────────────────────────────────────────────────────────
# SECTION 6: Concatenation / Stacking
# ───────────────────────────────────────────────────────────────────
print("\n── Cat / Stack ──")

sf_halves = [make_sf(x_np[:, :, :D//2]), make_sf(x_np[:, :, D//2:])]
pt_halves = [make_pt(x_np[:, :, :D//2]), make_pt(x_np[:, :, D//2:])]

bench("torch.cat([a,b],dim=-1)",
      lambda: sf.cat(sf_halves, dim=-1),
      lambda: torch.cat(pt_halves, dim=-1))

sf_pair = [make_sf(x_np[:, :1, :]), make_sf(y_np[:, :1, :])]
pt_pair = [make_pt(x_np[:, :1, :]), make_pt(y_np[:, :1, :])]
bench("torch.stack([a,b],dim=0)",
      lambda: sf.stack(sf_pair, dim=0),
      lambda: torch.stack(pt_pair, dim=0))

# ───────────────────────────────────────────────────────────────────
# SECTION 7: Matmul / BMM
# ───────────────────────────────────────────────────────────────────
print("\n── Matmul / BMM ──")

sf_w = make_sf(w_np)
pt_w = make_pt(w_np)

# 2D matmul (Linear layer core)
sf_2d = make_sf(x_np.reshape(B * S, D))
pt_2d = make_pt(x_np.reshape(B * S, D))

bench("matmul 2D (B*S,D)@(D,D)",
      lambda: sf.matmul(sf_2d, sf_w),
      lambda: torch.matmul(pt_2d, pt_w))

# 3D batched matmul (attention Q@K^T)
qk_np = np.random.randn(B * H, S, HD).astype(np.float32)
sf_qk = make_sf(qk_np)
pt_qk = make_pt(qk_np)
sf_kT = sf_qk.transpose(-2, -1)
pt_kT = pt_qk.transpose(-2, -1)

bench("bmm Q@K^T (B*H,S,HD)@(B*H,HD,S)",
      lambda: sf.matmul(sf_qk, sf_kT.contiguous()),
      lambda: torch.matmul(pt_qk, pt_kT))

# attention weights @ V
attn_np = np.random.randn(B * H, S, S).astype(np.float32)
v_4d_np = np.random.randn(B * H, S, HD).astype(np.float32)
sf_attn = make_sf(attn_np); sf_v4d = make_sf(v_4d_np)
pt_attn = make_pt(attn_np); pt_v4d = make_pt(v_4d_np)

bench("bmm attn@V (B*H,S,S)@(B*H,S,HD)",
      lambda: sf.matmul(sf_attn, sf_v4d),
      lambda: torch.matmul(pt_attn, pt_v4d))

# ───────────────────────────────────────────────────────────────────
# SECTION 8: Activations (nn.functional)
# ───────────────────────────────────────────────────────────────────
print("\n── Activations ──")

bench("F.silu [B,S,D]",
      lambda: sf_F.silu(sf_x),
      lambda: torch_F.silu(pt_x))

bench("F.softmax(dim=-1) [B,S,D]",
      lambda: sf_F.softmax(sf_x, dim=-1),
      lambda: torch_F.softmax(pt_x, dim=-1))

bench("F.softplus [B,S,D]",
      lambda: sf_F.softplus(sf_x),
      lambda: torch_F.softplus(pt_x))

bench("F.dropout(p=0.1,training) [B,S,D]",
      lambda: sf_F.dropout(sf_x, p=0.1, training=True),
      lambda: torch_F.dropout(pt_x, p=0.1, training=True))

# ───────────────────────────────────────────────────────────────────
# SECTION 9: Cross Entropy
# ───────────────────────────────────────────────────────────────────
print("\n── Cross Entropy ──")

sf_logits_nograd = make_sf(logits_np)
pt_logits_nograd = make_pt(logits_np)
sf_tgt = make_sf(targets_np)
pt_tgt = make_pt(targets_np)

bench("F.cross_entropy (no grad) [B*S,V]",
      lambda: sf_F.cross_entropy(sf_logits_nograd, sf_tgt),
      lambda: torch_F.cross_entropy(pt_logits_nograd, pt_tgt))

# With grad (training path)
sf_logits_grad = sf.tensor(logits_np, requires_grad=True).to(sf_device)
pt_logits_grad = np_to_pt(logits_np, requires_grad=True, device=pt_device)

bench("F.cross_entropy (with grad) [B*S,V]",
      lambda: sf_F.cross_entropy(sf_logits_grad, sf_tgt),
      lambda: torch_F.cross_entropy(pt_logits_grad, pt_tgt))

# ───────────────────────────────────────────────────────────────────
# SECTION 10: Indexing
# ───────────────────────────────────────────────────────────────────
print("\n── Indexing ──")

sf_data = make_sf(data_np)
pt_data = make_pt(data_np)
starts_np = np.random.randint(0, 9000, (B * 4,)).astype(np.int64)
offsets_np = np.arange(S, dtype=np.int64)
indices_np = starts_np[:, None] + offsets_np[None, :]  # (B*4, S)

sf_idx = make_sf(indices_np)
pt_idx = make_pt(indices_np)

bench("data[indices] gather (B*4,S)",
      lambda: sf_data[sf_idx],
      lambda: pt_data[pt_idx])

bench("tensor[:,:,0:HD] slice",
      lambda: sf_4d[:, :, :, :HD//2],
      lambda: pt_4d[:, :, :, :HD//2])

bench("tensor[...,0] ellipsis",
      lambda: sf_x[..., 0],
      lambda: pt_x[..., 0])

bench("tensor[-1] negative index",
      lambda: sf_x[-1],
      lambda: pt_x[-1])

# ───────────────────────────────────────────────────────────────────
# SECTION 11: Comparison / Where / Sort
# ───────────────────────────────────────────────────────────────────
print("\n── Comparison / Where / Sort ──")

bench("tensor > 0 comparison [B,S,D]",
      lambda: sf_x > 0,
      lambda: pt_x > 0)

sf_mask = make_sf((x_np > 0).astype(np.float32))
pt_mask = (pt_x > 0).float()

bench("torch.where(mask,x,0) [B,S,D]",
      lambda: sf.where(sf_x > 0, sf_x, sf.tensor(np.float32(0.0)).to(sf_device)),
      lambda: torch.where(pt_x > 0, pt_x, torch.tensor(0.0)))

sort_np = np.random.randn(S).astype(np.float32)
sf_sort = make_sf(sort_np)
pt_sort = make_pt(sort_np)

bench("torch.sort(descending) [S]",
      lambda: sf.sort(sf_sort, descending=True),
      lambda: torch.sort(pt_sort, descending=True))

bench("cumsum(dim=-1) [B,S,D]",
      lambda: sf.cumsum(sf_x, dim=-1),
      lambda: torch.cumsum(pt_x, dim=-1))

bench("torch.bernoulli [B,S,D]",
      lambda: sf.bernoulli(make_sf(np.full((B, S, D), 0.5, dtype=np.float32))),
      lambda: torch.bernoulli(torch.full((B, S, D), 0.5)))

# ───────────────────────────────────────────────────────────────────
# SECTION 12: Clamp
# ───────────────────────────────────────────────────────────────────
print("\n── Clamp ──")

bench("clamp(min=-1,max=1) [B,S,D]",
      lambda: sf_x.clamp(min=-1.0, max=1.0),
      lambda: pt_x.clamp(min=-1.0, max=1.0))

# ───────────────────────────────────────────────────────────────────
# SECTION 13: torch.outer / trig for RoPE
# ───────────────────────────────────────────────────────────────────
print("\n── RoPE Helpers ──")

freqs_np = np.random.randn(D // 2).astype(np.float32)
pos_np   = np.arange(S, dtype=np.float32)

sf_freqs = make_sf(freqs_np); sf_pos = make_sf(pos_np)
pt_freqs = make_pt(freqs_np); pt_pos = make_pt(pos_np)

bench("torch.outer(pos,freqs) [S]x[D/2]",
      lambda: sf.outer(sf_pos, sf_freqs),
      lambda: torch.outer(pt_pos, pt_freqs))

# ───────────────────────────────────────────────────────────────────
# SECTION 14: nn.Linear forward
# ───────────────────────────────────────────────────────────────────
print("\n── nn.Linear ──")

# Build matched layers
sf_linear = sf_nn.Linear(D, D, bias=True)
pt_linear = torch_nn.Linear(D, D, bias=True)
# Copy weights
with torch.no_grad():
    pt_linear.weight.copy_(np_to_pt(sf_linear.weight._data))
    pt_linear.bias.copy_(np_to_pt(sf_linear.bias._data))
sf_linear = sf_linear.to(sf_device)

sf_lin_in = make_sf(x_np.reshape(B * S, D))
pt_lin_in = make_pt(x_np.reshape(B * S, D))

bench("nn.Linear(D,D) fwd [B*S,D]",
      lambda: sf_linear(sf_lin_in),
      lambda: pt_linear(pt_lin_in))

# ───────────────────────────────────────────────────────────────────
# SECTION 15: nn.Embedding forward
# ───────────────────────────────────────────────────────────────────
print("\n── nn.Embedding ──")

sf_emb = sf_nn.Embedding(VOCAB, D).to(sf_device)
pt_emb = torch_nn.Embedding(VOCAB, D)
with torch.no_grad():
    pt_emb.weight.copy_(np_to_pt(sf_emb.weight._data))

sf_emb_idx = make_sf(idx_np)
pt_emb_idx = make_pt(idx_np)

bench("nn.Embedding(V,D) fwd [B,S]",
      lambda: sf_emb(sf_emb_idx),
      lambda: pt_emb(pt_emb_idx))

# ───────────────────────────────────────────────────────────────────
# SECTION 16: nn.Conv1d forward
# ───────────────────────────────────────────────────────────────────
print("\n── nn.Conv1d ──")

sf_conv = sf_nn.Conv1d(D, D, kernel_size=4, padding=3, groups=D).to(sf_device)
pt_conv = torch_nn.Conv1d(D, D, kernel_size=4, padding=3, groups=D)
with torch.no_grad():
    pt_conv.weight.copy_(np_to_pt(sf_conv.weight._data))
    if sf_conv.bias is not None:
        pt_conv.bias.copy_(np_to_pt(sf_conv.bias._data))

conv_in_np = np.random.randn(B, D, S).astype(np.float32)
sf_conv_in = make_sf(conv_in_np)
pt_conv_in = make_pt(conv_in_np)

bench("nn.Conv1d(D,D,k=4,g=D) fwd [B,D,S]",
      lambda: sf_conv(sf_conv_in),
      lambda: pt_conv(pt_conv_in))

# ───────────────────────────────────────────────────────────────────
# SECTION 17: nn.Dropout + F.pad
# ───────────────────────────────────────────────────────────────────
print("\n── Dropout / Pad ──")

sf_drop = sf_nn.Dropout(0.1)
sf_drop.train()
bench("nn.Dropout(0.1) fwd [B,S,D]",
      lambda: sf_drop(sf_x),
      lambda: torch_F.dropout(pt_x, 0.1, training=True))

pad_np = np.random.randn(B, S, D).astype(np.float32)
sf_pad_in = make_sf(pad_np)
pt_pad_in = make_pt(pad_np)

bench("F.pad((0,0,3,0)) [B,S,D]",
      lambda: sf_F.pad(sf_pad_in, (0, 0, 3, 0)),
      lambda: torch_F.pad(pt_pad_in, (0, 0, 3, 0)))

# ───────────────────────────────────────────────────────────────────
# SECTION 18: Scaled Dot-Product Attention
# ───────────────────────────────────────────────────────────────────
print("\n── Scaled Dot-Product Attention ──")

q_np = np.random.randn(B, H, S, HD).astype(np.float32)
k_np = np.random.randn(B, H, S, HD).astype(np.float32)
v_sdp_np = np.random.randn(B, H, S, HD).astype(np.float32)

sf_q = make_sf(q_np); sf_k = make_sf(k_np); sf_v_sdp = make_sf(v_sdp_np)
pt_q = make_pt(q_np); pt_k = make_pt(k_np); pt_v_sdp = make_pt(v_sdp_np)

bench("F.sdpa (B,H,S,HD) no mask",
      lambda: sf_F.scaled_dot_product_attention(sf_q, sf_k, sf_v_sdp),
      lambda: torch_F.scaled_dot_product_attention(pt_q, pt_k, pt_v_sdp))

# ───────────────────────────────────────────────────────────────────
# SECTION 19: In-place Operations
# ───────────────────────────────────────────────────────────────────
print("\n── In-place Operations ──")

def bench_inplace_mul_():
    t = make_sf(x_np.copy())
    t.mul_(0.9)
def bench_inplace_mul_pt():
    t = make_pt(x_np.copy())
    t.mul_(0.9)
bench("mul_(0.9) [B,S,D]", bench_inplace_mul_, bench_inplace_mul_pt)

def bench_inplace_add_():
    t = make_sf(x_np.copy())
    t2 = make_sf(y_np.copy())
    t.add_(t2, alpha=0.1)
def bench_inplace_add_pt():
    t = make_pt(x_np.copy())
    t2 = make_pt(y_np.copy())
    t.add_(t2, alpha=0.1)
bench("add_(t2,alpha=0.1) [B,S,D]", bench_inplace_add_, bench_inplace_add_pt)

def bench_zero_():
    t = make_sf(x_np.copy())
    t.zero_()
def bench_zero_pt():
    t = make_pt(x_np.copy())
    t.zero_()
bench("zero_() [B,S,D]", bench_zero_, bench_zero_pt)

def bench_fill_():
    t = make_sf(x_np.copy())
    t.fill_(float('-inf'))
def bench_fill_pt():
    t = make_pt(x_np.copy())
    t.fill_(float('-inf'))
bench("fill_(-inf) [B,S,D]", bench_fill_, bench_fill_pt)

def bench_copy_():
    t = make_sf(x_np.copy())
    t2 = make_sf(y_np.copy())
    t.copy_(t2)
def bench_copy_pt():
    t = make_pt(x_np.copy())
    t2 = make_pt(y_np.copy())
    t.copy_(t2)
bench("copy_(other) [B,S,D]", bench_copy_, bench_copy_pt)

# ───────────────────────────────────────────────────────────────────
# SECTION 20: Autograd (backward pass)
# ───────────────────────────────────────────────────────────────────
print("\n── Backward Pass ──")

def bench_linear_backward_sf():
    inp = sf.tensor(x_np.reshape(B * S, D).copy(), requires_grad=True).to(sf_device)
    out = sf_linear(inp)
    loss = out.sum()
    loss.backward()

def bench_linear_backward_pt():
    inp = np_to_pt(x_np.reshape(B * S, D), requires_grad=True, device=pt_device)
    out = pt_linear(inp)
    loss = out.sum()
    loss.backward()

bench("Linear fwd+bwd [B*S,D]", bench_linear_backward_sf, bench_linear_backward_pt)

def bench_ce_backward_sf():
    lg = sf.tensor(logits_np.copy(), requires_grad=True).to(sf_device)
    tg = make_sf(targets_np)
    loss = sf_F.cross_entropy(lg, tg)
    loss.backward()

def bench_ce_backward_pt():
    lg = np_to_pt(logits_np, requires_grad=True, device=pt_device)
    tg = make_pt(targets_np)
    loss = torch_F.cross_entropy(lg, tg)
    loss.backward()

bench("CrossEntropy fwd+bwd [B*S,V]", bench_ce_backward_sf, bench_ce_backward_pt)

# matmul backward
def bench_matmul_backward_sf():
    a = sf.tensor(x_np.reshape(B * S, D).copy(), requires_grad=True).to(sf_device)
    w = sf.tensor(w_np.copy(), requires_grad=True).to(sf_device)
    out = sf.matmul(a, w)
    out.sum().backward()

def bench_matmul_backward_pt():
    a = np_to_pt(x_np.reshape(B * S, D), requires_grad=True, device=pt_device)
    w = np_to_pt(w_np, requires_grad=True, device=pt_device)
    out = torch.matmul(a, w)
    out.sum().backward()

bench("MatMul fwd+bwd (B*S,D)@(D,D)", bench_matmul_backward_sf, bench_matmul_backward_pt)

# ───────────────────────────────────────────────────────────────────
# SECTION 21: Optimizer Step
# ───────────────────────────────────────────────────────────────────
print("\n── Optimizer Step ──")

# Build matched small models for optimizer benchmark
class TinyModel(sf_nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = sf_nn.Linear(D, D)
        self.fc2 = sf_nn.Linear(D, D)
    def forward(self, x):
        return self.fc2(sf_F.silu(self.fc1(x)))

class TinyModelPT(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch_nn.Linear(D, D)
        self.fc2 = torch_nn.Linear(D, D)
    def forward(self, x):
        return self.fc2(torch_F.silu(self.fc1(x)))

sf_model = TinyModel().to(sf_device)
pt_model = TinyModelPT().to(pt_device)

sf_opt = sf_optim.AdamW(sf_model.parameters(), lr=1e-3, weight_decay=0.01)
pt_opt = torch.optim.AdamW(pt_model.parameters(), lr=1e-3, weight_decay=0.01)

# Pre-populate grads
sf_tiny_in = sf.tensor(x_np.reshape(B * S, D)[:32].copy(), requires_grad=True).to(sf_device)
pt_tiny_in = np_to_pt(x_np.reshape(B * S, D)[:32], requires_grad=True, device=pt_device)

sf_model(sf_tiny_in).sum().backward()
pt_model(pt_tiny_in).sum().backward()

bench("AdamW.step() (2xLinear D→D)",
      lambda: sf_opt.step(),
      lambda: pt_opt.step())

bench("optimizer.zero_grad(set_to_none)",
      lambda: sf_opt.zero_grad(set_to_none=True),
      lambda: pt_opt.zero_grad(set_to_none=True))

# ───────────────────────────────────────────────────────────────────
# SECTION 22: Grad Clipping
# ───────────────────────────────────────────────────────────────────
print("\n── Grad Clipping ──")

# Repopulate grads
sf_model(sf_tiny_in).sum().backward()
pt_model(pt_tiny_in).sum().backward()

bench("clip_grad_norm_(1.0)",
      lambda: sf_nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0),
      lambda: torch_nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0))

# ───────────────────────────────────────────────────────────────────
# SECTION 23: Misc (detach, clone, item, numel, nonzero, to, float)
# ───────────────────────────────────────────────────────────────────
print("\n── Misc Tensor Methods ──")

bench(".detach() [B,S,D]",
      lambda: sf_x.detach(),
      lambda: pt_x.detach())

bench(".float() [B,S,D]",
      lambda: sf_x.float(),
      lambda: pt_x.float())

sf_scalar = sf.tensor(np.float32(3.14)).to(sf_device)
pt_scalar = torch.tensor(3.14)
bench(".item() scalar",
      lambda: sf_scalar.item(),
      lambda: pt_scalar.item())

bench(".numel() [B,S,D]",
      lambda: sf_x.numel(),
      lambda: pt_x.numel())

sf_bool = make_sf((x_np > 0).astype(np.float32))
pt_bool = (pt_x > 0).float()
bench(".nonzero(as_tuple=True) [B,S,D]",
      lambda: sf_bool.nonzero(as_tuple=True),
      lambda: pt_bool.nonzero(as_tuple=True))

bench("torch.multinomial [VOCAB]",
      lambda: sf.multinomial(sf_F.softmax(make_sf(logits_np[0]), dim=-1), 1),
      lambda: torch.multinomial(torch_F.softmax(make_pt(logits_np[0]), dim=-1), 1))

# ───────────────────────────────────────────────────────────────────
# SECTION 24: Model state_dict / load_state_dict
# ───────────────────────────────────────────────────────────────────
print("\n── State Dict ──")

bench("model.state_dict()",
      lambda: sf_model.state_dict(),
      lambda: pt_model.state_dict())

_sf_sd = sf_model.state_dict()
_pt_sd = pt_model.state_dict()

bench("model.load_state_dict()",
      lambda: sf_model.load_state_dict(_sf_sd),
      lambda: pt_model.load_state_dict(_pt_sd))

# ───────────────────────────────────────────────────────────────────
# SECTION 25: RMSNorm-like pattern
# ───────────────────────────────────────────────────────────────────
print("\n── RMSNorm Pattern ──")

sf_rms_w = make_sf(np.ones(D, dtype=np.float32))
pt_rms_w = torch.ones(D, dtype=torch.float32)

def rms_norm_sf():
    var = sf_x.pow(2).mean(dim=-1, keepdim=True)
    return sf_x * sf.rsqrt(var + sf.tensor(np.float32(1e-6)).to(sf_device)) * sf_rms_w

def rms_norm_pt():
    var = pt_x.pow(2).mean(dim=-1, keepdim=True)
    return pt_x * torch.rsqrt(var + 1e-6) * pt_rms_w

bench("RMSNorm pattern [B,S,D]", rms_norm_sf, rms_norm_pt)

# ───────────────────────────────────────────────────────────────────
# SECTION 26: Full Attention Pattern
# ───────────────────────────────────────────────────────────────────
print("\n── Full Attention Pattern ──")

scale = HD ** -0.5

def full_attn_sf():
    q = sf_q * scale
    scores = sf.matmul(q.reshape(B*H, S, HD), sf_k.reshape(B*H, S, HD).transpose(-2,-1).contiguous())
    weights = sf_F.softmax(scores, dim=-1)
    out = sf.matmul(weights, sf_v_sdp.reshape(B*H, S, HD))
    return out.reshape(B, H, S, HD)

def full_attn_pt():
    q = pt_q * scale
    scores = torch.matmul(q.reshape(B*H, S, HD), pt_k.reshape(B*H, S, HD).transpose(-2,-1))
    weights = torch_F.softmax(scores, dim=-1)
    out = torch.matmul(weights, pt_v_sdp.reshape(B*H, S, HD))
    return out.reshape(B, H, S, HD)

bench("Full attention pattern", full_attn_sf, full_attn_pt)

# ───────────────────────────────────────────────────────────────────
# SECTION 27: expm1 (Mamba-like pattern)
# ───────────────────────────────────────────────────────────────────
print("\n── Mamba Helpers ──")

bench("torch.expm1 [B,S,D]",
      lambda: sf.expm1(sf_x),
      lambda: torch.expm1(pt_x))

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  SUMMARY")
print("=" * 100)

# Sort by ratio (worst first)
results.sort(key=lambda r: -r[3])

slow    = [(l, s, p, r) for l, s, p, r in results if r >= 5.0]
medium  = [(l, s, p, r) for l, s, p, r in results if 2.0 <= r < 5.0]
ok      = [(l, s, p, r) for l, s, p, r in results if r < 2.0]

if slow:
    print(f"\n  VERY SLOW (>=5x): {len(slow)} operations")
    for l, s, p, r in slow:
        print(f"    {l:<45s}  {r:6.2f}x  (sf={s*1e3:.2f}ms, pt={p*1e3:.2f}ms)")

if medium:
    print(f"\n  SLOW (2-5x): {len(medium)} operations")
    for l, s, p, r in medium:
        print(f"    {l:<45s}  {r:6.2f}x  (sf={s*1e3:.2f}ms, pt={p*1e3:.2f}ms)")

if ok:
    print(f"\n  OK (<2x): {len(ok)} operations")
    for l, s, p, r in ok:
        print(f"    {l:<45s}  {r:6.2f}x  (sf={s*1e3:.2f}ms, pt={p*1e3:.2f}ms)")

total_sf = sum(s for _, s, _, _ in results)
total_pt = sum(p for _, _, p, _ in results)
print(f"\n  Total scaffolding time: {total_sf*1e3:.1f}ms")
print(f"  Total PyTorch time:    {total_pt*1e3:.1f}ms")
print(f"  Overall ratio:         {total_sf/total_pt:.2f}x")
print("=" * 100)
