#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — CUDA Performance Benchmark vs PyTorch                 ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
CUDA Performance Test — Scaffolding vs PyTorch

Benchmarks core operations on GPU and reports wall-clock times with
a head-to-head scoreboard.  Requires both scaffolding (with CUDA
extension built) and PyTorch with CUDA support.

Usage:
    python tests/test_cuda_perf.py
    python tests/test_cuda_perf.py --warmup 5 --trials 20
    python tests/test_cuda_perf.py --sizes small        # quick run
    python tests/test_cuda_perf.py --sizes large        # stress test
"""
from __future__ import annotations

import argparse
import sys
import os
import time
import statistics
from typing import Callable

import numpy as np

# ── Resolve imports ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Check availability ──
_HAS_SCAFFOLDING_CUDA = False
_HAS_PYTORCH_CUDA = False

try:
    import scaffolding as sf
    import scaffolding.nn as sf_nn
    import scaffolding.nn.functional as sf_F
    import scaffolding.optim as sf_optim
    from scaffolding.cuda import is_available as sf_cuda_available
    if sf_cuda_available():
        _HAS_SCAFFOLDING_CUDA = True
except ImportError:
    sf = None

try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as t_F
    if torch.cuda.is_available():
        _HAS_PYTORCH_CUDA = True
except ImportError:
    torch = None


# ================================================================
#  Timing utilities
# ================================================================

def _sync_cuda():
    """Synchronize both frameworks' GPU."""
    if _HAS_PYTORCH_CUDA:
        torch.cuda.synchronize()
    if _HAS_SCAFFOLDING_CUDA:
        sf.cuda.synchronize()


def bench(fn: Callable, warmup: int = 3, trials: int = 10,
          sync: bool = True) -> dict:
    """Time *fn* and return stats in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            _sync_cuda()

    times = []
    for _ in range(trials):
        if sync:
            _sync_cuda()
        t0 = time.perf_counter()
        fn()
        if sync:
            _sync_cuda()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ================================================================
#  Problem sizes
# ================================================================

SIZES = {
    'small': {
        'vec_n': 100_000,
        'mat_m': 256, 'mat_k': 256, 'mat_n': 256,
        'softmax_rows': 256, 'softmax_cols': 1024,
        'ce_batch': 128, 'ce_classes': 1000,
        'linear_batch': 64, 'linear_in': 256, 'linear_out': 256,
        'embed_vocab': 10_000, 'embed_dim': 128, 'embed_seq': 128,
        'ln_batch': 64, 'ln_dim': 256,
        'attn_batch': 4, 'attn_heads': 8, 'attn_seq': 128, 'attn_dim': 64,
        'adam_params': 100_000,
        'fwd_batch': 32, 'fwd_seq': 128, 'fwd_dim': 256, 'fwd_ffn': 1024,
        'fwd_vocab': 10_000,
    },
    'medium': {
        'vec_n': 1_000_000,
        'mat_m': 1024, 'mat_k': 1024, 'mat_n': 1024,
        'softmax_rows': 512, 'softmax_cols': 4096,
        'ce_batch': 256, 'ce_classes': 32000,
        'linear_batch': 128, 'linear_in': 1024, 'linear_out': 1024,
        'embed_vocab': 50_000, 'embed_dim': 512, 'embed_seq': 256,
        'ln_batch': 128, 'ln_dim': 1024,
        'attn_batch': 8, 'attn_heads': 16, 'attn_seq': 256, 'attn_dim': 64,
        'adam_params': 1_000_000,
        'fwd_batch': 16, 'fwd_seq': 256, 'fwd_dim': 512, 'fwd_ffn': 2048,
        'fwd_vocab': 32000,
    },
    'large': {
        'vec_n': 10_000_000,
        'mat_m': 2048, 'mat_k': 2048, 'mat_n': 2048,
        'softmax_rows': 1024, 'softmax_cols': 32000,
        'ce_batch': 512, 'ce_classes': 50000,
        'linear_batch': 256, 'linear_in': 2048, 'linear_out': 2048,
        'embed_vocab': 100_000, 'embed_dim': 1024, 'embed_seq': 512,
        'ln_batch': 256, 'ln_dim': 2048,
        'attn_batch': 16, 'attn_heads': 32, 'attn_seq': 512, 'attn_dim': 64,
        'adam_params': 10_000_000,
        'fwd_batch': 8, 'fwd_seq': 512, 'fwd_dim': 1024, 'fwd_ffn': 4096,
        'fwd_vocab': 50000,
    },
}


# ================================================================
#  Benchmark definitions
# ================================================================

class BenchmarkResult:
    __slots__ = ('name', 'sf_ms', 'pt_ms', 'speedup', 'winner')

    def __init__(self, name: str, sf_ms: float, pt_ms: float):
        self.name = name
        self.sf_ms = sf_ms
        self.pt_ms = pt_ms
        if pt_ms > 0 and sf_ms > 0:
            self.speedup = pt_ms / sf_ms
        else:
            self.speedup = float('nan')
        self.winner = 'Scaffolding' if sf_ms <= pt_ms else 'PyTorch'


def run_benchmarks(sz: dict, warmup: int, trials: int) -> list[BenchmarkResult]:
    results = []
    np.random.seed(42)

    # ── 1. Element-wise: exp ──
    print('  [1/14] Element-wise exp ...', end='', flush=True)
    n = sz['vec_n']
    x_np = np.random.randn(n).astype(np.float32)

    if _HAS_PYTORCH_CUDA:
        x_pt = torch.tensor(x_np, device='cuda')
        pt = bench(lambda: torch.exp(x_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        x_sf = sf.tensor(x_np, device='cuda')
        sf_t = bench(lambda: x_sf.exp(), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('exp (element-wise)', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 2. Element-wise: sigmoid ──
    print('  [2/14] Element-wise sigmoid ...', end='', flush=True)
    if _HAS_PYTORCH_CUDA:
        pt = bench(lambda: torch.sigmoid(x_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        sf_t = bench(lambda: x_sf.sigmoid(), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('sigmoid (element-wise)', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 3. Element-wise: SiLU ──
    print('  [3/14] SiLU activation ...', end='', flush=True)
    if _HAS_PYTORCH_CUDA:
        pt = bench(lambda: t_F.silu(x_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        sf_t = bench(lambda: sf_F.silu(x_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('SiLU activation', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 4. Element-wise: GELU ──
    print('  [4/14] GELU activation ...', end='', flush=True)
    if _HAS_PYTORCH_CUDA:
        pt = bench(lambda: t_F.gelu(x_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        sf_t = bench(lambda: sf_F.gelu(x_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('GELU activation', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 5. MatMul (SGEMM) ──
    print('  [5/14] MatMul (cuBLAS SGEMM) ...', end='', flush=True)
    M, K, N = sz['mat_m'], sz['mat_k'], sz['mat_n']
    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)

    if _HAS_PYTORCH_CUDA:
        a_pt = torch.tensor(a_np, device='cuda')
        b_pt = torch.tensor(b_np, device='cuda')
        pt = bench(lambda: torch.matmul(a_pt, b_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        a_sf = sf.tensor(a_np, device='cuda')
        b_sf = sf.tensor(b_np, device='cuda')
        sf_t = bench(lambda: sf.matmul(a_sf, b_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'MatMul {M}x{K}x{N}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 6. Softmax ──
    print('  [6/14] Softmax ...', end='', flush=True)
    R, C = sz['softmax_rows'], sz['softmax_cols']
    s_np = np.random.randn(R, C).astype(np.float32)

    if _HAS_PYTORCH_CUDA:
        s_pt = torch.tensor(s_np, device='cuda')
        pt = bench(lambda: t_F.softmax(s_pt, dim=-1), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        s_sf = sf.tensor(s_np, device='cuda')
        sf_t = bench(lambda: sf.softmax(s_sf, dim=-1), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'Softmax {R}x{C}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 7. Cross-Entropy Loss ──
    print('  [7/14] Cross-Entropy Loss ...', end='', flush=True)
    B_ce, C_ce = sz['ce_batch'], sz['ce_classes']
    logits_np = np.random.randn(B_ce, C_ce).astype(np.float32)
    targets_np = np.random.randint(0, C_ce, size=(B_ce,)).astype(np.int64)

    if _HAS_PYTORCH_CUDA:
        logits_pt = torch.tensor(logits_np, device='cuda')
        targets_pt = torch.tensor(targets_np, device='cuda')
        pt = bench(lambda: t_F.cross_entropy(logits_pt, targets_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        logits_sf = sf.tensor(logits_np, device='cuda')
        targets_sf = sf.tensor(targets_np, device='cuda')
        sf_t = bench(lambda: sf_F.cross_entropy(logits_sf, targets_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'CrossEntropy {B_ce}x{C_ce}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 8. Linear Layer (forward) ──
    print('  [8/14] Linear forward ...', end='', flush=True)
    B_l, D_in, D_out = sz['linear_batch'], sz['linear_in'], sz['linear_out']
    lin_x_np = np.random.randn(B_l, D_in).astype(np.float32)
    lin_w_np = np.random.randn(D_out, D_in).astype(np.float32)
    lin_b_np = np.random.randn(D_out).astype(np.float32)

    if _HAS_PYTORCH_CUDA:
        lin_x_pt = torch.tensor(lin_x_np, device='cuda')
        lin_w_pt = torch.tensor(lin_w_np, device='cuda')
        lin_b_pt = torch.tensor(lin_b_np, device='cuda')
        pt = bench(lambda: t_F.linear(lin_x_pt, lin_w_pt, lin_b_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        lin_x_sf = sf.tensor(lin_x_np, device='cuda')
        lin_w_sf = sf.tensor(lin_w_np, device='cuda')
        lin_b_sf = sf.tensor(lin_b_np, device='cuda')
        sf_t = bench(lambda: sf_F.linear(lin_x_sf, lin_w_sf, lin_b_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'Linear {B_l}x{D_in}→{D_out}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 9. Embedding lookup ──
    print('  [9/14] Embedding lookup ...', end='', flush=True)
    V, E_dim, S = sz['embed_vocab'], sz['embed_dim'], sz['embed_seq']
    emb_w_np = np.random.randn(V, E_dim).astype(np.float32)
    emb_idx_np = np.random.randint(0, V, size=(B_l, S)).astype(np.int64)

    if _HAS_PYTORCH_CUDA:
        emb_pt = tnn.Embedding(V, E_dim).cuda()
        emb_pt.weight.data = torch.tensor(emb_w_np, device='cuda')
        idx_pt = torch.tensor(emb_idx_np, device='cuda')
        pt = bench(lambda: emb_pt(idx_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        emb_sf = sf_nn.Embedding(V, E_dim)
        emb_sf.weight._data = emb_w_np.copy()
        idx_sf = sf.tensor(emb_idx_np, device='cuda')
        sf_t = bench(lambda: emb_sf(idx_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'Embedding {V}x{E_dim} seq={S}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 10. Layer Norm ──
    print('  [10/14] LayerNorm ...', end='', flush=True)
    LN_B, LN_D = sz['ln_batch'], sz['ln_dim']
    ln_np = np.random.randn(LN_B, LN_D).astype(np.float32)

    if _HAS_PYTORCH_CUDA:
        ln_pt_mod = tnn.LayerNorm(LN_D).cuda()
        ln_pt_x = torch.tensor(ln_np, device='cuda')
        pt = bench(lambda: ln_pt_mod(ln_pt_x), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        ln_sf_mod = sf_nn.LayerNorm(LN_D)
        ln_sf_x = sf.tensor(ln_np, device='cuda')
        sf_t = bench(lambda: ln_sf_mod(ln_sf_x), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'LayerNorm {LN_B}x{LN_D}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 11. ReLU ──
    print('  [11/14] ReLU ...', end='', flush=True)
    if _HAS_PYTORCH_CUDA:
        pt = bench(lambda: t_F.relu(x_pt), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        sf_t = bench(lambda: sf_F.relu(x_sf), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('ReLU (element-wise)', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 12. Dropout ──
    print('  [12/14] Dropout ...', end='', flush=True)
    if _HAS_PYTORCH_CUDA:
        x_pt_drop = torch.tensor(x_np, device='cuda')
        pt = bench(lambda: t_F.dropout(x_pt_drop, p=0.1, training=True), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        x_sf_drop = sf.tensor(x_np, device='cuda')
        sf_t = bench(lambda: sf_F.dropout(x_sf_drop, p=0.1, training=True), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult('Dropout p=0.1', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 13. AdamW optimizer step ──
    print('  [13/14] AdamW step ...', end='', flush=True)
    P = sz['adam_params']
    p_np = np.random.randn(P).astype(np.float32)
    g_np = np.random.randn(P).astype(np.float32) * 0.01

    if _HAS_PYTORCH_CUDA:
        p_pt_param = torch.nn.Parameter(torch.tensor(p_np, device='cuda'))
        opt_pt = torch.optim.AdamW([p_pt_param], lr=1e-3)
        p_pt_param.grad = torch.tensor(g_np, device='cuda')

        def pt_adam_step():
            p_pt_param.grad = torch.tensor(g_np, device='cuda')
            opt_pt.step()

        pt = bench(pt_adam_step, warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        p_sf_param = sf_nn.Parameter(sf.tensor(p_np.copy(), device='cuda', requires_grad=True))
        opt_sf = sf_optim.AdamW([p_sf_param], lr=1e-3)
        p_sf_param._grad = g_np.copy()

        def sf_adam_step():
            p_sf_param._grad = g_np.copy()
            opt_sf.step()

        sf_t = bench(sf_adam_step, warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(f'AdamW step ({P:,} params)', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    # ── 14. Transformer block forward (end-to-end) ──
    print('  [14/14] Transformer block forward ...', end='', flush=True)
    B_f, S_f, D_f, FFN = sz['fwd_batch'], sz['fwd_seq'], sz['fwd_dim'], sz['fwd_ffn']

    if _HAS_PYTORCH_CUDA:
        class PtBlock(tnn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = tnn.LayerNorm(D_f)
                self.attn_qkv = tnn.Linear(D_f, 3 * D_f, bias=False)
                self.attn_out = tnn.Linear(D_f, D_f, bias=False)
                self.ln2 = tnn.LayerNorm(D_f)
                self.ff1 = tnn.Linear(D_f, FFN, bias=False)
                self.ff2 = tnn.Linear(FFN, D_f, bias=False)

            def forward(self, x):
                h = self.ln1(x)
                qkv = self.attn_qkv(h).reshape(B_f, S_f, 3, 8, D_f // 8)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = q.permute(0, 3, 1, 2)
                k = k.permute(0, 3, 1, 2)
                v = v.permute(0, 3, 1, 2)
                scale = (D_f // 8) ** -0.5
                attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_w = torch.softmax(attn_w, dim=-1)
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.permute(0, 2, 1, 3).reshape(B_f, S_f, D_f)
                x = x + self.attn_out(attn_out)
                h2 = self.ln2(x)
                x = x + self.ff2(t_F.silu(self.ff1(h2)))
                return x

        pt_block = PtBlock().cuda().eval()
        pt_inp = torch.randn(B_f, S_f, D_f, device='cuda')
        with torch.no_grad():
            pt = bench(lambda: pt_block(pt_inp), warmup, trials)
    else:
        pt = {'median': float('inf')}

    if _HAS_SCAFFOLDING_CUDA:
        class SfBlock(sf_nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = sf_nn.LayerNorm(D_f)
                self.attn_qkv = sf_nn.Linear(D_f, 3 * D_f, bias=False)
                self.attn_out = sf_nn.Linear(D_f, D_f, bias=False)
                self.ln2 = sf_nn.LayerNorm(D_f)
                self.ff1 = sf_nn.Linear(D_f, FFN, bias=False)
                self.ff2 = sf_nn.Linear(FFN, D_f, bias=False)

            def forward(self, x):
                h = self.ln1(x)
                qkv = self.attn_qkv(h)
                # Attention head reshape requires CPU roundtrip for permute
                qkv_r = qkv.reshape(B_f, S_f, 3, 8, D_f // 8)
                q = qkv_r[:, :, 0].permute(0, 2, 1, 3).contiguous().to('cuda')
                k = qkv_r[:, :, 1].permute(0, 2, 1, 3).contiguous().to('cuda')
                v = qkv_r[:, :, 2].permute(0, 2, 1, 3).contiguous().to('cuda')
                scale = (D_f // 8) ** -0.5
                attn_w = sf_F.softmax(q.matmul(k.transpose(-2, -1)) * scale, dim=-1)
                attn_out = attn_w.matmul(v)
                # Merge heads back
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(
                    B_f, S_f, D_f).to('cuda')
                x = x + self.attn_out(attn_out)
                h2 = self.ln2(x)
                ff_out = self.ff2(sf_F.silu(self.ff1(h2)))
                return x + ff_out

        sf_block = SfBlock()
        sf_block.eval()
        sf_inp = sf.randn(B_f, S_f, D_f, device='cuda')
        sf_t = bench(lambda: sf_block(sf_inp), warmup, trials)
    else:
        sf_t = {'median': float('inf')}

    results.append(BenchmarkResult(
        f'Transformer Blk {B_f}x{S_f}x{D_f}', sf_t['median'], pt['median']))
    print(f' done  sf={sf_t["median"]:.3f}ms  pt={pt["median"]:.3f}ms')

    return results


# ================================================================
#  Scoreboard
# ================================================================

def print_scoreboard(results: list[BenchmarkResult], size_name: str):
    sf_wins = sum(1 for r in results if r.winner == 'Scaffolding')
    pt_wins = sum(1 for r in results if r.winner == 'PyTorch')
    total = len(results)

    hdr = f'  CUDA Performance Benchmark — size={size_name}'
    print()
    print('╔' + '═' * 80 + '╗')
    print(f'║{hdr:<80}║')
    print('╠' + '═' * 80 + '╣')
    print(f'║  {"Benchmark":<32} {"Scaffolding":>10} {"PyTorch":>10} '
          f'{"Speedup":>9} {"Winner":>12}  ║')
    print('╠' + '─' * 80 + '╣')

    for r in results:
        sf_str = f'{r.sf_ms:.3f}ms' if r.sf_ms < float('inf') else 'N/A'
        pt_str = f'{r.pt_ms:.3f}ms' if r.pt_ms < float('inf') else 'N/A'
        if r.speedup != r.speedup:  # NaN
            sp_str = 'N/A'
        elif r.speedup >= 1.0:
            sp_str = f'{r.speedup:.2f}x'
        else:
            sp_str = f'{1.0/r.speedup:.2f}x'
        w_str = r.winner
        marker = '>>>' if r.winner == 'Scaffolding' else '   '
        print(f'║{marker} {r.name:<31} {sf_str:>10} {pt_str:>10} '
              f'{sp_str:>9} {w_str:>12}  ║')

    print('╠' + '═' * 80 + '╣')
    print(f'║  FINAL SCORE:  Scaffolding {sf_wins}  —  PyTorch {pt_wins}'
          f'  (of {total} benchmarks)' + ' ' * (80 - 55 - len(str(sf_wins))
          - len(str(pt_wins)) - len(str(total))) + '║')

    # Geometric mean speedup
    valid = [r.speedup for r in results
             if r.speedup == r.speedup and r.speedup > 0]
    if valid:
        geo_mean = np.exp(np.mean(np.log(valid)))
        if geo_mean >= 1.0:
            print(f'║  Geometric mean speedup: Scaffolding is {geo_mean:.2f}x '
                  f'vs PyTorch' + ' ' * (80 - 51 - len(f'{geo_mean:.2f}')) + '║')
        else:
            inv = 1.0 / geo_mean
            print(f'║  Geometric mean speedup: PyTorch is {inv:.2f}x '
                  f'vs Scaffolding' + ' ' * (80 - 51 - len(f'{inv:.2f}')) + '║')

    print('╚' + '═' * 80 + '╝')
    print()


# ================================================================
#  GPU info
# ================================================================

def print_gpu_info():
    print()
    print('═' * 60)
    print('  GPU Information')
    print('═' * 60)

    if _HAS_PYTORCH_CUDA:
        dev = torch.cuda.current_device()
        print(f'  PyTorch CUDA:     {torch.version.cuda}')
        print(f'  PyTorch device:   {torch.cuda.get_device_name(dev)}')
        cap = torch.cuda.get_device_capability(dev)
        print(f'  Compute cap:      {cap[0]}.{cap[1]}')
        mem = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        print(f'  Total memory:     {mem:.1f} GB')
    else:
        print('  PyTorch CUDA:     NOT AVAILABLE')

    if _HAS_SCAFFOLDING_CUDA:
        print(f'  Scaffolding CUDA: available')
        print(f'  Scaffolding dev:  {sf.cuda.get_device_name(0)}')
        cap = sf.cuda.get_device_capability(0)
        print(f'  Compute cap:      {cap[0]}.{cap[1]}')
    else:
        print('  Scaffolding CUDA: NOT AVAILABLE')

    print('═' * 60)
    print()


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scaffolding vs PyTorch CUDA Performance Benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations (default: 3)')
    parser.add_argument('--trials', type=int, default=10,
                        help='Timing trials (default: 10)')
    parser.add_argument('--sizes', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'all'],
                        help='Problem size preset (default: medium)')
    args = parser.parse_args()

    if not _HAS_SCAFFOLDING_CUDA and not _HAS_PYTORCH_CUDA:
        print('ERROR: No CUDA support detected in either framework.')
        print('  - Scaffolding: build with CUDA (python setup.py build_ext --inplace)')
        print('  - PyTorch: install with CUDA (pip install torch --index-url ...)')
        sys.exit(1)

    if not _HAS_SCAFFOLDING_CUDA:
        print('WARNING: Scaffolding CUDA not available. Only PyTorch times shown.')
    if not _HAS_PYTORCH_CUDA:
        print('WARNING: PyTorch CUDA not available. Only Scaffolding times shown.')

    print_gpu_info()

    if args.sizes == 'all':
        for name in ('small', 'medium', 'large'):
            print(f'\n▶ Running benchmarks: size={name}, '
                  f'warmup={args.warmup}, trials={args.trials}')
            results = run_benchmarks(SIZES[name], args.warmup, args.trials)
            print_scoreboard(results, name)
    else:
        name = args.sizes
        print(f'▶ Running benchmarks: size={name}, '
              f'warmup={args.warmup}, trials={args.trials}')
        results = run_benchmarks(SIZES[name], args.warmup, args.trials)
        print_scoreboard(results, name)


if __name__ == '__main__':
    main()
