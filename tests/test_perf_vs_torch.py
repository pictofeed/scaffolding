"""
Scaffolding vs PyTorch — Comprehensive Performance Benchmark
=============================================================
Compares median latency across multiple trials with GC disabled during
timed sections.  Tests multiple tensor sizes and covers matmul, softmax,
activations, element-wise ops, cross-entropy, backward pass, and AdamW.

Run:  python3 tests/test_perf_vs_torch.py
"""

import os, sys, gc, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
import numpy as np

import scaffolding
import scaffolding.nn as snn
import scaffolding.nn.functional as sF
from scaffolding import autograd as sag

try:
    import torch as pt
    import torch.nn as pnn
    import torch.nn.functional as pF
    HAS_PT = True
except ImportError:
    print("PyTorch not installed -- skipping PyTorch benchmarks.")
    HAS_PT = False

np.random.seed(42)

# ── Results collector ─────────────────────────────────────────────────
RESULTS = []          # list of (name, t_scaffolding, t_pytorch)
SEP = "-" * 72


# ── Helpers ───────────────────────────────────────────────────────────
def bench(fn, n=20, warmup=5):
    """Return median and stdev of *n* trials (seconds) with GC off."""
    for _ in range(warmup):
        fn()
    gc.disable()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    gc.enable()
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


def np_to_pt(arr):
    """Convert numpy array to PyTorch tensor (workaround for broken from_numpy)."""
    return pt.tensor(arr.tolist(), dtype=pt.float32)


def report(name, med_s, std_s, med_p=None, std_p=None):
    """Pretty-print one benchmark row and record it."""
    print(f"  Scaffolding: {med_s*1000:9.2f} ms  (std {std_s*1000:.2f})")
    if med_p is not None:
        print(f"  PyTorch:     {med_p*1000:9.2f} ms  (std {std_p*1000:.2f})")
        ratio = med_p / med_s
        winner = "Scaffolding" if ratio > 1 else "PyTorch"
        print(f"  -> {winner} is {max(ratio, 1/ratio):.2f}x faster")
    RESULTS.append((name, med_s, med_p))
    print()


def print_summary():
    """Print a final summary table with wins/losses."""
    print("\n" + "=" * 72)
    print(f"{'BENCHMARK SUMMARY':^72}")
    print("=" * 72)
    print(f"  {'Test':<40} {'Scaffolding':>10} {'PyTorch':>10} {'Winner':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    s_wins = p_wins = ties = 0
    for name, ts, tp in RESULTS:
        ts_ms = f"{ts*1000:.1f}ms"
        if tp is not None:
            tp_ms = f"{tp*1000:.1f}ms"
            ratio = tp / ts
            if ratio > 1.02:
                winner = "Scaff."
                s_wins += 1
            elif ratio < 0.98:
                winner = "PyTorch"
                p_wins += 1
            else:
                winner = "TIE"
                ties += 1
        else:
            tp_ms = "N/A"
            winner = ""
        print(f"  {name:<40} {ts_ms:>10} {tp_ms:>10} {winner:>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  Scaffolding wins: {s_wins}   PyTorch wins: {p_wins}   Ties: {ties}")
    print("=" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════
#  BENCHMARKS
# ══════════════════════════════════════════════════════════════════════

# ── 1. 2D Matmul (single) ────────────────────────────────────────────
for sz in [256, 1024, 2048]:
    label = f"2D Matmul ({sz}x{sz})"
    print(f"\n=== {label} ===")
    a_np = np.random.randn(sz, sz).astype(np.float32)
    b_np = np.random.randn(sz, sz).astype(np.float32)

    a_s = scaffolding.tensor(a_np)
    b_s = scaffolding.tensor(b_np)
    ms, ss = bench(lambda: scaffolding.matmul(a_s, b_s))

    mp = sp = None
    if HAS_PT:
        a_p = np_to_pt(a_np)
        b_p = np_to_pt(b_np)
        mp, sp = bench(lambda: pt.matmul(a_p, b_p))

    report(label, ms, ss, mp, sp)


# ── 2. Batched Matmul ────────────────────────────────────────────────
for bsz, dim in [(16, 256), (64, 512)]:
    label = f"Batched Matmul ({bsz}x{dim}x{dim})"
    print(f"=== {label} ===")
    a_np = np.random.randn(bsz, dim, dim).astype(np.float32)
    b_np = np.random.randn(bsz, dim, dim).astype(np.float32)

    a_s = scaffolding.tensor(a_np)
    b_s = scaffolding.tensor(b_np)
    ms, ss = bench(lambda: scaffolding.matmul(a_s, b_s))

    mp = sp = None
    if HAS_PT:
        a_p = np_to_pt(a_np)
        b_p = np_to_pt(b_np)
        mp, sp = bench(lambda: pt.matmul(a_p, b_p))

    report(label, ms, ss, mp, sp)


# ── 3. Softmax ───────────────────────────────────────────────────────
for shape, tag in [((64, 512), "2D 64x512"), ((64, 512, 512), "3D 64x512x512")]:
    label = f"Softmax ({tag})"
    print(f"=== {label} ===")
    x_np = np.random.randn(*shape).astype(np.float32)

    x_s = scaffolding.tensor(x_np)
    ms, ss = bench(lambda: sF.softmax(x_s, dim=-1))

    mp = sp = None
    if HAS_PT:
        x_p = np_to_pt(x_np)
        mp, sp = bench(lambda: pF.softmax(x_p, dim=-1))

    report(label, ms, ss, mp, sp)


# ── 4. Activations (SiLU, GELU, ReLU) ────────────────────────────────
x_np = np.random.randn(64, 512, 512).astype(np.float32)

for act_name, s_fn, p_fn in [
    ("SiLU", sF.silu, pF.silu if HAS_PT else None),
    ("GELU", sF.gelu, pF.gelu if HAS_PT else None),
    ("ReLU", sF.relu, pF.relu if HAS_PT else None),
]:
    label = f"{act_name} (64x512x512)"
    print(f"=== {label} ===")
    x_s = scaffolding.tensor(x_np)
    ms, ss = bench(lambda _f=s_fn: _f(x_s))

    mp = sp = None
    if HAS_PT and p_fn is not None:
        x_p = np_to_pt(x_np)
        mp, sp = bench(lambda _f=p_fn: _f(x_p))

    report(label, ms, ss, mp, sp)


# ── 5. Element-wise chain (add + mul + exp) ──────────────────────────
for shape, tag in [((1024, 1024), "1Kx1K"), ((64, 512, 512), "64x512x512")]:
    label = f"Elem add+mul+exp ({tag})"
    print(f"=== {label} ===")
    x_np = np.random.randn(*shape).astype(np.float32)
    y_np = np.random.randn(*shape).astype(np.float32)

    x_s = scaffolding.tensor(x_np)
    y_s = scaffolding.tensor(y_np)
    def _elem_s():
        z = (x_s + y_s) * y_s
        return z.exp()
    ms, ss = bench(_elem_s)

    mp = sp = None
    if HAS_PT:
        x_p = np_to_pt(x_np)
        y_p = np_to_pt(y_np)
        def _elem_p():
            z = (x_p + y_p) * y_p
            return z.exp()
        mp, sp = bench(_elem_p)

    report(label, ms, ss, mp, sp)


# ── 6. Cross-Entropy Loss ────────────────────────────────────────────
label = "Cross-Entropy (B=256, C=1000)"
print(f"=== {label} ===")
logits_np = np.random.randn(256, 1000).astype(np.float32)
targets_np = np.random.randint(0, 1000, size=(256,)).astype(np.int64)

logits_s = scaffolding.tensor(logits_np)
targets_s = scaffolding.tensor(targets_np)
ms, ss = bench(lambda: sF.cross_entropy(logits_s, targets_s))

mp = sp = None
if HAS_PT:
    logits_p = np_to_pt(logits_np)
    targets_p = pt.tensor(targets_np.tolist(), dtype=pt.long)
    mp, sp = bench(lambda: pF.cross_entropy(logits_p, targets_p))

report(label, ms, ss, mp, sp)


# ── 7. MLP Forward ───────────────────────────────────────────────────
D_in, D_hid, D_out = 512, 2048, 512

for bsz, seq in [(16, 128), (64, 512)]:
    label = f"MLP Fwd ({bsz}x{seq}x{D_in}->{D_hid}->{D_out})"
    print(f"=== {label} ===")
    x_np = np.random.randn(bsz, seq, D_in).astype(np.float32)

    class _SMLP(snn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = snn.Linear(D_in, D_hid)
            self.l2 = snn.Linear(D_hid, D_out)
        def forward(self, x):
            return self.l2(sF.silu(self.l1(x)))

    mlp_s = _SMLP()
    x_s = scaffolding.tensor(x_np)
    ms, ss = bench(lambda: mlp_s(x_s))

    mp = sp = None
    if HAS_PT:
        class _PMLP(pnn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = pnn.Linear(D_in, D_hid)
                self.l2 = pnn.Linear(D_hid, D_out)
            def forward(self, x):
                return self.l2(pF.silu(self.l1(x)))

        mlp_p = _PMLP()
        mlp_p.eval()
        x_p = np_to_pt(x_np)

        @pt.no_grad()
        def _fwd_p():
            return mlp_p(x_p)

        mp, sp = bench(_fwd_p)

    report(label, ms, ss, mp, sp)


# ── 8. Backward Pass (Linear + SiLU) ─────────────────────────────────
label = "Backward (Linear+SiLU, 64x512x512)"
print(f"=== {label} ===")

lin_s = snn.Linear(512, 512)
x_np = np.random.randn(64, 512, 512).astype(np.float32)
x_s = scaffolding.tensor(x_np, requires_grad=True)

def bwd_s():
    out = sF.silu(lin_s(x_s))
    loss = out.sum()
    loss.backward()

ms, ss = bench(bwd_s, n=5, warmup=2)

mp = sp = None
if HAS_PT:
    lin_p = pnn.Linear(512, 512)
    x_p = np_to_pt(x_np).requires_grad_(True)

    def bwd_p():
        out = pF.silu(lin_p(x_p))
        loss = out.sum()
        loss.backward()
        lin_p.zero_grad()
        if x_p.grad is not None:
            x_p.grad = None

    mp, sp = bench(bwd_p, n=5, warmup=2)

report(label, ms, ss, mp, sp)


# ── 9. AdamW Step ─────────────────────────────────────────────────────
label = "AdamW step (2-layer MLP, 2M params)"
print(f"=== {label} ===")

mlp_s2 = _SMLP()
opt_s = scaffolding.optim.AdamW(mlp_s2.parameters(), lr=1e-3)
x_s2 = scaffolding.tensor(np.random.randn(16, 128, D_in).astype(np.float32), requires_grad=True)

def adam_s():
    opt_s.zero_grad()
    out = mlp_s2(x_s2)
    loss = out.sum()
    loss.backward()
    opt_s.step()

ms, ss = bench(adam_s, n=5, warmup=2)

mp = sp = None
if HAS_PT:
    mlp_p2 = _PMLP()
    opt_p = pt.optim.AdamW(mlp_p2.parameters(), lr=1e-3)
    x_p2 = np_to_pt(np.random.randn(16, 128, D_in).astype(np.float32)).requires_grad_(True)

    def adam_p():
        opt_p.zero_grad()
        out = mlp_p2(x_p2)
        loss = out.sum()
        loss.backward()
        opt_p.step()

    mp, sp = bench(adam_p, n=5, warmup=2)

report(label, ms, ss, mp, sp)


# ══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print_summary()
