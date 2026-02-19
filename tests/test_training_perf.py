#!/usr/bin/env python3
"""
Large-scale training benchmark: scaffolding vs PyTorch.

Builds identical Transformer-based language models in both frameworks,
trains them on the same synthetic data for multiple steps, and compares
wall-clock time across every phase of training.

Tests:
  1. Model construction & init
  2. Forward pass (varying batch sizes & sequence lengths)
  3. Loss computation (cross-entropy on large vocabularies)
  4. Backward pass (full gradient computation)
  5. Optimizer step (AdamW with weight decay)
  6. Grad clipping
  7. LR scheduling
  8. Full training step (fwd + loss + bwd + clip + optim + sched)
  9. Multi-step training loop (many iterations, large data)
 10. Inference / generation pass (no_grad)

Usage:
    python tests/test_training_perf.py
    python tests/test_training_perf.py --steps 200 --batch 8
"""
import sys, os, time, argparse, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Import both frameworks ──────────────────────────────────────────
import scaffolding as sf
import scaffolding.nn as sf_nn
import scaffolding.nn.functional as sf_F
import scaffolding.optim as sf_optim

import torch
import torch.nn as pt_nn
import torch.nn.functional as pt_F

# ── CLI args ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Training perf benchmark")
parser.add_argument("--steps",   type=int, default=100,  help="Training steps for the full loop")
parser.add_argument("--batch",   type=int, default=4,    help="Batch size")
parser.add_argument("--seq",     type=int, default=256,  help="Sequence length")
parser.add_argument("--dim",     type=int, default=512,  help="Model dimension")
parser.add_argument("--heads",   type=int, default=8,    help="Attention heads")
parser.add_argument("--layers",  type=int, default=4,    help="Transformer layers")
parser.add_argument("--vocab",   type=int, default=8192, help="Vocabulary size")
parser.add_argument("--warmup",  type=int, default=3,    help="Warmup iterations (untimed)")
parser.add_argument("--repeats", type=int, default=10,   help="Timed iterations for per-phase benchmarks")
parser.add_argument("--lr",      type=float, default=3e-4, help="Learning rate")
args = parser.parse_args()

B      = args.batch
S      = args.seq
D      = args.dim
H      = args.heads
HD     = D // H
N_LAYERS = args.layers
VOCAB  = args.vocab
WARMUP = args.warmup
REPEATS = args.repeats
STEPS  = args.steps
LR     = args.lr

sf_device = sf.device("mps")
pt_device = torch.device("cpu")

np.random.seed(42)
sf.manual_seed(42)
torch.manual_seed(42)

# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

_NP_TO_TORCH_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int64:   torch.int64,
    np.int32:   torch.int32,
    np.bool_:   torch.bool,
}

def np_to_pt(arr, requires_grad=False):
    a = np.ascontiguousarray(arr)
    dt = _NP_TO_TORCH_DTYPE.get(a.dtype.type, torch.float32)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype=dt).reshape(a.shape)
    if requires_grad:
        t = t.requires_grad_(True)
    return t

def np_to_sf(arr):
    return sf.tensor(arr).to(sf_device)


def time_fn(fn, warmup=WARMUP, repeats=REPEATS):
    """Time a function, return median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def fmt_row(label, t_sf, t_pt):
    ratio = t_sf / t_pt if t_pt > 0 else float('inf')
    tag = "FASTER" if ratio < 0.95 else "OK" if ratio < 1.5 else "SLOW" if ratio < 3.0 else "VERY SLOW"
    bar = ""
    if ratio < 1.0:
        bar = "◀ " + "█" * min(int((1.0/ratio - 1) * 20), 30)
    else:
        bar = "▶ " + "█" * min(int((ratio - 1) * 20), 30)
    return (f"  {label:<50s}  sf={t_sf*1e3:9.2f}ms  pt={t_pt*1e3:9.2f}ms  "
            f"{ratio:6.2f}x  [{tag:^9s}]  {bar}")


results = []
def record(label, t_sf, t_pt):
    ratio = t_sf / t_pt if t_pt > 0 else float('inf')
    results.append((label, t_sf, t_pt, ratio))
    print(fmt_row(label, t_sf, t_pt))


# ═════════════════════════════════════════════════════════════════════
#  Model Definitions — identical architectures in both frameworks
# ═════════════════════════════════════════════════════════════════════

class SF_MultiHeadAttention(sf_nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = sf_nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = sf_nn.Linear(dim, dim, bias=False)
        self.dropout = sf_nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape[0], x.shape[1], x.shape[2]
        qkv = self.qkv_proj(x)                           # (B, S, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)                   # 3x (B, S, D)

        q = q.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,S,HD)
        k = k.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = sf_F.scaled_dot_product_attention(q, k, v)  # (B,H,S,HD)
        attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, S, D)
        return self.out_proj(attn)


class SF_FeedForward(sf_nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = dim * mult
        self.w1 = sf_nn.Linear(dim, hidden, bias=False)
        self.w2 = sf_nn.Linear(hidden, dim, bias=False)
        self.act = sf_nn.SiLU()
        self.dropout = sf_nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))


class SF_TransformerBlock(sf_nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = sf_nn.RMSNorm(dim)
        self.attn = SF_MultiHeadAttention(dim, n_heads, dropout)
        self.ln2 = sf_nn.RMSNorm(dim)
        self.ff = SF_FeedForward(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SF_TransformerLM(sf_nn.Module):
    def __init__(self, vocab_size, dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.tok_emb = sf_nn.Embedding(vocab_size, dim)
        self.layers = sf_nn.ModuleList([
            SF_TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = sf_nn.RMSNorm(dim)
        self.head = sf_nn.Linear(dim, vocab_size, bias=False)
        self.dropout = sf_nn.Dropout(dropout)

    def forward(self, idx):
        x = self.tok_emb(idx)                             # (B, S, D)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)                             # (B, S, V)
        return logits


# ── PyTorch mirror ──────────────────────────────────────────────────

class PT_RMSNorm(pt_nn.Module):
    """RMSNorm for PyTorch versions that lack nn.RMSNorm (< 2.4)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = pt_nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class PT_MultiHeadAttention(pt_nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = pt_nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = pt_nn.Linear(dim, dim, bias=False)
        self.dropout = pt_nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = pt_F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, S, D)
        return self.out_proj(attn)


class PT_FeedForward(pt_nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = dim * mult
        self.w1 = pt_nn.Linear(dim, hidden, bias=False)
        self.w2 = pt_nn.Linear(hidden, dim, bias=False)
        self.act = pt_nn.SiLU()
        self.dropout = pt_nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))


class PT_TransformerBlock(pt_nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = PT_RMSNorm(dim)
        self.attn = PT_MultiHeadAttention(dim, n_heads, dropout)
        self.ln2 = PT_RMSNorm(dim)
        self.ff = PT_FeedForward(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PT_TransformerLM(pt_nn.Module):
    def __init__(self, vocab_size, dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.tok_emb = pt_nn.Embedding(vocab_size, dim)
        self.layers = pt_nn.ModuleList([
            PT_TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = PT_RMSNorm(dim)
        self.head = pt_nn.Linear(dim, vocab_size, bias=False)
        self.dropout = pt_nn.Dropout(dropout)

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ═════════════════════════════════════════════════════════════════════
#  Sync weights between the two models so they start identically
# ═════════════════════════════════════════════════════════════════════

def sync_weights(sf_model, pt_model):
    """Copy scaffolding model weights into PyTorch model (same init)."""
    sf_sd = sf_model.state_dict()
    pt_sd = pt_model.state_dict()
    for key in pt_sd:
        if key in sf_sd:
            arr = sf_sd[key]._data if hasattr(sf_sd[key], '_data') else sf_sd[key]
            pt_sd[key] = np_to_pt(np.array(arr))
    pt_model.load_state_dict(pt_sd)


# ═════════════════════════════════════════════════════════════════════
#  Generate synthetic training data
# ═════════════════════════════════════════════════════════════════════

def generate_data(total_tokens, vocab_size, seq_len, batch_size):
    """Generate enough token sequences for the full training run."""
    n_batches = max(total_tokens // (batch_size * seq_len), STEPS + WARMUP + 5)
    # Random token IDs as training data
    all_input_ids  = np.random.randint(0, vocab_size, (n_batches, batch_size, seq_len)).astype(np.int64)
    all_target_ids = np.random.randint(0, vocab_size, (n_batches, batch_size, seq_len)).astype(np.int64)
    return all_input_ids, all_target_ids


# ═════════════════════════════════════════════════════════════════════
#  Build models, optimizers, schedulers
# ═════════════════════════════════════════════════════════════════════

print("=" * 100)
print(f"  Training Performance Benchmark: scaffolding vs PyTorch")
print(f"  Model: TransformerLM — {N_LAYERS} layers, {D} dim, {H} heads, {VOCAB} vocab")
print(f"  Training: batch={B}, seq={S}, steps={STEPS}, lr={LR}")
total_data_tokens = STEPS * B * S
print(f"  Total training tokens: {total_data_tokens:,}")
print("=" * 100)

# ── Build models ────────────────────────────────────────────────────
print("\n── Building models ──")
t0 = time.perf_counter()
sf_model = SF_TransformerLM(VOCAB, D, H, N_LAYERS, dropout=0.1)
sf_model.to(sf_device)
sf_model.train()
t_sf_build = time.perf_counter() - t0

t0 = time.perf_counter()
pt_model = PT_TransformerLM(VOCAB, D, H, N_LAYERS, dropout=0.1)
pt_model.to(pt_device)
pt_model.train()
t_pt_build = time.perf_counter() - t0

record("Model construction", t_sf_build, t_pt_build)

n_params_sf = sum(p.numel() for p in sf_model.parameters())
n_params_pt = sum(p.numel() for p in pt_model.parameters())
print(f"  Parameters: sf={n_params_sf:,}  pt={n_params_pt:,}")

# Sync weights
sync_weights(sf_model, pt_model)

# ── Build optimizers ────────────────────────────────────────────────
sf_opt = sf_optim.AdamW(sf_model.parameters(), lr=LR, weight_decay=0.01)
pt_opt = torch.optim.AdamW(pt_model.parameters(), lr=LR, weight_decay=0.01)

# ── Build schedulers ───────────────────────────────────────────────
sf_sched = sf_optim.lr_scheduler.CosineAnnealingLR(sf_opt, T_max=STEPS, eta_min=LR * 0.1)
pt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pt_opt, T_max=STEPS, eta_min=LR * 0.1)

# ── Generate data ──────────────────────────────────────────────────
print("\n── Generating synthetic training data ──")
all_inputs, all_targets = generate_data(total_data_tokens, VOCAB, S, B)
print(f"  Data: {all_inputs.shape[0]} batches x ({B},{S}) = {all_inputs.shape[0]*B*S:,} tokens")

# ═════════════════════════════════════════════════════════════════════
#  PHASE 1: Forward pass benchmarks (various sizes)
# ═════════════════════════════════════════════════════════════════════

print("\n── Phase 1: Forward Pass (no grad) ──")

for label, b, s in [("Small  (B=1, S=64)", 1, 64),
                     ("Medium (B=4, S=256)", 4, 256),
                     ("Large  (B=8, S=512)", 8, 512),
                     (f"Config (B={B}, S={S})", B, S)]:
    inp_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    sf_inp = np_to_sf(inp_np)
    pt_inp = np_to_pt(inp_np)

    sf_model.eval()
    pt_model.eval()

    def sf_fwd():
        with sf.no_grad():
            sf_model(sf_inp)

    def pt_fwd():
        with torch.no_grad():
            pt_model(pt_inp)

    t_sf = time_fn(sf_fwd)
    t_pt = time_fn(pt_fwd)
    record(f"Forward (no_grad) {label}", t_sf, t_pt)

    sf_model.train()
    pt_model.train()


# ═════════════════════════════════════════════════════════════════════
#  PHASE 2: Forward + loss
# ═════════════════════════════════════════════════════════════════════

print("\n── Phase 2: Forward + Cross-Entropy Loss ──")

for label, b, s in [("Small  (B=2, S=128)", 2, 128),
                     ("Medium (B=4, S=256)", 4, 256),
                     ("Large  (B=8, S=512)", 8, 512)]:
    inp_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    tgt_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    sf_inp = np_to_sf(inp_np)
    sf_tgt = np_to_sf(tgt_np)
    pt_inp = np_to_pt(inp_np)
    pt_tgt = np_to_pt(tgt_np)

    def sf_fwd_loss():
        logits = sf_model(sf_inp)
        loss = sf_F.cross_entropy(logits.reshape(-1, VOCAB), sf_tgt.reshape(-1))
        return loss

    def pt_fwd_loss():
        logits = pt_model(pt_inp)
        loss = pt_F.cross_entropy(logits.reshape(-1, VOCAB), pt_tgt.reshape(-1))
        return loss

    t_sf = time_fn(sf_fwd_loss)
    t_pt = time_fn(pt_fwd_loss)
    record(f"Fwd + CE loss {label}", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 3: Full training step (fwd + loss + bwd + clip + optim)
# ═════════════════════════════════════════════════════════════════════

print("\n── Phase 3: Full Training Step (fwd + loss + bwd + clip + optim + sched) ──")

for label, b, s in [("Small  (B=2, S=128)", 2, 128),
                     ("Medium (B=4, S=256)", 4, 256),
                     ("Large  (B=8, S=512)", 8, 512)]:
    inp_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    tgt_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    sf_inp = np_to_sf(inp_np)
    sf_tgt = np_to_sf(tgt_np)
    pt_inp = np_to_pt(inp_np)
    pt_tgt = np_to_pt(tgt_np)

    def sf_train_step():
        sf_opt.zero_grad()
        logits = sf_model(sf_inp)
        loss = sf_F.cross_entropy(logits.reshape(-1, VOCAB), sf_tgt.reshape(-1))
        loss.backward()
        sf_nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
        sf_opt.step()

    def pt_train_step():
        pt_opt.zero_grad()
        logits = pt_model(pt_inp)
        loss = pt_F.cross_entropy(logits.reshape(-1, VOCAB), pt_tgt.reshape(-1))
        loss.backward()
        pt_nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)
        pt_opt.step()

    t_sf = time_fn(sf_train_step)
    t_pt = time_fn(pt_train_step)
    record(f"Full train step {label}", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 4: Component-level training breakdown
# ═════════════════════════════════════════════════════════════════════

print("\n── Phase 4: Training Step Breakdown (B={}, S={}) ──".format(B, S))

inp_np = np.random.randint(0, VOCAB, (B, S)).astype(np.int64)
tgt_np = np.random.randint(0, VOCAB, (B, S)).astype(np.int64)
sf_inp = np_to_sf(inp_np)
sf_tgt = np_to_sf(tgt_np)
pt_inp = np_to_pt(inp_np)
pt_tgt = np_to_pt(tgt_np)

# 4a. Forward only (with grad)
def sf_fwd_grad():
    return sf_model(sf_inp)

def pt_fwd_grad():
    return pt_model(pt_inp)

t_sf = time_fn(sf_fwd_grad)
t_pt = time_fn(pt_fwd_grad)
record("Forward (with grad)", t_sf, t_pt)

# 4b. Loss computation
sf_logits = sf_model(sf_inp)
pt_logits = pt_model(pt_inp)

def sf_loss_only():
    return sf_F.cross_entropy(sf_logits.reshape(-1, VOCAB), sf_tgt.reshape(-1))

def pt_loss_only():
    return pt_F.cross_entropy(pt_logits.reshape(-1, VOCAB), pt_tgt.reshape(-1))

t_sf = time_fn(sf_loss_only)
t_pt = time_fn(pt_loss_only)
record("Loss computation only", t_sf, t_pt)

# 4c. Backward only
def sf_bwd():
    sf_opt.zero_grad()
    logits = sf_model(sf_inp)
    loss = sf_F.cross_entropy(logits.reshape(-1, VOCAB), sf_tgt.reshape(-1))
    loss.backward()

def pt_bwd():
    pt_opt.zero_grad()
    logits = pt_model(pt_inp)
    loss = pt_F.cross_entropy(logits.reshape(-1, VOCAB), pt_tgt.reshape(-1))
    loss.backward()

t_sf = time_fn(sf_bwd)
t_pt = time_fn(pt_bwd)
record("Fwd + Bwd (backward portion)", t_sf, t_pt)

# 4d. Grad clipping
sf_opt.zero_grad()
sf_logits2 = sf_model(sf_inp)
sf_loss2 = sf_F.cross_entropy(sf_logits2.reshape(-1, VOCAB), sf_tgt.reshape(-1))
sf_loss2.backward()

pt_opt.zero_grad()
pt_logits2 = pt_model(pt_inp)
pt_loss2 = pt_F.cross_entropy(pt_logits2.reshape(-1, VOCAB), pt_tgt.reshape(-1))
pt_loss2.backward()

def sf_clip():
    sf_nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)

def pt_clip():
    pt_nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)

t_sf = time_fn(sf_clip)
t_pt = time_fn(pt_clip)
record("Grad clipping (clip_grad_norm_)", t_sf, t_pt)

# 4e. Optimizer step
def sf_optim_step():
    sf_opt.step()

def pt_optim_step():
    pt_opt.step()

t_sf = time_fn(sf_optim_step)
t_pt = time_fn(pt_optim_step)
record("Optimizer step (AdamW)", t_sf, t_pt)

# 4f. Scheduler step
def sf_sched_step():
    sf_sched.step()

def pt_sched_step():
    pt_sched.step()

t_sf = time_fn(sf_sched_step)
t_pt = time_fn(pt_sched_step)
record("Scheduler step (CosineAnnealing)", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 5: Multi-step training loop — the BIG test
# ═════════════════════════════════════════════════════════════════════

print(f"\n── Phase 5: Full Training Loop ({STEPS} steps, B={B}, S={S}) ──")

# Rebuild fresh models and optimizers for a clean run
sf_model2 = SF_TransformerLM(VOCAB, D, H, N_LAYERS, dropout=0.1)
sf_model2.to(sf_device)
sf_model2.train()
sf_opt2 = sf_optim.AdamW(sf_model2.parameters(), lr=LR, weight_decay=0.01)
sf_sched2 = sf_optim.lr_scheduler.CosineAnnealingLR(sf_opt2, T_max=STEPS, eta_min=LR * 0.1)

pt_model2 = PT_TransformerLM(VOCAB, D, H, N_LAYERS, dropout=0.1)
pt_model2.to(pt_device)
pt_model2.train()
sync_weights(sf_model2, pt_model2)
pt_opt2 = torch.optim.AdamW(pt_model2.parameters(), lr=LR, weight_decay=0.01)
pt_sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(pt_opt2, T_max=STEPS, eta_min=LR * 0.1)

# Warmup
print(f"  Warming up ({WARMUP} steps)...")
for i in range(WARMUP):
    inp_np = all_inputs[i]
    tgt_np = all_targets[i]

    sf_opt2.zero_grad()
    sf_out = sf_model2(np_to_sf(inp_np))
    sf_l = sf_F.cross_entropy(sf_out.reshape(-1, VOCAB), np_to_sf(tgt_np).reshape(-1))
    sf_l.backward()
    sf_nn.utils.clip_grad_norm_(sf_model2.parameters(), 1.0)
    sf_opt2.step()
    sf_sched2.step()

    pt_opt2.zero_grad()
    pt_out = pt_model2(np_to_pt(inp_np))
    pt_l = pt_F.cross_entropy(pt_out.reshape(-1, VOCAB), np_to_pt(tgt_np).reshape(-1))
    pt_l.backward()
    pt_nn.utils.clip_grad_norm_(pt_model2.parameters(), 1.0)
    pt_opt2.step()
    pt_sched2.step()

# ── Scaffolding training loop (timed) ───────────────────────────────
print(f"  Training scaffolding for {STEPS} steps ...")
sf_losses = []
gc.collect()
t0_sf = time.perf_counter()
for step in range(STEPS):
    inp_np = all_inputs[WARMUP + step]
    tgt_np = all_targets[WARMUP + step]

    sf_opt2.zero_grad()
    logits = sf_model2(np_to_sf(inp_np))
    loss = sf_F.cross_entropy(logits.reshape(-1, VOCAB), np_to_sf(tgt_np).reshape(-1))
    loss.backward()
    sf_nn.utils.clip_grad_norm_(sf_model2.parameters(), 1.0)
    sf_opt2.step()
    sf_sched2.step()

    if step % max(1, STEPS // 10) == 0 or step == STEPS - 1:
        l_val = loss.item()
        sf_losses.append(l_val)
        elapsed = time.perf_counter() - t0_sf
        tps = (step + 1) * B * S / elapsed
        print(f"    step {step:4d}/{STEPS}  loss={l_val:.4f}  "
              f"elapsed={elapsed:.1f}s  tok/s={tps:,.0f}")

t_sf_loop = time.perf_counter() - t0_sf

# ── PyTorch training loop (timed) ──────────────────────────────────
print(f"  Training PyTorch for {STEPS} steps ...")
pt_losses = []
gc.collect()
t0_pt = time.perf_counter()
for step in range(STEPS):
    inp_np = all_inputs[WARMUP + step]
    tgt_np = all_targets[WARMUP + step]

    pt_opt2.zero_grad()
    logits = pt_model2(np_to_pt(inp_np))
    loss = pt_F.cross_entropy(logits.reshape(-1, VOCAB), np_to_pt(tgt_np).reshape(-1))
    loss.backward()
    pt_nn.utils.clip_grad_norm_(pt_model2.parameters(), 1.0)
    pt_opt2.step()
    pt_sched2.step()

    if step % max(1, STEPS // 10) == 0 or step == STEPS - 1:
        l_val = loss.item()
        pt_losses.append(l_val)
        elapsed = time.perf_counter() - t0_pt
        tps = (step + 1) * B * S / elapsed
        print(f"    step {step:4d}/{STEPS}  loss={l_val:.4f}  "
              f"elapsed={elapsed:.1f}s  tok/s={tps:,.0f}")

t_pt_loop = time.perf_counter() - t0_pt

record(f"Training loop ({STEPS} steps)", t_sf_loop, t_pt_loop)

sf_tps = STEPS * B * S / t_sf_loop
pt_tps = STEPS * B * S / t_pt_loop
print(f"\n  Throughput: scaffolding {sf_tps:,.0f} tok/s  |  PyTorch {pt_tps:,.0f} tok/s  "
      f"|  ratio {sf_tps/pt_tps:.2f}x")


# ═════════════════════════════════════════════════════════════════════
#  PHASE 6: Inference benchmark (no grad, larger batches)
# ═════════════════════════════════════════════════════════════════════

print(f"\n── Phase 6: Inference (no_grad, eval mode) ──")

sf_model2.eval()
pt_model2.eval()

for label, b, s in [("Small  (B=1, S=128)", 1, 128),
                     ("Medium (B=8, S=256)", 8, 256),
                     ("Large  (B=16, S=512)", 16, 512),
                     ("XLarge (B=32, S=512)", 32, 512)]:
    inp_np = np.random.randint(0, VOCAB, (b, s)).astype(np.int64)
    sf_inp = np_to_sf(inp_np)
    pt_inp = np_to_pt(inp_np)

    def sf_infer():
        with sf.no_grad():
            sf_model2(sf_inp)

    def pt_infer():
        with torch.no_grad():
            pt_model2(pt_inp)

    t_sf = time_fn(sf_infer)
    t_pt = time_fn(pt_infer)
    record(f"Inference {label}", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 7: State dict save / load
# ═════════════════════════════════════════════════════════════════════

print(f"\n── Phase 7: State Dict Operations ──")

def sf_save_load():
    sd = sf_model2.state_dict()
    sf_model2.load_state_dict(sd)

def pt_save_load():
    sd = pt_model2.state_dict()
    pt_model2.load_state_dict(sd)

t_sf = time_fn(sf_save_load)
t_pt = time_fn(pt_save_load)
record("state_dict() + load_state_dict()", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 8: Gradient accumulation (multiple micro-batches)
# ═════════════════════════════════════════════════════════════════════

print(f"\n── Phase 8: Gradient Accumulation (4 micro-batches) ──")

ACCUM_STEPS = 4
micro_inputs = [np.random.randint(0, VOCAB, (B, S)).astype(np.int64) for _ in range(ACCUM_STEPS)]
micro_targets = [np.random.randint(0, VOCAB, (B, S)).astype(np.int64) for _ in range(ACCUM_STEPS)]

sf_model.train()
pt_model.train()

def sf_grad_accum():
    sf_opt.zero_grad()
    for mi, mt in zip(micro_inputs, micro_targets):
        logits = sf_model(np_to_sf(mi))
        loss = sf_F.cross_entropy(logits.reshape(-1, VOCAB), np_to_sf(mt).reshape(-1))
        (loss / ACCUM_STEPS).backward()
    sf_nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
    sf_opt.step()

def pt_grad_accum():
    pt_opt.zero_grad()
    for mi, mt in zip(micro_inputs, micro_targets):
        logits = pt_model(np_to_pt(mi))
        loss = pt_F.cross_entropy(logits.reshape(-1, VOCAB), np_to_pt(mt).reshape(-1))
        (loss / ACCUM_STEPS).backward()
    pt_nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)
    pt_opt.step()

t_sf = time_fn(sf_grad_accum)
t_pt = time_fn(pt_grad_accum)
record(f"Grad accumulation ({ACCUM_STEPS}x micro-batch)", t_sf, t_pt)


# ═════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SUMMARY")
print("=" * 100)

# Categorize
faster = [(l, s, p, r) for l, s, p, r in results if r < 0.95]
ok     = [(l, s, p, r) for l, s, p, r in results if 0.95 <= r < 1.5]
slow   = [(l, s, p, r) for l, s, p, r in results if 1.5 <= r < 3.0]
v_slow = [(l, s, p, r) for l, s, p, r in results if r >= 3.0]

if v_slow:
    print(f"\n  VERY SLOW (>=3x): {len(v_slow)} tests")
    for l, s, p, r in sorted(v_slow, key=lambda x: -x[3]):
        print(f"    {l:<52s}  {r:5.2f}x  (sf={s*1e3:.1f}ms, pt={p*1e3:.1f}ms)")

if slow:
    print(f"\n  SLOW (1.5-3x): {len(slow)} tests")
    for l, s, p, r in sorted(slow, key=lambda x: -x[3]):
        print(f"    {l:<52s}  {r:5.2f}x  (sf={s*1e3:.1f}ms, pt={p*1e3:.1f}ms)")

if ok:
    print(f"\n  OK (0.95-1.5x): {len(ok)} tests")
    for l, s, p, r in sorted(ok, key=lambda x: -x[3]):
        print(f"    {l:<52s}  {r:5.2f}x  (sf={s*1e3:.1f}ms, pt={p*1e3:.1f}ms)")

if faster:
    print(f"\n  FASTER (<0.95x — scaffolding wins): {len(faster)} tests")
    for l, s, p, r in sorted(faster, key=lambda x: x[3]):
        print(f"    {l:<52s}  {r:5.2f}x  (sf={s*1e3:.1f}ms, pt={p*1e3:.1f}ms)")

# Totals
total_sf = sum(s for _, s, _, _ in results)
total_pt = sum(p for _, _, p, _ in results)
overall = total_sf / total_pt if total_pt > 0 else float('inf')
print(f"\n  Total scaffolding time: {total_sf*1e3:.1f}ms")
print(f"  Total PyTorch time:    {total_pt*1e3:.1f}ms")
print(f"  Overall ratio:         {overall:.2f}x")

# Training loop specific
loop_results = [(l, s, p, r) for l, s, p, r in results if "Training loop" in l]
if loop_results:
    l, s, p, r = loop_results[0]
    print(f"\n  ★ Training loop ({STEPS} steps): scaffolding {s:.1f}s  vs  PyTorch {p:.1f}s  →  {r:.2f}x")
    if r < 1.0:
        print(f"    → scaffolding is {(1-r)*100:.1f}% FASTER for end-to-end training!")
    else:
        print(f"    → scaffolding is {(r-1)*100:.1f}% slower for end-to-end training")

print("\n" + "=" * 100)
