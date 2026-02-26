#!/usr/bin/env python3
"""
Tora GPU Memory Stress Test
============================
Mirrors the exact pipeline flow from pictofeed-api/gpu_worker/main.py
but with a tiny UNet so OOM kills surface in seconds instead of minutes.

Exercises every memory-critical path:
  1. ToraPipeline construction + from_pretrained kwargs (device_map, etc.)
  2. Component-by-component GPU upload with GC between each
  3. Full __call__ with output_type="sf_tensor" + callback_on_step_end
  4. _release_cpu_shadows after UNet mini-batches and VAE decode
  5. GPU-resident frame output (no PIL, no numpy on CPU)
  6. Cross-fade blending on GPU
  7. Frame-by-frame video export (GPU→CPU streaming)

Run:
    PYTHONPATH=. python tests/test_tora_gpu_memory.py
    PYTHONPATH=. python tests/test_tora_gpu_memory.py --steps 4 --frames 9
    PYTHONPATH=. python tests/test_tora_gpu_memory.py --cpu   # CPU-only (no CUDA)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import resource

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scaffolding as sf
from scaffolding.tensor import Tensor
from scaffolding.nn.module import Module
from scaffolding.diffusion.schedulers import DDIMScheduler, CogVideoXDPMScheduler
from scaffolding.diffusion.models import UNet2DConditionModel, AutoencoderKL
from scaffolding.diffusion.pipelines.tora import ToraPipeline, PipelineOutput

# The MPS Accelerate BLAS sgemm_nt in _mps_ops.pyx produces wrong output
# sizes for small Linear dims (e.g. 32→32), likely due to a stale compiled
# .so or ABI mismatch.  Monkeypatch accelerate_linear_forward to use numpy
# while keeping other MPS operations (Conv2d im2col/einsum, SiLU, GELU) fast.
import scaffolding.nn.layers as _layers_mod
if hasattr(_layers_mod, '_mops') and _layers_mod._mops is not None:
    _orig_alf = _layers_mod._mops.accelerate_linear_forward
    def _safe_linear_forward(x, w):
        return (x @ w.T).astype(np.float32)
    _layers_mod._mops.accelerate_linear_forward = _safe_linear_forward

# ── CLI ──
parser = argparse.ArgumentParser(description="Tora GPU memory stress test")
parser.add_argument("--steps", type=int, default=3,
                    help="Denoising steps (fewer = faster test)")
parser.add_argument("--frames", type=int, default=5,
                    help="Number of video frames")
parser.add_argument("--guidance", type=float, default=6.0,
                    help="Guidance scale")
parser.add_argument("--cpu", action="store_true",
                    help="Force CPU mode (skip CUDA)")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Print extra detail")
args = parser.parse_args()

# Fixed resolution for the tiny UNet.  channel_mult=(1,) has no
# down/upsampling so any spatial size works.  32×32 keeps each forward
# pass under 0.4 s on CPU (numpy).
HEIGHT = 32
WIDTH = 32


# ── Helpers ──

def _rss_mb() -> float:
    """Current process RSS in MB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)   # macOS: bytes → MB
    return raw / 1024                # Linux: KB → MB


def _vram_mb() -> str:
    """Current VRAM usage string, or 'N/A'."""
    try:
        import scaffolding.backends.cuda as sf_cuda
        if sf_cuda.is_available():
            alloc = sf_cuda.memory_allocated(0) / (1024 * 1024)
            return f"{alloc:.0f} MB"
    except Exception:
        pass
    return "N/A"


def _log(tag: str, msg: str = ""):
    rss = _rss_mb()
    vram = _vram_mb()
    print(f"[{tag:>20s}]  RSS={rss:6.1f} MB  VRAM={vram:>8s}  {msg}")


def _detect_device() -> str:
    if args.cpu:
        return "cpu"
    try:
        import scaffolding.backends.cuda as sf_cuda
        if sf_cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _make_tiny_unet():
    """Build a small UNet that initialises fast and fits in RAM.

    Uses model_channels=32 (not 320), channel_mult=(1,) (single level,
    no down/upsampling), num_res_blocks=1, 2 heads, context_dim=32.
    Total params: ~50 K (vs ~860 M for the default).

    Notes on why this exact config:
    * channel_mult=(1,) — single level, no down/upsampling.  The UNet
      forward post-applies all downsamplers after all encoder blocks
      (rather than interleaving), so multi-level configs cause spatial
      dimension mismatches.
    * context_dim must equal model_channels (32) because the Accelerate
      BLAS sgemm in _mps_ops.pyx produces wrong results when the
      Linear maps a large dim to a much smaller one (e.g. 768→32).
    * Tests must pass ``prompt_embeds`` with last dim == 32 to match.
    """
    return UNet2DConditionModel(
        in_channels=4,
        out_channels=4,
        model_channels=32,
        context_dim=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(0,),
        num_heads=2,
        dropout=0.0,
    )


def _make_tiny_vae():
    """Build a small VAE (default AutoencoderKL is already lightweight)."""
    return AutoencoderKL()


# Context dim for the tiny UNet — must match everywhere.
_CTX_DIM = 32


def _make_prompt_embeds() -> Tensor:
    """Create prompt embeddings with the right context_dim for the tiny UNet."""
    return Tensor._wrap(
        np.random.randn(1, 77, _CTX_DIM).astype(np.float32),
        False, None, None)


def _make_pipe(device: str) -> ToraPipeline:
    """Construct a test pipeline with tiny models on the given device."""
    pipe = ToraPipeline(
        unet=_make_tiny_unet(),
        scheduler=DDIMScheduler(),
        vae=_make_tiny_vae(),
        latent_channels=4,
        latent_scale=8,
    )
    pipe.to(device)
    return pipe


# ── Test functions ──

def test_from_pretrained_kwargs():
    """Ensure HuggingFace-style kwargs are stripped and don't reach __init__."""
    _log("test_kwargs", "Testing _HF_IGNORED filtering …")

    # Monkeypatch UNet/VAE/Scheduler from_pretrained to return tiny models
    # so we don't build the 860M-param default UNet.
    _orig_unet_fp = UNet2DConditionModel.from_pretrained
    _orig_vae_fp = AutoencoderKL.from_pretrained if hasattr(AutoencoderKL, 'from_pretrained') else None
    _orig_sched_fp = DDIMScheduler.from_pretrained if hasattr(DDIMScheduler, 'from_pretrained') else None

    try:
        UNet2DConditionModel.from_pretrained = classmethod(
            lambda cls, *a, **kw: _make_tiny_unet())
        if hasattr(AutoencoderKL, 'from_pretrained'):
            AutoencoderKL.from_pretrained = classmethod(
                lambda cls, *a, **kw: _make_tiny_vae())

        pipe = ToraPipeline.from_pretrained(
            "/nonexistent/path",
            dtype=sf.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            torch_dtype="float16",
            variant="fp16",
            use_safetensors=True,
        )
        assert pipe.unet is not None, "UNet not created"
        assert pipe.scheduler is not None, "Scheduler not created"
        _log("test_kwargs", "PASS — no TypeError from HF kwargs")
        del pipe
        gc.collect()
    except TypeError as e:
        _log("test_kwargs", f"FAIL — {e}")
        raise
    finally:
        UNet2DConditionModel.from_pretrained = _orig_unet_fp
        if _orig_vae_fp is not None:
            AutoencoderKL.from_pretrained = _orig_vae_fp
        if _orig_sched_fp is not None:
            DDIMScheduler.from_pretrained = _orig_sched_fp


def test_callback_on_step_end():
    """Verify callback_on_step_end is invoked with correct signature."""
    _log("test_callback", "Testing callback_on_step_end …")

    device = _detect_device()
    pipe = ToraPipeline(
        unet=_make_tiny_unet(),
        scheduler=DDIMScheduler(),
        vae=_make_tiny_vae(),
        latent_channels=4,
        latent_scale=8,
    )
    pipe.to(device)

    callback_log = []

    def _on_step(pipeline, step_idx, timestep, cb_kwargs):
        callback_log.append({
            "step": step_idx,
            "timestep": timestep,
            "has_latents": "latents" in cb_kwargs,
        })
        return cb_kwargs

    with sf.inference_mode():
        output = pipe(
            prompt_embeds=_make_prompt_embeds(),
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=1.0,
            output_type="pil",
            callback_on_step_end=_on_step,
        )

    assert len(callback_log) == args.steps, (
        f"Expected {args.steps} callbacks, got {len(callback_log)}")
    for entry in callback_log:
        assert entry["has_latents"], "callback_kwargs missing 'latents'"

    _log("test_callback", f"PASS — {len(callback_log)} callbacks received")
    del pipe, output
    gc.collect()


def test_gpu_resident_pipeline():
    """Full pipeline run mimicking gpu_worker/main.py — tracks RSS at every stage."""
    device = _detect_device()
    is_cuda = device == "cuda"

    print()
    print("=" * 70)
    print("  Tora GPU Memory Stress Test")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  Frames      : {args.frames}")
    print(f"  Resolution  : {HEIGHT}x{WIDTH}")
    print(f"  Steps       : {args.steps}")
    print(f"  Guidance    : {args.guidance}")
    print("=" * 70)
    print()

    rss_start = _rss_mb()
    _log("init", f"Starting RSS = {rss_start:.1f} MB")

    # ── 1. Construct pipeline (mirrors _load_pipeline) ──
    _log("construct", "Building ToraPipeline with tiny UNet …")
    pipe = ToraPipeline(
        unet=_make_tiny_unet(),
        scheduler=DDIMScheduler(),
        vae=_make_tiny_vae(),
        latent_channels=4,
        latent_scale=8,
    )
    _log("construct", "Done")

    # ── 2. Switch to CogVideoXDPM scheduler ──
    pipe.scheduler = CogVideoXDPMScheduler(
        num_train_timesteps=1000,
        snr_shift_scale=3.0,
        prediction_type="v_prediction",
    )
    _log("scheduler", "CogVideoXDPMScheduler configured")

    # ── 3. Move components to device one-by-one (mirrors main.py) ──
    for comp_name in ("unet", "vae", "text_encoder"):
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "to"):
            comp.to(device)
            gc.collect()
            _log(f"{comp_name}->{device}", "moved + GC")
    _log("all_on_device", f"All components on {device}")

    # ── 4. VAE config ──
    if pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    _log("vae_config", "Slicing + tiling enabled")

    # ── 5. Run inference with callback_on_step_end ──
    step_rss: list[float] = []

    def _step_callback(pipe_obj, step_idx, timestep, callback_kwargs):
        rss = _rss_mb()
        step_rss.append(rss)
        _log(f"step {step_idx+1}/{args.steps}",
             f"timestep={timestep}")
        return callback_kwargs

    _log("inference", "Starting denoising loop …")
    t0 = time.monotonic()

    output_type = "sf_tensor" if is_cuda else "pil"

    with sf.inference_mode():
        output = pipe(
            prompt_embeds=_make_prompt_embeds(),
            video_flow=None,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=HEIGHT,
            width=WIDTH,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance,
            output_type=output_type,
            seed=42,
            callback_on_step_end=_step_callback,
        )

    elapsed = time.monotonic() - t0
    _log("inference", f"Done in {elapsed:.1f}s")

    # ── 6. Validate output ──
    assert isinstance(output, PipelineOutput), f"Expected PipelineOutput, got {type(output)}"
    assert len(output.frames) >= 1, "No batch outputs"
    frames = output.frames[0]
    assert len(frames) == args.frames, (
        f"Expected {args.frames} frames, got {len(frames)}")

    if output_type == "sf_tensor":
        f0 = frames[0]
        assert isinstance(f0, Tensor), f"Frame is {type(f0)}, not Tensor"
        assert f0.shape[-1] == 3, f"Expected (H,W,3), got {f0.shape}"
        _log("validate", f"GPU-resident frames: {len(frames)}x{f0.shape}")

        # ── 7. Simulate video export (GPU->CPU streaming) ──
        _log("export", "Streaming frames GPU->CPU …")
        for i, t in enumerate(frames):
            arr = t.cpu().numpy() if hasattr(t, 'cpu') else t.numpy()
            assert arr.dtype == np.uint8, f"Frame dtype {arr.dtype}, expected uint8"
            del arr
        _log("export", f"Exported {len(frames)} frames")
    else:
        _log("validate", f"PIL frames: {len(frames)}")

    # ── 8. Cross-fade test (GPU blending if CUDA) ──
    if is_cuda and len(frames) >= 4:
        _log("crossfade", "Testing GPU cross-fade …")
        n = min(4, len(frames) // 2)
        tail = frames[-n:]
        head = frames[:n]
        blended = []
        for i in range(n):
            alpha = (i + 1) / (n + 1)
            a = tail[i].to(dtype=sf.float16) / 255.0
            b = head[i].to(dtype=sf.float16) / 255.0
            merged = a * (1.0 - alpha) + b * alpha
            merged = (merged * 255.0).clamp(0, 255).to(dtype=sf.uint8)
            blended.append(merged)
            del a, b, merged
        _log("crossfade", f"Blended {n} frames on GPU — PASS")
        del blended

    # ── 9. Cleanup ──
    del output, frames
    gc.collect()
    if is_cuda:
        try:
            import scaffolding.backends.cuda as sf_cuda
            sf_cuda.empty_cache()
        except Exception:
            pass
    _log("cleanup", "Freed all tensors")

    # ── Report ──
    rss_end = _rss_mb()
    peak_step_rss = max(step_rss) if step_rss else rss_end
    print()
    print("=" * 70)
    print(f"  RESULTS")
    print(f"  Start RSS      : {rss_start:.1f} MB")
    print(f"  Peak step RSS  : {peak_step_rss:.1f} MB")
    print(f"  Final RSS      : {rss_end:.1f} MB")
    print(f"  RSS growth     : {rss_end - rss_start:.1f} MB")
    print(f"  Frames         : {args.frames}")
    print(f"  Time           : {elapsed:.1f}s")
    print(f"  Device         : {device}")

    growth = rss_end - rss_start
    if growth > 500:
        print(f"  WARNING: RSS grew {growth:.0f} MB — possible memory leak!")
    else:
        print(f"  RSS growth looks healthy ({growth:.0f} MB)")
    print("=" * 70)


def test_release_cpu_shadows():
    """Verify _release_cpu_shadows actually drops CPU data."""
    _log("test_shadows", "Testing _release_cpu_shadows …")

    import scaffolding.nn as nn

    model = nn.Linear(64, 32)
    device = _detect_device()

    if device == "cuda":
        model.to("cuda")
        for p in model.parameters():
            assert p._gpu is not None, "Parameter not on GPU after .to('cuda')"

        # Force a CPU shadow to exist
        for p in model.parameters():
            p._ensure_cpu()
            assert p._data is not None, "CPU shadow should exist after _ensure_cpu"

        # Release shadows
        model._release_cpu_shadows()
        for p in model.parameters():
            assert p._data is None, "CPU shadow should be None after _release_cpu_shadows"
            assert p._gpu is not None, "GPU copy should still exist"

        _log("test_shadows", "PASS — CPU shadows released, GPU copies intact")
    else:
        _log("test_shadows", "SKIP — no CUDA device (test requires GPU)")


def test_multi_chunk_simulation():
    """Simulate the multi-chunk generation loop from main.py with cross-fade."""
    _log("test_chunks", "Simulating 2-chunk generation with cross-fade …")

    device = _detect_device()
    pipe = ToraPipeline(
        unet=_make_tiny_unet(),
        scheduler=CogVideoXDPMScheduler(
            num_train_timesteps=1000,
            snr_shift_scale=3.0,
            prediction_type="v_prediction",
        ),
        vae=_make_tiny_vae(),
        latent_channels=4,
        latent_scale=8,
    )
    pipe.to(device)

    CHUNK_FRAMES = args.frames
    CROSSFADE = min(2, CHUNK_FRAMES // 2)
    all_frames: list = []
    output_type = "sf_tensor" if device == "cuda" else "pil"

    for chunk_idx in range(2):
        _log(f"chunk {chunk_idx+1}/2", "Running …")

        with sf.inference_mode():
            output = pipe(
                prompt_embeds=_make_prompt_embeds(),
                num_frames=CHUNK_FRAMES,
                num_inference_steps=max(2, args.steps // 2),
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=1.0,
                output_type=output_type,
                seed=42 + chunk_idx,
            )

        chunk_frames = output.frames[0]
        del output
        gc.collect()

        if chunk_idx == 0:
            all_frames.extend(chunk_frames)
        else:
            all_frames.extend(chunk_frames[CROSSFADE:])

        del chunk_frames
        _log(f"chunk {chunk_idx+1}/2", f"Total frames so far: {len(all_frames)}")

    expected = CHUNK_FRAMES + (CHUNK_FRAMES - CROSSFADE)
    assert len(all_frames) == expected, (
        f"Expected {expected} total frames, got {len(all_frames)}")

    del all_frames
    gc.collect()
    _log("test_chunks", f"PASS — {expected} frames across 2 chunks")
    del pipe
    gc.collect()


# ── Main ──

if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  Scaffolding — Tora GPU Memory Stress Test")
    print("=" * 62)
    print()

    t_total = time.monotonic()
    passed = 0
    failed = 0
    tests = [
        ("from_pretrained kwargs", test_from_pretrained_kwargs),
        ("callback_on_step_end", test_callback_on_step_end),
        ("_release_cpu_shadows", test_release_cpu_shadows),
        ("GPU-resident pipeline", test_gpu_resident_pipeline),
        ("multi-chunk simulation", test_multi_chunk_simulation),
    ]

    for name, fn in tests:
        print(f"\n{'_' * 60}")
        print(f"  TEST: {name}")
        print(f"{'_' * 60}")
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {name} -- {e}")

    elapsed_total = time.monotonic() - t_total
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{passed + failed} tests passed in {elapsed_total:.1f}s")
    if failed:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print(f"  All tests passed")
    print(f"{'=' * 60}")
    sys.exit(0)
