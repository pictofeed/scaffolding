#!/usr/bin/env python3
"""Test: Generate a video using a HuggingFace text-to-video diffusion model.

Downloads and runs the ModelScope text-to-video pipeline
(``damo-vilab/text-to-video-ms-1.7b``) via the HuggingFace ``diffusers``
library, then saves frames and an MP4 to ``tests/output_video/``.

Requirements (already satisfied in the current venv):
    pip install torch diffusers transformers accelerate imageio imageio-ffmpeg

Usage:
    PYTHONPATH=. python tests/test_hf_video.py
    PYTHONPATH=. python tests/test_hf_video.py --prompt "a dog running on the beach"
    PYTHONPATH=. python tests/test_hf_video.py --steps 25 --frames 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HuggingFace text-to-video test")
parser.add_argument("--prompt", type=str, default="a cat walking on grass, photorealistic",
                    help="Text prompt for video generation")
parser.add_argument("--model", type=str, default="damo-vilab/text-to-video-ms-1.7b",
                    help="HuggingFace model ID")
parser.add_argument("--steps", type=int, default=25,
                    help="Number of inference steps (fewer = faster)")
parser.add_argument("--frames", type=int, default=16,
                    help="Number of video frames to generate")
parser.add_argument("--height", type=int, default=256,
                    help="Frame height in pixels")
parser.add_argument("--width", type=int, default=256,
                    help="Frame width in pixels")
parser.add_argument("--fps", type=int, default=8,
                    help="Frames per second for the output MP4")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Directory to save output (default: tests/output_video)")
parser.add_argument("--device", type=str, default=None,
                    help="Force device (cpu, mps, cuda). Auto-detected if omitted.")
args = parser.parse_args()


def detect_device() -> str:
    """Pick the best available device."""
    import torch
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    import imageio

    device = detect_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output_video"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  HuggingFace Text-to-Video Test")
    print(f"{'=' * 60}")
    print(f"  Model   : {args.model}")
    print(f"  Prompt  : {args.prompt}")
    print(f"  Device  : {device}  (dtype={dtype})")
    print(f"  Frames  : {args.frames}")
    print(f"  Size    : {args.height}×{args.width}")
    print(f"  Steps   : {args.steps}")
    print(f"  Seed    : {args.seed}")
    print(f"  Output  : {output_dir}")
    print(f"{'=' * 60}")

    # ── Load pipeline ────────────────────────────────────────────────
    print("\n[1/4] Loading pipeline …")
    t0 = time.time()

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    # Use DPM-Solver for faster sampling
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # Move to device — MPS may not support all ops, fall back to CPU
    try:
        pipe = pipe.to(device)
    except Exception as e:
        print(f"  ⚠ Could not move to {device}: {e}")
        print(f"  Falling back to CPU …")
        device = "cpu"
        pipe = pipe.to(device)

    # Enable memory-efficient attention if available
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    load_time = time.time() - t0
    print(f"  Pipeline loaded in {load_time:.1f}s")

    # ── Generate video ───────────────────────────────────────────────
    print(f"\n[2/4] Generating video ({args.steps} steps) …")
    t1 = time.time()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
    )

    gen_time = time.time() - t1
    print(f"  Generation completed in {gen_time:.1f}s")

    # ── Extract frames ───────────────────────────────────────────────
    # diffusers returns .frames as a list of lists of PIL images
    frames = result.frames
    if isinstance(frames, list) and len(frames) > 0:
        if isinstance(frames[0], list):
            frames = frames[0]  # unwrap batch dimension

    print(f"\n[3/4] Saving {len(frames)} frames …")

    # Save individual frames as PNG
    frame_arrays = []
    for i, frame in enumerate(frames):
        import numpy as np
        if hasattr(frame, "save"):
            # PIL Image
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            frame_arrays.append(np.array(frame))
        else:
            # numpy array
            frame_arrays.append(np.asarray(frame))

    print(f"  Saved {len(frames)} frames to {output_dir}/")

    # ── Save MP4 ─────────────────────────────────────────────────────
    print(f"\n[4/4] Encoding MP4 …")
    mp4_path = os.path.join(output_dir, "output.mp4")
    try:
        writer = imageio.get_writer(mp4_path, fps=args.fps, codec="libx264",
                                    quality=8)
        for fa in frame_arrays:
            writer.append_data(fa)
        writer.close()
        mp4_size = os.path.getsize(mp4_path) / 1024
        print(f"  Saved MP4: {mp4_path} ({mp4_size:.0f} KB)")
    except Exception as e:
        print(f"  ⚠ MP4 encoding failed: {e}")
        print(f"  Individual frames are still available in {output_dir}/")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  Load time      : {load_time:.1f}s")
    print(f"  Generation time : {gen_time:.1f}s")
    print(f"  Frames          : {len(frames)}")
    print(f"  Output          : {output_dir}")
    print(f"{'=' * 60}")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
