"""Tests for Conv2d/Conv3d layers, supporting layers, and text-to-video pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_imports():
    """Test all new imports."""
    print("=== Test Imports ===")
    import scaffolding as torch
    import scaffolding.nn as nn
    from scaffolding.nn import (
        Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
        BatchNorm2d, BatchNorm3d, GroupNorm,
        AvgPool2d, MaxPool2d, AdaptiveAvgPool2d,
        Upsample, PixelShuffle, Tanh, Sigmoid,
    )
    import scaffolding.nn.functional as F
    print("  All imports OK")
    return True


def test_conv2d_forward():
    """Test Conv2d forward pass shapes."""
    print("=== Test Conv2d Forward ===")
    import scaffolding as torch
    from scaffolding.nn import Conv2d

    # Basic Conv2d
    conv = Conv2d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    out = conv(x)
    assert out.shape == (2, 16, 8, 8), f"Expected (2,16,8,8), got {out.shape}"
    print(f"  Basic Conv2d: {out.shape} OK")

    # Conv2d with stride
    conv_s2 = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    out_s2 = conv_s2(x)
    assert out_s2.shape == (2, 16, 4, 4), f"Expected (2,16,4,4), got {out_s2.shape}"
    print(f"  Conv2d stride=2: {out_s2.shape} OK")

    # Conv2d no padding
    conv_np = Conv2d(3, 8, kernel_size=3)
    out_np = conv_np(x)
    assert out_np.shape == (2, 8, 6, 6), f"Expected (2,8,6,6), got {out_np.shape}"
    print(f"  Conv2d no padding: {out_np.shape} OK")

    # Conv2d with groups
    conv_g = Conv2d(16, 32, kernel_size=3, padding=1, groups=4)
    x_g = torch.randn(2, 16, 8, 8)
    out_g = conv_g(x_g)
    assert out_g.shape == (2, 32, 8, 8), f"Expected (2,32,8,8), got {out_g.shape}"
    print(f"  Conv2d groups=4: {out_g.shape} OK")

    return True


def test_conv3d_forward():
    """Test Conv3d forward pass shapes."""
    print("=== Test Conv3d Forward ===")
    import scaffolding as torch
    from scaffolding.nn import Conv3d

    # Basic Conv3d
    conv = Conv3d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 4, 8, 8)
    out = conv(x)
    assert out.shape == (1, 8, 4, 8, 8), f"Expected (1,8,4,8,8), got {out.shape}"
    print(f"  Basic Conv3d: {out.shape} OK")

    # Conv3d with stride
    conv_s = Conv3d(3, 8, kernel_size=3, stride=(1, 2, 2), padding=1)
    out_s = conv_s(x)
    assert out_s.shape == (1, 8, 4, 4, 4), f"Expected (1,8,4,4,4), got {out_s.shape}"
    print(f"  Conv3d stride=(1,2,2): {out_s.shape} OK")

    return True


def test_conv2d_backward():
    """Test Conv2d backward pass."""
    print("=== Test Conv2d Backward ===")
    import scaffolding as torch
    from scaffolding.nn import Conv2d
    from scaffolding.autograd import backward

    conv = Conv2d(3, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 4, 4)
    x._requires_grad = True

    out = conv(x)
    loss = out.sum()
    backward(loss)

    assert conv.weight.grad is not None, "weight gradient is None"
    assert conv.weight.grad.shape == conv.weight.shape, \
        f"weight grad shape {conv.weight.grad.shape} != {conv.weight.shape}"
    assert conv.bias.grad is not None, "bias gradient is None"
    assert x.grad is not None, "input gradient is None"
    assert x.grad.shape == x.shape, f"input grad shape {x.grad.shape} != {x.shape}"
    print(f"  weight grad shape: {conv.weight.grad.shape} OK")
    print(f"  bias grad shape: {conv.bias.grad.shape} OK")
    print(f"  input grad shape: {x.grad.shape} OK")

    return True


def test_conv3d_backward():
    """Test Conv3d backward pass."""
    print("=== Test Conv3d Backward ===")
    import scaffolding as torch
    from scaffolding.nn import Conv3d
    from scaffolding.autograd import backward

    conv = Conv3d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 2, 3, 4, 4)
    x._requires_grad = True

    out = conv(x)
    loss = out.sum()
    backward(loss)

    assert conv.weight.grad is not None, "weight gradient is None"
    assert conv.weight.grad.shape == conv.weight.shape, \
        f"weight grad shape {conv.weight.grad.shape} != {conv.weight.shape}"
    assert conv.bias.grad is not None, "bias gradient is None"
    assert x.grad is not None, "input gradient is None"
    assert x.grad.shape == x.shape, f"input grad shape {x.grad.shape} != {x.shape}"
    print(f"  weight grad shape: {conv.weight.grad.shape} OK")
    print(f"  bias grad shape: {conv.bias.grad.shape} OK")
    print(f"  input grad shape: {x.grad.shape} OK")

    return True


def test_supporting_layers():
    """Test BatchNorm, GroupNorm, pooling, etc."""
    print("=== Test Supporting Layers ===")
    import scaffolding as torch
    from scaffolding.nn import (
        BatchNorm2d, BatchNorm3d, GroupNorm,
        AvgPool2d, MaxPool2d, AdaptiveAvgPool2d,
        Upsample, PixelShuffle, Tanh, Sigmoid,
        ConvTranspose2d, ConvTranspose3d,
    )

    x4d = torch.randn(2, 8, 4, 4)
    x5d = torch.randn(1, 8, 3, 4, 4)

    # BatchNorm2d
    bn2 = BatchNorm2d(8)
    out = bn2(x4d)
    assert out.shape == x4d.shape, f"BN2d shape {out.shape}"
    print(f"  BatchNorm2d: {out.shape} OK")

    # BatchNorm3d
    bn3 = BatchNorm3d(8)
    out3 = bn3(x5d)
    assert out3.shape == x5d.shape, f"BN3d shape {out3.shape}"
    print(f"  BatchNorm3d: {out3.shape} OK")

    # GroupNorm
    gn = GroupNorm(4, 8)
    out_gn = gn(x4d)
    assert out_gn.shape == x4d.shape, f"GN shape {out_gn.shape}"
    print(f"  GroupNorm: {out_gn.shape} OK")

    # AvgPool2d
    ap = AvgPool2d(kernel_size=2, stride=2)
    out_ap = ap(x4d)
    assert out_ap.shape == (2, 8, 2, 2), f"AvgPool2d shape {out_ap.shape}"
    print(f"  AvgPool2d: {out_ap.shape} OK")

    # MaxPool2d
    mp = MaxPool2d(kernel_size=2, stride=2)
    out_mp = mp(x4d)
    assert out_mp.shape == (2, 8, 2, 2), f"MaxPool2d shape {out_mp.shape}"
    print(f"  MaxPool2d: {out_mp.shape} OK")

    # AdaptiveAvgPool2d
    aap = AdaptiveAvgPool2d((1, 1))
    out_aap = aap(x4d)
    assert out_aap.shape == (2, 8, 1, 1), f"AdaptiveAvgPool2d shape {out_aap.shape}"
    print(f"  AdaptiveAvgPool2d: {out_aap.shape} OK")

    # Upsample
    up = Upsample(scale_factor=2, mode='nearest')
    out_up = up(x4d)
    assert out_up.shape == (2, 8, 8, 8), f"Upsample shape {out_up.shape}"
    print(f"  Upsample: {out_up.shape} OK")

    # PixelShuffle
    x_ps = torch.randn(1, 4, 4, 4)  # 4 = 1 * 2^2
    ps = PixelShuffle(2)
    out_ps = ps(x_ps)
    assert out_ps.shape == (1, 1, 8, 8), f"PixelShuffle shape {out_ps.shape}"
    print(f"  PixelShuffle: {out_ps.shape} OK")

    # Tanh/Sigmoid
    t = Tanh()
    out_t = t(x4d)
    assert out_t.shape == x4d.shape
    print(f"  Tanh: OK")

    s = Sigmoid()
    out_s = s(x4d)
    assert out_s.shape == x4d.shape
    print(f"  Sigmoid: OK")

    # ConvTranspose2d
    ct2 = ConvTranspose2d(8, 4, kernel_size=2, stride=2)
    out_ct2 = ct2(x4d)
    assert out_ct2.shape == (2, 4, 8, 8), f"ConvT2d shape {out_ct2.shape}"
    print(f"  ConvTranspose2d: {out_ct2.shape} OK")

    return True


def test_functional_ops():
    """Test functional API for conv ops."""
    print("=== Test Functional Ops ===")
    import scaffolding as torch
    import scaffolding.nn.functional as F

    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(16, 3, 3, 3)

    # functional conv2d
    out = F.conv2d(x, w, padding=1)
    assert out.shape == (1, 16, 8, 8), f"F.conv2d shape {out.shape}"
    print(f"  F.conv2d: {out.shape} OK")

    # functional conv3d
    x3 = torch.randn(1, 3, 4, 8, 8)
    w3 = torch.randn(8, 3, 3, 3, 3)
    out3 = F.conv3d(x3, w3, padding=1)
    assert out3.shape == (1, 8, 4, 8, 8), f"F.conv3d shape {out3.shape}"
    print(f"  F.conv3d: {out3.shape} OK")

    # functional group_norm
    x_gn = torch.randn(2, 8, 4, 4)
    w_gn = torch.ones(8)
    b_gn = torch.zeros(8)
    out_gn = F.group_norm(x_gn, 4, w_gn, b_gn)
    assert out_gn.shape == x_gn.shape, f"F.group_norm shape {out_gn.shape}"
    print(f"  F.group_norm: {out_gn.shape} OK")

    # functional tanh
    out_t = F.tanh(x)
    assert out_t.shape == x.shape
    print(f"  F.tanh: OK")

    return True


def test_video_pipeline_init():
    """Test TextToVideoPipeline instantiation."""
    print("=== Test Video Pipeline Init ===")
    from scaffolding.nn.video import TextToVideoPipeline

    pipeline = TextToVideoPipeline(
        num_frames=4,
        frame_height=16,
        frame_width=16,
        text_embed_dim=64,
        model_channels=16,
        num_heads=2,
        use_latent=False,
    )
    print(f"  Pipeline created OK")
    return True


def test_video_pipeline_generate():
    """Test TextToVideoPipeline.generate() with very small sizes."""
    print("=== Test Video Pipeline Generate ===")
    from scaffolding.nn.video import TextToVideoPipeline

    pipeline = TextToVideoPipeline(
        num_frames=2,
        frame_height=8,
        frame_width=8,
        text_embed_dim=32,
        model_channels=8,
        num_heads=2,
        use_latent=False,
    )
    try:
        video = pipeline.generate("a cat", num_steps=3, guidance_scale=1.0)
        print(f"  Generated video shape: {video.shape} OK")
        return True
    except Exception as e:
        print(f"  Generate failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = {}
    tests = [
        ('imports', test_imports),
        ('conv2d_forward', test_conv2d_forward),
        ('conv3d_forward', test_conv3d_forward),
        ('conv2d_backward', test_conv2d_backward),
        ('conv3d_backward', test_conv3d_backward),
        ('supporting_layers', test_supporting_layers),
        ('functional_ops', test_functional_ops),
        ('video_pipeline_init', test_video_pipeline_init),
        ('video_pipeline_generate', test_video_pipeline_generate),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = False
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")

    all_pass = all(results.values())
    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED'}")
    sys.exit(0 if all_pass else 1)
