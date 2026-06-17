#!/usr/bin/env python3
"""
Verify FusedVisionEncoder + VLJEPA forward shapes with synthetic input.
Downloads VGGT-1B weights (~5 GB), runs checks, then deletes them.

Usage (in docker vjepa21:v2):
    python scripts/test_vggt_fusion.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import hf_hub_download

CKPT_DIR  = ROOT / "ckpts" / "VGGT-1B"
CKPT_PATH = CKPT_DIR / "model.pt"

# ── Download ──────────────────────────────────────────────────────────────────

def download_vggt() -> None:
    print("==> Downloading facebook/VGGT-1B  model.pt (~5 GB)...")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="facebook/VGGT-1B",
        filename="model.pt",
        local_dir=str(CKPT_DIR),
    )
    print(f"    saved → {CKPT_PATH}")


# ── Shape test ────────────────────────────────────────────────────────────────

def test_fused_encoder_shape() -> None:
    from vljepa.models.vljepa import FusedVisionEncoder

    B, T, C, H, W = 2, 1, 3, 224, 224
    output_dim = 2048

    print("\n==> FusedVisionEncoder forward shape test")
    enc = FusedVisionEncoder(
        backbone="vit_l_16",
        output_dim=output_dim,
        vggt_ckpt=str(CKPT_PATH),
    ).cuda().eval()

    frames = torch.randn(B, T, C, H, W, device="cuda")
    with torch.no_grad():
        out = enc(frames)

    print(f"    input : {tuple(frames.shape)}")
    print(f"    output: {tuple(out.shape)}  (expect ({B}, {T}, {output_dim}))")
    assert out.shape == (B, T, output_dim), f"FAIL shape {out.shape}"
    print("    [PASS]")


def test_vljepa_full_forward() -> None:
    from vljepa.models.vljepa import VLJEPA, VLJEPAConfig

    B, T, C, H, W = 2, 1, 3, 224, 224

    print("\n==> VLJEPA full forward test (toy text encoders)")
    cfg = VLJEPAConfig(
        vision_backbone="vit_l_16",
        predictor_hidden_size=512,
        predictor_layers=2,
        predictor_heads=8,
        predictor_ffn_mult=4,
        predictor_dropout=0.0,
        query_model_name="toy",
        y_encoder_name="toy",
        max_query_tokens=16,
        max_target_tokens=16,
        shared_embed_dim=256,
        freeze_x_encoder=True,
        y_encoder_lr_multiplier=0.05,
        use_vggt=True,
        vggt_ckpt=str(CKPT_PATH),
    )
    model = VLJEPA(cfg).cuda().eval()

    frames  = torch.randn(B, T, C, H, W, device="cuda")
    queries = ["what is this?"] * B
    targets = ["a room"] * B

    with torch.no_grad():
        out = model(frames, queries, targets)

    pred_shape   = tuple(out["pred"].shape)
    target_shape = tuple(out["target"].shape)
    expect = (B, cfg.shared_embed_dim)
    print(f"    pred  : {pred_shape}  (expect {expect})")
    print(f"    target: {target_shape}  (expect {expect})")
    assert out["pred"].shape   == expect, f"FAIL pred {pred_shape}"
    assert out["target"].shape == expect, f"FAIL target {target_shape}"
    print("    [PASS]")
    return model


# ── Frozen checks ─────────────────────────────────────────────────────────────

def test_frozen(model) -> None:
    print("\n==> Frozen parameter check")

    enc = model.x_encoder   # FusedVisionEncoder

    vit_grad  = [n for n, p in enc.vit.named_parameters()  if p.requires_grad]
    vggt_grad = [n for n, p in enc.vggt.named_parameters() if p.requires_grad]
    proj_grad = [n for n, p in enc.proj.named_parameters() if p.requires_grad]

    print(f"    ViT  trainable params : {len(vit_grad)}   (expect 0)")
    print(f"    VGGT trainable params : {len(vggt_grad)}  (expect 0)")
    print(f"    proj trainable params : {len(proj_grad)}  (expect >0)")

    assert len(vit_grad)  == 0, f"ViT NOT frozen: {vit_grad[:3]}"
    assert len(vggt_grad) == 0, f"VGGT NOT frozen: {vggt_grad[:3]}"
    assert len(proj_grad)  > 0, "proj has NO trainable params"
    print("    [PASS]")


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup() -> None:
    print(f"\n==> Deleting {CKPT_DIR}  (save disk space)...")
    shutil.rmtree(CKPT_DIR)
    print("    done.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_vggt()
    test_fused_encoder_shape()
    model = test_vljepa_full_forward()
    test_frozen(model)
    cleanup()
    print("\n==> All checks passed.")
