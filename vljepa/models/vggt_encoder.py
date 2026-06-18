from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VGGT_ROOT = Path(__file__).resolve().parents[2] / "vggt"
if _VGGT_ROOT.exists() and str(_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402


class FrozenVGGT(nn.Module):
    """
    Frozen VGGT aggregator encoder.

    Input:  (B, T, 3, H, W)  ImageNet-normalised float32
    Output: (B, T, 2048)     mean-pooled final-layer patch features

    The aggregator expects [0, 1] unnormalised images and handles
    ImageNet normalisation internally.  This wrapper un-normalises
    our pre-normalised frames, resizes to 518 × 518, then forwards
    through the aggregator and returns mean-pooled patch tokens from
    the final cached layer (round 23 of 24).
    """

    _VGGT_SIZE = 518
    OUT_DIM = 2048  # 2 × embed_dim (frame_inter ∥ global_inter)

    def __init__(self, ckpt_path: str) -> None:
        super().__init__()

        vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        vggt.load_state_dict(state, strict=False)

        self.aggregator = torch.compile(vggt.aggregator)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("_un_mean", mean, persistent=False)
        self.register_buffer("_un_std",  std,  persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, 3, H, W)  ImageNet-normalised
        returns: (B, T, 2048)
        """
        B, T, C, H, W = frames.shape

        imgs = frames * self._un_std.to(frames) + self._un_mean.to(frames)
        imgs = imgs.clamp(0.0, 1.0)

        if H != self._VGGT_SIZE or W != self._VGGT_SIZE:
            imgs = imgs.view(B * T, C, H, W)
            imgs = F.interpolate(
                imgs, size=self._VGGT_SIZE, mode="bilinear", align_corners=False
            )
            imgs = imgs.view(B, T, C, self._VGGT_SIZE, self._VGGT_SIZE)

        # token_list: 24 entries, non-None shape (B, T, P, 2048)
        # P = 1 camera + 4 register + 1369 patch = 1374
        token_list, patch_start_idx = self.aggregator(imgs)

        # Last entry is always the final cached round (depth-1 = 23)
        final = token_list[-1]                              # (B, T, 1374, 2048)
        patches = final[:, :, patch_start_idx:, :]         # (B, T, 1369, 2048)
        return patches.mean(dim=2)                          # (B, T, 2048)
