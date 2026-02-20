from __future__ import annotations

"""Dataset and collation utilities for VL-JEPA experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import v2


class VisionLanguageJsonlDataset(Dataset):
    """
    Expected JSONL rows:
    {
      "image": "path/to/image.jpg" | null,
      "video": "path/to/video.mp4" | null,
      "query": "question or prompt",
      "target": "answer text",
      "candidates": ["a", "b", ...]  # optional for discriminative tasks
    }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 8,
        image_size: int = 256,
    ) -> None:
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be > 0, got {num_frames}")
        if image_size <= 0:
            raise ValueError(f"`image_size` must be > 0, got {image_size}")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.num_frames = num_frames
        self.transform = v2.Compose(
            [
                # Match common ImageNet-style normalization for ViT backbones.
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"Manifest has no valid samples: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_as_frames(self, image_path: str) -> torch.Tensor:
        """Load an image then duplicate it across T frames (paper Sec. 3.2)."""

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
        frame = self.transform(img)  # [3, H, W]
        return frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)  # [T, 3, H, W]

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Uniformly sample num_frames from a video clip."""

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video, _, _ = read_video(video_path, pts_unit="sec")
        # read_video returns [T, H, W, C] uint8
        if video.numel() == 0:
            raise RuntimeError(f"Empty video: {video_path}")
        total = video.shape[0]
        # Uniform temporal sampling to keep frame selection deterministic.
        idx = torch.linspace(0, total - 1, steps=self.num_frames).long()
        frames = video[idx]  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = torch.stack([self.transform(f) for f in frames], dim=0)
        return frames

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one sample with normalized frames and text fields."""

        row = self.samples[index]
        if "target" not in row:
            raise KeyError(f"Row {index} missing required `target` field.")
        if row.get("video"):
            frames = self._load_video(row["video"])
        elif row.get("image"):
            frames = self._load_image_as_frames(row["image"])
        else:
            raise ValueError("Each row must have either `image` or `video`.")

        return {
            "frames": frames,
            "query": row.get("query", ""),
            "target": row["target"],
            "candidates": row.get("candidates"),
        }


def vl_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch collator with explicit shape conventions."""

    return {
        "frames": torch.stack([x["frames"] for x in batch], dim=0),  # [B, T, C, H, W]
        "query": [x["query"] for x in batch],
        "target": [x["target"] for x in batch],
        "candidates": [x["candidates"] for x in batch],
    }
