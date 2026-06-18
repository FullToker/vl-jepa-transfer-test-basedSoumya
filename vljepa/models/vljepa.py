from __future__ import annotations

"""Core VL-JEPA model components and paper-faithful data flow."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency in tiny/offline mode
    AutoModel = None
    AutoTokenizer = None

from vljepa.models.vggt_encoder import FrozenVGGT

# References:
# - Chen et al., "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
#   arXiv:2512.10942v2, Feb 2, 2026
# - Paper mapping:
#   - Fig. 1 / Fig. 2: overall architecture
#   - Sec. 3.1: X-Encoder, Predictor, Y-Encoder, projection heads (1536-d)
#   - Sec. 3.2 + Tab. 7(b): Y-Encoder LR multiplier ~= 0.05


@dataclass
class VLJEPAConfig:
    """Configuration for all trainable and frozen VL-JEPA modules."""

    vision_backbone: str
    predictor_hidden_size: int
    predictor_layers: int
    predictor_heads: int
    predictor_ffn_mult: int
    predictor_dropout: float
    query_model_name: str
    y_encoder_name: str
    max_query_tokens: int
    max_target_tokens: int
    shared_embed_dim: int
    freeze_x_encoder: bool
    y_encoder_lr_multiplier: float
    # VGGT fusion (last-layer concat)
    use_vggt: bool = False
    vggt_ckpt: str = "./ckpts/VGGT-1B/model.pt"


class _TokenBatch(dict):
    """Simple tensor dictionary with `.to(device)` convenience."""

    def to(self, device: torch.device) -> "_TokenBatch":
        return _TokenBatch({k: v.to(device) for k, v in self.items()})


class ToyTokenizer:
    """
    Lightweight whitespace tokenizer for fully offline smoke tests.

    This is not meant to replicate paper quality; it guarantees local runtime.
    """

    def __init__(self, vocab_size: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "[PAD]"
        self.eos_token = "[EOS]"

    def _encode_text(self, text: str, max_length: int) -> List[int]:
        words = text.lower().strip().split()
        ids = [((abs(hash(w)) % (self.vocab_size - 2)) + 2) for w in words]
        ids = ids[: max(1, max_length - 1)]
        ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: List[str],
        max_length: int,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> _TokenBatch:
        del truncation  # Truncation already handled by _encode_text max_length.
        if return_tensors != "pt":
            raise ValueError("ToyTokenizer only supports return_tensors='pt'.")
        ids = [self._encode_text(t, max_length=max_length) for t in texts]
        tgt_len = max(len(x) for x in ids) if padding else None
        padded = []
        mask = []
        for seq in ids:
            if padding and tgt_len is not None:
                pad = tgt_len - len(seq)
                padded.append(seq + [self.pad_token_id] * pad)
                mask.append([1] * len(seq) + [0] * pad)
            else:
                padded.append(seq)
                mask.append([1] * len(seq))
        return _TokenBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        )


class ToyTextEncoder(nn.Module):
    """Simple text encoder returning token embeddings as `last_hidden_state`."""

    def __init__(self, vocab_size: int = 4096, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        x = self.embedding(input_ids)
        x = self.norm(x)
        return SimpleNamespace(last_hidden_state=x)


class VisionEncoder(nn.Module):
    """X-Encoder approximation used for S_V extraction (paper Sec. 3.1)."""

    def __init__(self, backbone: str, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"`output_dim` must be > 0, got {output_dim}")
        self.backbone_name = backbone
        if backbone == "vit_b_16":
            net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            feat_dim = net.hidden_dim
            self.backbone = net
        elif backbone == "vit_l_16":
            net = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
            feat_dim = net.hidden_dim
            self.backbone = net
        elif backbone == "vit_b_16_rand":
            net = vit_b_16(weights=None)
            feat_dim = net.hidden_dim
            self.backbone = net
        elif backbone == "vit_l_16_rand":
            net = vit_l_16(weights=None)
            feat_dim = net.hidden_dim
            self.backbone = net
        elif backbone == "toy_cnn":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            feat_dim = 64
        elif backbone.startswith("hf:"):
            if AutoModel is None:
                raise ImportError("transformers is required for `hf:` vision backbones.")
            model_name = backbone.split("hf:", 1)[1]
            self.backbone = AutoModel.from_pretrained(model_name)
            feat_dim = getattr(self.backbone.config, "hidden_size", None)
            if feat_dim is None:
                raise ValueError(f"HF vision model missing `hidden_size`: {model_name}")
        else:
            raise ValueError(f"Unsupported vision backbone: {backbone}")
        self.proj = nn.Linear(feat_dim, output_dim)

    def _forward_torchvision_vit(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        b = x.shape[0]
        x = self.backbone._process_input(x)
        cls_token = self.backbone.class_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.backbone.encoder(x)
        x = x[:, 1:, :].mean(dim=1)  # average patch tokens
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        if self.backbone_name in {"vit_b_16", "vit_l_16", "vit_b_16_rand", "vit_l_16_rand"}:
            feat = self._forward_torchvision_vit(x)
        elif self.backbone_name == "toy_cnn":
            feat = self.backbone(x).flatten(1)
        else:
            out = self.backbone(pixel_values=x)
            if hasattr(out, "last_hidden_state"):
                feat = out.last_hidden_state[:, 1:, :].mean(dim=1)
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                raise RuntimeError("Vision model output has no supported feature tensor.")
        feat = self.proj(feat).view(b, t, -1)
        return feat


class FusedVisionEncoder(nn.Module):
    """
    Torchvision ViT + frozen VGGT last-layer concat encoder.

    Both branches receive the same ImageNet-normalised frames.  VGGT
    features are mean-pooled over spatial tokens; ViT features are
    mean-pooled over patch tokens.  The two are concatenated and
    projected to output_dim.

    Input:  [B, T, C, H, W]  ImageNet-normalised
    Output: [B, T, output_dim]

    Supported backbones: vit_b_16, vit_l_16, vit_b_16_rand, vit_l_16_rand
    """

    def __init__(self, backbone: str, output_dim: int, vggt_ckpt: str) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone in {"vit_b_16", "vit_b_16_rand"}:
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if backbone == "vit_b_16" else None
            net = vit_b_16(weights=weights)
        elif backbone in {"vit_l_16", "vit_l_16_rand"}:
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if backbone == "vit_l_16" else None
            net = vit_l_16(weights=weights)
        else:
            raise ValueError(
                f"FusedVisionEncoder supports only torchvision ViTs, got: {backbone}"
            )

        self._vit_feat_dim: int = net.hidden_dim
        self.vit = net
        # Freeze pretrained ViT — only proj is learnable in this encoder
        for p in self.vit.parameters():
            p.requires_grad_(False)

        self.vggt = FrozenVGGT(vggt_ckpt)  # always frozen internally
        self.proj = nn.Linear(self._vit_feat_dim + FrozenVGGT.OUT_DIM, output_dim)

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) → (N, vit_feat_dim) mean-pooled patch tokens."""
        n = x.shape[0]
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 1:, :].mean(dim=1)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, C, H, W] → [B, T, output_dim]"""
        B, T, C, H, W = frames.shape

        vit_feat = self._vit_features(frames.view(B * T, C, H, W))
        vit_feat = vit_feat.view(B, T, self._vit_feat_dim)

        vggt_feat = self.vggt(frames)  # (B, T, 2048)

        fused = torch.cat([vit_feat, vggt_feat], dim=-1)
        return self.proj(fused).float()  # (B, T, output_dim)


class VLJEPA(nn.Module):
    """
    VL-JEPA core model.

    Implements:
    - X-Encoder: visual-to-latent sequence S_V
    - Predictor: bidirectional query-conditioned transformer, (S_V, X_Q) -> S^_Y
    - Y-Encoder: target text embedding S_Y
    - Shared projection space for loss computation
    """

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.predictor_hidden_size
        if h <= 0:
            raise ValueError("`predictor_hidden_size` must be > 0.")

        # Paper Sec. 3.1: frozen visual encoder by default.
        if cfg.use_vggt:
            self.x_encoder = FusedVisionEncoder(cfg.vision_backbone, h, cfg.vggt_ckpt)
        else:
            self.x_encoder = VisionEncoder(cfg.vision_backbone, h)
        if cfg.freeze_x_encoder:
            for p in self.x_encoder.parameters():
                p.requires_grad = False
            # FusedVisionEncoder: proj bridges the two frozen backbones and must train
            if cfg.use_vggt:
                for p in self.x_encoder.proj.parameters():
                    p.requires_grad = True

        if cfg.query_model_name == "toy":
            self.query_tokenizer = ToyTokenizer()
            self.query_encoder = ToyTextEncoder(hidden_size=128)
        else:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("transformers is required for non-toy query models.")
            self.query_tokenizer = AutoTokenizer.from_pretrained(cfg.query_model_name, use_fast=True)
            if self.query_tokenizer.pad_token is None:
                self.query_tokenizer.pad_token = self.query_tokenizer.eos_token
            self.query_encoder = AutoModel.from_pretrained(cfg.query_model_name)
        self.query_embed = self.query_encoder.get_input_embeddings()
        for p in self.query_encoder.parameters():
            p.requires_grad = False

        # Paper Sec. 3.1: Predictor uses bidirectional attention (no causal mask).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.predictor_heads,
            dim_feedforward=h * cfg.predictor_ffn_mult,
            dropout=cfg.predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=cfg.predictor_layers)

        self.query_in_proj = nn.Linear(self.query_embed.embedding_dim, h)
        if cfg.y_encoder_name == "toy":
            self.y_encoder = ToyTextEncoder(hidden_size=128)
            self.y_tokenizer = ToyTokenizer()
        else:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("transformers is required for non-toy y-encoders.")
            self.y_encoder = AutoModel.from_pretrained(cfg.y_encoder_name)
            self.y_tokenizer = AutoTokenizer.from_pretrained(cfg.y_encoder_name, use_fast=True)
            if self.y_tokenizer.pad_token is None:
                self.y_tokenizer.pad_token = self.y_tokenizer.eos_token
        y_hidden = getattr(self.y_encoder.config, "hidden_size", None)
        if y_hidden is None:
            raise ValueError(f"Y-Encoder `{cfg.y_encoder_name}` missing `hidden_size`.")

        # Paper Sec. 3.1: Predictor and Y-Encoder are projected into a shared
        # embedding space where InfoNCE is computed (dimension set by config).
        self.pred_proj = nn.Linear(h, cfg.shared_embed_dim)
        self.y_proj = nn.Linear(y_hidden, cfg.shared_embed_dim)

    def encode_target_text(self, targets: List[str], device: torch.device) -> torch.Tensor:
        """Y-Encoder branch: Y -> S_Y (paper Fig. 1, Sec. 2 and Sec. 3.1)."""

        if not targets:
            raise ValueError("`targets` must be a non-empty list.")
        tok = self.y_tokenizer(
            targets,
            max_length=self.cfg.max_target_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = self.y_encoder(**tok)
        hs = out.last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.y_proj(pooled)

    def predict_embedding(self, frames: torch.Tensor, queries: List[str]) -> torch.Tensor:
        """Predictor branch: (S_V, X_Q) -> S^_Y with non-causal joint attention."""

        if frames.ndim != 5:
            raise ValueError(f"`frames` must be [B, T, C, H, W], got shape {tuple(frames.shape)}")
        if len(queries) != frames.shape[0]:
            raise ValueError("`queries` length must match batch size in `frames`.")
        device = frames.device
        vis = self.x_encoder(frames)  # [B, T, H]
        q = self.query_tokenizer(
            queries,
            max_length=self.cfg.max_query_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        q_emb = self.query_embed(q["input_ids"])
        q_emb = self.query_in_proj(q_emb)

        x = torch.cat([vis, q_emb], dim=1)
        b, t, _ = vis.shape
        q_mask = q["attention_mask"]
        vis_mask = torch.ones((b, t), device=device, dtype=q_mask.dtype)
        mask = torch.cat([vis_mask, q_mask], dim=1)  # 1 means valid
        # No causal mask: visual/query tokens can attend bidirectionally.
        x = self.predictor(x, src_key_padding_mask=(mask == 0))

        # Mean-pool non-pad query tokens; fallback to full sequence if query is empty.
        q_tokens = x[:, t:, :]
        q_mask = q_mask.unsqueeze(-1)
        pooled_q = (q_tokens * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp_min(1)
        fallback = x.mean(dim=1)
        pooled = torch.where((q_mask.sum(dim=1) > 0), pooled_q, fallback)
        return self.pred_proj(pooled)

    def forward(self, frames: torch.Tensor, queries: List[str], targets: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass returning predicted and target embeddings for loss."""

        pred = self.predict_embedding(frames, queries)
        target = self.encode_target_text(targets, frames.device)
        return {"pred": pred, "target": target}

    def parameter_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        # Paper Sec. 3.1 + Tab. 7(b): slower Y-Encoder updates improve stability.
        if lr <= 0:
            raise ValueError("`lr` must be > 0.")
        y_params = []
        base_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("y_encoder."):
                y_params.append(p)
            else:
                base_params.append(p)

        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": lr, "weight_decay": weight_decay})
        if y_params:
            groups.append(
                {
                    "params": y_params,
                    "lr": lr * self.cfg.y_encoder_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )
        return groups
