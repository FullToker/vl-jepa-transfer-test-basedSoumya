from __future__ import annotations

"""Factory helpers for model, dataloader, optimizer, and scheduler."""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader

from vljepa.data.dataset import VisionLanguageJsonlDataset, vl_collate
from vljepa.models.vljepa import VLJEPA, VLJEPAConfig


def build_model(cfg: dict) -> VLJEPA:
    """Instantiate VL-JEPA from config values."""

    m = cfg["model"]
    model_cfg = VLJEPAConfig(
        vision_backbone=m["vision_backbone"],
        predictor_hidden_size=m["predictor_hidden_size"],
        predictor_layers=m["predictor_layers"],
        predictor_heads=m["predictor_heads"],
        predictor_ffn_mult=m["predictor_ffn_mult"],
        predictor_dropout=m["predictor_dropout"],
        query_model_name=m["query_model_name"],
        y_encoder_name=m["y_encoder_name"],
        max_query_tokens=m["max_query_tokens"],
        max_target_tokens=m["max_target_tokens"],
        shared_embed_dim=m["shared_embed_dim"],
        freeze_x_encoder=m["freeze_x_encoder"],
        y_encoder_lr_multiplier=m["y_encoder_lr_multiplier"],
    )
    return VLJEPA(model_cfg)


def build_train_loader(cfg: dict) -> DataLoader:
    """Build train dataloader from one or multiple JSONL manifests."""

    dcfg = cfg["data"]
    datasets = []
    for manifest in dcfg["train_manifests"]:
        if not Path(manifest).exists():
            raise FileNotFoundError(f"Training manifest not found: {manifest}")
        datasets.append(
            VisionLanguageJsonlDataset(
                manifest_path=manifest,
                num_frames=dcfg["num_frames"],
                image_size=dcfg["image_size"],
            )
        )
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        pin_memory=True,
        collate_fn=vl_collate,
        drop_last=True,
    )


def build_optimizer_and_scheduler(model: VLJEPA, cfg: dict) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """
    Build optimizer/scheduler pair.

    Paper mapping:
    - Pretraining: constant LR (Sec. 3.2)
    - SFT: cosine LR annealing (Sec. 3.2)
    """

    tcfg = cfg["train"]
    optim = torch.optim.AdamW(
        model.parameter_groups(lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]),
        betas=(tcfg["adam_beta1"], tcfg["adam_beta2"]),
        eps=tcfg["adam_eps"],
    )
    sched_name = tcfg.get("scheduler", "constant")
    if sched_name == "constant":
        scheduler = None
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=tcfg["max_steps"],
            eta_min=tcfg.get("min_lr", 0.0),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")
    return optim, scheduler
