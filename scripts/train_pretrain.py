#!/usr/bin/env python3
from __future__ import annotations

"""Entry point for VL-JEPA stage-1 pretraining."""

import argparse

import torch

from vljepa.train.build import build_model, build_optimizer_and_scheduler, build_train_loader
from vljepa.train.trainer import Trainer
from vljepa.utils.config import dump_resolved_config, load_yaml, validate_train_config
from vljepa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VL-JEPA pretraining stage")
    parser.add_argument("--config", type=str, default="vljepa/configs/pretrain.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    validate_train_config(cfg)
    set_seed(cfg["train"]["seed"], deterministic=True)
    dump_resolved_config(cfg, cfg["train"]["output_dir"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # Build all run components from config for explicit experiment control.
    model = build_model(cfg)
    loader = build_train_loader(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        output_dir=cfg["train"]["output_dir"],
        max_steps=cfg["train"]["max_steps"],
        grad_accum_steps=cfg["train"]["gradient_accumulation_steps"],
        clip_grad_norm=cfg["train"]["clip_grad_norm"],
        log_every=cfg["train"]["log_every"],
        save_every=cfg["train"]["save_every"],
        temperature=cfg["train"]["temperature"],
        precision=cfg["train"]["precision"],
    )
    trainer.fit(device)


if __name__ == "__main__":
    main()
