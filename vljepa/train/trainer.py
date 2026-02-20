from __future__ import annotations

"""Training loop with checkpointing and traceability logs."""

import json
import platform
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vljepa.models.losses import bidirectional_infonce


class Trainer:
    """
    Generic VL-JEPA trainer shared by pretraining and SFT.

    Stage mapping to paper Sec. 3.2:
    - Pretraining config uses constant LR.
    - SFT config uses cosine LR annealing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_loader: DataLoader,
        output_dir: str | Path,
        max_steps: int,
        grad_accum_steps: int = 1,
        clip_grad_norm: float = 1.0,
        log_every: int = 20,
        save_every: int = 1000,
        temperature: float = 0.07,
        precision: str = "bf16",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("`max_steps` must be > 0.")
        if grad_accum_steps <= 0:
            raise ValueError("`grad_accum_steps` must be > 0.")
        if log_every <= 0:
            raise ValueError("`log_every` must be > 0.")
        if save_every <= 0:
            raise ValueError("`save_every` must be > 0.")
        if precision not in {"bf16", "fp16"}:
            raise ValueError("`precision` must be either `bf16` or `fp16`.")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.log_every = log_every
        self.save_every = save_every
        self.temperature = temperature
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        self.autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"
        self.meta_path = self.output_dir / "run_meta.json"

    def _save_ckpt(self, step: int) -> None:
        """Save model/optimizer/scheduler state for exact continuation."""

        ckpt = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(ckpt, self.output_dir / f"step_{step:07d}.pt")

    def _write_run_metadata(self, device: torch.device) -> None:
        """Log environment info to make runs auditable/reproducible."""

        meta = {
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "hostname": platform.node(),
            "platform": platform.platform(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def fit(self, device: torch.device) -> None:
        """Execute optimization steps until max_steps is reached."""

        self.model.to(device)
        self.model.train()
        self._write_run_metadata(device)
        step = 0
        running_loss = 0.0

        pbar = tqdm(total=self.max_steps, desc="training")
        while step < self.max_steps:
            for batch in self.train_loader:
                if "frames" not in batch or "query" not in batch or "target" not in batch:
                    raise KeyError("Batch must contain `frames`, `query`, and `target` keys.")
                # All batch fields are explicit for easier debugging.
                frames = batch["frames"].to(device, non_blocking=True)
                queries = batch["query"]
                targets = batch["target"]

                with torch.autocast(device_type=device.type, dtype=self.autocast_dtype):
                    out = self.model(frames, queries, targets)
                    # Main JEPA objective in embedding space (paper Sec. 2).
                    loss = bidirectional_infonce(
                        out["pred"],
                        out["target"],
                        temperature=self.temperature,
                    )
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                running_loss += loss.item() * self.grad_accum_steps

                if (step + 1) % self.grad_accum_steps == 0:
                    # Standard mixed-precision optimizer update.
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

                step += 1
                pbar.update(1)
                if step % self.log_every == 0:
                    info: Dict[str, float | int] = {
                        "step": step,
                        "loss": running_loss / self.log_every,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(info) + "\n")
                    running_loss = 0.0
                if step % self.save_every == 0:
                    self._save_ckpt(step)
                if step >= self.max_steps:
                    break

        self._save_ckpt(step)
        pbar.close()
