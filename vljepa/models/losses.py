from __future__ import annotations

import torch
import torch.nn.functional as F


def bidirectional_infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Bi-directional InfoNCE used by VL-JEPA (paper Sec. 2, "Training Objective").

    This combines:
    - alignment between matched (pred_i, target_i)
    - uniformity/anti-collapse via in-batch negatives
    and is applied in both directions as described in the paper text.

    pred: [B, D]
    target: [B, D]
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"`pred` and `target` must be rank-2 tensors, got {pred.ndim} and {target.ndim}.")
    if pred.shape != target.shape:
        raise ValueError(f"`pred` and `target` shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}.")
    if pred.shape[0] < 2:
        raise ValueError("InfoNCE requires batch size >= 2 for negatives.")
    if temperature <= 0:
        raise ValueError("`temperature` must be > 0.")

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (pred @ target.T) / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss_xy = F.cross_entropy(logits, labels)
    loss_yx = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_xy + loss_yx)
