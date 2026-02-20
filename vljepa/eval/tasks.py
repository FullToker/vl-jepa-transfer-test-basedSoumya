from __future__ import annotations

"""Task utilities for discriminative matching and retrieval evaluation."""

from typing import List, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def discriminative_match(model, pred_emb: torch.Tensor, candidates: Sequence[Sequence[str]]) -> List[str]:
    """
    Select candidate answer with highest cosine similarity to predicted embedding.

    Paper mapping:
    - Sec. 2 multi-tasking and Sec. 4 discriminative VQA setup.
    """

    device = pred_emb.device
    outputs = []
    for i, cand in enumerate(candidates):
        if not cand:
            raise ValueError(f"Empty candidate list at index {i}.")
        c_emb = model.encode_target_text(list(cand), device)
        sim = F.normalize(pred_emb[i : i + 1], dim=-1) @ F.normalize(c_emb, dim=-1).T
        outputs.append(cand[int(sim.argmax().item())])
    return outputs


@torch.no_grad()
def retrieval_scores(
    model,
    video_embs: torch.Tensor,
    query_texts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Compute query-to-video cosine similarity matrix for retrieval."""

    q_emb = model.encode_target_text(list(query_texts), device)
    v = F.normalize(video_embs, dim=-1)
    q = F.normalize(q_emb, dim=-1)
    return q @ v.T
