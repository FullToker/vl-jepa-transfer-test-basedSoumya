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
    for i, cand in enumerate(candidates):
        if not cand:
            raise ValueError(f"Empty candidate list at index {i}.")

    # Encode all candidates from the entire batch in one y_encoder forward pass.
    sizes = [len(cand) for cand in candidates]
    flat_cands = [c for cand in candidates for c in cand]
    flat_emb = model.encode_target_text(flat_cands, device)  # [sum(sizes), D]

    outputs = []
    offset = 0
    for i, size in enumerate(sizes):
        c_emb = flat_emb[offset : offset + size]  # [n_cand, D]
        sim = F.normalize(pred_emb[i : i + 1], dim=-1) @ F.normalize(c_emb, dim=-1).T
        outputs.append(candidates[i][int(sim.argmax().item())])
        offset += size
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
