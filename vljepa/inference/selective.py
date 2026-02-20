from __future__ import annotations

"""Selective decoding utilities for long-form video streams."""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def _temporal_connectivity(n: int) -> np.ndarray:
    """Create chain-graph connectivity so clusters stay temporally local."""

    conn = np.zeros((n, n), dtype=np.int8)
    idx = np.arange(n - 1)
    conn[idx, idx + 1] = 1
    conn[idx + 1, idx] = 1
    return conn


def selective_decode_points(
    embedding_stream: torch.Tensor,
    num_segments: int,
    use_avg_pool: bool = True,
) -> Tuple[List[int], torch.Tensor]:
    """
    Selective decoding helper (paper Sec. 4.6, Fig. 4).

    Implementation mirrors the paper description:
    - cluster embedding stream with temporal connectivity constraints
    - Ward linkage to favor low intra-segment variance (monosemantic segments)
    - decode one point per segment (midpoint), optionally on avg-pooled segment embedding

    embedding_stream: [T, D]
    Returns:
      decode_indices: midpoint frame index per temporal segment
      decode_embeddings: [N, D], either midpoint embedding or avg pooled segment embedding
    """
    x = embedding_stream.detach().cpu().numpy()
    t = x.shape[0]
    if embedding_stream.ndim != 2:
        raise ValueError(f"`embedding_stream` must be [T, D], got shape {tuple(embedding_stream.shape)}")
    if num_segments <= 0:
        raise ValueError("`num_segments` must be > 0.")
    if num_segments >= t:
        # Degenerate case: decode at every timestamp.
        idx = list(range(t))
        return idx, embedding_stream

    clustering = AgglomerativeClustering(
        n_clusters=num_segments,
        linkage="ward",
        connectivity=_temporal_connectivity(t),
    )
    labels = clustering.fit_predict(x)

    points = []
    pooled = []
    for lab in np.unique(labels):
        seg_idx = np.where(labels == lab)[0]
        mid = int(seg_idx[len(seg_idx) // 2])
        points.append(mid)
        if use_avg_pool:
            pooled.append(embedding_stream[seg_idx].mean(dim=0))
        else:
            pooled.append(embedding_stream[mid])
    points_sorted = np.argsort(points)
    decode_idx = [points[i] for i in points_sorted]
    decode_emb = torch.stack([pooled[i] for i in points_sorted], dim=0)
    return decode_idx, decode_emb
