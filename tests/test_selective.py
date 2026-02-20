"""Unit tests for selective decoding segmentation logic."""

import torch

from vljepa.inference.selective import selective_decode_points


def test_selective_decode_points_shapes() -> None:
    stream = torch.randn(12, 16)
    idx, emb = selective_decode_points(stream, num_segments=4, use_avg_pool=True)
    assert len(idx) == 4
    assert emb.shape == (4, 16)
    assert idx == sorted(idx)


def test_selective_decode_points_all_points_when_segments_large() -> None:
    stream = torch.randn(5, 16)
    idx, emb = selective_decode_points(stream, num_segments=10, use_avg_pool=False)
    assert idx == [0, 1, 2, 3, 4]
    assert emb.shape == (5, 16)

