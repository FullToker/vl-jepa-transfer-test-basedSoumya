"""Unit tests for loss correctness and edge-case handling."""

import torch

from vljepa.models.losses import bidirectional_infonce


def test_bidirectional_infonce_runs_and_is_finite() -> None:
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    loss = bidirectional_infonce(pred, target, temperature=0.07)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bidirectional_infonce_rejects_small_batch() -> None:
    pred = torch.randn(1, 8)
    target = torch.randn(1, 8)
    try:
        bidirectional_infonce(pred, target, temperature=0.07)
        assert False, "Expected ValueError for batch size < 2"
    except ValueError as exc:
        assert "batch size" in str(exc)

