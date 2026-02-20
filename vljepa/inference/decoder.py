from __future__ import annotations

"""Lightweight readout decoders for embedding-to-text inference."""

from typing import List

import torch
import torch.nn.functional as F


class NearestNeighborDecoder:
    """
    Lightweight readout decoder that maps predicted embeddings to text
    by nearest-neighbor search in a text bank.
    """

    def __init__(self, model, text_bank: List[str], batch_size: int = 128) -> None:
        if not text_bank:
            raise ValueError("`text_bank` cannot be empty.")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        self.model = model
        self.text_bank = text_bank
        self.batch_size = batch_size
        self.bank_emb = None

    @torch.no_grad()
    def build(self, device: torch.device) -> None:
        """Encode all candidate strings into a normalized search bank."""

        embs = []
        for i in range(0, len(self.text_bank), self.batch_size):
            chunk = self.text_bank[i : i + self.batch_size]
            e = self.model.encode_target_text(chunk, device)
            embs.append(F.normalize(e, dim=-1))
        self.bank_emb = torch.cat(embs, dim=0)

    @torch.no_grad()
    def decode(self, pred_embeddings: torch.Tensor, topk: int = 1) -> List[str] | List[List[str]]:
        """Return nearest strings by cosine similarity in embedding space."""

        if self.bank_emb is None:
            raise RuntimeError("Decoder bank not built. Call build(device) first.")
        if topk <= 0:
            raise ValueError("`topk` must be > 0.")
        topk = min(topk, self.bank_emb.shape[0])
        x = F.normalize(pred_embeddings, dim=-1)
        sim = x @ self.bank_emb.T
        idx = sim.topk(k=topk, dim=-1).indices
        if topk == 1:
            return [self.text_bank[i] for i in idx[:, 0].tolist()]
        return [[self.text_bank[j] for j in row.tolist()] for row in idx]
