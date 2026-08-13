"""Frozen LightGCN. K=3, d=64 are scientific freezes; not searched."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        d: int,
        K: int,
        hat_a: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.d = d
        self.K = K
        self.embedding = nn.Embedding(self.n_nodes, d)
        nn.init.xavier_uniform_(self.embedding.weight)
        # hat_A is a buffer so it moves with .to(device) and is not trained.
        # MPS has incomplete sparse.mm; we densify on that device (ML-1M/Beauty fit in 16+ GB).
        if hat_a.is_sparse:
            hat_a = hat_a.coalesce()
        self.register_buffer("hat_a", hat_a)

    def _matmul(self, hat: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if e.device.type == "mps" and hat.is_sparse:
            hat = hat.to_dense()
        if hat.is_sparse:
            return torch.sparse.mm(hat, e)
        return hat @ e

    def propagate(self) -> list[torch.Tensor]:
        e0 = self.embedding.weight
        layers = [e0]
        e = e0
        hat = self.hat_a
        for _ in range(self.K):
            e = self._matmul(hat, e)
            layers.append(e)
        return layers

    def fuse_uniform(self, layers: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        if layers is None:
            layers = self.propagate()
        return torch.stack(list(layers), dim=0).mean(dim=0)

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fused is None:
            fused = self.fuse_uniform()
        u = fused[users]
        it = fused[self.n_users + items]
        return (u * it).sum(dim=-1)

    def l2_batch(
        self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> torch.Tensor:
        w = self.embedding.weight
        return w[users].pow(2).sum() + w[self.n_users + pos].pow(2).sum() + w[
            self.n_users + neg
        ].pow(2).sum()

    def cache_layers(self) -> list[torch.Tensor]:
        self.eval()
        with torch.no_grad():
            return [t.detach().cpu().contiguous() for t in self.propagate()]
