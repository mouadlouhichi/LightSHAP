"""BPR-MF. Exactly the six listed (d, lr, reg) tuples; not searched beyond that."""

from __future__ import annotations

import torch
import torch.nn as nn


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, d: int) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.d = d
        self.user = nn.Embedding(n_users, d)
        self.item = nn.Embedding(n_items, d)
        nn.init.xavier_uniform_(self.user.weight)
        nn.init.xavier_uniform_(self.item.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(dim=-1)

    def full_scores(self, users: torch.Tensor) -> torch.Tensor:
        return (self.user(users) @ self.item.weight.t()).to(torch.float32)

    def l2(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return (
            self.user(users).pow(2).sum()
            + self.item(pos).pow(2).sum()
            + self.item(neg).pow(2).sum()
        )
