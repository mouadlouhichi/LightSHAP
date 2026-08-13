"""Symmetrically normalized bipartite adjacency. Built from train only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class GraphBundle:
    hat_a: torch.Tensor  # sparse coalesced (N, N)
    n_users: int
    n_items: int
    n_nodes: int
    nnz: int
    user_degree: np.ndarray
    item_degree: np.ndarray
    isolated_users: int
    isolated_items: int

    def to_meta(self) -> dict[str, int]:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_nodes": self.n_nodes,
            "nnz": self.nnz,
            "isolated_users": self.isolated_users,
            "isolated_items": self.isolated_items,
        }


def build_hat_a(train_df: pd.DataFrame, n_users: int, n_items: int) -> GraphBundle:
    """hat_A = D^{-1/2} A D^{-1/2} on the undirected bipartite graph, no self-loops."""
    u = train_df["user_idx"].to_numpy(dtype=np.int64)
    it = train_df["item_idx"].to_numpy(dtype=np.int64)
    # undirected: user→item and item→user
    rows = np.concatenate([u, it + n_users])
    cols = np.concatenate([it + n_users, u])
    data = np.ones(len(rows), dtype=np.float32)
    n = n_users + n_items
    a = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    # collapse possible duplicate edges
    a.sum_duplicates()
    deg = np.asarray(a.sum(axis=1)).flatten().astype(np.float64)
    inv = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    inv[np.isinf(inv)] = 0.0
    inv[deg == 0] = 0.0
    d_inv = sp.diags(inv.astype(np.float32))
    hat = (d_inv @ a @ d_inv).tocoo()
    indices = torch.tensor(np.vstack([hat.row, hat.col]), dtype=torch.long)
    values = torch.tensor(hat.data, dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        sparse = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    user_deg = deg[:n_users].astype(np.int64)
    item_deg = deg[n_users:].astype(np.int64)
    return GraphBundle(
        hat_a=sparse,
        n_users=n_users,
        n_items=n_items,
        n_nodes=n,
        nnz=int(sparse._nnz()),
        user_degree=user_deg,
        item_degree=item_deg,
        isolated_users=int((user_deg == 0).sum()),
        isolated_items=int((item_deg == 0).sum()),
    )


def assert_symmetric_sparse(mat: torch.Tensor, atol: float = 1e-5) -> None:
    t = mat.transpose(0, 1).coalesce()
    m = mat.coalesce()
    if m._nnz() != t._nnz():
        raise ValueError("hat_A is not structurally symmetric")
    # compare by converting a small dense only if tiny; else check indices
    mi = set(map(tuple, m.indices().t().tolist()))
    ti = set(map(tuple, t.indices().t().tolist()))
    if mi != ti:
        raise ValueError("hat_A index set is not symmetric")
