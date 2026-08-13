"""Load, implicitize, 5-core to a fixed point, temporal LOO split, index align."""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from lightshap.config import DatasetConfig
from lightshap.data.synthetic import make_synthetic_interactions
from lightshap.exceptions import DataIntegrityError


def load_raw_interactions(cfg: DatasetConfig, raw_dir: Path) -> pd.DataFrame:
    if cfg.name == "synthetic":
        return make_synthetic_interactions()
    path = raw_dir / cfg.filename
    if not path.is_file():
        raise DataIntegrityError(f"raw file missing: {path}")
    if cfg.name == "ml1m":
        return _load_ml1m(path)
    if cfg.name == "beauty":
        return _load_beauty(path)
    raise DataIntegrityError(f"unknown dataset {cfg.name}")


def _load_ml1m(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("ratings.dat"))
        raw = zf.read(name).decode("latin-1")
    df = pd.read_csv(
        io.StringIO(raw),
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
        header=None,
    )
    df["user_raw"] = df["user_raw"].astype(str)
    df["item_raw"] = df["item_raw"].astype(str)
    return df


def _load_beauty(path: Path) -> pd.DataFrame:
    rows = []
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "user_raw": str(rec.get("reviewerID") or rec.get("userId")),
                    "item_raw": str(rec.get("asin") or rec.get("itemId")),
                    "rating": float(rec.get("overall") or rec.get("rating") or 5.0),
                    "timestamp": int(rec.get("unixReviewTime") or rec.get("timestamp") or 0),
                }
            )
    return pd.DataFrame(rows)


def implicitize(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    out = df.copy()
    if threshold is not None:
        out = out[out["rating"] >= float(threshold)]
    out = out.dropna(subset=["user_raw", "item_raw", "timestamp"])
    # collapse duplicate (user, item) keeping the latest timestamp
    out = (
        out.sort_values("timestamp")
        .groupby(["user_raw", "item_raw"], as_index=False)
        .agg({"timestamp": "max", "rating": "last"})
    )
    return out.reset_index(drop=True)


def iterative_kcore(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Iterate 5-core to a fixed point. Log per-iteration drop."""
    cur = df.copy()
    history: list[dict[str, int]] = []
    while True:
        u_counts = cur["user_raw"].value_counts()
        i_counts = cur["item_raw"].value_counts()
        keep_u = set(u_counts[u_counts >= k].index)
        keep_i = set(i_counts[i_counts >= k].index)
        nxt = cur[cur["user_raw"].isin(keep_u) & cur["item_raw"].isin(keep_i)]
        rec = {
            "users": int(cur["user_raw"].nunique()),
            "items": int(cur["item_raw"].nunique()),
            "rows": int(len(cur)),
            "users_after": int(nxt["user_raw"].nunique()),
            "items_after": int(nxt["item_raw"].nunique()),
            "rows_after": int(len(nxt)),
        }
        history.append(rec)
        if len(nxt) == len(cur):
            return nxt.reset_index(drop=True), history
        cur = nxt


def temporal_loo_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per user: last = test, second-last = val, rest = train. Timestamp ties → row order."""
    ordered = df.sort_values(["user_raw", "timestamp"], kind="mergesort").reset_index(drop=True)
    ordered["_pos"] = ordered.groupby("user_raw").cumcount(ascending=False)
    # _pos 0 = last, 1 = second last
    test = ordered[ordered["_pos"] == 0].drop(columns="_pos")
    val = ordered[ordered["_pos"] == 1].drop(columns="_pos")
    train = ordered[ordered["_pos"] >= 2].drop(columns="_pos")
    # users without 3 interactions should have been removed by 5-core + LOO needs ≥3
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def reindex(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """Users [0, U) from *train* users; items [0, I) from *train* items.

    Val/test edges whose user or item is unseen in train are dropped (they
    cannot enter hat_A and have no embedding).
    """
    users = sorted(train["user_raw"].unique().tolist())
    items = sorted(train["item_raw"].unique().tolist())
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: i for i, it in enumerate(items)}

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[frame["user_raw"].isin(u_map) & frame["item_raw"].isin(i_map)].copy()
        out["user_idx"] = out["user_raw"].map(u_map).astype(int)
        out["item_idx"] = out["item_raw"].map(i_map).astype(int)
        return out.reset_index(drop=True)

    return _apply(train), _apply(val), _apply(test), u_map, i_map


def assert_no_split_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    tr = set(zip(train["user_idx"], train["item_idx"], strict=False))
    va = set(zip(val["user_idx"], val["item_idx"], strict=False))
    te = set(zip(test["user_idx"], test["item_idx"], strict=False))
    if tr & va or tr & te:
        raise DataIntegrityError("train overlaps val/test — leakage")
    # val and test may theoretically collide only if duplicates survived; forbid
    if va & te:
        raise DataIntegrityError("val overlaps test")
