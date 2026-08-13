"""Centralized RNG. No hidden seeds."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _numpy_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state().cpu().numpy().tolist()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                t.cpu().numpy().tolist() for t in torch.cuda.get_rng_state_all()
            ]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        py = state["python"]
        # JSON turns tuples into lists
        if isinstance(py, list) and len(py) == 3:
            py = (py[0], tuple(py[1]), py[2])
        random.setstate(py)
    if "numpy" in state:
        np.random.set_state(_numpy_state_from_json(state["numpy"]))
    if torch is not None and "torch" in state:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]]
            )


def _numpy_state() -> dict[str, Any]:
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "kind": kind,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached": float(cached),
    }


def _numpy_state_from_json(obj: Any) -> tuple[Any, ...]:
    if isinstance(obj, dict):
        return (
            obj["kind"],
            np.array(obj["keys"], dtype=np.uint32),
            obj["pos"],
            obj["has_gauss"],
            obj["cached"],
        )
    return obj
