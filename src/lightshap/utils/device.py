"""Resolve cpu / cuda / mps. ``auto`` prefers MPS (Apple Silicon) then CUDA."""

from __future__ import annotations

from typing import Any


def resolve_device(requested: str = "auto") -> dict[str, Any]:
    req = (requested or "auto").lower()
    cuda = False
    mps = False
    gpu_name = None
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
        elif mps:
            gpu_name = "Apple MPS"
    except Exception:
        pass

    if req == "auto":
        if mps:
            device = "mps"
        elif cuda:
            device = "cuda"
        else:
            device = "cpu"
    elif req in {"cpu", "cuda", "mps"}:
        if req == "cuda" and not cuda:
            device = "cpu"
        elif req == "mps" and not mps:
            device = "cpu"
        else:
            device = req
    else:
        device = "cpu"

    return {
        "requested": requested,
        "device": device,
        "cuda_available": cuda,
        "mps_available": mps,
        "gpu_name": gpu_name,
    }
