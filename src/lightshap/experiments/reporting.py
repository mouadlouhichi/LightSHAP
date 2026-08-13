"""Tables, figures, and the human-readable summary. No fabricated numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lightshap.utils.io import atomic_write_json, atomic_write_text, ensure_dir


def write_tables(run_dir: Path, payload: dict[str, Any]) -> list[str]:
    tdir = ensure_dir(run_dir / "tables")
    written: list[str] = []
    for name in ("rq1", "rq2", "rq3", "rq4", "bpr", "shapley", "datasets"):
        if name in payload:
            path = tdir / f"{name}.json"
            atomic_write_json(path, payload[name])
            written.append(str(path))
    return written


def write_figures(run_dir: Path, payload: dict[str, Any]) -> list[str]:
    fdir = ensure_dir(run_dir / "figures")
    written: list[str] = []
    # Figure: v(C) 8-bar if present
    rq1 = payload.get("rq1") or {}
    for ds, rec in rq1.items() if isinstance(rq1, dict) else []:
        vecs = rec.get("v_C_vectors") if isinstance(rec, dict) else None
        if not vecs:
            continue
        mean = [sum(col) / len(col) for col in zip(*vecs, strict=False)]
        labels = rec.get("coalition_order") or [str(i) for i in range(len(mean))]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(range(len(mean)), mean, color="#3b6d9a")
        ax.set_xticks(range(len(mean)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("v(C)  (val NDCG@10 uplift)")
        ax.set_title(f"RQ1 coalition values — {ds}")
        fig.tight_layout()
        path = fdir / f"rq1_vC_{ds}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    # Figure: φ bars
    shap = payload.get("shapley") or {}
    if isinstance(shap, dict):
        for key, rec in shap.items():
            rows = rec if isinstance(rec, list) else [rec]
            for rec_i in rows:
                if not isinstance(rec_i, dict) or "phi" not in rec_i:
                    continue
                phi = rec_i["phi"]
                seed = rec_i.get("seed", "mean")
                fig, ax = plt.subplots(figsize=(4, 3))
                ks = ["1", "2", "3"]
                ax.bar(ks, [float(phi.get(k, 0.0)) for k in ks], color="#c47b2b")
                ax.axhline(0.0, color="black", linewidth=0.6)
                ax.set_xlabel("extra hop")
                ax.set_ylabel("φ")
                ax.set_title(f"Game A Shapley — {key} seed={seed}")
                fig.tight_layout()
                path = fdir / f"phi_{key}_seed{seed}.png"
                fig.savefig(path, dpi=120)
                plt.close(fig)
                written.append(str(path))
    return written


def write_summary_md(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "# LightShap run summary",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- profile: `{summary.get('profile')}`",
        f"- status: `{summary.get('status')}`",
        "",
        "## Stages",
        "",
    ]
    for name, rec in (summary.get("stages") or {}).items():
        lines.append(f"- **{name}**: {rec.get('state')} ({rec.get('status', '')})")
    lines += ["", "## Warnings", ""]
    warns = summary.get("warnings") or []
    if not warns:
        lines.append("_none_")
    else:
        lines.extend(f"- {w}" for w in warns)
    lines += [
        "",
        "## Scientific reminders",
        "",
        "- Flag is not a prune claim.",
        "- Skip-pattern coalitions are scoring rules on a frozen cache.",
        "- Never write “2018 All-Beauty”.",
        "- I_12 = 1/2 (the claim I_12 = 1 is wrong).",
        "",
    ]
    path = run_dir / "summary.md"
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path
