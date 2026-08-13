"""Honest spec-compliance report. Never fabricate PASS."""

from __future__ import annotations

from typing import Any, Literal

Status = Literal["PASS", "FAIL", "NOT_EXECUTED", "NOT_APPLICABLE"]


REQUIREMENTS: list[dict[str, str]] = [
    {"id": "FIX.PHI", "section": "A.0/A.8", "name": "fixture phi = (5/3, 8/3, 11/3)"},
    {"id": "FIX.I", "section": "A.0/A.8", "name": "fixture I_ij = 1/2"},
    {"id": "FIX.I12_NOT_1", "section": "A.0", "name": "I_12 = 1 is rejected"},
    {"id": "FIX.PROP2", "section": "A.0", "name": "both Prop. 2 gadgets"},
    {"id": "BPR.GRID6", "section": "A.5", "name": "exactly 6 listed BPR tuples"},
    {"id": "BPR.NEG_TOGGLE", "section": "A.5", "name": "--neg-pool heldout_excluded|train_only"},
    {"id": "LGCN.K3D64", "section": "frozen", "name": "LightGCN K=3 d=64 lr=1e-3 reg=1e-4 patience=20"},
    {"id": "RANK.LEX", "section": "A.6", "name": "lex Top-K score desc, item_id asc, no jitter"},
    {"id": "SCORE.GEMM", "section": "A.6", "name": "GEMM + dense mask + item I-1"},
    {"id": "GAMEA.3P", "section": "frozen", "name": "Game A players {1,2,3}, E0 always on, 8 coalitions"},
    {"id": "GAMEA.MODEA", "section": "frozen", "name": "Mode A mask-and-mean"},
    {"id": "V.VAL", "section": "frozen", "name": "v = val NDCG@10 uplift over E0"},
    {"id": "FLAG.FLOOR", "section": "A.11", "name": "flag UNDEFINED if v(L)<0.005"},
    {"id": "FLAG.CLAUSES", "section": "A.11", "name": "I<0 and three 0.15 relative clauses"},
    {"id": "FLAG.NOT_PRUNE", "section": "A.11", "name": "flag is not a prune claim"},
    {"id": "COSINE.FULL", "section": "A.11", "name": "cosine: full side, row-norm, flattened, no subsample"},
    {"id": "GAMEB.TRIGGER", "section": "A.11", "name": "Game B strict unique-max + 0.05 same-side"},
    {"id": "VALALPHA.286", "section": "A.12", "name": "exactly 286 val-alpha candidates"},
    {"id": "VALALPHA.TIE", "section": "A.12", "name": "tie: closest to uniform then lex"},
    {"id": "ADAPT.FLOOR", "section": "A.10", "name": "adaptive E0-only if v_m<0.005"},
    {"id": "ADAPT.BETA", "section": "A.10", "name": "beta=1/4, no beta_m in RQ4"},
    {"id": "RQ3.DELTA", "section": "RQ3", "name": "Δ only if both groups v_m>=0.005"},
    {"id": "RQ3.NEG_D", "section": "RQ3", "name": "negative D_m = deep hops hurt"},
    {"id": "Q.ONCE", "section": "frozen", "name": "quartile cuts once from train degrees"},
    {"id": "SEEDS.5", "section": "frozen", "name": "five scientific seeds"},
    {"id": "STAT.USER", "section": "frozen", "name": "average users across seeds then bootstrap users"},
    {"id": "STAT.NOT_5U", "section": "frozen", "name": "never treat 5|U| as iid"},
    {"id": "RQ1.SPEARMAN_V", "section": "RQ1", "name": "Spearman on 8-vector v(C), average ranks"},
    {"id": "RQ1.NO_SPEARMAN_PHI", "section": "RQ1", "name": "no Spearman on phi"},
    {"id": "RQ2.RETRAIN_K", "section": "RQ2", "name": "retrained K=1..4 in RQ2"},
    {"id": "RQ4.WILCOXON", "section": "RQ4", "name": "two-sided Wilcoxon"},
    {"id": "RQ4.HOLM", "section": "RQ4", "name": "Holm within dataset, 3 tests, no pooling"},
    {"id": "BEAUTY.2014", "section": "A.3", "name": "Amazon Beauty 2014 5-core"},
    {"id": "BEAUTY.FLOOR", "section": "A.3", "name": "|U|<1000 exploratory; RQ3 confirmatory=ML-1M"},
    {"id": "DATA.SHA", "section": "A.0", "name": "SHA256 + mirror before scientific tag"},
    {"id": "RESUME", "section": "infra", "name": "stage/seed/epoch resume"},
    {"id": "NO_CORR_N3", "section": "3.4", "name": "flag sub-conditions: co-occurrence, not r"},
]


def blank_compliance() -> dict[str, Any]:
    return {
        req["id"]: {
            "id": req["id"],
            "section": req["section"],
            "name": req["name"],
            "status": "NOT_EXECUTED",
            "evidence": None,
        }
        for req in REQUIREMENTS
    }


def set_status(
    report: dict[str, Any],
    req_id: str,
    status: Status,
    evidence: Any = None,
) -> None:
    if req_id not in report:
        raise KeyError(req_id)
    # never upgrade FAIL to PASS silently; never invent PASS without evidence
    prev = report[req_id]["status"]
    if prev == "FAIL" and status == "PASS":
        return
    report[req_id]["status"] = status
    if evidence is not None:
        report[req_id]["evidence"] = evidence


def apply_static_passes(report: dict[str, Any]) -> None:
    """Things that are true by construction of this codebase (unit-tested)."""
    # These stay NOT_EXECUTED until the corresponding tests/stages actually run.
    # The test suite / runner sets PASS with evidence.
    return
