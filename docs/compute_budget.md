# Compute budget vs scientific gain (Q1-facing)

## What the paper actually needs

The boxed estimand is **one frozen** uniform `K=3`, `d=64` LightGCN.
Game A is **exact Shapley on 8 coalitions** (seconds–minutes after the
cache exists). Monte Carlo is not used and would be wasted energy.

The expensive work is ordinary RecSys **training**, not attribution.

| Work | Why it exists | Necessary for the estimand? |
|---|---|---|
| LightGCN `K=3` × 5 seeds × 2 graphs | the frozen checkpoint | **yes** |
| Exact 8-coalition Game A + SII/flag | the paper | **yes**, cheap |
| RQ3 quartiles / RQ4 Adaptive+Val-α | confirmatory stats | yes, cheap if eval is vectorized |
| BPR 6-point grid × 5 seeds | listed-effort baseline, *not* equal community prior | **control**, not the game |
| Retrained `K=1,2,3,4` | RQ2 “flag ≠ prune / masking ≠ architecture” | **control** |

A reviewer who asks “why a week of GPU?” should be answered:

> We do not search LightGCN. Attribution is an 8-forward-pass exact game
> on a cached fusion. Wall-clock is dominated by reproducing the standard
> BPR/LightGCN trainers used as controls. The Shapley step is not the
> energy cost.

## What we changed (engineering, not a freeze)

1. **Rejection-sampled negatives** — same uniform-over-allowed law as the
   spec’s `--neg-pool` toggle, without building an O(I) Python list per
   triple. This was the ~2 min/epoch sink on ML-1M.
2. **`eval_every: 5`** — early-stop metric is still val NDCG@10;
   **patience is still 20 epochs** without improvement. Reported numbers
   remain full-catalog. Evaluating every epoch is not a scientific freeze.
3. **Checkpoint on evals, not every epoch** — less disk I/O.
4. **Dense `hat_A` cached once on MPS** — do not densify every batch.
5. **LightGCN val fusion computed once per eval**, not once per user batch.

## Recommended run order

1. `configs/experiment_core.yaml` — frozen LightGCN + Game A + RQ1/RQ3/RQ4.
   This is the paper’s confirmatory spine. Overnight on an M4 / T4.
2. `configs/experiment.yaml` — add BPR 6-grid + retrained K for RQ2.
   Resume-safe; does not recompute a finished core if you use a new run id
   only for the control wave (or keep one run and leave core artifacts).

Do **not** Monte Carlo Shapley to “save” compute. That would spend energy
on a noisier estimator of an 8-point game.

## What not to cut

- 5 seeds (user-average then bootstrap; never `5|U|` iid)
- Full-catalog **reported** NDCG
- Exact 8 coalitions
- The 6 listed BPR tuples *when* the BPR control is run
- Retrained K *when* the prune-control table is run
