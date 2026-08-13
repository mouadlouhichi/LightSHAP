# LightShap

Attribution of **inference-time hop inclusion** in one frozen LightGCN fusion
(He et al. 2020). Skip-pattern coalitions are scoring rules on a cached
`E0..E3` table, not independently trained depth architectures. Attribution is
licensed **only through the Shapley average**.

This repository is the implementation of
`specs/LightShap_Implementation_Spec (5).md`. The earlier `specs/start/`
draft is **not** the source of truth.

## Estimand (boxed)

We do **not** attribute architectures. We attribute inference-time inclusion
of extra hops `{1,2,3}` in one frozen uniform-`K=3`, `d=64` checkpoint. `E0`
is always kept. `v` is **validation** NDCG@10 uplift over LightGCN-`E0`
(selection-biased; owned). The official flag is **not** a prune license;
retrained `K=1,2,3,4` is the architecture control.

Frozen numbers that the code will refuse to silently change under
`profile: scientific`:

| Item | Value |
|---|---|
| LightGCN | `K=3`, `d=64`, `lr=1e-3`, `reg=1e-4`, patience 20 |
| BPR-MF | the **six** listed `(d, lr, reg)` tuples |
| Seeds | 5, then **average users** and bootstrap users (never `5\|U\|` iid) |
| Stability floor | `0.005` (flag UNDEFINED, Adaptive E0-only, RQ3 Δ gate) |
| Flag | `I_ab<0` and three relative `0.15` clauses |
| Val-α | **286** simplex points; tie: closest to uniform, then lex |
| Top-K | lex `(score desc, item_id asc)` in float32 — **no** `1e-12` jitter |
| Beauty | **2014** `reviews_Beauty_5.json.gz`. Never “2018 All-Beauty”. |
| Fixture | `φ=(5/3, 8/3, 11/3)`, `I_ij=1/2` (`I_12=1` is wrong) |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

CPU PyTorch is enough. No PyTorch Geometric.

### Apple Silicon (M4 Pro)

`device: auto` in the yaml files selects **MPS**. Sparse `hat_A` is densified
on MPS (ML-1M ~0.4 GB, Beauty ~5 GB — fine on 48 GB). First-time data fetch:

```bash
python scripts/fetch_data.py
# paste the printed SHA256s into configs/ml1m.yaml and configs/beauty.yaml
```

Then open `notebooks/run_all.ipynb`, set `PROFILE = "synthetic"` for a smoke
run, or `PROFILE = "experiment"` for the scientific ML-1M + Beauty pipeline.

## CLI

```bash
# synthetic end-to-end (cheap, deterministic)
python scripts/run_experiment.py --config configs/synthetic_fast.yaml

# resume after a crash
python scripts/run_experiment.py --config configs/synthetic_fast.yaml --resume

# status
python scripts/run_experiment.py --run-id RUN_ID --status

# validate config + frozen invariants
python scripts/run_experiment.py --config configs/experiment.yaml --validate

# BPR negative-pool sensitivity toggle (must exist in the tagged commit)
python scripts/run_experiment.py --config configs/synthetic.yaml --neg-pool train_only

# stage window
python scripts/run_experiment.py --config configs/synthetic_fast.yaml \
    --from-stage RQ1 --until-stage RQ4

# fixtures oracle
python scripts/gen_shapley_fixtures.py

# cache
python scripts/run_experiment.py --clean-cache
```

Equivalent module form: `python -m lightshap.cli ...`

## Scientific run (do not skip the tag order)

1. Download MovieLens-1M and Amazon Beauty **2014** 5-core.
2. Compute SHA256, paste into `configs/ml1m.yaml` and `configs/beauty.yaml`,
   **mirror the gz on OSF/Zenodo**.
3. `python scripts/gen_shapley_fixtures.py` and the A.0 tests must be green
   (including the `--neg-pool` toggle).
4. Signed git tag + OSF deposit of the spec pair + `tests/` + configs.
5. **Then** `python scripts/run_experiment.py --config configs/experiment.yaml`.

Never fill a hash during the science run. If Beauty `|U|<1000` after 5-core,
every Beauty RQ3/RQ4 claim is **exploratory** and confirmatory RQ3 is ML-1M.

## Tests

```bash
pytest
pytest -m unit
pytest -m integration
pytest -m resume
```

Gates, in order: static validation → unit → fixtures → synthetic e2e →
checkpoint/resume → real-data integrity → smoke → full experiment.

Real ML-1M / Beauty training is **not** fabricated when data are absent.
Those stages stay `NOT_EXECUTED` in `results/spec_compliance.json`.

## Notebook

`notebooks/run_all.ipynb` is the user-facing orchestrator. It calls the same
Python implementation as the CLI (no duplicated science).

## Layout

```
src/lightshap/     scientific primitives, experiments, pipeline
configs/           frozen yaml (bpr_grid, ml1m, beauty, experiment, synthetic)
scripts/           run_experiment.py, gen_shapley_fixtures.py
tests/             unit / integration / regression / resume
docs/implementation_matrix.md
results/spec_compliance.json
notebooks/run_all.ipynb
```

## Engineering fills (documented, not silent)

The polish spec leaves three formulae underspecified. They are implemented
exactly as recorded in `docs/implementation_matrix.md` and `src/lightshap/constants.py`:

1. **R_ab** — max over C ⊆ L\{a,b} of the leftover marginals; grand-coalition
   slice = LOO.
2. **Adaptive q** — map 3-vector φ to 4 weights, mix with uniform at β=1/4.
3. **RQ4 Holm contrasts** — Adaptive vs uniform, Val-α vs uniform, Adaptive vs Val-α.
4. **Game B zero** — empty scores are the zero vector (rank by item_id).

## What this is not

- A new GNN.
- A license to drop hops at **training** time.
- A literature correction of He et al.
- A claim that skip patterns are models.
