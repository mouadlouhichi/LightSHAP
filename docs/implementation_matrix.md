# LightShap implementation matrix

Authoritative science: `specs/LightShap_Implementation_Spec (5).md`
(and the companion paper freeze `specs/LightShap_Paper_Structure (6).md`).
`specs/start/` is the retracted earlier draft and is **not** implemented.

Status values used in `results/spec_compliance.json`:
`PASS | FAIL | NOT_EXECUTED | NOT_APPLICABLE`.

---

## Estimand

| ID | Section | Meaning | Module | Inputs | Outputs | Tests | Checkpoint |
|---|---|---|---|---|---|---|---|
| EST.BOX | boxed estimand | Inference-time extra-hop inclusion on one frozen uniform K=3,d=64 LightGCN. Skip patterns are cache scoring rules. Attribution only via Shapley average. | `shapley/`, `scoring/fusion.py` | cached E0..E3 | Game A v, φ | `test_fusion.py`, `test_shapley.py` | shapley JSON |
| EST.NOT_ARCH | boxed / RQ2 | Do not treat skip patterns as trained depths. | `experiments/rq2.py` | frozen {1} vs retrained K=1 | comparison row | integration | rq2.json |

## Fixtures

| ID | Section | Meaning | Module | Tests |
|---|---|---|---|---|
| FIX.PHI | A.0 / A.8 | φ=(5/3,8/3,11/3) | `shapley/fixtures.py`, `scripts/gen_shapley_fixtures.py` | `test_fixtures.py` |
| FIX.I | A.0 / A.8 | I_12=I_13=I_23=1/2 | same | `test_fixtures.py` |
| FIX.I12_NOT_1 | A.0 | I_12=1 is a retracted wrong claim | fixtures + regression | `test_frozen_invariants.py` |
| FIX.PROP2 | A.0 | Two substitution gadgets; LOO=0 and φ_a=φ_b; n=3 half-split is not general | fixtures | `test_fixtures.py` |

## Data

| ID | Section | Meaning | Module | Outputs |
|---|---|---|---|---|
| DATA.ML1M | A.3 | GroupLens ml-1m.zip, rating≥4, 5-core, temporal LOO | `data/` | processed bundle |
| DATA.BEAUTY | A.3 | 2014 `reviews_Beauty_5.json.gz`. Never “2018 All-Beauty”. | `data/`, `gates/beauty.py` | bundle + status |
| DATA.SHA | A.0 / A.3 | SHA256 pasted before tag; mirror bytes; refuse scientific run if missing | `data/download.py` | download manifest |
| DATA.KCORE | A.3 | Iterative 5-core to a fixed point | `data/preprocess.py` | core_history |
| DATA.SPLIT | A.3 | Last=test, 2nd-last=val, rest=train; graph from train only | `data/preprocess.py` | train/val/test |
| DATA.INDEX | A.3 | Users [0,U), items [U,U+I) / item ids 0..I-1 | `data/bundle.py`, `scoring/fusion.py` | aligned indices |
| DATA.Q | frozen | Quartile cuts **once** from frozen train degrees, reused across seeds | `data/stats.py` | QuartileCuts |
| DATA.BEAUTY_FLOOR | A.3 | \|U\|<1000 ⇒ exploratory; RQ3 confirmatory = ML-1M; abstract template B | `gates/beauty.py` | beauty_status |

## Models

| ID | Section | Meaning | Module |
|---|---|---|---|
| LGCN.FREEZE | frozen | K=3,d=64,lr=1e-3,reg=1e-4,patience=20,max 1000. Not searched. | `models/lightgcn.py`, `config.py` |
| LGCN.CACHE | A.5 / 3.2 | Train once with uniform mean; cache E0..E3; coalitions mean subsets | `models/lightgcn.py`, `experiments/lightgcn_run.py` |
| BPR.GRID6 | A.5 | Exactly the 6 listed (d,lr,reg) tuples; select by val NDCG@10 | `configs/bpr_grid.yaml`, `experiments/bpr_grid.py` |
| BPR.NEG | A.5 | `--neg-pool {heldout_excluded,train_only}` exists in the tagged trainer | `models/negatives.py`, `cli.py` |
| RQ2.K | RQ2 | Retrained K=1,2,3,4 as architecture control | `experiments/lightgcn_run.py`, `rq2.py` |

## Scoring

| ID | Section | Meaning | Module |
|---|---|---|---|
| SCORE.GEMM | A.6 | GEMM user×item + dense [U,I] mask + item index I-1 | `scoring/fusion.py`, `scoring/mask.py` |
| RANK.LEX | A.6 | `lexsort((item_id, -score))`; float32; **no** 1e-12 jitter | `scoring/rank.py` |
| MASK.VAL | 4.2 | Val uses train-mask | `data/bundle.py` |
| MASK.TEST | 4.2 | Test uses train+val mask | `data/bundle.py` |
| FUSE.MODEA | frozen | Mode A = mask-and-mean. Game B ≠ Mode B. | `scoring/fusion.py` |

## Game A / flag / SII

| ID | Section | Meaning | Module | Engineering fill? |
|---|---|---|---|---|
| GAMEA.3P | frozen | Players {1,2,3}, E0 always on, 8 coalitions | `shapley/exact.py`, `experiments/shapley_run.py` | no |
| V.VAL | frozen | v(C)=val NDCG@10(C)−val NDCG@10(E0); selection-biased, owned | `shapley_run.py` | no |
| SHAP.EXACT | A.8 | Exact Shapley + efficiency | `shapley/exact.py` | no |
| SII.3x3 | A.11 | 2-SII on Game A pairs | `shapley/interactions.py` | no |
| LOO | A.11 | LOO_i=v(N)−v(N\{i}) | `interactions.py` | no |
| RAB | A.11 | Substitution residual | `interactions.py` | **yes** — see below |
| FLAG | A.11 | UNDEFINED if v(L)<0.005; else I<0 and three 0.15 clauses | `shapley/flag.py` | no |
| FLAG.COOCCUR | 3.4 | Co-occurrence table of 4 clauses; **not** a correlation | `flag.py` | no |
| FLAG.NOT_PRUNE | A.11 / RQ2 | Flag is not a training-prune license | `rq2.py` | no |

**R_ab (engineering fill).** The polish spec uses R_ab but does not write the formula. It says the LOO clause is “a redundant-by-design restatement of the grand-coalition slice of R_ab”. Implemented:

```
R_ab = max_{C ⊆ L\{a,b}} max(|v(C∪{a,b})−v(C∪a)|, |v(C∪{a,b})−v(C∪b)|)
```

The C=L\{a,b} term is exactly max(|LOO_a|,|LOO_b|).

## Cosine / Game B

| ID | Section | Meaning | Module |
|---|---|---|---|
| COS.4x4 | A.11 | Similarity is 4×4 over {E0..E3} | `gates/cosine.py` |
| COS.RULE | A.11 | Side-specific; drop train-degree 0; no subsample; row ℓ2; flattened cosine; no RNG | `gates/cosine.py` |
| GAMEB.TRIG | A.11 | No Game-A flag AND some side has unique off-diag max at (E0,E2) with ≥0.05 vs same-side second. Ties do not fire. | `gates/game_b.py` |
| GAMEB.GAME | 3.3 | All-slot {0,1,2,3}, different zero, credits incomparable | `shapley_run.game_b_values` |

**Game B zero (engineering fill).** Empty coalition scores are the zero vector; lex ranking then follows item_id. v_B(C)=NDCG(mean of selected slots)−NDCG(zeros).

## Adaptive / Val-α / RQ3–RQ4

| ID | Section | Meaning | Module | Engineering fill? |
|---|---|---|---|---|
| ADAPT.FLOOR | A.10 | v_m<0.005 → q=(1,0,0,0) | `gates/adaptive.py` | no |
| ADAPT.BETA | A.10 | β=1/4, no β_m in RQ4 | `adaptive.py`, `rq4.py` | q-map yes |
| VALA.286 | A.12 | Exactly 286 candidates w_i=k/10, Σw=1 | `gates/val_alpha.py` | no |
| VALA.TIE | A.12 | max val NDCG → closest to uniform L2 → lex (w0,w1,w2) | `val_alpha.py` | no |
| RQ1 | 4.3 | v(L); Spearman of 8-vector (average ranks); φ order+signs+cosine; no Spearman on φ | `experiments/rq1.py` | no |
| RQ2 | 4.4 | flag, LOO vs φ, retrained K, frozen {1} vs K=1 | `rq2.py` | no |
| RQ3 | 4.5 | Δ=D_Q4−D_Q1 iff both v_m≥0.005; neg D = deep hops hurt | `gates/rq3.py` | no |
| RQ4 | 4.6 / A.12 | Transfer check; two-sided Wilcoxon; Holm within dataset (3 tests); no pooling | `rq4.py` | contrast names yes |
| STAT.UNIT | frozen | Average each user across 5 seeds, then bootstrap **users**. Never 5\|U\| iid. | `statistics/` | no |

**Adaptive 4-vector (engineering fill).** Game A φ is a 3-vector. Mapping to 4 fusion weights:

- relu(φ_k) for k=1,2,3; attr[0]=mean(relu(φ)); if all relu=0 then (1,0,0,0); L1-normalize;
- q=(1−β)·uniform_4 + β·attr; renormalize.

**RQ4 Holm contrasts (engineering fill).** Spec says “3 tests” without naming them. Implemented:

1. `adaptive_vs_uniform`
2. `val_alpha_vs_uniform`
3. `adaptive_vs_val_alpha`

## Infrastructure

| ID | Meaning | Module |
|---|---|---|
| PIPE.STAGES | Explicit DAG, refuse missing deps | `pipeline/stages.py`, `runner.py` |
| PIPE.RESUME | Stage / seed / epoch resume; invalid artifacts recomputed | `runner.py`, `models/train.py` |
| PIPE.CKPT | Atomic writes; hash metadata; compatibility check | `checkpoints/store.py` |
| PIPE.LOG | Structured run.log + events.jsonl + metrics.jsonl | `logging/structured.py` |
| PIPE.PROV | git, packages, device, config hash, dataset hash | `utils/provenance.py` |
| PIPE.CACHE | Hash-aware cache | `cache/store.py` |
| CLI | run, resume, status, validate, clean-cache, from-stage, until-stage, --neg-pool | `cli.py`, `scripts/run_experiment.py` |
| NB | `notebooks/run_all.ipynb` calls the real implementation | notebook |

## Pipeline DAG

```
ENVIRONMENT → CONFIG_VALIDATION → DATA_DOWNLOAD → DATA_VALIDATION → DATA_PREPROCESSING
CONFIG_VALIDATION → FIXTURE_GENERATION → FIXTURE_TESTS
DATA_PREPROCESSING + FIXTURE_TESTS → BPR_GRID
DATA_PREPROCESSING + FIXTURE_TESTS → LIGHTGCN_TRAINING → SHAPLEY_COMPUTATION
SHAPLEY → RQ1, RQ2, RQ3, RQ4 → STATISTICS → TABLE_GENERATION
                                              → FIGURE_GENERATION → FINAL_REPORT
```

## What is NOT implemented (on purpose)

- `specs/start/` 4-player architecture game, 2018 Beauty, Shapley-as-improvement claims, Spearman on φ.
- Filling SHA256 during a science run.
- Fabricating real-data metrics. Real ML-1M / Beauty stages stay `NOT_EXECUTED` until data+hashes exist.
