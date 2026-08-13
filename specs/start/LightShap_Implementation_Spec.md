# LightShap — Technical Implementation Specification and Registered Predictions

**Companion to:** `LightShap_Paper_Structure.md` (the paper blueprint). That file says *what the paper argues*; this file says *what to build and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` and clustering/quality diagnostics from `ActionShap/code/` and `SignalShap/code/` port over with essentially no change. LightGCN backbone code is public PyTorch. Nothing from DyHuCoG is used.

---

# PART A — IMPLEMENTATION

## A.1 Repository layout

```
LightShap/code/
├── requirements.txt
├── configs/
│   ├── ml1m.yaml
│   └── beauty.yaml
├── lightshap/
│   ├── __init__.py
│   ├── data.py              # loaders, 5-core filtering, temporal split, graph building
│   ├── graph.py             # normalized adjacency hat_A, sparse helpers
│   ├── backbone.py          # LightGCN propagation, caching E0..EK
│   ├── fusion.py            # layer-weight fusion head, coalition masking
│   ├── normalize.py         # per-user per-layer z-normalization + sigma=0 guard
│   ├── game.py              # characteristic function v(C), exact Shapley, per-user
│   ├── segments.py          # behavioural (degree) + attribution segmentation
│   ├── adaptive.py          # LightShap-Weight, LightShap-Adaptive
│   ├── baselines.py         # uniform, LOO, attention-over-layers, JK-Net
│   ├── metrics.py           # NDCG@K, Recall@K, HR@K, MRR, coverage
│   ├── stats.py             # PORTED from ActionShap/SignalShap
│   └── report.py            # LaTeX table + figure emitters
├── scripts/
│   ├── build_cache.py       # stages 1-2: data + graph + cached E^k
│   ├── run_game.py          # stages 3-5: coalitions, Shapley, per-user
│   ├── run_segments.py      # stages 6-7: segments + Weight/Adaptive
│   └── run_all.py
├── tests/
└── results/{raw,tables,figures}/
```

## A.2 Environment

```
python = 3.12
torch >= 2.4  (CPU build sufficient)
numpy >= 2.4, < 2.5          # same pin as SignalShap/SHAPER; avoids macOS Accelerate segfault
scipy >= 1.18  (scipy.sparse for hat_A)
scikit-learn >= 1.6
pandas >= 2.2
matplotlib >= 3.9
pyyaml, tqdm, pytest
```

No GPU. No PyTorch Geometric needed — LightGCN uses only `torch.sparse.mm` and `scipy.sparse`. Everything runs on a laptop CPU — this is a claim the paper makes, so keep it true. Pin `torch` + `numpy` together; mismatched builds reproduce the SignalShap segfault.

## A.3 Data layer (`data.py`)

| | MovieLens-1M | Amazon-Beauty |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | Amazon Reviews 2018, Beauty 5-core |
| Raw interactions | 1,000,209 | ~198,502 |
| Implicit conversion | rating ≥ 4 → positive | all reviews → positive |
| Filtering | 5-core, **iterative to convergence** | 5-core, iterative |
| Graph | Bipartite user–item, undirected, no self-loops | same |

Implementation requirements:

1. **Iterate 5-core to fixed point.** Loop until users and items stable, then record final counts for Table 2 — never pre-filter numbers. Beauty loses more mass than ML-1M; log per-iteration drop.
2. **Temporal leave-one-out split per user.** Sort by timestamp; last = test, second-last = validation, rest = train. Break ties by row order. Build graph **only from train** — test/validation edges must not enter `hat_A`. This is the most common leakage in GNN RecSys and a reviewer will check.
3. **Freeze splits + graph to disk** with a config hash, so every downstream stage reads identical splits and identical `hat_A`. Re-deriving per stage is how silent inconsistencies enter.
4. Emit `DatasetStats`: users, items, interactions, density, mean/median degree per user and per item, degree quantile boundaries for segmentation, and graph sparsity (`nnz / (U+I)^2`).
5. **Index alignment.** LightGCN stacks users `[0, U)` and items `[U, U+I)` into one node list. Keep this ordering fixed everywhere; a single off-by-`U` offset silently swaps users and items.

## A.4 Graph construction (`graph.py`)

Build once, reuse for all coalitions:

```python
import scipy.sparse as sp
import numpy as np
import torch

def build_hat_A(train_df, n_users, n_items):
    """
    train_df: DataFrame with columns user_idx in [0,U), item_idx in [0,I)
    Returns: torch.sparse_coo_tensor hat_A of shape (U+I, U+I), symmetric, normalized
    """
    U, I = n_users, n_items
    N = U + I
    # Edges both directions: user->item and item->user
    rows = np.concatenate([train_df.user_idx.values, train_df.item_idx.values + U])
    cols = np.concatenate([train_df.item_idx.values + U, train_df.user_idx.values])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    # Symmetric normalization: hat_A = D^-0.5 A D^-0.5
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    hat_A = D_inv_sqrt @ A @ D_inv_sqrt
    hat_A = hat_A.tocoo()
    # To torch sparse
    indices = torch.tensor(np.vstack([hat_A.row, hat_A.col]), dtype=torch.long)
    values = torch.tensor(hat_A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
```

Log: `N`, `nnz`, degree histogram (especially `deg==0` isolated nodes after filtering — they exist on Beauty and should stay, with `deg_inv_sqrt=0`).

## A.5 Backbone and layer cache (`backbone.py`)

Standard LightGCN, no tricks:

- Embedding: `nn.Embedding(N, d)` with `d=64` main, `d=32,128` as ablations, `N=U+I`. Xavier uniform init.
- Propagation: `E^{k+1} = hat_A @ E^{k}` via `torch.sparse.mm`, no `W`, no `σ`. Do `K=3` steps, cache `E0, E1, E2, E3` as dense tensors `[(N,d) x4]`.
- Training: BPR loss over `(user, pos_item, neg_item)` triples sampled from train. `neg` sampled uniformly from items not in user's train set. `lr=1e-3` (Adam), `reg=1e-4`, `batch=2048` (ML-1M) / `1024` (Beauty), `epochs=1000` with early stopping on validation NDCG@10 (patience 20).

Crucial split:

1.  **Grand coalition training:** Train once with uniform fusion `E = mean(E0..E3)`, cache all four `E^k`. This costs the only heavy training in the paper.
2.  **Coalition evaluation:** For coalition `C`, compute `E_C = mean_{k in C} E^k` (or weighted, see A.7) **using cached matrices only** — no re-propagation, no re-embedding. This is what makes 16 coalitions trivial.

```python
# Pseudo: after grand coalition converges
model.eval()
with torch.no_grad():
    E0 = model.embedding.weight  # (N,d)
    Es = [E0]
    for k in range(1, K+1):
        Es.append(torch.sparse.mm(hat_A, Es[-1]))
    torch.save(Es, "cache/ml1m_Es.pt")  # list of 4 tensors
```

Pin the padding: no padding node exists in LightGCN (unlike sequence models), so no mask is needed.

## A.6 Fusion and coalition masking (`fusion.py`)

Two modes, pre-committed in paper:

**Mode A — Mask-and-mean (main):** For coalition `C`, `E_C = (1/|C|) sum_{k in C} E^k`. No learned weights — the value `v(C)` is directly the ranking quality of that mean.

**Mode B — Mask-and-refit-weights (robustness in appendix):** For coalition `C`, learn weights `w_k` for `k in C` via a tiny convex fit: `E_C = sum_{k in C} w_k E^k`, where `w` is fitted by logistic BPR over cached scores (same fusion trick as SignalShap). Mode B can only improve over Mode A, so Mode A is the honest baseline.

Implementation: dropping layers vs zeroing their weight is mathematically equivalent under L2 (separable penalty) — verified numerically for SignalShap. Drop columns to keep matrix small.

```python
def score_for_coalition(Es, coalition, user_idx, item_idx):
    """
    Es: list of tensors [E0, E1, E2, E3], each (N,d)
    coalition: tuple of ints e.g. (0,2)
    Returns: scores (n_pairs,) = dot(E_C[user], E_C[item])
    """
    if len(coalition) == 0:
        return None  # v(empty)=0 via pi0
    E_C = torch.stack([Es[k] for k in coalition], dim=0).mean(dim=0)  # (N,d)
    return (E_C[user_idx] * E_C[item_idx]).sum(dim=1)
```

For `C` containing only `{0}`, score is BPR-MF (`E0` only) — that's `pi0`.

## A.7 Normalization (`normalize.py`)

Per-user, per-layer, across that user's candidate items — mirrors SignalShap's per-user per-source z-norm. Needed only if scoring head uses per-layer scores before fusion; if fusion is over embeddings directly (dot then mean), normalization is not needed. Pre-commit to **mean-of-embeddings** fusion (no per-layer normalization) in main text, and report per-score normalization as an ablation.

If normalization is used, guard `sigma=0` (isolated node's deeper layers are zero after propagation) by returning zeros, not `eps`-noised values. Log degenerate rate per layer per dataset — it is direct evidence for which depths are uninformative for cold users.

## A.8 The game (`game.py`)

```python
v(C) = ndcg_at_10(rank_by(f_theta^C)) - ndcg_at_10(pi0)
```

with `f_theta[empty] = f_theta[{0}] = pi0` so $v(\emptyset)=v(\{0\})=0$. Fix `pi0` seed and report its NDCG so subtraction is auditable.

The no-propagation baseline `pi0` is BPR-MF (`E0` only, same embedding table, same BPR loss, no `hat_A`). Its NDCG is the expected LightGCN-without-propagation quality.

Exact Shapley over all $2^{K+1}$ coalitions:

- $K=3$ → players `{0,1,2,3}`, 16 coalitions
- Include `[]` and `[0]` both mapping to 0 for correctness, but deduplicate in efficiency check

Plus per-user values from same sweep by Proposition 3: $v_u(C)$ restricted to user $u$.

```python
def shapley(values, n_players=4):
    """values: dict coalition_tuple -> v(C), includes empty"""
    import math, itertools
    phi = {k: 0.0 for k in range(n_players)}
    for k in range(n_players):
        for C in coalitions_without(k, n_players):
            weight = math.factorial(len(C)) * math.factorial(n_players - len(C) - 1) / math.factorial(n_players)
            marginal = values[tuple(sorted(C + (k,)))] - values[tuple(sorted(C))]
            phi[k] += weight * marginal
    return phi
```

Assert $\sum_k \varphi_k = v(\mathcal{L})$ and $\frac1{|\mathcal{U}|}\sum_u \varphi_k(u)=\varphi_k$ to machine precision.

## A.9 Test suite (`tests/`)

Non-negotiable, priority order:

1. **Efficiency identity.** $\sum_k \varphi_k = v(\mathcal{L})$ to `1e-6`. Catches most bugs; `v(empty)=0` definition is the usual failure.
2. **Per-user consistency.** $\frac1{|\mathcal{U}|}\sum_u \varphi_k(u) = \varphi_k$ to `1e-6`.
3. **Empty and singleton.** $v(\emptyset)=0$ and $v(\{0\})=0$ exactly — both map to `pi0`.
4. **Symmetry on synthetic data.** Duplicate a layer (copy `E1` as `E1'`), run game with `K=4` duplicate — the two copies must receive equal $\varphi$, and their individual LOO ≈ 0. This is Prop.2 as an executable test.
5. **Isolated node.** A user with degree 0 has `E1=E2=E3=0` after propagation; its per-user $\varphi_0$ must be 1.0 (all credit to memorization), deeper layers 0. If not, propagation leak.
6. **Degenerate normalization.** Constant score column → all-zero normalized column, no NaN/inf.
7. **Backbone frozen.** Assert embedding table and `hat_A` identical across all coalition evaluations (only fusion weight changed).
8. **Dummy layer.** A pure-noise layer (random Gaussian, not propagated) must receive $\varphi \approx 0$ or negative.

Tests 1,2,4 are the ones that would catch a wrong paper rather than a crashed run.

## A.10 Adaptive modules (`adaptive.py`)

**LightShap-Weight:** Convert $\bar\varphi_k$ to fusion weights with shrinkage:

```python
phi_bar = phi / phi.sum()  # normalized share, sum 1
# clip negatives to 0 before renorm if any phi < 0 (oversmoothing harm)
phi_bar = np.maximum(phi_bar, 0)
phi_bar /= phi_bar.sum()
phi_tilde = (1 - alpha) * (1/(K+1)) + alpha * phi_bar
# renormalize after shrinkage (still sum 1)
phi_tilde /= phi_tilde.sum()
E_weighted = sum(phi_tilde[k] * Es[k] for k in range(K+1))
```

Sweep $\alpha \in [0,1]$ step $0.1$; $\alpha=0$ recovers uniform LightGCN (nested). Also support per-segment $\alpha_m$.

**LightShap-Adaptive:** Fit per-segment weights with shrinkage toward global:

$$w_m = (1-\lambda) w_{global} + \lambda w_{segment(m)}$$

Segments fitted on **training users only**; held-out users assigned by frozen segmenter (degree quantile lookup). Report $\lambda$ sweep.

## A.11 Runtime budget (CPU, single laptop)

| Stage | ML-1M | Beauty |
|---|---|---|
| Load, filter, split, build hat_A | < 1 min | < 1 min |
| LightGCN backbone training (grand coalition, ~1000 epochs early-stopped) | 20–40 min CPU / 8–15 min GPU | 10–20 min CPU / 5–10 min GPU |
| Cache E0..E3 | < 1 min | < 1 min |
| 16 coalition evaluations (Weight) | 1–3 min (scoring over cached Es) | 1–2 min |
| Per-user decomposition (free) | < 1 min | < 1 min |
| Segments + Adaptive sweep | 2–4 min | 2–4 min |
| **Full pipeline, one seed** | **~30 min CPU / ~15 min GPU** | **~20 min CPU / ~10 min GPU** |
| Five seeds, both datasets | ~2.5 hr CPU / ~1 hr GPU | |

If any stage is an order of magnitude over these, suspect dense `hat_A` (should be sparse) or re-running propagation per coalition instead of using cache.

---

# PART B — REGISTERED PREDICTIONS

> **Pre-registration.** Everything below is expected *before* running. When real numbers arrive, report them against this table and flag every miss explicitly. A miss discussed is a strength; a quietly revised prediction is misconduct.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Beauty | Confidence |
|---|---|---|---|
| Users after 5-core | ~6,040 | ~22,000 | high |
| Items after 5-core | ~3,400–3,700 | ~12,000 | high |
| Density after filtering | ~4.8–5.2% | ~0.08–0.10% | high |
| Mean degree (user) | ~150–170 | ~9–11 | high |
| NDCG@10, no-prop baseline $\pi_0$ ($E_0$ only = BPR-MF) | 0.26–0.31 | 0.11–0.16 | medium |
| NDCG@10, uniform LightGCN $f^{\mathcal{L}}$ (K=3 mean) | 0.30–0.36 | 0.13–0.19 | medium |
| Uplift $v(\mathcal{L}) = f^{\mathcal{L}} - \pi_0$ | **0.03–0.06** | **0.01–0.03** | medium |
| Isolated degree-0 users after split | ~0% | 2–6% | medium |

If $v(\mathcal{L}) < 0.01$ on both datasets, propagation is not helping and the game explains *why* but Weight cannot beat a weak grand coalition — still report as negative result.

## B.2 Layer shares $\bar\varphi_k$ (headline result)

| Layer | ML-1M (dense, oversmoothing regime) | Beauty (sparse, shallow regime) | Reasoning |
|---|---|---|---|
| **$E_0$ (0-hop)** | **20–30%** | **35–50%** | Only signal for isolated/cold nodes; washed out on dense where neighbors are informative |
| **$E_1$ (1-hop)** | **25–35%** | **25–35%** | Direct collaborators — universally useful, least redundant |
| **$E_2$ (2-hop)** | **15–25%** | 15–25% | Second-order smoothing — helpful but starts to correlate with $E_3$ |
| **$E_3$ (3-hop)** | **15–25%** | **5–15%** (often ~0 or negative) | Deep smoothing: valuable on dense (reaches far), harmful on sparse (oversmoothing + noise) |

**Prediction that carries the paper:** the *ordering* inverts between datasets — deep layers ($E_2+E_3$ jointly ~35–45% on ML-1M) collapse on Beauty (jointly ~20–30%, with $E_3$ near zero). Joint share of the redundant pair $E_2$+$E_3$ is the RQ2 story; their cosine is the diagnostic.

**Single most testable prediction:** $\cos(E_2, E_3)$ flattened across all nodes will be **0.85–0.95 on ML-1M** and **0.60–0.80 on Beauty**. If this does not hold, the oversmoothing premise is wrong.

## B.3 Shapley versus leave-one-layer-out / uniform

| Pair | Predicted cosine $E_a$↔$E_b$ | Predicted LOO | Predicted $\varphi$ | Confidence |
|---|---|---|---|---|
| **$E_2$ ↔ $E_3$** | **0.85–0.95 (ML-1M)**, 0.60–0.80 (Beauty) | both LOO < 0.005 (near zero) | both 15–25% (ML-1M), both 5–15% (Beauty) | **high — oversmoothing guarantees it on dense** |
| $E_0$ ↔ $E_1$ | 0.35–0.55 | moderate | moderate | medium |
| $E_1$ ↔ $E_2$ | 0.55–0.75 | moderate | moderate | medium |

Predicted headline: uniform weighting over-weights the redundant pair by ~1.5× (gives 50% weight to $E_2+E_3$ jointly, Shapley says ~35% is earned), and $\sum \mathrm{LOO}$ falls **30–50% short** of $v(\mathcal{L})$ — visible efficiency gap. That gap is the cleanest single number in the paper and must be reported.

## B.4 Segment heterogeneity

Using degree quartiles Q1 (coldest, lowest degree) → Q4 (heaviest, highest degree):

| Segment | Predicted dominant layers | Predicted $E_0$ share |
|---|---|---|
| Q1 cold (degree 5–15) | **$E_0, E_1$** | **40–60%** |
| Q2 (degree 15–40) | $E_1, E_2$ | 25–35% |
| Q3 (degree 40–120) | $E_1, E_2$ | 18–28% |
| Q4 heavy (degree >120) | **$E_2, E_3$** | **10–20%** |

Prediction: **global depth ordering inverts in Q1 on both datasets** — on ML-1M, global says $E_1$ leads but Q1 says $E_0$ leads; on Beauty, global says $E_0$ leads but Q4 says $E_1$ leads. Permutation test on between-segment variance expected $p < 0.01$, high confidence — heterogeneity by degree is textbook in GNNs.

Lower confidence: attribution segments (clustering $\varphi(u)$ vectors) vs behavioural segments (degree quartiles). Adjusted Rand predicted **0.4–0.7**: related but not identical — if ARI ≈ 0, view credit is not reducible to degree, which is publishable either way. Be prepared to drop the claim if ARI < 0.2.

## B.5 Weighted fusion and adaptive depth gains

| Intervention | Predicted gain over uniform LightGCN | Confidence |
|---|---|---|
| **LightShap-Weight** (global $\alpha$) | **+2% to +5% relative NDCG@10**, **+2% to +4% Recall@10** overall | medium-high |
| LightShap-Weight, Q1 cold | +5% to +12% relative | medium-high |
| LightShap-Weight, Q4 heavy | +1% to +3% relative | medium |
| **LightShap-Adaptive** (per-segment $\lambda$) | **+3% to +6% relative** overall (adds ~1% over global Weight) | medium |
| Best $\alpha$ | 0.6–0.9 (dense needs more Shapley, sparse needs more uniform) | medium |
| Best $\lambda$ | 0.5–0.8 | medium |

Be honest that overall gain is modest — the *shape* (large gains where depth importance diverges most from uniform, parity where uniform was already near-optimal) is the convincing part and is exactly what the mechanism predicts.

If Weight gives no gain on Beauty, it is still a finding: oversmoothing is not the bottleneck on sparse graphs, and the negative result should be reported — paper still stands on RQ1–RQ3.

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| $v(\mathcal{L}) < 0.01$ on both datasets | propagation not helping | Report as negative result; paper still stands on attribution (RQ1–RQ3), but RQ4 weakens to *why* uniform fails; check $K=1$ vs $K=3$ gap |
| $E_2$↔$E_3$ cosine < 0.5 on ML-1M | engineered redundancy did not materialize | Report honestly; redundancy demo shifts to synthetic appendix where $E_3$ is forced as copy of $E_2$ |
| LOO and Shapley agree everywhere | no redundancy in this system | **Genuine negative result** — report it; framing in §1.3 softens; heterogeneity may still carry |
| Shares identical across datasets | density/degree does not drive depth attribution | Two-dataset justification collapses; add LastFM-2K or vary $d$ and reframe around embedding size |
| Segments homogeneous ($p > 0.05$) | depth credit is population-uniform | Drop behavioural-vs-attribution claim; paper still has three contributions |
| Adaptive gains ≤ 0 | attribution does not transfer to improvement | Report negative result — gap between explanation and intervention is informative; still publishable |
| Some $\varphi_k < 0$ (especially $E_3$ on Beauty) | a layer harms ranking via oversmoothing | **Do not hide** — it is a finding; clip to 0 for weighting but report raw values; normalized shares reported only when all $\varphi\ge0$ |
| Efficiency check fails | implementation bug | **Stop**; do not interpret anything until tests 1–2 pass |
| $E_0$ share >50% on ML-1M | memorization beats propagation on dense — contradicts LightGCN premise | Check leakage (test edges in hat_A) or $d$ too small; genuine if reproducible, and very publishable as a contrarian result |

Two failures that would wound the paper are homogeneous segments + no redundancy. Both are cheap to check early — **run RQ2 and RQ3 before writing prose** (first full pipeline run answers B.3 + B.4).

## B.7 Suggested milestone order

1. Data, splits, graph `hat_A`. **Gate: degree distributions sane, no test leakage, `hat_A` symmetric.**
2. LightGCN backbone (grand coalition). **Gate: $v(\mathcal{L}) > 0.01$ and uniform LightGCN matches published NDCG (ML-1M ~0.32, Beauty ~0.15).**
3. Cache `E0..E3` + coalition sweep (16 coalitions). **Gate: tests 1–4 pass.**
4. Shapley + B.2 check. **Gate: does $E_2$↔$E_3$ cosine predict redundancy and does ordering invert between datasets?**
5. LOO/uniform comparison + B.3 check. **Gate: does redundancy appear?**
6. Segments + B.4 check. **Gate: is heterogeneity significant?**
7. Weight/Adaptive sweeps, statistics, LaTeX emitters.

Steps 4, 5, 6 are decision points. If all three land as predicted, the paper is essentially written. If any fails, the contingency table says what to do without improvising.

