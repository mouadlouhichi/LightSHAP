# LightShap — Full Paper Structure, TOC & Embedded Content

**New name:** **LightShap** — **LightGCN via Shapley-Weighted Layer Attribution** (Cooperative Layer Credit in Graph Collaborative Filtering)
**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** six-section layout of *Game Theory Meets Explainable AI* (IJACSA 2025), mirroring *SignalShap* and *SHAPER* — not the nine-section DyHuCoG layout.
**Target:** ≈ 8,500 words, 7 figures, 8 tables.

> **Why this is the easiest accept after SignalShap/SHAPER.** It needs no new dataset and no new architecture. LightGCN is the *most-cited, most-reproduced* GNN recommender (He et al. SIGIR 2020) — every reviewer knows it, every codebase has it. The standard LightGCN averages four embeddings `E0, E1, E2, E3` equally. LightShap makes the **layers themselves the players** of a cooperative game. There are **4 players → 16 coalitions → exact Shapley**, no sampling error to defend. Embeddings are trained once, only a 4-weight head is refitted per coalition — seconds of compute. Novelty is in *what the players are* (depth, not features) and in proving equal weighting is provably wrong under oversmoothing.

> **Critical dependency note.** This paper is **independent of DyHuCoG**. SignalShap avoided DyHuCoG by using only standard signal sources; SHAPER by using only standard augmentations; LightShap does the same by using only the public LightGCN implementation (PyTorch + scipy.sparse) that runs on CPU in minutes. Nothing from DyHuCoG is reused.

---

## Working Title (primary + alternates)

- **Primary (selected):** *LightShap: Exact Shapley Attribution over Propagation Depth in Light Graph Collaborative Filtering*
- Alt 1: *Which Layer Recommends? Cooperative Credit Assignment over GNN Depth in LightGCN*
- Alt 2: *From Mean Pooling to Earned Pooling: A Cooperative-Game View of Light Graph Convolution*
- Alt 3: *Depth Attribution for Graph Recommendation: Shapley-Weighted Layer Fusion in LightGCN*

*(LightShap is the method and codebase name throughout. LightGCN remains the backbone name — the paper is a wrapper, not a replacement.)*

## One-paragraph thesis (the spine)

LightGCN learns by propagating embeddings through a user–item bipartite graph and **averaging the embeddings from every depth** — the initial embedding `E0`, one-hop `E1`, two-hop `E2`, and three-hop `E3` — with equal weight. The equal weight is a convention, not a result, and it is violated the moment oversmoothing makes `E2` and `E3` near-identical: two mutually redundant layers each receive full weight while their joint contribution is counted twice, washing out the memorization signal in `E0` that cold users depend on. We recast LightGCN as a **cooperative game whose players are the propagation layers** and whose value is ranking quality (NDCG@10), and compute the **exact Shapley value** over that tiny player set. Because the Shapley operator is linear, the same 16-coalition game yields **exact per-user and per-segment attributions at no extra cost**, revealing that the layer that serves heavy users (`E3`) is not the layer that serves cold users (`E0`). We close the loop with **Shapley-weighted layer fusion** and **segment-adaptive depth**, which improve NDCG@10 and Recall@10 at zero additional inference cost — fusion is a four-number dot product.

## Research questions

| RQ | Question | Thesis link |
|---|---|---|
| **RQ1** | Can LightGCN be posed as a cooperative game whose players are propagation layers, such that the Shapley allocation exactly decomposes ranking-quality uplift and is computable exactly? | Extends the thesis spine from *source attribution* (SignalShap) and *view attribution* (SHAPER) to *depth attribution* |
| **RQ2** | Does Shapley layer weighting differ from equal weighting / leave-one-layer-out, and is the difference explained by oversmoothing-induced redundancy among deep layers? | The redundancy-failure argument applied to GNN depth |
| **RQ3** | Is layer attribution homogeneous across users, or does a global mean conceal opposing segment-level stories (cold vs heavy)? | Heterogeneity line from IJACSA 2025 + SignalShap — segments as explanatory unit |
| **RQ4** | Can attribution be turned into improvement — does Shapley-weighted fusion and segment-adaptive depth beat uniform LightGCN? | Attribution → intervention closure |

---

# TABLE OF CONTENTS

```
Abstract / Keywords
1. Introduction
   1.1 Background: LightGCN and the equal-weight convention
   1.2 The depth-attribution gap in GNN recommendation
   1.3 Why leave-one-layer-out fails under oversmoothing
   1.4 Contributions
   1.5 Organization
2. Literature Review
   2.1 Graph collaborative filtering: NGCF → LightGCN → UltraGCN
   2.2 Depth, oversmoothing, and layer combination in GNNs
   2.3 Shapley values in ML and XAI
   2.4 Cooperative games over model components and layers
   2.5 Heterogeneity in graph recommendation
   2.6 Positioning and differentiation (comparison table)
3. Methodology
   3.1 Notation and problem formulation
   3.2 LightGCN backbone and its uniform fusion
   3.3 The layer-players (E0, E1, E2, E3)
   3.4 The layer-attribution game
   3.5 Exact Shapley computation and its cost
   3.6 Per-user and per-segment decomposition
   3.7 LightShap-Weight: Shapley-weighted layer fusion
   3.8 LightShap-Adaptive: segment-adaptive depth
   3.9 Comparative analysis against alternative weightings
   3.10 Theoretical justification
   3.11 Complexity analysis
   3.12 Practical implementation
4. Experimental Results
   4.1 Datasets and preprocessing
   4.2 Protocol, metrics, and baselines
   4.3 RQ1 — exact layer attribution
   4.4 RQ2 — Shapley versus equal / leave-one-layer-out under redundancy
   4.5 RQ3 — segment heterogeneity of layer credit
   4.6 RQ4 — weighted fusion and adaptive depth gains
   4.7 Sensitivity, stability, and ablations
   4.8 Statistical significance
5. Discussion and Broader Implications
   5.1 What layer attributions mean for GNN design
   5.2 Oversmoothing as a credit-assignment phenomenon
   5.3 Relation to source-level and view-level attribution
   5.4 Limitations and threats to validity
6. Conclusion and Future Work
Declarations
Appendices
```

---

# ABSTRACT (draft, ~218 words)

Light Graph Convolution (LightGCN) improves collaborative filtering by propagating embeddings through a user–item graph and averaging the representations from every propagation depth. The averaging is uniform by convention — the initial embedding and each smoothed layer receive identical weight — despite the fact that deeper layers become mutually redundant under oversmoothing and that the layer most useful for a cold user with few interactions is not the layer most useful for a heavy user with a dense neighborhood. We recast LightGCN as a cooperative game in which the players are the **propagation layers** and the characteristic function is ranking quality, and we compute the **exact Shapley allocation**. With three to four layers the game has 8–16 coalitions, so no Monte-Carlo approximation is required, and because embeddings are trained once and only a four-weight fusion is refitted per coalition, the full game costs seconds. We prove that efficiency yields an exact additive decomposition of ranking-quality uplift, that equal weighting and leave-one-layer-out collapse under perfect redundancy where Shapley splits credit evenly, and that linearity gives per-user Shapley values at no extra cost. Aggregating per-user values into behavioural segments shows that global layer credit conceals opposing segment-level stories. Exploiting this, Shapley-weighted fusion and segment-adaptive depth improve NDCG@10 and Recall@10 over uniform LightGCN at no additional inference cost. Experiments span MovieLens-1M (dense) and Amazon-Beauty (sparse) and hold across embedding sizes and depths.

**Keywords:** Shapley value; cooperative game theory; graph neural networks; LightGCN; recommender systems; depth attribution; oversmoothing; user segmentation

---

# 1. INTRODUCTION

## 1.1 Background: LightGCN and the equal-weight convention

Open flat, IJACSA voice. LightGCN (He et al. 2020) simplified NGCF by removing feature transforms and nonlinearities, leaving only neighborhood aggregation: `E_{k+1}= \hat A E_k`. State the convention that made it famous: `E_final = mean(E0, E1, ..., EK)` with `K=3`. Note that the paper ablated `K` but never ablated *how* layers are combined — mean was inherited, not justified.

## 1.2 The depth-attribution gap

The maintenance framing becomes a capacity framing. Each layer defines a receptive field — `E0` is pure memorization, `E1` is direct neighbors, `E3` is three-hop smoothing — and treating them equally is only defensible if each contributes equally, which oversmoothing guarantees they do not. Name the second audience: the system owner choosing embedding size and depth under a latency budget.

## 1.3 Why leave-one-layer-out fails under oversmoothing

Concrete intuition before formalism: if `E2` and `E3` are near-identical due to three rounds of smoothing, removing either alone changes the mean little (` (E0+E1+E2+E3)/4 ≈ (E0+E1+E2)/3`), so ablation reports *neither matters*, while removing both drops from 4-layer to 2-layer and is catastrophic. Ablation violates additivity and can report zero for a layer that is load-bearing whenever its smoothed twin remains. Forward-reference Proposition 2.

## 1.4 Contributions

1. **A cooperative game over propagation layers.** We formalize LightGCN as a transferable-utility game whose players are layers `E0...EK` and whose value is NDCG@10 uplift over a no-propagation baseline (`E0` only). The player set is 4 by construction, so Shapley is exact, with no sampling variance.

2. **Three short results that make the allocation usable.** Efficiency gives exact additive decomposition of uplift (Prop. 1). Under perfect oversmoothing-induced redundancy, leave-one-layer-out assigns zero while Shapley splits evenly (Prop. 2). Linearity over a per-user-mean value gives per-user attribution for free (Prop. 3).

3. **Segment-level depth profiles.** Aggregating per-user layer Shapley vectors over behavioural segments (cold/light/heavy by degree) shows global depth credit averages over heterogeneous populations. Quantified with a permutation test.

4. **Two attribution-to-improvement interventions.** **LightShap-Weight:** Shapley-weighted layer fusion; **LightShap-Adaptive:** per-segment depth weights. Both improve NDCG@10/Recall@10 at no inference cost — fusion is a 4-number dot product and segment assignment is a lookup.

5. **A reproducible artifact.** The backbone is the public LightGCN PyTorch implementation; full game is reproducible on a laptop CPU. Code and configs released.

## 1.5 Organization

Roadmap paragraph naming Sections 2–6.

---

# 2. LITERATURE REVIEW

Thematic subsections, each closing with a gap sentence. ~1,400 words.

## 2.1 Graph collaborative filtering
NGCF's feature transforms, LightGCN's simplification (the most influential simplification in RecSys GNNs), UltraGCN's infinite-layer approximation, recent Mamba/GNN hybrids. *Gap: every method studies *how deep* but not *how to weight* depth.*

## 2.2 Depth, oversmoothing, and layer combination
Oversmoothing in GNNs, Dirichlet energy, JK-Nets and jumping knowledge, attention over layers (GAT-like). Establish that oversmoothing is *expected*, not exceptional, at `K=3` on dense graphs. *Gap: attention over layers is learned jointly and has no axiomatic guarantee, unlike Shapley.*

## 2.3 Shapley values in ML and XAI
SHAP, KernelSHAP, Data Shapley; axiomatic case; computational objection and sampling estimators. Point: objection is a function of treating *features* as players (hundreds), evaporates when players are *layers* (4).

## 2.4 Cooperative games over model components and layers
Layer Shapley in vision (ResNet), ensemble-member attribution, Data Shapley. Position layer attribution as an uninstantiated member for recommendation.

## 2.5 Heterogeneity in graph recommendation
Degree heterogeneity, cold vs heavy performance gaps in LightGCN, popularity bias. Cite the group's prior segmentation work.

Cite the group's *Explanation Drift* and *SignalShap/SHAPER* papers in §2.3 and §2.5. Drift asks whether attributions are stable **across time**; SignalShap asks whether **source** credit is stable **across population**; LightShap asks whether **depth** credit is stable **across population** — three orthogonal axes, a coherent trilogy.

## 2.6 Positioning and differentiation
**Table 1.** Rows: LightGCN, UltraGCN, JK-Net, attention-over-layers, layer Shapley. Columns: *unit of attribution* (feature / layer), *axiomatic guarantee*, *exact or approximate*, *heterogeneity-aware*, *closes loop to improvement*. LightShap is the only row with layer-level + exact + heterogeneity-aware + loop-closing.

---

# 3. METHODOLOGY

## 3.1 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | users, items |
| $\mathbf{E}^{(k)} \in \mathbb{R}^{(|\mathcal{U}|+|\mathcal{I}|)\times d}$ | embeddings after $k$ propagations; $\mathbf{E}^{(0)}$ = learnable |
| $\hat{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ | symmetrically normalized adjacency |
| $\mathcal{L} = \{0,1,...,K\}$ | layer players; $K=3$ main, $K=4$ robustness; $|\mathcal{L}|=K+1$ |
| $C \subseteq \mathcal{L}$ | coalition of layers |
| $f_\theta^C$ | LightGCN refitted using only layers in $C$ |
| $v(C)$ | characteristic function: NDCG@10 uplift of $f_\theta^C$ |
| $\varphi_k$ | Shapley value of layer $k$ |
| $\varphi_k(u)$ | per-user Shapley value of layer $k$ for user $u$ |
| $\mathcal{T}_1,...,\mathcal{T}_M$ | user segments by degree/activity |
| $\pi_0$ | no-propagation baseline ($E_0$ only, $C=\{0\}$) |

## 3.2 LightGCN backbone and its uniform fusion

Recall LightGCN propagation: $\mathbf{E}^{(k+1)} = \hat{\mathbf{A}} \mathbf{E}^{(k)}$, no $\mathbf{W}$, no $\sigma$. Standard fusion: $\mathbf{E} = \frac{1}{K+1}\sum_{k=0}^{K} \mathbf{E}^{(k)}$, score$(u,i) = \mathbf{e}_u^\top \mathbf{e}_i$, trained with BPR loss.

Critical design property: **embeddings $\mathbf{E}^{(0)}$ and propagation are executed once in the grand coalition and all $\mathbf{E}^{(k)}$ are cached**. A coalition $C$ is realized by recomputing $\mathbf{E}_C = \sum_{k\in C} w_k \mathbf{E}^{(k)}$ with refitted $w_k$ (or simply masking and averaging) — no graph re-propagation, no embedding retraining. This makes the exact game cheap.

## 3.3 The layer-players ($E_0, E_1, E_2, E_3$)

| $k$ | Layer | Receptive field | What it encodes |
|---|---|---|---|
| 0 | $E_0$ | 0-hop (itself) | Pure memorization; only signal for isolated nodes |
| 1 | $E_1$ | 1-hop neighbors | Direct collaborative signal; co-interacted users/items |
| 2 | $E_2$ | 2-hop | Second-order smoothing; user→item→user→item |
| 3 | $E_3$ | 3-hop | Deep smoothing; approaches oversmoothing on dense graphs |

$K=3$ (4 players) in main paper; $K=2$ and $K=4$ as ablations in appendix. Every player is a *cached matrix*, not a learned parameter — masking is well-defined.

## 3.4 The layer-attribution game

**Definition 1 (Layer-attribution game).** $(\mathcal{L}, v)$ with
$$v(C) = \mathrm{NDCG@10}(f_\theta^C) - \mathrm{NDCG@10}(\pi_0), \qquad v(\emptyset)=0, \; v(\{0\})=0.$$

Subtracting the no-propagation baseline turns the value into *uplift attributable to propagation*, which is the quantity a system owner cares about when choosing $K$. Fix the seed and report $\pi_0$ NDCG so subtraction is auditable.

Define the empty coalition and the singleton $\{0\}$ both as $\pi_0$ (BPR-MF equivalent), so $v(\emptyset)=v(\{0\})=0$ follows rather than being imposed. A coalition $C$ with $0\notin C$ is allowed but expected to be weak — include it; the game will discover it.

**Definition 2 (Layer Shapley value).**
$$\varphi_k = \sum_{C \subseteq \mathcal{L}\setminus\{k\}} \frac{|C|!\,(|\mathcal{L}|-|C|-1)!}{|\mathcal{L}|!} [v(C\cup\{k\}) - v(C)].$$

**Definition 3 (Normalized layer share).** $\bar\varphi_k = \varphi_k / \sum_h \varphi_h$ as percentage, reported alongside raw $\varphi_k$. Treat negative $\varphi_k$ as a finding (a layer that harms ranking via oversmoothing) rather than hiding it. Clip to $0$ only for weighting, with renormalization.

## 3.5 Exact Shapley computation and its cost

$2^{|\mathcal{L}|}=16$ coalitions for $K=3$, $8$ for $K=2$, $32$ for $K=4$. Each requires one **fusion-weight refit** over cached $\mathbf{E}^{(k)}$ — a tiny convex fit, not a GNN retraining. Give wall-clock in results. Contrast with feature-level SHAP where $2^{|F|}$ is infeasible — exactness is a *design choice to make layers the players*.

## 3.6 Per-user and per-segment decomposition

Define $v_u(C)$ as NDCG@10 uplift restricted to user $u$, so $v(C)=\frac{1}{|\mathcal{U}|}\sum_u v_u(C)$. Proposition 3 gives $\varphi_k = \frac{1}{|\mathcal{U}|}\sum_u \varphi_k(u)$ with **no extra coalitions** — per-user values fall out of the same 16 refits.

> **Granularity warning.** With leave-one-out evaluation each user has exactly one held-out item, so $v_u(C)$ takes ~11 values: $1/\log_2(r+1)$ if ranked $r\le 10$, else $0$. Per-user values are *exact* w.r.t. the defined game but high-variance as estimates. **Report segment aggregates as primary**; treat per-user vectors as intermediates.

Segments: (a) **behavioural segments** from node degree / activity quartiles (cold / light / heavy), and (b) **attribution segments** from clustering per-user $\varphi(u)$ vectors. If they disagree, depth importance is not reducible to degree.

> Heterogeneity test: same permutation test as SignalShap/SHAPER — cluster observed $\varphi(u)$, record between-cluster variance, recompute with shuffled labels many times, report $p$. Companion check: ARI across five seeds for cluster stability.

## 3.7 LightShap-Weight: Shapley-weighted layer fusion

Replace uniform mean:

$$\mathbf{E}_{uniform} = \frac{1}{K+1}\sum_{k=0}^{K} \mathbf{E}^{(k)}$$

with Shapley-weighted fusion:

$$\mathbf{E}_{LightShap} = \sum_{k=0}^{K} \bar\varphi_k \, \mathbf{E}^{(k)}, \qquad \sum_k \bar\varphi_k=1$$

with shrinkage toward uniform:

$$\tilde\varphi_k(\alpha) = (1-\alpha)\cdot \frac{1}{K+1} + \alpha \cdot \bar\varphi_k, \quad \alpha \in [0,1]$$

Sweep $\alpha\in[0,1]$ and report the curve; $\alpha=0$ recovers LightGCN, so the comparison is nested and honest. Clip negatives to $0$ and renormalize before weighting.

No inference cost: fusion is a 4-number weighted sum, precomputed.

## 3.8 LightShap-Adaptive: segment-adaptive depth

Fit $\alpha$ or per-layer weights per segment rather than globally, with shrinkage:

$$\mathbf{w}_m = (1-\lambda)\mathbf{w}_{global} + \lambda \mathbf{w}_{segment(m)}, \quad \lambda\in[0,1]$$

Segments fitted on **training users only**; held-out users assigned by frozen segmenter (degree lookup). Inference is a table lookup + 4-number dot product.

This is the paper's most practical contribution: cold users get shallow depth (trust $E_0$), heavy users get deep depth (trust $E_3$).

## 3.9 Comparative analysis against alternative weightings

Compare on: uniform (LightGCN baseline), learned attention over layers (GAT-style, jointly trained), JK-Net max/concat, leave-one-layer-out reweighting, forward selection, permutation importance over layers, and MC-Shapley. For each state what it measures, what axiom it violates, and its cost. Table must make oversmoothing-induced redundancy visible as an empirical column (cosine similarity between $E_2$ and $E_3$).

## 3.10 Theoretical justification

Mirror the group's calibration: *Explanation Drift* (zero props), *SignalShap/SHAPER* (two props + remark). Keep it light: two propositions + one remark, proofs in Appendix A.

**Proposition 1 (Exact additive decomposition).** $\sum_{k}\varphi_k = v(\mathcal{L}) = \mathrm{NDCG@10}(f^{\mathcal{L}}) - \mathrm{NDCG@10}(\pi_0)$. Immediate from efficiency.

**Proposition 2 (Redundancy collapse of leave-one-layer-out).** Let layers $a$ and $b$ be perfectly redundant (oversmoothed), i.e. $v(C\cup\{a\})=v(C\cup\{b\})=v(C\cup\{a,b\})$ for all $C\subseteq \mathcal{L}\setminus\{a,b\}$. Then $\mathrm{LOO}(a)=\mathrm{LOO}(b)=0$, while $\varphi_a=\varphi_b=\tfrac12 [v(\mathcal{L})-v(\mathcal{L}\setminus\{a,b\})]$. *Theoretical centre* — equal weighting double-counts redundancy, ablation reports zero for load-bearing deep layers.

**Proposition 3 (Free per-user decomposition).** Since $v=\frac1{|\mathcal{U}|}\sum_u v_u$ and Shapley is linear, $\varphi_k=\frac1{|\mathcal{U}|}\sum_u \varphi_k(u)$.

**Remark 1 (Scale invariance).** Because LightGCN prediction is an inner product, scaling a layer's embeddings by $c>0$ and re-normalizing fusion weights leaves $\varphi_k$ unchanged after refitting — shares are about *direction*, not magnitude.

> Do not claim invariance to nonlinear rescalings — oversmoothing changes the embedding subspace, not just its scale, and the paper must measure, not claim, nonlinear sensitivity.

## 3.11 Complexity analysis

Match the house style of *SignalShap/SHAPER/Explanation Drift*. Let $B$ be cost of training LightGCN embeddings + propagation, $\Phi$ cost of one fusion-weight refit over cached $\mathbf{E}^{(k)}$, $P=(|\mathcal{U}|+|\mathcal{I}|)d$ cached size.

| Stage | Cost | Note |
|---|---|---|
| Embedding training + propagation | $O(B)$ | paid **once**; sparse matmuls $\hat A E$ |
| Layer cache | $O((K+1)P)$ | enabling choice; 4 matrices for $K=3$ |
| Coalition sweep | $O(2^{K+1} \Phi)$ | $16\Phi$ for $K=3$; $\Phi$ is a 4-parameter fit |
| Shapley aggregation | $O((K+1)2^{K+1})$ | negligible |
| Per-user decomposition | $O(1)$ extra | free by Prop. 3 |

Headline: total $O(B + 2^{K+1}\Phi)$, dominated by $B$ — **attribution is cheaper than training the GNN it explains**. Contrast with MC sampling required for feature-level $2^{|F|}$ games.

## 3.12 Practical implementation

Library versions, seeds, hardware, wall-clock, embedding size $d$, depth $K$, negative sampling ratio, hyperparameter grids, caching. Pin: empty coalition definition $f_\theta^\emptyset\equiv \pi_0$, the $v(\{0\})=0$ convention, and that full game is reproducible on laptop CPU (no GPU required).

Give a hyperparameter table plus **two numbered algorithm blocks** — one for cached propagation + coalition sweep, one for Shapley aggregation and weighted/adaptive fusion.

---

# 4. EXPERIMENTAL RESULTS

## 4.1 Datasets and preprocessing

| Dataset | Users | Items | Interactions | Density | Avg. degree |
|---|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | ~4.5% dense | ~165 |
| Amazon-Beauty | ~22,363 | ~12,101 | ~198,502 | ~0.07% sparse | ~9 |

Two datasets chosen for **density and degree contrast** — dense, high-degree graph vs sparse, low-degree graph — the axis along which oversmoothing and depth attribution are predicted to move. 5-core iterative filtering to convergence. Leave-one-out temporal split (last interaction = test).

## 4.2 Protocol, metrics, and baselines

Metrics: NDCG@10 (primary, defines $v$), Recall@10, HR@10, MRR. Report $\pi_0$ ($E_0$ only = BPR-MF) NDCG explicitly — LightGCN's value must be shown as uplift over BPR-MF, otherwise a reviewer notes the baseline is missing.

Baselines for recommender: BPR-MF ($\pi_0$), LightGCN-uniform ($K=3$ mean), UltraGCN, JK-Net (concat), GAT-weighted layers (attention), $K=1,2,4$ ablations to show depth matters.

Baselines for weighting: leave-one-layer-out, forward selection, permutation importance over layers, learned attention, MC-Shapley.

## 4.3 RQ1 — exact layer attribution

**Table 4 + Figure 2:** $\varphi_k$ and $\bar\varphi_k$ per dataset with efficiency check $\sum_k\varphi_k=v(\mathcal{L})$ to machine precision.

Expected narrative (to confirm): on dense ML-1M, $\bar\varphi_2+\bar\varphi_3$ jointly dominate but individually modest due to redundancy; $E_0$ retains ~25–30% (memorization still matters). On sparse Beauty, $\bar\varphi_0+\bar\varphi_1$ dominate; $\bar\varphi_3$ is small or negative (oversmoothing hurts).

## 4.4 RQ2 — Shapley versus equal / leave-one-layer-out under redundancy

**Table 5:** $\varphi_k$ beside uniform weights ($1/4$) and LOO, with layer-layer redundancy diagnostic (cosine similarity between $\mathbf{E}^{(2)}$ and $\mathbf{E}^{(3)}$ flattened) as final column. **Figure 3:** scatter LOO vs $\varphi$, annotated at maximal disagreement + heatmap of $E_k$ cosine matrix.

Report as **mean over five seeds with std** — both values inherit embedding-training noise.

Look for: **$E_2$ ↔ $E_3$ cosine 0.85–0.95 on ML-1M**, LOO near zero for both, Shapley ~15–20% each. Report efficiency gap $\sum \mathrm{LOO}$ vs $v(\mathcal{L})$ — leave-one-out sum falls short by 30–50%.

Word claim as *consistent with* Prop. 2 direction, not *instance of* it — real data is approximate.

## 4.5 RQ3 — segment heterogeneity

**Figure 4:** stacked layer shares per degree segment (Q1 cold → Q4 heavy). **Figure 5:** attribution vs behavioural segments (Sankey). **Table 6:** per-segment shares + permutation test $p$.

Headline to test: Q1 cold users derive uplift from **$E_0/E_1$** (shallow), Q4 heavy users from **$E_2/E_3$** (deep), so global ordering inverts in Q1. Permutation test $p<0.01$ expected.

## 4.6 RQ4 — weighted fusion and adaptive depth gains

**Table 7:** NDCG@10, Recall@10, MRR for uniform LightGCN, LightShap-Weight ($\alpha$ sweep as **Figure 6** left), LightShap-Adaptive ($\lambda$ sweep as **Figure 6** right), and combined. Report per-segment gains — expect large gain on cold and heavy extremes, parity in middle.

**Figure 6:** dual-axis $\alpha$ and $\lambda$ sweeps; $\alpha=0$/$\lambda=0$ recover baseline (nested). **Figure 7:** heatmap of coalition values $v(C)$ (16 bars).

## 4.7 Sensitivity, stability, and ablations

Seed stability of $\varphi$ across five seeds (report std — with exact Shapley only variance source is embedding training); sensitivity to embedding size $d\in\{32,64,128\}$, depth $K\in\{2,3,4\}$, regularization, negative sampling ratio; NDCG@$k$ for $k\in\{5,10,20\}$ to show not cutoff artifact; measurement of layer-layer cosine vs depth as the oversmoothing diagnostic.

## 4.8 Statistical significance

Paired tests over users (not runs), Holm–Bonferroni across method family, Wilcoxon signed-rank, effect sizes (Cohen's $d_z$). Pair over **users**, state unit explicitly.

---

# 5. DISCUSSION AND BROADER IMPLICATIONS

## 5.1 What layer attributions mean for GNN design
A layer with small $\bar\varphi$ and nonzero cost (sparse matmul at inference) is a depth-pruning candidate; a layer that dominates only on sparse graphs suggests dataset-conditional depth.

## 5.2 Oversmoothing as a credit-assignment phenomenon
Reframe oversmoothing not as a smoothing problem but as *double-counting of redundant credit*, which Shapley quantifies directly. The $E_2$–$E_3$ cosine column in Table 5 is the diagnostic before any theorem.

## 5.3 Relation to source-level and view-level attribution
Layer attribution (LightShap), source attribution (SignalShap), and view attribution (SHAPER) answer different questions and compose hierarchically: once SignalShap identifies that the graph signal matters, LightShap identifies *which depth within that graph* teaches. SignalShap → LightShap is the natural two-paper zoom for graph recommenders.

## 5.4 Limitations and threats to validity
Be forthright: layer game assumes fusion is a linear combination — valid for LightGCN but not for JK-Net concat or GAT nonlinear fusion (different paper, Myerson values); masking a layer at fusion time is not identical to never having propagated it (no cascading smoothing); offline NDCG is a proxy; $K=3$ is a design choice and exactness degrades beyond $K\approx 6$; per-user values are exact but coarse, so segments are the reliable unit; two datasets are two datasets.

---

# 6. CONCLUSION AND FUTURE WORK

Restate four contributions against four RQs. Future work: Owen values when layers are grouped (shallow vs deep families); temporal extension to dynamic graphs (drift of $\varphi_k$ as graph densifies — bridge to *Explanation Drift*); online validation of adaptive depth; extension to heterogeneous GNNs where edge types are players (Myerson machinery); depth attribution for UltraGCN's infinite-layer approximation.

---

# DECLARATIONS

Funding · Competing interests · Ethics approval (public secondary data) · Data availability (links) · Code availability (repository) · Author contributions (CRediT) · **Use of AI tools** — include same declaration as *SignalShap/SHAPER/Explanation Drift* for consistency.

---

# APPENDICES

- **A.** Proofs of Props. 1–3 and Remark 1 — one page total
- **B.** Full coalition value tables $v(C)$ for all $2^{K+1}$ coalitions, per dataset — transparency showpiece (16 rows)
- **C.** Hyperparameter grids and selected values
- **D.** Layer-layer cosine / Dirichlet energy per depth (oversmoothing diagnostics)
- **E.** Per-segment attribution tables in full
- **F.** $K=2$ and $K=4$ robustness results + attention-over-layers comparison

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Diagram | Architecture: graph → E0-E3 → Shapley-weighted fusion → LightShap |
| Fig 2 | Bar | Layer shares $\bar\varphi_k$ per dataset |
| Fig 3 | Scatter + Heatmap | LOO vs Shapley + E2–E3 cosine |
| Fig 4 | Stacked bar | Layer shares per degree segment |
| Fig 5 | Sankey | Attribution vs behavioural segments |
| Fig 6 | Line (dual) | $\alpha$ (Weight) and $\lambda$ (Adaptive) sweeps |
| Fig 7 | Heatmap | Coalition value surface $v(C)$ |
| Tab 1 | Comparison | Positioning vs LightGCN family |
| Tab 2 | Descriptive | Dataset statistics |
| Tab 3 | Descriptive | The four layers and their receptive fields |
| Tab 4 | Results | $\varphi_k$, $\bar\varphi_k$, efficiency check |
| Tab 5 | Results | Shapley vs LOO/uniform/attention, redundancy (cosine) |
| Tab 6 | Results | Per-segment shares + heterogeneity test |
| Tab 7 | Results | Recommendation quality, LightShap variants |
| Tab 8 | Results | Significance tests and effect sizes |

---

# PLANNING NOTES (NOT part of manuscript)

## Why this is likely to be accepted

Narrow claim, fully supported; no oversold theorem. Exact Shapley removes the most common reviewer attack. LightGCN is the most standard GNN recommender — every reviewer knows it, reproducibility objections are weak. Two short propositions are easy to verify and non-trivial. Prop. 2 lifts this above an empirical note and directly predicts the observable $E_2$–$E_3$ redundancy.

**Complexity calibrated to group's published level.** *Explanation Drift* has zero propositions; *SignalShap/SHAPER* have two-three + remark. This blueprint mirrors that exactly, with the same selling point: exact $2^{K+1}=16$ game over layers vs infeasible $2^{|F|}$ over features.

## Build order

1. Data loaders and 5-core filtering; freeze splits and build `hat_A` sparse matrix.
2. LightGCN backbone (uniform) + caching of `E0...E3` for grand coalition.
3. Fusion-weight head + coalition-masking harness; verify $v(\emptyset)=0$ and $v(\{0\})=0$.
4. Exact Shapley over $K$ layers; **assert efficiency in unit test**.
5. Per-user decomposition; assert averaging.
6. LightShap-Weight and LightShap-Adaptive; $\alpha$/$\lambda$ sweeps.
7. Segmentation (reuse ActionShap pipeline) + degree diagnostics.
8. Statistics and LaTeX emitters (port `stats.py`).

## What can be reused

`stats.py` (paired tests, Holm–Bonferroni, Cohen's $d_z$) and k-means diagnostics from ActionShap/SignalShap transfer unchanged. `data.py` and `normalize.py` patterns from SignalShap port directly. LightGCN PyTorch code is public and already used for the 8-year leaderboard.

## Estimated effort

~3–4 days for someone with SignalShap codebase in hand — dominated by LightGCN backbone verification rather than game theory. Game itself is 16 coalitions on two benchmarks, trivial.

## Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Players | Propagation layers $E_0...E_3$, not features | $K=3$ → 16 coalitions, exact, no sampling |
| $K$ | 3 main (4 players), 2 and 4 as ablations | Keeps exactness; oversmoothing visible at K=3 |
| Backbone | LightGCN | Most standard, CPU-runnable, LightShap is wrapper-agnostic — UltraGCN appendix is cheap |

## Remaining open questions

- Whether to evaluate $v(C)$ by refitting only fusion weights $w_k$ (cleanest, seconds) vs refitting a tiny adapter on BPR loss (slightly more expressive, still cheap). Pre-commit to fusion-weight-only in main text, adapter in appendix.
- Whether behavioural segments by degree quartiles vs k-means on degree + clustering coeff — consider quantile split in main (easier to explain), k-means in appendix (same as SignalShap decision).

