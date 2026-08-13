# LightShap — Technical Implementation Specification and Registered Predictions

**Companion to:** `LightShap_Paper_Structure.md`.

**Status:** last polish before the tag. Math fixtures are independently verified (\(I_{**}=1/2\), \(\varphi=(5/3,8/3,11/3)\)). If anything claims \(I_{12}=1\), that claim is wrong.

**Tag order:** (1) download Beauty + ML-1M, compute SHA256, paste into yaml, **mirror the gz on OSF/Zenodo**; (2) `gen_shapley_fixtures.py` + A.0 tests green, including BPR-neg sensitivity **toggle existing in code**; (3) signed git tag + OSF deposit of this pair + `tests/` + configs; (4) `build_cache.py`. Never fill a hash during the science run.

---

## A.0 Before the tag

1. Fixtures generated: \(\varphi=(5/3,8/3,11/3)\), \(I_{12}=I_{13}=I_{23}=1/2\), both Prop. 2 gadgets.  
2. GEMM advanced indexing + dense-mask + item \(I-1\).  
3. Beauty 2014 SHA256 **pasted** in `configs/beauty.yaml`; file mirrored.  
4. `configs/bpr_grid.yaml` lists the **actual** 6 tuples (A.5), not “equal cardinality.”  
5. Val-\(\alpha\) tie-break coded (A.12).  
6. Adaptive uses \(v_m<0.005\) → \(E_0\)-only (aligned with the stability floor).  
7. RQ3 \(\Delta\) only if both extreme groups have \(v_m\ge 0.005\).  
8. Quartile cuts computed **once** from frozen `train_df`, reused across seeds.  
9. Game B trigger = A.11 one-sentence freeze (either side, same-side margin).  
10. BPR-neg sensitivity flag exists in the tagged commit (even if run after G3).  
11. Top-K ties = lexicographic `(score desc, item_id asc)` in float32, **not** \(10^{-12}\) jitter.

---

## Frozen decisions

| Item | Freeze |
|---|---|
| Estimand | Extra-hop inclusion on one frozen \(K=3,d=64\) LightGCN |
| Skip patterns | Valid cache scoring; not trained depths; **attribution only via the Shapley average**, not as standalone models |
| Game A | `{1,2,3}`, Mode A, val \(v\), train-mask at val |
| Game B | Secondary **all-slot** game, different zero **and** different player set. Credits **not comparable** to Game A |
| Game B trigger | See A.11. Either side sufficient; margin vs second-max **on that same side**; no trigger on a tie for max |
| Test mask (main) | Train+val |
| Beauty | 2014 `reviews_Beauty_5.json.gz`. If post-filter \(\lvert U\rvert<1000\): **exploratory only**; RQ3 confirmatory = ML-1M only; abstract uses the “secondary check” template |
| Fixtures | \(\varphi=(5/3,8/3,11/3)\), \(I_{**}=1/2\) |
| Flag | Approx. substitution in \(v\) **plus** \(I_{ab}<0\). Motivated by the substitution lemma, **not** a nested special case of it. Not a pruning license |
| Flag UNDEFINED | \(v(\mathcal L)<0.005\) (covers \(\le 0\)). Log may distinguish \(v\le 0\) vs \(0<v<0.005\). Always emit raw numbers. 0.005 = **stability floor**, not a meaningful-NDCG cutoff |
| LOO clause | Redundant-by-design restatement of the grand-coalition slice of \(R_{ab}\) |
| Sub-conditions | Report **co-occurrence** of the four clauses (tiny table). Do **not** report a correlation coefficient (\(N=3\) pairs) |
| LightGCN | Fixed \(K=3,d=64,\mathrm{lr}=10^{-3},\mathrm{reg}=10^{-4}\), batch 2048/1024, patience 20 |
| BPR-MF | The 6-point grid in A.5. Residual asymmetry (LightGCN defaults are community-tuned) acknowledged in Discussion |
| BPR-neg main | Exclude train∪{val,test}. Offline convention, **not** anti-leakage. Sensitivity toggle in tagged code |
| Seeds | Average each user across 5 seeds, then bootstrap **users**. Seed SD separate. Never \(5\lvert U\rvert\) iid |
| RQ1 | Spearman of the **8-vector** \(v(C)\) (average ranks for ties). For \(\varphi\): **order pattern** + per-hop **sign agreement** + cosine of the 3-vectors. No Spearman on \(\varphi\) (too short) |
| RQ3 primary | \(\Delta=D_{Q4}-D_{Q1}\), \(D_m=\varphi_2^{(m)}+\varphi_3^{(m)}\), only if \(v_{Q1}\ge 0.005\) and \(v_{Q4}\ge 0.005\). Else: “uplift too small for a mix comparison,” raw \(v_m\) only. Negative \(D_m\) = “deep extra hops **hurt** that group,” not a small share |
| Adaptive | \(\beta=1/4\); \(v_m<0.005\) → \(E_0\)-only. No \(\beta_m\) in RQ4 |
| Val-\(\alpha\) | All **286** points. Tie: smallest \(\lVert w-\mathrm{uniform}\rVert_2\), then lex \((w_0,w_1,w_2)\) |
| RQ4 tests | Two-sided Wilcoxon on seed-averaged per-user diffs. Holm **within dataset** (3 tests). No pooling |
| Quartiles | One cut from frozen train degrees; reused across seeds |
| Cosine (gate) | All **positive-train-degree** rows of that side; no subsample; row \(\ell_2\)-normalize then cosine of flattened matrix; deterministic |
| Top-K | Lex `(score desc, item_id asc)` |
| Null plan | Short methods note if RQ2+RQ3+RQ4 fail and Game B off |

---

# PART A — IMPLEMENTATION

## A.3 Data

2014 Beauty URL: `http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz`.  
SHA256: **paste before tag** (do not leave blank). Also deposit the bytes on OSF next to the pre-reg.  
ML-1M: GroupLens `ml-1m.zip`, hash in `ml1m.yaml` before tag.

If Beauty \(\lvert U\rvert<1000\) after the pipeline: keep the numbers, mark every Beauty RQ3/RQ4 claim **exploratory**, confirmatory RQ3 = ML-1M only.

Quartile cuts: once, from that frozen `train_df`.

## A.5 BPR-MF grid (`configs/bpr_grid.yaml`)

```yaml
# 6 configurations. Selected by val NDCG@10. Locked before any LightGCN φ is interpreted.
# batch: 2048 on ML-1M, 1024 on Beauty (dataset scale, not a search dimension).
# patience: 20. epochs max: 1000.
grid:
  - {d: 32,  lr: 1.0e-3, reg: 1.0e-4}
  - {d: 64,  lr: 1.0e-3, reg: 1.0e-4}
  - {d: 128, lr: 1.0e-3, reg: 1.0e-4}
  - {d: 32,  lr: 5.0e-4, reg: 1.0e-4}
  - {d: 64,  lr: 5.0e-4, reg: 1.0e-4}
  - {d: 128, lr: 5.0e-4, reg: 1.0e-4}
```

LightGCN itself is **not** searched. Discussion: equal *listed* effort ≠ equal *prior* tuning quality.

`--neg-pool {heldout_excluded,train_only}` must exist in the tagged trainer.

## A.6 Scoring

`mask_seen` via advanced indexing. Top-K:

```python
# float32 scores. Do not add 1e-12*id (vanishes vs eps ~ 1e-7).
order = torch.lexsort(torch.stack((item_ids.expand_as(scores), -scores), 0), dim=-1)
```

or equivalent stable sort by `(-score, item_id)`.

## A.8 Fixtures and inference

Hard-coded rationals as above. Generator is the oracle.

Spearman on \(v(C)\): SciPy default **average** ranks, deterministic.

\(\varphi\) stability: modal order pattern across seeds; sign agreement per hop; mean pairwise cosine of the 3-vectors.

## A.10 Adaptive

```python
if v_m < 0.005:
    q = np.array([1.0, 0.0, 0.0, 0.0])
else:
    # β = 1/4 attribution-guided q
```

Threshold matches the stability floor. Not Shapley-dictated. Report per-seed \(w^{(m)}\) in the appendix.

## A.11 Flag and Game B

```
if v_L < 0.005:
    flag = UNDEFINED          # log "nonpositive" vs "tiny positive" if useful
else:
    flag = (I_ab < 0)
       and max_C |v(C∪a)-v(C∪b)| / v_L < 0.15
       and R_ab / v_L < 0.15
       and max(|LOO_a|, |LOO_b|) / v_L < 0.15   # redundant-by-design, RQ2
```

**Not a pruning claim.** Retrained-\(K\) (RQ2 main text) is the architecture control.

**Cosine for the gate** (and the \(4\times 4\) table):

- Side-specific: users `[0,U)` or items `[U,U+I)`.  
- Drop rows with train-degree 0 on that side.  
- No subsample (all remaining rows).  
- Row \(\ell_2\)-normalize; cosine = flattened normalized matrices’ cosine (equiv. mean of row-wise cosines after that normalize — **freeze: cosine of the concatenated flattened normalized rows**).  
- Deterministic; no RNG.

**Game B trigger (no leftover ambiguity):**

> Compute `cos_user` and `cos_item` as \(4\times 4\). Trigger iff **no** Game-A pair is flagged **and** there exists a side on which the \((E_0,E_2)\) entry is **strictly** the unique maximum off-diagonal **and** exceeds the second-largest off-diagonal on **that same side** by \(\ge 0.05\). If two off-diagonals tie for max, that side does not fire.

Popularity = train frequency, ties = lower item id.

## A.12 Val-\(\alpha\)

Enumerate all 286 points (`w_i = k/10`, \(k\in\mathbb{N}_0\), \(\sum w=1\)).  
Tie on val NDCG@10: smallest \(\lVert w-(1/4)1\rVert_2\); remaining ties: lexicographic \((w_0,w_1,w_2)\).

## A.13 Runtime

Seconds-to-minutes if vectorized. No paper wall-clock until measured.

---

# PART B — PREDICTIONS

Unchanged bands. Add:

- Negative \(D_m\): deep extra hops **hurt** group \(m\). Secondary display \(D_m/v_m\) only when \(v_m\ge 0.005\).  
- Beauty \(\lvert U\rvert<1000\): exploratory; RQ3 confirmatory ML-1M-only.  
- RQ2 main table includes retrained \(K=1,2,3,4\) test NDCG **and** frozen prefix coalitions.

## Contingencies (add)

| Event | Action |
|---|---|
| Beauty \(\lvert U\rvert<1000\) | Exploratory Beauty; RQ3 confirmatory = ML-1M; abstract template B |
| \(v_{Q1}\) or \(v_{Q4}<0.005\) | No \(\Delta\); raw \(v_m\) only |
| Val-\(\alpha\) ties | Uniform-closest then lex |
| Cosine sides disagree | Either side may fire; if neither meets the strict unique-max+0.05, no Game B |

---

# PART C — LAST PASS

Reviewers independently verified fixtures and retracted \(I_{12}=1\). Remaining items were hygiene: listed BPR grid, val-\(\alpha\) ties, Adaptive/RQ3 floors at 0.005, Beauty fallback + abstract template, cosine gate, lex Top-K, Holm within-dataset two-sided, retrained-\(K\) in RQ2, flag ≠ prune, co-occurrence not correlation, mirror the data file.
