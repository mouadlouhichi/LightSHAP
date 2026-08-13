# LightShap — Full Paper Structure, TOC & Embedded Content

**Name:** LightShap — attribution of inference-time hop inclusion in a frozen LightGCN fusion  
**Target:** *Discover Artificial Intelligence*  
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹  
**Affiliation:** ¹ ENSIAS, Mohammed V University in Rabat, Morocco  
**Corresponding:** mouad_louhichi@um5.ac.ma  
**~8,000 words, 7 figures, 8 tables.**

**Status:** last polish before timestamp. Companion spec has the code freezes.

**Pre-registration.** Paste SHA256s → mirror Beauty on OSF → A.0 tests green (`gen_shapley_fixtures.py` + BPR-neg toggle in the commit) → signed tag + OSF → then `build_cache.py`.

---

## Title

*LightShap: Attribution of Inference-Time Hop Inclusion in a Frozen LightGCN Fusion*

**Boxed estimand:**

> We do not attribute architectures. We attribute inference-time inclusion of extra hops in one frozen uniform-\(K=3\), \(d=64\) checkpoint. Skip-pattern coalitions are valid scoring rules on that cache, not independently trained depth architectures. Their attribution is meaningful **only in aggregate through the Shapley average**.

---

## Thesis

He et al. (2020) average hops uniformly; learned global \(\alpha\) helped &lt; 1%. We define a three-player game on extra-hop inclusion in one **fixed** LightGCN (\(K=3\), \(d=64\)): \(E_0\) always kept; \(v\) = val NDCG@10 uplift over LightGCN-\(E_0\). That \(v(\mathcal L)\) is **selection-biased upward** (early-stop on the same metric). Shapley is the LMG *operator* on a nonlinear ranking set function. Under exact substitution, LOO \(=0\) and \(\varphi_a=\varphi_b\); the two-player half-split is not generally true for \(n=3\). We operationalize a conservative **flag** — approximate substitution in \(v\) plus negative SII — **motivated by** the lemma, not a nested special case of it (empirical NDCG never meets exact equality). The flag is **not** a license to prune at training time; retrained \(K\) is that control. Similarity is \(4\times 4\); SII is \(3\times 3\). A secondary all-slot game (different zero, different players) runs only under a tight \((E_0,E_2)\) trigger. RQ3’s one confirmatory contrast is extra-deep mix \(D=\varphi_2+\varphi_3\) in Q4 vs Q1, and only if both groups have \(v_m\ge 0.005\). RQ4 is a 286-point grid transfer check. Joint null ⇒ short methods note.

---

## Research questions

| RQ | Statistic |
|---|---|
| **RQ1** | \(v(\mathcal L)\) size; Spearman of the **8-vector** \(v(C)\) across seeds (average ranks); \(\varphi\) **order pattern** + sign agreement + cosine of 3-vectors. \(v\approx 0\) = effect size. |
| **RQ2** | Official flag; LOO vs \(\varphi\); **main-text** retrained \(K=1,2,3,4\) vs frozen prefix coalitions. Flag ≠ prune. |
| **RQ3** | \(\Delta=D_{Q4}-D_{Q1}\) iff both \(v_m\ge 0.005\); else no mix claim. Negative \(D_m\) = deep hops **hurt**. |
| **RQ4** | Two-sided Wilcoxon, seed-averaged users, Holm **within dataset** (3 tests). Prior: null. |

**Beauty \(\lvert U\rvert<1000\):** exploratory; RQ3 confirmatory = ML-1M only.

---

## Abstract templates (pick after G0)

**A — both graphs powered:**  
… Experiments use MovieLens-1M and Amazon Beauty (**2014** 5-core, `reviews_Beauty_5`). Temporal LOO, full-catalog ranking. *[gates]*

**B — Beauty underpowered:**  
… Primary experiments on MovieLens-1M; Amazon Beauty (2014 5-core) is reported as a **secondary check** after filtering left fewer than a preregistered 1,000 users. …

Never write “2018 All-Beauty” in the abstract.

---

# 1. INTRODUCTION

1.1 He et al. convention.  
1.2 Boxed estimand + fixed checkpoint (not a val-searched architecture).  
1.3 Substitution lemma as a **diagnostic pattern**, not a literature correction.  
1.4 Four contributions (no “we correct Prop. 2”).  
1.5 Organization.

---

# 2. RELATED WORK

Hop polynomials; LightGCN / LayerGCN. LMG = same operator, different \(v\). Novelty = instantiation. Discussion sentence: BPR-MF’s 6-point grid is equal *listed* effort, not equal *community prior*.

---

# 3. METHODOLOGY

**3.2** Fixed LightGCN hparams. BPR-MF: the **listed** 6 tuples in the spec. Negatives: offline convention; sensitivity toggle in the artifact. Val \(v\) selection-biased.

**3.3** Game B: different question, different zero, **different player set**; hop credits not comparable. Trigger: either side, unique max \((E_0,E_2)\), +0.05 vs second-max **on that side**. Cosine: all positive-degree rows, row-normalized, flattened cosine, no subsample.

**3.4** Fixtures in appendix. Flag wording: *operationalization motivated by* the lemma. LOO clause = redundant-by-design. Co-occurrence table, not a correlation. **The flag does not imply a pair is safe to drop from training; Table RQ2-retrained-\(K\) is the pruning-relevant control.**

**3.5** Keep: “\(\varphi^{(m)}=\mathrm{mean}_{T_m}\varphi(u)\) exactly — same quantity as averaging per-user Shapley, used as a reporting structure, not a cleaner estimator.” Quartile cuts once from frozen train degrees. \(\Delta\) floor. Secondary \(D_m/v_m\) only when \(v_m\ge 0.005\).

**3.6** Adaptive: \(v_m<0.005\) → \(E_0\)-only (aligned floor). No \(\beta_m\) in RQ4.

**3.7** 286 points; ties → closest to uniform, then lex. Wilcoxon **two-sided**. Holm within-dataset only.

---

# 4. EXPERIMENTS

**4.1** ML-1M; **Amazon Beauty (2014 5-core)**. SHA256 + OSF mirror. Threshold 1000 users → template B. Quartiles frozen once.

**4.2** Full catalog; lex Top-K; test = train+val mask.

**4.3 RQ1** as table above.

**4.4 RQ2 (main text includes retrained \(K\)):**  
- Frozen Game A: \(v(C)\), \(\varphi\), flag.  
- Test NDCG of retrained \(K=1,2,3,4\).  
- Row: frozen coalition \(\{1\}\) vs retrained \(K=1\) (masking ≠ architecture).

**4.5 RQ3** \(\Delta\) first if defined.

**4.6 RQ4** one subsection.

**4.7** Sensitivities: \(d\), cutoffs, BPR-neg, test-mask, flag 0.10/0.20.

**4.8** Two-layer uncertainty.

---

# 5. DISCUSSION

Estimand; aggregate-only skip patterns; selection bias; 0.005 = floor; flag ≠ lemma ≠ prune; Game B incomparable; Adaptive discontinuity owned; BPR-neg convention; BPR grid ≠ equal prior; two graphs ≠ density.

---

# 6. CONCLUSION

Honest instantiation + diagnostics. Short note if G2 and G3 fail.

---

# DECLARATIONS

Pre-reg tag / OSF · SHA256s · **data mirror URL** · lockfile · seeds · CRediT · AI.

---

# APPENDICES

A. Gadgets + \(\varphi=(5/3,8/3,11/3)\), \(I=1/2\)  
B. Eight \(v(C)\)  
C. LightGCN fixed; BPR 6-tuple list  
D. Similarity, SII, flag co-occurrence (not \(r\))  
E. \(\Delta\), mix identity  
F. Game B if triggered; BPR-neg sensitivity; \(\beta\) appendix; per-seed Adaptive weights

---

# LAST-PASS FREEZES

| Item | Decision |
|---|---|
| \(I_{12}\) | \(1/2\) (reviews retracted \(1\)) |
| Abstract | “Amazon Beauty (2014 5-core)” |
| Beauty \(\lvert U\rvert<1000\) | Exploratory; RQ3 confirmatory ML-1M |
| RQ3 \(\Delta\) | Only if both \(v_m\ge 0.005\) |
| Adaptive | Same 0.005 floor |
| Val-\(\alpha\) ties | Closest to uniform, then lex |
| Holm | Within-dataset; Wilcoxon two-sided |
| RQ1 \(\varphi\) | Order + signs + cosine, not Spearman |
| Top-K | Lex sort, not \(10^{-12}\) jitter |
| Cosine gate | Full side, row-norm, no subsample |
| Retrained \(K\) | RQ2 main text |
| Flag | Not a prune claim |
| BPR grid | 6 listed tuples |
| Quartiles | Once, reused |
| Data | SHA256 + OSF mirror before tag |
