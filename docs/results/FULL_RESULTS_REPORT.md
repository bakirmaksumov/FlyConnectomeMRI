# FlyConnectomeMRI: full results report (Phases 8–10)

*Compiled 2026-10-14 from the frozen result files only. No new run, re-analysis or tuning was done to write this report.*

**Sources:** [RESEARCH_PLAN.md](../RESEARCH_PLAN.md), [phase8/summary.md](phase8/summary.md), [phase9/summary.md](phase9/summary.md), [phase10/summary.md](phase10/summary.md), [phase10_prerun/](phase10_prerun/) and the CSV files next to them. Each phase folder has a `SHA256SUMS.txt`.

**Reading rules for every number below:**
- The values are descriptive. Seeds are training seeds, not independent samples, so **no statistical significance is claimed**.
- **No clinical claim** is made. This is a research prototype, not a medical device.
- **Possible site confound.** All 67 `MR…`-prefixed patients are glioma, so absolute scores may partly reflect hospital or scanner differences rather than tumour appearance.
- **CNN vs graph models is not a pure topology comparison.** The CNN has 41,921 parameters, while the graph models have 48,527 (EB) or 44,411 (visual).

---

## 1. Short answer

| Question | Result |
|---|---|
| Does the real *Drosophila* wiring beat its degree-preserving rewired null? | **No.** Across 3 experiments the difference was −0.015 (EB, Phase 8), +0.017 (visual, Phase 9) and −0.002 (visual, 5-fold CV, Phase 10). Every difference is well within the run-to-run spread. |
| Do graph-constrained models beat the CNN baseline on ROC AUC? | Yes, in this setup. But random/control graphs did so equally or better, so the gain comes from the **graph-model architecture**, not from biological topology. |
| Which model had the highest mean? | It was a control graph every time: BA (Phase 8), WS (Phase 9) and WS (Phase 10, fold level). |
| Did any run fail? | No. 155 of 155 runs succeeded (30 + 25 + 100), and none was repeated or replaced. |

**Frozen conclusion (Phase 10, the most robust experiment):**

> Leakage-safe five-fold cross-validation across the full 171-patient cohort did not demonstrate a topology-specific advantage for the MaleCNS visual-system graph. Patient-level ROC AUC difference (Visual Drosophila − rewired) across 25 paired fold×seed runs: −0.002 ± 0.030; fold level −0.002 ± 0.008; the biological graph was higher in 12 of 25 paired runs and lower in 13 of 25.

---

## 2. Setup

### 2.1 Data

| Item | Value |
|---|---|
| Dataset | Cheng J., *brain tumor dataset*, Figshare v8, doi:10.6084/m9.figshare.1512427, CC BY 4.0 (original `.mat` files, MD5-verified) |
| Task | Glioma (1) vs meningioma (0); pituitary slices excluded |
| Cohort | 171 patients and 2,134 slices: 89 glioma (1,426 slices) and 82 meningioma (708 slices) |
| Patient linking | `^(?P<base>MR\d{6})[A-Z]?$` puts 42 IDs into 18 families, giving 147 independent groups. All linked IDs are glioma. |
| Preprocessing | Per-image min-max to [0, 1], resize to 128×128, normalisation with train-split statistics. No augmentation. |
| Phases 8–9 split | Seed 0, 70/15/15 stratified group split: train 124 patients / 1,506 slices, val 26 / 326, **test 21 / 302** |
| Phase 10 split | Official Figshare `cvind.mat` outer folds, leakage-safe (Section 5.1) |

### 2.2 Graphs

| Graph | Selection | Nodes | Edges | Density | Role |
|---|---|---|---|---|---|
| EB-128 **w20** | Top-128 MaleCNS v1.0 neurons by ellipsoid-body synapses, edges with weight ≥ 20 | 128 | 3,047 | 0.187 | Engineering/provenance graph (**not visual**) |
| Visual-128 **raw** | Local retinotopic columnar patch in ME(R); 11 types (L1, L2, L3, Mi1, Mi4, Mi9, C3, Tm1, Tm2, Tm4, Tm9); centre column (18, 20); edges with weight ≥ 1 | 128 | 989 | 0.061 | Biologically motivated graph |

Each biological graph has four matched controls with the same node and edge counts, seeds 0–4:
- **Degree-preserving rewired.** This is the **primary null**. It keeps every neuron's in- and out-degree.
- **ER.**
- **Directed WS** (rewiring probability 0.1).
- **Directed BA.**

### 2.3 Models and training (frozen; identical for every model)

```
CNN baseline : MRI -> CNN encoder (128-d) -> Linear(128, 1)                                   41,921 params
Graph model  : MRI -> same CNN encoder -> BatchNorm1d(128) -> ConnectomeLayer x2 -> Linear(128, 1)
               ConnectomeLayer: h_j = ReLU( sum_{i->j in graph} w_ij * h_i + b_j )            EB 48,527 / visual 44,411
```

| Setting | Value |
|---|---|
| Encoder | Conv(1→16)–ReLU–Pool → Conv(16→32)–ReLU–Pool → Conv(32→128)–ReLU → AdaptiveAvgPool |
| Loss | BCEWithLogitsLoss with `pos_weight` computed from the training set only |
| Optimiser | Adam with lr 1e-3, weight decay 1e-4, batch size 32, at most 30 epochs, early-stopping patience 7 |
| Checkpoint selection | Best validation slice ROC AUC |
| Threshold | 0.5, fixed and never tuned |
| Patient score | Mean of the patient's slice probabilities |
| Validation gate | Technical degeneration only (constant output, non-finite values, dead graph output); no performance threshold |

---

## 3. Phase 8: EB-128 w20 graph (single split, 5 seeds)

Batch `results/phase8_20260928-235044`, commit `a3c966b`. **30/30 runs succeeded.** Test set: 21 patients.

### 3.1 Mean ± SD over 5 seeds

| Model | Patient ROC AUC | Slice ROC AUC | Patient bal. acc. | Patient sensitivity | Patient specificity |
|---|---|---|---|---|---|
| CNN baseline | 0.900 ± 0.009 | 0.806 ± 0.017 | 0.839 ± 0.029 | 0.754 ± 0.114 | 0.925 ± 0.112 |
| **Drosophila w20** | **0.927 ± 0.044** | **0.905 ± 0.029** | **0.862 ± 0.044** | **0.723 ± 0.088** | **1.000 ± 0.000** |
| Rewired (primary null) | 0.942 ± 0.041 | 0.913 ± 0.019 | 0.860 ± 0.074 | 0.769 ± 0.196 | 0.950 ± 0.112 |
| ER | 0.960 ± 0.018 | 0.930 ± 0.011 | 0.853 ± 0.031 | 0.831 ± 0.126 | 0.875 ± 0.177 |
| WS | 0.933 ± 0.039 | 0.911 ± 0.023 | 0.785 ± 0.169 | 0.569 ± 0.338 | 1.000 ± 0.000 |
| BA | 0.963 ± 0.036 | 0.937 ± 0.036 | 0.838 ± 0.177 | 0.677 ± 0.354 | 1.000 ± 0.000 |

### 3.2 Paired differences, Drosophila w20 − control (per seed, then mean ± SD)

| Comparison | Patient ROC AUC | Seeds Drosophila higher / lower / tied | Slice ROC AUC | Patient bal. acc. |
|---|---|---|---|---|
| **− rewired (primary)** | **−0.015 ± 0.041** | 2 / 3 / 0 | −0.007 ± 0.024 | +0.002 ± 0.091 |
| − ER | −0.033 ± 0.037 | 1 / 4 / 0 | −0.025 ± 0.038 | +0.009 ± 0.052 |
| − WS | −0.006 ± 0.041 | 2 / 3 / 0 | −0.006 ± 0.043 | +0.077 ± 0.174 |
| − BA | −0.037 ± 0.036 | 0 / 4 / 1 | −0.032 ± 0.036 | +0.023 ± 0.186 |

**Phase 8 finding:** the EB wiring showed no advantage over any control. On ROC AUC it was the lowest graph model on average.

Figures: [phase8/](phase8/), `fig1`–`fig5`.

---

## 4. Phase 9: visual-system graph (single split, 5 seeds)

Batch `results/phase9_20260929-110630`, commit `77eead1`. **25/25 runs succeeded.** The CNN was not retrained; its Phase 8 numbers are shown for reference.

### 4.1 Mean ± SD over 5 seeds

| Model | Patient ROC AUC (primary) | Slice ROC AUC | Patient bal. acc. | Patient sensitivity | Patient specificity |
|---|---|---|---|---|---|
| CNN baseline (Phase 8 reference) | 0.900 ± 0.009 | 0.806 ± 0.017 | 0.839 ± 0.029 | 0.754 ± 0.114 | 0.925 ± 0.112 |
| **Visual Drosophila** | **0.944 ± 0.029** | **0.922 ± 0.021** | **0.857 ± 0.039** | **0.738 ± 0.069** | **0.975 ± 0.056** |
| Visual rewired (primary null) | 0.927 ± 0.053 | 0.914 ± 0.034 | 0.838 ± 0.157 | 0.677 ± 0.314 | 1.000 ± 0.000 |
| Visual ER | 0.931 ± 0.037 | 0.907 ± 0.027 | 0.769 ± 0.119 | 0.538 ± 0.237 | 1.000 ± 0.000 |
| Visual WS | 0.960 ± 0.025 | 0.930 ± 0.023 | 0.872 ± 0.074 | 0.769 ± 0.154 | 0.975 ± 0.056 |
| Visual BA | 0.927 ± 0.044 | 0.918 ± 0.022 | 0.762 ± 0.137 | 0.523 ± 0.274 | 1.000 ± 0.000 |

### 4.2 Paired differences, Visual Drosophila − control

| Comparison | Patient ROC AUC | Seeds Drosophila higher / lower / tied | Slice ROC AUC | Patient bal. acc. |
|---|---|---|---|---|
| **− rewired (primary)** | **+0.017 ± 0.061** | 3 / 2 / 0 | +0.008 ± 0.029 | +0.018 ± 0.181 |
| − ER | +0.013 ± 0.049 | 3 / 2 / 0 | +0.015 ± 0.038 | +0.087 ± 0.134 |
| − WS | −0.015 ± 0.033 | 1 / 3 / 1 | −0.008 ± 0.027 | −0.015 ± 0.100 |
| − BA | +0.017 ± 0.055 | 3 / 2 / 0 | +0.004 ± 0.030 | +0.095 ± 0.142 |

**Phase 9 finding:** Visual Drosophila had a slightly higher mean than its rewired null, but the difference was well within the seed-to-seed spread. **No topology-specific advantage was demonstrated.**
- Visual WS had the highest mean.
- Visual Drosophila had the smallest spread in patient balanced accuracy (±0.039) and sensitivity (±0.069). In contrast, rewired, ER and BA each had seeds that called almost every patient meningioma. This is descriptive only.

### 4.3 EB vs visual (side by side only; not a controlled comparison)

| Family | EB w20 patient ROC AUC | Visual raw patient ROC AUC |
|---|---|---|
| Drosophila | 0.927 ± 0.044 | 0.944 ± 0.029 |
| Rewired | 0.942 ± 0.041 | 0.927 ± 0.053 |
| ER | 0.960 ± 0.018 | 0.931 ± 0.037 |
| WS | 0.933 ± 0.039 | 0.960 ± 0.025 |
| BA | 0.963 ± 0.036 | 0.927 ± 0.044 |

The two graphs differ in density (0.19 vs 0.06) and parameter count (48,527 vs 44,411).

Figures: [phase9/](phase9/), `fig1`–`fig6`.

---

## 5. Phase 10: leakage-safe 5-fold cross-validation (official Figshare outer folds)

Batch `results/phase10_20260929-134229`, run at commit `45133b9`, after the protocol (`c81615d`) and the pre-registered analysis code (`feac0c3`) were committed. **100/100 runs succeeded:** 4 models × 5 outer folds × 5 seeds.

### 5.1 Fold construction

- **Test:** official fold *k*, unchanged.
- **Validation:** official fold `k % 5 + 1`.
- **Training:** the other three folds.
- **Leakage exclusion:** any patient whose `patient_link_key` occurs in test is removed from training and validation, and any patient linked to the cleaned validation set is removed from training.
- **Never done:** no patient was moved to another fold.

| Outer fold | Train patients (removed) | Val patients (removed) | Test patients | Train `pos_weight` |
|---|---|---|---|---|
| 1 | 91 (13) | 33 (1) | 33 | 0.679 |
| 2 | 89 (13) | 34 (1) | 34 | 0.663 |
| 3 | 86 (15) | 31 (4) | 35 | 0.779 |
| 4 | 88 (14) | 30 (4) | 35 | 0.798 |
| 5 | 95 (9) | 31 (2) | 34 | 0.597 |

- **Exclusions:** every removed patient is a glioma with an `MR` ID.
- **Overlap check:** zero patient-ID and zero link-key overlap between train, validation and test in every fold.
- **OOF coverage:** all 171 patients were tested exactly once.

### 5.2 Patient ROC AUC (primary metric)

| Model | Fold level: mean ± SD over 5 folds (seeds averaged first) | Run level: 25 runs, mean ± SD [min–max] | Pooled OOF: mean ± SD over 5 seeds |
|---|---|---|---|
| CNN baseline | 0.865 ± 0.041 | 0.865 ± 0.040 [0.760–0.915] | 0.834 ± 0.028 |
| **Visual Drosophila** | **0.925 ± 0.027** | **0.925 ± 0.038 [0.833–0.997]** | **0.864 ± 0.033** |
| Visual rewired | 0.926 ± 0.029 | 0.926 ± 0.041 [0.847–1.000] | 0.874 ± 0.054 |
| Visual WS | 0.934 ± 0.030 | 0.934 ± 0.042 [0.868–0.993] | 0.874 ± 0.025 |

**Per fold** (patient ROC AUC, mean over 5 seeds):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| CNN baseline | 0.905 | 0.863 | 0.899 | 0.856 | 0.803 |
| Visual Drosophila | 0.938 | 0.881 | 0.919 | 0.936 | 0.951 |
| Visual rewired | 0.939 | 0.881 | 0.915 | 0.951 | 0.946 |
| Visual WS | 0.944 | 0.883 | 0.931 | 0.951 | 0.958 |

### 5.3 Secondary metrics (fold level, mean ± SD over 5 folds)

| Model | Slice ROC AUC | Patient bal. acc. | Slice bal. acc. |
|---|---|---|---|
| CNN baseline | 0.802 ± 0.091 | 0.766 ± 0.075 | 0.706 ± 0.085 |
| Visual Drosophila | 0.917 ± 0.023 | 0.732 ± 0.078 | 0.739 ± 0.084 |
| Visual rewired | 0.921 ± 0.026 | 0.749 ± 0.061 | 0.747 ± 0.057 |
| Visual WS | 0.921 ± 0.023 | 0.761 ± 0.063 | 0.753 ± 0.072 |

### 5.4 Pooled out-of-fold patient metrics (171 patients, mean ± SD over 5 seeds, threshold 0.5)

| Model | ROC AUC | Bal. acc. | Sensitivity | Specificity | F1 | Precision |
|---|---|---|---|---|---|---|
| CNN baseline | 0.834 ± 0.028 | 0.766 ± 0.033 | 0.791 ± 0.036 | 0.741 ± 0.067 | 0.780 ± 0.026 | 0.771 ± 0.044 |
| Visual Drosophila | 0.864 ± 0.033 | 0.733 ± 0.055 | 0.879 ± 0.045 | 0.588 ± 0.075 | 0.778 ± 0.044 | 0.699 ± 0.047 |
| Visual rewired | 0.874 ± 0.054 | 0.749 ± 0.066 | 0.870 ± 0.106 | 0.629 ± 0.156 | 0.786 ± 0.055 | 0.727 ± 0.085 |
| Visual WS | 0.874 ± 0.025 | 0.762 ± 0.060 | 0.872 ± 0.058 | 0.651 ± 0.174 | 0.797 ± 0.028 | 0.743 ± 0.083 |

### 5.5 Paired comparisons (matched by outer fold + seed; 25 pairs, 0 NA)

| Comparison | Patient ROC AUC, 25 pairs | Higher / lower / tied | Fold-level mean ± SD | Slice ROC AUC | Patient bal. acc. |
|---|---|---|---|---|---|
| **Drosophila − rewired (primary)** | **−0.002 ± 0.030** | **12 / 13 / 0** | **−0.002 ± 0.008** | −0.004 ± 0.021 | −0.016 ± 0.171 |
| Drosophila − WS | −0.009 ± 0.040 | 10 / 14 / 1 | −0.009 ± 0.005 | −0.004 ± 0.026 | −0.029 ± 0.191 |
| Drosophila − CNN (architectural, not topological) | +0.060 ± 0.063 | 24 / 1 / 0 | +0.060 ± 0.055 | +0.115 ± 0.076 | −0.033 ± 0.172 |

**Primary comparison per fold** (Drosophila − rewired, patient ROC AUC, mean of 5 seeds):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|
| −0.001 | −0.001 | +0.004 | −0.015 | +0.005 |

### 5.6 Training cost (CPU)

| Model | Mean training time per run, s [min–max] | Mean selected epoch [min–max] |
|---|---|---|
| CNN baseline | 147.1 [47.8–199.0] | 18.5 [1–30] |
| Visual Drosophila | 166.0 [96.3–218.8] | 19.6 [8–30] |
| Visual rewired | 168.5 [84.9–220.7] | 20.6 [7–30] |
| Visual WS | 166.2 [87.3–219.3] | 19.6 [7–30] |

The total training time across all 100 runs was about 4.5 h.

### 5.7 Phase 10 observations (descriptive; recorded, not acted on)

- **Specificity at the fixed 0.5 threshold.** The graph models had lower pooled OOF patient specificity (0.59–0.65) than the CNN (0.74). As a result, the CNN's balanced accuracy was comparable even though its ROC AUC was lower. Several graph runs predicted almost every test patient as glioma (specificity 0.000–0.125). The threshold was not tuned, as the protocol required.
- **OOF vs fold-level AUC.** Pooled OOF ROC AUC is lower than fold-level ROC AUC because it mixes probabilities from five separately trained models.
- **CNN on folds 4–5.** The CNN was weakest on fold 5 (0.803) and stopped very early on fold 4 (selected epochs 1–4). This drives most of the Drosophila − CNN gap (+0.080 and +0.148 on those folds).

Figures: [phase10/](phase10/), `fig1`–`fig9`.

---

## 6. All primary comparisons together

| Experiment | Biological graph | Evaluation | Drosophila − rewired (patient ROC AUC) | Drosophila higher in |
|---|---|---|---|---|
| Phase 8 | EB-128 w20 | 21-patient test, 5 seeds | −0.015 ± 0.041 | 2/5 seeds |
| Phase 9 | Visual-128 raw | 21-patient test, 5 seeds | +0.017 ± 0.061 | 3/5 seeds |
| Phase 10 | Visual-128 raw | 171 patients, 5 folds × 5 seeds | −0.002 ± 0.030 (fold level −0.002 ± 0.008) | 12/25 pairs |

The sign of the difference changes between experiments. In every experiment its magnitude is far smaller than its spread, and the largest and most robust experiment (Phase 10) puts it at essentially zero.

---

## 7. Conclusions

1. **No topology-specific advantage.** The specific *Drosophila* wiring, both the EB graph and the visual columnar circuit, gave no measurable advantage over a degree-preserving rewired graph with the same degrees for glioma-vs-meningioma MRI classification.
2. **The graph architecture helps ROC AUC, but not because of biology.** Graph-constrained models had higher ROC AUC than the CNN baseline, by +0.060 in Phase 10. Random and control graphs reached the same or higher values, so this gain belongs to the architecture, not to biological topology. The comparison also differs in parameter count.
3. **Threshold-dependent metrics favour neither.** At the fixed 0.5 threshold the graph models traded specificity for sensitivity, and balanced accuracy did not favour them over the CNN.
4. **No significance or clinical claims.** None are made.

## 8. Limitations

- **Site confound:** class is aligned with the ID format, possibly reflecting the acquisition site. A numeric-ID subgroup analysis is pre-declared but **not yet run**.
- **Seeds:** training seeds are not independent experiments, and Phases 8–9 used only a 21-patient test set.
- **Leakage-safe folds:** the folds remove linked glioma patients from training and validation, which slightly changes class balance per fold.
- **Visual graph coverage:** the visual graph is the columnar input stage only. T4/T5 and Tm3 are not included because MaleCNS has no column assignment for them.
- **One task, one architecture:** only one dataset, task and graph-layer architecture was tested, at 128×128 resolution.
- **Not yet run:** the pre-declared visual w ≥ 5 sensitivity analysis.

## 9. Reproducibility pointers

| Item | Location |
|---|---|
| Protocols and frozen decisions (dated, pre-registered) | [docs/RESEARCH_PLAN.md](../RESEARCH_PLAN.md) |
| Phase 8 tables, figures, run index | [docs/results/phase8/](phase8/) |
| Phase 9 tables, figures, run index | [docs/results/phase9/](phase9/) |
| Phase 10 tables, figures, OOF predictions | [docs/results/phase10/](phase10/) |
| Phase 10 fold construction, exclusions, checksums | [docs/results/phase10_prerun/](phase10_prerun/) |
| Frozen-decision guard tests | `tests/test_frozen_decisions.py` |
