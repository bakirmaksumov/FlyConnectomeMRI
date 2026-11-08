# Proposed paper tables

All values come from frozen files. The source of each table is given below it. Values are mean ± SD unless stated otherwise.

## TABLE I: Dataset and evaluation protocol

| Item | Value |
|---|---|
| Dataset | Cheng et al. brain tumour dataset, Figshare 1512427 v8 (original `.mat`, CC BY 4.0) |
| Full release | 3,064 T1-CE slices, 233 patients (meningioma 708/82, glioma 1,426/89, pituitary 930/62) |
| Binary task | glioma (positive) vs meningioma; pituitary excluded |
| Cohort | 171 patients, 2,134 slices (1,426 glioma, 708 meningioma) |
| Patient grouping | published `patient_id`, plus a conservative `patient_link_key` (42 IDs → 18 families) |
| Input | 1×128×128; per-slice min–max scaling; train-only normalisation; no augmentation |
| Fixed split (Phases 8–9) | train 124 / val 26 / test 21 patients (1,506 / 326 / 302 slices) |
| Main evaluation (Phase 10) | Leakage-safe CV using the official Figshare outer test folds: test = official fold *k*; val = fold *k*%5+1; linked relatives excluded from train/val |
| Official test folds (patients) | 33 / 34 / 35 / 35 / 34 (slices 357 / 505 / 370 / 449 / 453) |
| Linked patients excluded from train+val per fold | 14 / 14 / 19 / 18 / 11 (all `MR`-prefixed gliomas) |
| Leakage checks | zero `patient_id` and zero `patient_link_key` overlap; each of 171 patients tested exactly once |
| Primary metric | patient-level ROC AUC (mean of slice probabilities) |

*Sources: `data/splits/figshare_glioma_meningioma/dataset_metadata.json`, `docs/results/phase10_prerun/counts.csv`, `cv_prerun_report.json`.*

## TABLE II: Visual graph and control graph properties (128 nodes, 989 edges)

| Graph | Density | Max in / out degree | Reciprocity | Clustering | Strongly conn. components | Edge overlap with biological |
|---|---|---|---|---|---|---|
| Visual Drosophila (raw, w≥1) | 0.061 | 38 / 20 | 0.510 | 0.381 | 12 (largest 117) | 1.000 |
| Degree-preserving rewired* | 0.061 | 38 / 20 | 0.084 | 0.097 | 11.2 | 0.130 |
| ER* | 0.061 | 14.4 / 15.6 | 0.057 | 0.062 | 1.2 | 0.057 |
| WS (directed variant)* | 0.061 | 11.2 / 8.0 | 0.795 | 0.479 | 1.0 | 0.065 |
| BA (directed variant)* | 0.061 | 29.2 / 28.6 | 0.000 | 0.102 | 1.0 | 0.056 |
| *Reference:* EB w20 (Phase 8) | 0.187 | 53 / 71 | 0.519 | 0.496 | 1 | n/a (3,047 edges) |

\* Mean over control seeds 0–4. The rewired graph keeps every node's exact in- and out-degree.

*Sources: `data/connectome/*_metadata.json` (frozen checksums in `docs/RESEARCH_PLAN.md` and `docs/results/phase10_prerun/graph_checksums.txt`).*

## TABLE III: Phase 10 pooled out-of-fold patient-level performance (mean ± SD across 5 seeds; threshold 0.5)

| Model | Params | ROC AUC | Balanced acc. | Sensitivity | Specificity | F1 | Precision |
|---|---|---|---|---|---|---|---|
| CNN baseline | 41,921 | 0.834 ± 0.028 | 0.766 ± 0.033 | 0.791 ± 0.036 | 0.741 ± 0.067 | 0.780 ± 0.026 | 0.771 ± 0.044 |
| Visual Drosophila | 44,411 | 0.864 ± 0.033 | 0.733 ± 0.055 | 0.879 ± 0.045 | 0.588 ± 0.075 | 0.778 ± 0.044 | 0.699 ± 0.047 |
| Visual rewired | 44,411 | 0.874 ± 0.054 | 0.749 ± 0.066 | 0.870 ± 0.106 | 0.629 ± 0.156 | 0.786 ± 0.055 | 0.727 ± 0.085 |
| Visual WS | 44,411 | 0.874 ± 0.025 | 0.762 ± 0.060 | 0.872 ± 0.058 | 0.651 ± 0.174 | 0.797 ± 0.028 | 0.743 ± 0.083 |

Recall equals sensitivity. Each seed's OOF set contains all 171 patients exactly once.

*Source: `docs/results/phase10/oof_summary.csv`.*

## TABLE IV: Phase 10 fold-level patient ROC AUC (mean over 5 seeds per fold)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD | Median [min, max] |
|---|---|---|---|---|---|---|---|
| CNN baseline | 0.905 | 0.862 | 0.899 | 0.856 | 0.803 | 0.865 ± 0.041 | 0.862 [0.803, 0.905] |
| Visual Drosophila | 0.938 | 0.881 | 0.919 | 0.936 | 0.951 | 0.925 ± 0.027 | 0.936 [0.881, 0.951] |
| Visual rewired | 0.939 | 0.881 | 0.915 | 0.951 | 0.946 | 0.926 ± 0.029 | 0.939 [0.881, 0.951] |
| Visual WS | 0.944 | 0.883 | 0.931 | 0.951 | 0.958 | 0.934 ± 0.030 | 0.944 [0.883, 0.958] |

Fold-level slice ROC AUC: CNN 0.802 ± 0.091, Drosophila 0.917 ± 0.023, rewired 0.921 ± 0.026, WS 0.921 ± 0.023.

*Sources: `docs/results/phase10/fold_level_metrics.csv`, `fold_level_summary.csv`.*

## TABLE V: Primary paired topology comparison, Visual Drosophila − Visual rewired (Phase 10)

| Metric | All 25 fold×seed pairs | Higher / lower / tied | Fold-level (5 values) |
|---|---|---|---|
| Patient ROC AUC | −0.002 ± 0.030 | 12 / 13 / 0 | −0.002 ± 0.008 |
| Slice ROC AUC | −0.004 ± 0.021 | 8 / 17 / 0 | −0.004 ± 0.013 |
| Patient balanced accuracy | −0.016 ± 0.171 | 10 / 15 / 0 | −0.016 ± 0.029 |

- **Per-fold patient ROC AUC differences:** −0.001, −0.001, +0.004, −0.015, +0.005.
- **Secondary comparisons (patient ROC AUC):**
  - Drosophila − WS: −0.009 ± 0.040 (10 / 14 / 1; fold-level −0.009 ± 0.005).
  - Drosophila − CNN: +0.060 ± 0.063 (24 / 1 / 0). This is an **architectural** comparison, not a topology comparison.

*Sources: `docs/results/phase10/paired_summary.csv`, `paired_fold_means.csv`, `paired_drosophila_rewired.csv`.*

## Supplementary TABLE S1: Earlier fixed-split experiments (test = 21 patients; 5 seeds)

| Experiment | Drosophila | Rewired | ER | WS | BA | CNN | Drosophila − rewired |
|---|---|---|---|---|---|---|---|
| Phase 8, EB w20 (48,527 params) | 0.927 ± 0.044 | 0.942 ± 0.041 | 0.960 ± 0.018 | 0.933 ± 0.039 | 0.963 ± 0.036 | 0.900 ± 0.009 | −0.015 ± 0.041 (2/5 higher) |
| Phase 9, visual raw (44,411 params) | 0.944 ± 0.029 | 0.927 ± 0.053 | 0.931 ± 0.037 | 0.960 ± 0.025 | 0.927 ± 0.044 | 0.900 ± 0.009† | +0.017 ± 0.061 (3/5 higher) |

Patient ROC AUC. † Phase 8 CNN reference, not retrained.

*Sources: `docs/results/phase8/`, `docs/results/phase9/` (`summary_by_model.csv`, `paired_summary.csv`).*
