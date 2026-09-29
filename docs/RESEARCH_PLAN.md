# Research plan

*A living document. Every change that affects how results are interpreted is recorded here with a date, **before** the experiments it affects are run.*

## Question

Does constraining a neural network's connectivity with a graph derived from the real *Drosophila* connectome change its performance on binary brain-MRI classification (tumor vs normal)? The comparison is against the same network constrained by random control graphs, and against a conventional CNN.

The wording used throughout is: *a neural network whose connectivity is constrained by a Drosophila-derived graph*. No claim is made that "the fly brain detects tumors", and nothing here is a medical device.

## Status of the biological graph

> **The EB graph is an engineering / provenance validation graph. It is not the final, biologically motivated visual-system graph.**

- The current graph is `data/connectome/drosophila_128.csv`: the 128 MaleCNS v1.0 neurons with the most synapses in the ellipsoid body (EB). EB is part of the central complex, a navigation and head-direction region, and is not a visual-system region.
- It was selected by a generic rule ("most synapses in one ROI"). Its purpose is to prove the whole pipeline works on real data: download, provenance, validation, filtering and control generation.
- **Before the final MRI experiments**, a visual-system extraction step will select neurons using MaleCNS visual ROIs (for example `ME`, `LO`, `LOP`, `LA`, `AME`, where appropriate). Its selection rule will be written into this document before it is run. Results obtained with the EB graph will be labelled as pipeline-validation results.

## Pre-declared decisions

### 2026-09-28: primary edge-weight threshold `w20`

Declared **before any MRI training**.

| Variant | Rule | Edges | Density | Role |
|---|---|---|---|---|
| raw | weight ≥ 1 | 12,899 | 0.793 | sensitivity analysis |
| w5 | weight ≥ 5 | 8,518 | 0.524 | sensitivity analysis |
| w10 | weight ≥ 10 | 5,804 | 0.357 | sensitivity analysis |
| **w20** | **weight ≥ 20** | **3,047** | **0.187** | **PRIMARY** |

- **Reason:** in the raw EB graph 79% of all possible connections exist, so a biological mask would barely differ from a dense layer, and from a random mask of the same density. Keeping connections with at least 20 synapses lowers the density to about 19%. That makes the topology meaningfully sparse while still keeping more than 3,000 biological connections, which carry 71% of the total synapse weight.
- **Rule:** the threshold uses only the biological synapse count. **It must not be tuned later based on MRI performance.** The other thresholds exist only for sensitivity analysis.
- **Where it's recorded:** `configs/graph_variants.yaml`, and each variant's `*_metadata.json`.

### 2026-09-28: control graphs (Phase 4)

All controls are matched to the primary `w20` graph. Each has the same node count (128) and **exactly** the same edge count (3,047), is directed, has no self-loops, and is generated deterministically from a seed (seeds 0–4). They use no MRI data.

| Control | Preserves | Randomises | Role |
|---|---|---|---|
| **rewired** (degree-preserving) | every neuron's exact in- and out-degree; each neuron's total outgoing synapse weight | which neurons connect to which | **most important null model** |
| ER | node and edge count | everything else | structurally different null model |
| WS (directed variant) | node and edge count; lattice-like local structure | 10% of edges | structurally different null model |
| BA (directed variant) | node and edge count; hub-forming growth | everything else | structurally different null model |

ER, WS and BA are *structurally different* null models, not random copies of the fly graph. Only the rewired control isolates the question "does the **specific** biological wiring matter, beyond its degree distribution?"

The rewired control's edges still overlap about 35% with the biological edges. That's the expected chance level for random graphs with exactly these degrees (computed analytically at 0.353), and it stops decreasing after about 3 swaps per edge, so it's not a sign of incomplete mixing.

## FROZEN decisions (2026-09-28, before any MRI experiment)

These choices are final for all MRI experiments on the EB graph and must not be changed after seeing MRI results:

| Decision | Frozen value |
|---|---|
| Primary biological graph | Drosophila EB-128 **w20** (weight ≥ 20) |
| Sensitivity-only variants | raw (≥1), w5, w10 |
| **Primary null model** | **degree-preserving rewired graph** |
| Additional controls | ER, WS (directed variant), BA (directed variant) |
| Control seeds | 0, 1, 2, 3, 4 |
| WS rewiring probability | 0.1 |
| Rewiring swaps per edge | 10 |

The custom directed WS and BA construction rules are documented in `src/flyconnectome_mri/connectome/random_graphs.py` and must remain unchanged. `tests/test_frozen_decisions.py` fails if a frozen config value or any construction rule's output changes.

Frozen data files (SHA-256; the files are git-ignored but can be regenerated deterministically):

| File | SHA-256 |
|---|---|
| drosophila_128.csv (raw download) | `ae4aa6c4a51209455389cbfed18686a10133eb55a4367853507aa23c82458c22` |
| drosophila_128_w20.csv | `878db46d53584f31a9f699fd951b1730e6f04c4d9a0b35ca4989d64acfe118ad` |
| rewired_128_w20_seed0.csv | `ecc4f2441747edf698e1f1fec7ce49e1cdade5d9a5f5dacc3eb64f7397ebc202` |
| er_128_w20_seed0.csv | `c422f106aa52259edd5cf4de82f436ec20adda983dabf1f6ce591c5a8e0c84fb` |
| ws_128_w20_seed0.csv | `41070c2b46dfd37a7238dee4d089050dd450f1db9203e99417a458d77e5e8d1b` |
| ba_128_w20_seed0.csv | `914ada71361b87087104c358316202c5f3be05c708ef86f4ccde219676b7f61c` |

All derived and control files were regenerated after commit `a3669f1`, and their CSV checksums were identical to the first generation.

**Provenance note.** The raw download's metadata records git commit `2c00739`. The fetch was run before the Phase 3 fetch code was committed; that code is in commit `4863ba5`. The raw metadata is deliberately left unmodified.

## First real MRI dataset (declared 2026-09-28, before any training)

- **Source:** Cheng J., *brain tumor dataset*, Figshare v8, https://doi.org/10.6084/m9.figshare.1512427, CC BY 4.0. This is the original `.mat` files, not a Kaggle conversion. All six files were MD5-verified against Figshare, and the originals and extracted `.mat` files are read-only.
- **Task:** **Glioma (1) vs Meningioma (0)**, a binary task. Pituitary slices (930 slices, 62 patients) are excluded from this dataset definition, and their files are untouched. Tumor-vs-Normal isn't possible here, because this dataset contains no normal scans.
- **Patients:** `cjdata.PID` is used exactly as published. 171 patients: 89 glioma (1,426 slices) and 82 meningioma (708 slices).
- **Conservative linking.** 42 patient IDs share an `MR` base number with other IDs (for example `MR024780B`/`D`/`E`/`G`, or `MR040240` with `MR040240B`/`C`/`D`). They may be different exams of one person. `patient_link_regex: ^(?P<base>MR\d{6})[A-Z]?$` keeps each such family in one split, which gives 147 independent groups. All 42 linked IDs are glioma.
- **Split:** seed 0, 70/15/15 stratified group split.
  - train: 124 patients, 1,506 slices
  - val: 26 patients, 326 slices
  - test: 21 patients, 302 slices
  - No patient ID and no link key is shared between splits.
- **Official 5-fold CV** (`cvind.mat`): it maps cleanly to patients, since no patient spans two folds. Each slice's fold is stored in the manifest (`cv_fold`) for later experiments. The train/val/test split above does not use it.
- **Preprocessing:** per-image min-max scaling to [0, 1] (as in the authors' own conversion code), then a resize to 128×128 (15 slices are natively 256×256), then normalization with train-split mean/std. There's no augmentation.
- **Class imbalance:** about 67% of slices are glioma. Accuracy alone is therefore misleading, so ROC AUC, sensitivity and specificity must be reported (Phase 6).

### Potential confound found 2026-09-28 (after the first seed-0 baseline run)

The patient-ID *format* is almost perfectly aligned with the class:

| ID format | Glioma patients | Meningioma patients |
|---|---|---|
| `MR…` (e.g. MR024780B) | 67 | **0** |
| numeric (e.g. 100360) | 22 | 82 |

The README says the images came from two hospitals but doesn't say which IDs belong to which. **If** the ID format marks the hospital or scanner, class and acquisition site are strongly confounded, and any model (the CNN baseline and the graph models alike) could separate the classes partly by site-specific image properties rather than tumour appearance. This is not established; it's a risk that has to be reported.

- **Status:** documented limitation. The split and the protocol are unchanged, and nothing was modified in response to test results.
- **Proposed follow-up** (to be pre-declared before it is run): a subgroup evaluation restricted to numeric-ID patients (22 glioma vs 82 meningioma). The current test split holds only 2 numeric-ID glioma patients, so this would need the official 5-fold CV or pooled seeds, not the single test split.
- Because every model sees the same data, this confound affects all models alike. The graph-vs-control comparison stays fair, but absolute performance must not be read as "tumour-type recognition" alone.

## Training protocol (declared 2026-09-28, before the first training run)

These settings are shared by every model: the CNN baseline and all graph models. They're in `configs/base.yaml`.

- **Input and resolution:** 1 × 128 × 128. The resolution is **frozen**; 224×224 may be added later only as a sensitivity analysis.
- **Encoder:** Conv(1→16)–ReLU–MaxPool → Conv(16→32)–ReLU–MaxPool → Conv(32→128)–ReLU → AdaptiveAvgPool, giving a 128-d feature vector. It has no batch norm, no dropout, no pretraining and no augmentation.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` = #meningioma / #glioma counted on the **train split only** (496 / 1010 = 0.491).
- **Optimisation:** Adam, learning rate 1e-3, weight decay 1e-4, batch size 32, at most 30 epochs.
- **Model selection:** the best validation slice-level ROC AUC (ties keep the earlier epoch). Early stopping after 7 epochs without improvement. The test split is loaded only after training and checkpoint selection are finished.
- **Decision threshold:** 0.5, fixed; never tuned on test data.
- **Metrics:**
  - Primary: ROC AUC, sensitivity, specificity, balanced accuracy, F1, precision, recall.
  - Secondary: accuracy and the confusion matrix.
  - Levels: reported at **slice level** and at **patient level** (a patient's probability = the mean of their slice probabilities). Patient level is the primary level for the paper.
- **Seed 0 alone is a pipeline-validation run.** Research results require multiple seeds (Phase 8).

## FROZEN Phase 6 baseline settings (2026-09-28, before any connectome model was trained)

Verified by an exact re-run at commit `f289329`: identical predictions and metrics. These settings are final and must not be changed after seeing Phase 7 or later results:

| Setting | Frozen value |
|---|---|
| Image size | 1 × 128 × 128 |
| Encoder | Conv(1→16)–ReLU–Pool → Conv(16→32)–ReLU–Pool → Conv(32→128)–ReLU → AdaptiveAvgPool → 128-d |
| Loss | BCEWithLogitsLoss with train-only `pos_weight` (496/1010) |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Batch size / max epochs | 32 / 30 |
| Early stopping | patience 7 |
| Checkpoint selection | validation slice-level ROC AUC (ties → earlier epoch) |
| Decision threshold | 0.5 |
| Split | Figshare Glioma vs Meningioma, seed 0, conservative `patient_link_key` grouping |

`tests/test_frozen_decisions.py` fails if any of these values in `configs/base.yaml` changes.

## FROZEN confound checks (declared 2026-09-28, before any connectome model was trained)

**Interpretation rule.** The patient-ID format is almost perfectly aligned with the class (every `MR…` patient is glioma), which may reflect an acquisition-site confound. **Absolute tumour-classification scores of every model must therefore be interpreted cautiously.** They may partly reflect site or scanner differences rather than tumour appearance.

These analyses are declared now and will be run with the **same model settings for every architecture**:

1. **Numeric-ID-only subgroup analysis:** metrics restricted to numeric-ID patients (82 meningioma, 22 glioma). This is a sensitivity/confound analysis only. **It is never used to tune or select the primary model.**
2. **Official 5-fold cross-validation** (`cvind.mat`, which maps cleanly to patients): patient-level folds, identical settings for every architecture, reporting both slice-level and patient-level metrics. The exact inner train/validation split within each fold will be specified before the CV runs.

## FROZEN graph architecture (Phase 7, 2026-09-28)

**History.** The first specified version (128 scalar nodes → 2 graph layers → **mean pooling** → Linear(1→1)) **collapsed**. At initialisation the image-dependent signal shrank to about 3·10⁻⁴ after pooling. Validation AUC was exactly 0.500 at every epoch, every test slice got the same probability, and all layer-2 nodes were dead at the selected epoch. This was an architectural information bottleneck, not a Drosophila result, and it would have affected every graph.

**Correction (Option A, chosen before any comparison and verified on validation only):**

```
MRI → SAME frozen CNN encoder → 128 features → BatchNorm1d(128)
    → ConnectomeLayer 1 → ConnectomeLayer 2 → all 128 final node states → Linear(128, 1)
```

- **Not added:** no skip or residual connections, no node embeddings, no extra layers, and no changes to the optimizer, learning rate, loss or selection rule. The classifier's only input is the output of the last graph layer.
- **Tests:**
  - the output depends on the wiring;
  - removing one edge changes the output;
  - silencing the graph makes the output independent of the image (no bypass);
  - the classifier receives exactly the 128 final node states;
  - forbidden edges contribute nothing and get zero gradient, and allowed edges get gradients.
- **Validation gate:** every run checks the selected checkpoint on validation (predictions not constant, validation AUC ≠ 0.5, the last graph layer varies with the image, and not all of its nodes are dead). The test split is evaluated **only** if the gate passes; otherwise the run stops with `status: collapsed_on_validation`.
- **Parameters for a 128-node, 3,047-edge graph:** encoder 41,792 + BatchNorm 256 + graph block 6,350 + classifier 129 = **48,527**, identical for Drosophila w20 and all its controls.
- **Seed-0 pipeline check (Drosophila w20):** the gate passed (validation AUC 0.915 at epoch 7; logit std 2.9; 3.9% of layer-2 nodes dead). **A single seed is not evidence.**
- **Observation, recorded and not acted on:** the validation loss fluctuates strongly between epochs (for example 0.34 → 1.10 → 0.33 → 2.36), while training loss falls below the baseline's. Possible causes include the BatchNorm train/eval difference and overfitting. The frozen settings are not changed because of it; Phase 8's multiple seeds will show how much results vary.

**Phase 8 must use exactly this architecture for Drosophila w20, rewired, ER, WS and BA.** `tests/test_frozen_decisions.py` guards the structure.

## Phase 8 protocol (declared 2026-09-28, BEFORE any Phase 8 run)

- **Models:** CNN baseline, Drosophila w20, degree-preserving rewired (**primary null**), ER, WS, BA. All graph models use the frozen Phase 7 architecture; all settings are the frozen Phase 6/7 settings.
- **Seeds:** 0, 1, 2, 3, 4, which gives 6 × 5 = **30 runs**. Training seed *s* uses control graph seed *s* (`{control}_128_w20_seed{s}.csv`, generated in Phase 4 and unchanged). The Drosophila w20 graph is the same for every seed.
- **Primary comparison:** Drosophila − rewired, as paired per-seed differences (seed *i* vs seed *i*). Secondary: Drosophila − ER, − WS, − BA. The CNN baseline has a different parameter count (41,921 vs 48,527), so CNN-vs-graph is **not** a pure topology comparison.
- **Reported per model:** mean ± SD over seeds, plus every individual seed, for slice ROC AUC, patient ROC AUC, balanced accuracy, sensitivity, specificity and F1. **No significance claims from 5 training seeds, and no clinical claims.**
- **Failed-run policy:**
  - A run that numerically fails or degenerates is **recorded as a failed run** and its logs are kept.
  - The seed is **not replaced**, the hyperparameters are **not changed**, and the run is **not repeated with special settings**.
  - A failed run is excluded from that model's mean ± SD and from the paired differences for its seed, and it is **reported explicitly**.
- **Validation sanity gate (technical degeneration only):**
  - It fails a run on constant predictions, non-finite values (loss, probabilities or activations), or a completely dead graph output (the last graph layer has no image-dependent variation, or all its nodes are 0 for every image).
  - **No performance threshold is used.** A low-performing but functioning model is a valid result.
  - The earlier "validation AUC exactly 0.5" criterion is dropped for this reason.
- **Not in Phase 8:** the official 5-fold CV and the numeric-ID subgroup analysis (later phases).

## FROZEN Phase 8 results (EB-128 w20 graph; 2026-09-29)

- **Batch:** `results/phase8_20260928-235044`, run at commit `a3c966b`. 30/30 runs succeeded, and no run was repeated or replaced. Frozen copies of the tables, figures and run index are in [docs/results/phase8/](results/phase8/), with `SHA256SUMS.txt`.
- **Primary comparison (Drosophila w20 − degree-preserving rewired, paired by seed):**
  - slice ROC AUC −0.007 ± 0.024 (Drosophila higher in 2/5 seeds);
  - patient ROC AUC −0.015 ± 0.041 (2/5 seeds);
  - patient balanced accuracy +0.002 ± 0.091.
- **Observed result:** no advantage of the EB-derived biological wiring over its degree-preserving rewired null, or over ER/WS/BA. On ROC AUC, Drosophila w20 was the lowest of the graph models on average. All graph models exceeded the CNN baseline, but random graphs did so equally, so that gain is not attributable to biological topology (and CNN-vs-graph is not a pure topology comparison).
- **Interpretation limits:** 5 training seeds, a 21-patient test split, the possible site confound, and EB being an engineering/provenance graph. No significance or clinical claims.
- These results are frozen and are not re-run or modified. Later graphs (such as the visual-system graph) get their own, separately declared experiments.

## Rules (from the project specification)

1. The test set is never used for tuning or model selection.
2. All graph models share the same split, preprocessing, encoder, node count, optimizer, learning rate, epochs, batch size and seeds. Only the topology changes.
3. Multiple seeds, reported as mean ± std.
4. Results where the biological graph does worse than the controls are reported, not dropped.
5. MRI images from the same patient never cross splits.
6. A generic MaleCNS subgraph is never called the "visual system".
