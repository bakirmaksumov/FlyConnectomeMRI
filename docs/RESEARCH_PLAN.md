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

## Visual-system graph (declared 2026-09-29, BEFORE fetching the final files and before any MRI training on it)

**Exploration (MaleCNS v1.0, right optic lobe).**
- **Neurons per region:** ME 43,191; LO 26,555; LOP 15,193; LA 9,509; AME 716.
- **Heavy overlap:** columnar neurons span several regions (ME∩LO 18,909; LO∩LOP 11,124; ME∩LOP 10,293), so a region union is not a meaningful selection by itself.
- **Types and columns:** neurons carry cell types and, for many columnar types, an assigned retinotopic column (`assignedOlHex1/2`). The column grid's neighbour offsets (0,±1), (±1,0), (1,1), (−1,−1) were verified on L1 soma positions.

**Candidates** (128 neurons each; all edges, w ≥ 1):

| | A: ME(R) top-128 by ME synapses | **B: local retinotopic columnar patch** | C: right visual-projection neurons, top-128 by synapses |
|---|---|---|---|
| Content | wide-field neurons (MeLo9, Pm, Dm…) | 11 columnar types × ~12 columns | large projection neurons (VS, LT, LPT…) |
| Edges / density | 2,731 / 0.168 | 989 / 0.061 | 1,323 / 0.081 |
| Reciprocity / clustering | 0.59 / 0.44 | 0.51 / 0.38 | 0.46 / 0.33 |
| Strongly connected components | 1 | 12 (largest 117) | 1 |

**Chosen: B**, on biological relevance and graph properties only (no MRI data involved):
- It is a canonical, well-characterised circuit: the input stage of the ON/OFF motion pathways, repeated in every retinotopic column.
- It is selected purely by verified annotations (cell type plus assigned column), not by synapse-count ranking.
- It is naturally sparse.

A and C are ranked by neuron size and mix unrelated types.

**Exact rule** (`configs/malecns_visual.yaml`):
- **Types:** L1, L2, L3, Mi1, Mi4, Mi9, C3, Tm1, Tm2, Tm4, Tm9, with synapses in ME(R) and a non-null column assignment.
- **Centre:** the median (hex1, hex2) of the columns containing all 11 types.
- **Order:** hex distance from the centre, then hex1, hex2, the type order above, then bodyId.
- **Keep:** the first 128.

**Limitation:** T4/T5 (the motion detectors) and Tm3 have no column assignment in MaleCNS, so they are not included, and we don't assign columns ourselves. The graph therefore represents the columnar input stage, not the full motion detector.

**Frozen threshold and variants:**
- **PRIMARY:** `raw` (w ≥ 1, no threshold). The graph is already sparse (density ≈ 0.06, below EB w20's 0.19), so no density-motivated threshold is needed.
- **Sensitivity only:** `w5` (w ≥ 5).
- **Controls** matched to the primary graph: degree-preserving rewired (**primary null**), ER, WS and BA, with seeds 0–4 and the same frozen generators and settings as Phase 4.

The EB graph stays an engineering/provenance graph and is never relabelled as visual.

**Fetched 2026-09-29** at commit `b1ab832` (MaleCNS database uuid `4b2087c0…`).
- **Centre and patch:** the central column is (18, 20). The patch has 12 columns: the centre (11 neurons) and ring 1 (66) are complete, and ring 2 is partial (51 of 132 neurons, taken in the documented order).
- **Cell types:** 12 each of L1, L2, L3, Mi1, Mi4, Mi9 and C3, and 11 each of Tm1, Tm2, Tm4 and Tm9.
- **Neurotransmitters (consensus):** ACh 80, Glu 24, GABA 24.
- **Biological sanity check:** the strongest type-to-type links are the known pathways: L2→Tm2/Tm1/Tm4 (OFF), L1→Mi1 (ON), L3→Mi9, Mi4→Mi9, and C3→L2 feedback.
- **The 12 strongly connected components are biological, not errors:** 7 L3 cells have no inputs inside the graph (their input is photoreceptors, which are not included), and 3 Tm9 cells have no outputs inside it (their targets, T5, are not included).
- **Weights:** median 3 synapses; 31% of edges carry a single synapse (kept, per the frozen `raw` decision).

| Frozen file | SHA-256 |
|---|---|
| drosophila_visual_128.csv (raw download) = drosophila_visual_128_raw.csv (PRIMARY) | `d77053f852ccd83b496173e0a5f47bdbb6c90469cec859ca7797cf1193a2e015` |
| drosophila_visual_128_w5.csv (sensitivity) | `859013f264b51218f13326c93a1ccbdea626e6bea84cf4e3d182d202425762ae` |
| rewired_visual128_raw_seed0.csv | `5799bf61a9a967dae544bd5b15699bfb79a564280a951d4406e41e02a7ed37aa` |
| er_visual128_raw_seed0.csv | `e1448cfd6d54e5403f8277e06764fafa9d8d35ce94cb465b22b20266361db68b` |
| ws_visual128_raw_seed0.csv | `842ae60f09ad90d0c3b595b645eae9766ae43bfb584f873e67a3fa321d627ae6` |
| ba_visual128_raw_seed0.csv | `eaa2fc67f75ba154328f98d20433bd9939692c695d65f479876be658a1e4ea30` |

No MRI model has been trained on the visual graph or its controls.

## Phase 9 protocol: visual-graph topology comparison (declared 2026-09-29, BEFORE any Phase 9 run)

- **Question:** does the biologically selected MaleCNS visual-system topology give any advantage over matched control topologies?
- **Graphs:**
  - primary biological: `drosophila_visual_128_raw` (128 nodes, 989 edges, w ≥ 1);
  - **primary null:** degree-preserving rewired (`rewired_visual128_raw_seed{s}`);
  - additional controls: ER, WS and BA (`{er,ws,ba}_visual128_raw_seed{s}`).
  
  The w ≥ 5 visual variant is **not** used in this experiment.
- **Seeds:** 0–4, with training seed *s* paired with control-graph seed *s*. That gives **25 graph-model runs**.
- **Unchanged:** the frozen Phase 7 architecture and every frozen Phase 6/7 setting (dataset, split, preprocessing, 128×128, encoder, optimizer, learning rate, loss, train-only `pos_weight`, batch size, max epochs, early stopping, checkpoint rule, threshold 0.5, validation gate).
- **CNN baseline:** **not retrained.** The frozen Phase 8 CNN results (`docs/results/phase8/`) are shown as a reference only.
- **PRIMARY metric:** **patient-level ROC AUC**.
- **Secondary metrics:** slice ROC AUC, patient and slice balanced accuracy, patient sensitivity and specificity, F1, precision, recall.
- **Primary comparison:** visual Drosophila − rewired, as paired per-seed differences. Report individual seeds, the mean, the SD and the per-seed differences.
- **Secondary comparisons:** visual Drosophila − ER, − WS and − BA. **No significance claims from 5 seeds.**
- **Failed-run policy:** as in Phase 8. Failed runs are recorded, never replaced or retuned, and low performance is a valid result.
- **EB vs visual:** described side by side only. It is **not** a controlled head-to-head experiment, because the two biological graphs differ in density and structure.
- **Not in Phase 9:** the official 5-fold CV, the numeric-ID subgroup analysis and the visual w ≥ 5 sensitivity analysis.

## FROZEN Phase 9 results (visual-system graph; 2026-09-29)

- **Batch:** `results/phase9_20260929-110630`. The protocol was declared and committed at `77eead1`, and the batch ran at that commit. 25/25 runs succeeded, with none repeated or replaced. Frozen copies of the tables, figures, run index and batch info are in [docs/results/phase9/](results/phase9/), with `SHA256SUMS.txt`. The figures were redrawn from the same run data after two layout-only fixes (`ff3e3e0`); no numbers changed.
- **Primary metric (patient ROC AUC, mean ± SD over 5 seeds):**

  | Model | Patient ROC AUC |
  |---|---|
  | Visual Drosophila | 0.944 ± 0.029 |
  | Visual rewired | 0.927 ± 0.053 |
  | Visual ER | 0.931 ± 0.037 |
  | Visual WS | 0.960 ± 0.025 |
  | Visual BA | 0.927 ± 0.044 |
  | CNN baseline (Phase 8 reference, not retrained) | 0.900 ± 0.009 |

- **Primary comparison (Visual Drosophila − visual rewired, paired by seed):**
  - patient ROC AUC +0.017 ± 0.061 (Drosophila higher in 3/5 seeds);
  - slice ROC AUC +0.008 ± 0.029 (4/5);
  - patient balanced accuracy +0.018 ± 0.181.
- **Observed result:** Visual Drosophila had a slightly higher mean than its degree-preserving null, but the difference is well within the seed-to-seed variation. **No topology-specific advantage is demonstrated.** Visual WS had the highest mean patient ROC AUC. Visual Drosophila had the smallest seed-to-seed spread in patient balanced accuracy (±0.039) and sensitivity (±0.069), while rewired, ER and BA each had seeds that called almost everything meningioma. This is descriptive only and not evidence from 5 seeds.
- **EB vs visual:** described side by side only (the graphs differ in density, 0.19 vs 0.06, and in parameter count, 48,527 vs 44,411). Drosophila − rewired was −0.015 on EB and +0.017 on the visual graph.
- **Limits:** 5 seeds, a 21-patient test split, the possible site confound; no significance or clinical claims. These results are frozen and are not re-run or modified.

## Phase 10 protocol: leakage-safe cross-validation using the official Figshare outer test folds (declared 2026-09-29, BEFORE any Phase 10 training)

**Name.** This experiment is "**Leakage-safe cross-validation using the official Figshare outer test folds**". It is **not** an exact reproduction of the dataset authors' original five-fold protocol.

**Purpose.** A robustness check that removes the dependence on the single 21-patient test split. Every patient in the Glioma/Meningioma cohort (171 patients) is tested exactly once.

**Fold rule** (the official `cv_fold` values from `cvind.mat` are used and never altered):
- **Test:** official fold *k*, **unchanged**.
- **Validation:** the official fold `k % 5 + 1` (1→2, 2→3, 3→4, 4→5, 5→1), minus every patient whose `patient_link_key` occurs in test.
- **Training:** the other three official folds, minus every patient whose `patient_link_key` occurs in test or in the cleaned validation set.
- **Not done:** no patient is moved to another fold, no replacement patients are added, and the class balance is not restored.
- **Per-fold files:** every exclusion is listed in `data/splits/figshare_cv_leakage_safe/fold{k}/exclusions.csv` (reason `linked_to_test` or `linked_to_validation`).
- **Per-fold weighting and normalisation:** `pos_weight` and the normalisation statistics are recomputed for each fold from the **cleaned training set only**.

**Limitation (recorded before training).** The official Figshare folds were constructed using the published patient_id. Our conservative patient_link_key grouping indicates that some potentially related examinations span different official folds. To avoid possible subject-level leakage, official outer test folds were preserved unchanged while linked examinations were excluded from the corresponding training and validation sets.

This is not a defect in the original dataset. We do not know whether the letter-suffixed IDs are definitely the same person; the grouping is a **conservative leakage-prevention assumption**. All removed linked-family patients are gliomas with MR-prefixed IDs, which slightly lowers the glioma share of the affected training/validation sets.

**Models (4):**
1. CNN baseline;
2. Visual Drosophila (the fixed biological graph for every run);
3. Visual rewired (**primary null**; graph seed = training seed);
4. Visual WS (the strongest non-biological control in Phase 9; graph seed = training seed).

ER and BA are not included. Architectures and every setting are the frozen Phase 6/7 ones (see above), and so are the graph definitions and graph seeds.

**Runs:** 4 models × 5 outer folds × 5 seeds (0–4) = **100 runs**. The planned list is in `data/splits/figshare_cv_leakage_safe/planned_runs.csv`.

**Test access:** train on the training folds → select the best checkpoint by validation slice ROC AUC → pass the technical validation gate → only then load the test fold → evaluate it once.

**Failed-run policy:** as in Phases 8 and 9. Only technical degeneration counts as failure, never low AUC. A failed run is recorded and is never replaced or retuned.

**Primary metric:** **patient-level ROC AUC**, where a patient's probability is the mean of their slice probabilities, never an average of hard labels.

**Secondary metrics:** slice ROC AUC, patient and slice balanced accuracy, patient sensitivity and specificity, patient F1, precision, recall, and the confusion matrices.

**Aggregation:**
- **Run level:** all 25 fold × seed results per model, with mean, SD, median, min and max. These are not 25 independent experiments.
- **Fold level (the main robustness summary):** average the 5 seeds within each fold, then report the 5 fold values and their mean ± SD.
- **Pooled out-of-fold (OOF):** for each model and seed, combine the five test folds into one full-cohort prediction file (`oof_predictions_seed{s}.csv`). Compute the patient-level metrics on it and report mean ± SD over seeds. The OOF files keep `patient_link_key` and the ID format (MR-prefixed vs numeric), so the numeric-ID subgroup can be analysed later without retraining.

**Comparisons:**
- **Primary:** Visual Drosophila − Visual rewired patient ROC AUC. Report all 25 paired fold × seed differences, their mean ± SD, the number positive/negative/tied, and the 5 fold-level paired means.
- **Secondary:** Drosophila − WS, and Drosophila − CNN. The CNN comparison is **not** a pure topology comparison (different architecture and parameter count).
- **Wording:** neutral ("had a higher/lower mean in this experiment"). No significance claims from training seeds, and no clinical claims. The possible site confound still applies.

**Not in Phase 10:** the numeric-ID subgroup analysis and the visual w ≥ 5 sensitivity analysis.

**Pre-registered analysis code:** the Phase 10 aggregation and reporting code (`evaluation/cv_summary.py`, `evaluation/cv_plots.py`, `scripts/summarize_phase10.py`) and its mock-data tests were committed at **`feac0c3`**, before any Phase 10 model was trained. The summary rules (run level, fold level with seeds averaged first, pooled OOF per model and seed, pairing strictly by outer fold + seed, the nine figures, and failed runs kept as NA) must not be changed after results exist.

## Rules (from the project specification)

1. The test set is never used for tuning or model selection.
2. All graph models share the same split, preprocessing, encoder, node count, optimizer, learning rate, epochs, batch size and seeds. Only the topology changes.
3. Multiple seeds, reported as mean ± std.
4. Results where the biological graph does worse than the controls are reported, not dropped.
5. MRI images from the same patient never cross splits.
6. A generic MaleCNS subgraph is never called the "visual system".
