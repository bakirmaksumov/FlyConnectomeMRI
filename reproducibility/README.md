# Reproducibility artifacts (Phases 8–10)

This folder holds the graph and data-split files used by the frozen FlyConnectomeMRI experiments: 155 runs in Phase 8 (30), Phase 9 (25) and Phase 10 (100). Each file was copied unchanged from the working directory. Nothing was regenerated, and no experiment was re-run.

**Not included:** MRI images, model checkpoints, and run logs or run folders.

## Verifying the package

```
python scripts/verify_reproducibility.py
```

Run it from the repository root. It needs only the Python standard library. It does not use the MRI data or the unpublished `results/` folder. `--package` and `--repo-root` set other locations. It exits with status 1 if any check fails. The checks are:

1. **Integrity.** Every file in `SHA256SUMS.txt` is present and matches, and no file is missing from the list. `sha256sum -c SHA256SUMS.txt`, run from this folder, does the same check.
2. **Run records.** Every SHA-256 that a run recorded matches the packaged file. The runs recorded these in their `metrics.json`, and `run_artifact_usage.csv` holds a copy.
3. **Frozen reports.** The packaged files match the checksums frozen in `docs/RESEARCH_PLAN.md` and `docs/results/phase10_prerun/`.
4. **Internal consistency.** This covers the hashes that metadata files declare, node order against the node tables, and the per-fold exclusions against the full list.

**What `SHA256SUMS.txt` proves.** It was created when this package was assembled (2026-11-08). It proves that the package you have is the package that was published. It does **not** prove that a file is unchanged since the experiments. That depends on the evidence category of each file, listed below.

**Historical check (optional).** With `--results-dir path/to/results`, the script also checks the files against the original run folders:

- It rebuilds `run_artifact_usage.csv` from the `metrics.json` of all 155 runs and compares the two.
- It checks that the `node_order.csv` in every run equals the packaged `node_order` file.
- It checks that no run is missing from the table.

This needs the original, unpublished `results/` folder. It passed on the author's machine on 2026-11-08. Without that folder you cannot repeat this check, but checks 1–4 still work.

`README.md`, `SHA256SUMS.txt` and `.gitattributes` are not listed in `SHA256SUMS.txt`. The `.gitattributes` file turns off Git line-ending conversion for this folder, so the files stay byte-identical on every OS.

## Evidence for each file

### A. Confirmed by the runs' own checksums (90 files)

Each run wrote the SHA-256 of the graph and split files it loaded into its `metrics.json` at training time. These files match those values in every run that used them.

| Files | Count |
|---|---|
| Graph edge lists `<graph>.csv` and node tables `<graph>_nodes.csv` for the 42 trained graphs: EB w20, visual raw and their 40 controls | 84 |
| `splits/fixed_split/manifest.csv` (Phases 8–9, 55 runs) | 1 |
| `splits/cv_leakage_safe/fold{1-5}/manifest.csv` (Phase 10, 20 runs each) | 5 |

Several of these are also in the frozen reports in category B: 12 graph files in `RESEARCH_PLAN.md` and 22 in `graph_checksums.txt`. The six manifests are also in `cv_prerun_report.json`.

### B. Confirmed only by other frozen reports (6 files)

No run recorded these files. Their checksums were committed to Git before the Phase 10 runs, in `docs/RESEARCH_PLAN.md` and in commit `c81615d`.

| File | Evidence |
|---|---|
| `graphs/eb128/source_download/drosophila_128.csv` | SHA-256 frozen in `docs/RESEARCH_PLAN.md` |
| `graphs/visual128/source_download/drosophila_visual_128.csv` | SHA-256 frozen in `docs/RESEARCH_PLAN.md`. Byte-identical to the trained `drosophila_visual_128_raw.csv` |
| `splits/cv_leakage_safe/{counts,exclusions_all,planned_runs}.csv`, `cv_prerun_report.json` | SHA-256 frozen in `docs/results/phase10_prerun/SHA256SUMS.txt` (commit `c81615d`). Copies are committed in the same folder, but Git stores them with LF line endings, so two of them differ byte by byte from the CRLF originals kept here. Compare with the checksums, not the copies |

### C. Unchanged since the runs NOT confirmed (100 files)

These are the files present in the working directory and should be the originals. No run or frozen report recorded their checksums, so their historical immutability is **not confirmed**. Check 4 shows only that they agree with the confirmed files.

| Files | Count | Consistency evidence |
|---|---|---|
| Graph `*_metadata.json`: 42 trained graphs plus 2 source downloads | 44 | Each declares the SHA-256 of its own edge and node files and of its source, and all 174 match. The w20 metadata also declares the SHA-256 of the EB source metadata |
| Source-download node tables `drosophila_128_nodes.csv` and `drosophila_visual_128_nodes.csv` | 2 | Their SHA-256 is declared in the derived graphs' metadata, which is itself category C |
| `splits/cv_leakage_safe/fold{1-5}/exclusions.csv` | 5 | Together their rows equal the category-B `exclusions_all.csv` (76 rows) |
| `splits/*/dataset_metadata.json` (fixed split and 5 folds) | 6 | Each declares the SHA-256 of its manifest, and the folds also declare the fixed-split manifest |
| `node_order/*_node_order.csv` | 42 | Copied from the run folders. Each is the ascending node ID list of its graph, so it can be re-derived from category A. The historical check matched it to every run |
| `run_artifact_usage.csv` | 1 | **Created** on 2026-11-08 from the runs' `metrics.json`; the historical check rebuilds and compares it |

## Contents

| Path | What it is | Used by |
|---|---|---|
| `graphs/eb128/drosophila_128_w20*` | EB-128 graph, edges with ≥ 20 synapses: 128 nodes, 3,047 edges | Phase 8 |
| `graphs/eb128/source_download/drosophila_128*` | Raw neuPrint download (weight ≥ 1, 12,899 edges) that the w20 graph is filtered from | Source of the Phase 8 graph |
| `graphs/eb128/controls/{rewired,er,ws,ba}_128_w20_seed{0-4}*` | Control graphs matched to EB w20 | Phase 8 |
| `graphs/visual128/drosophila_visual_128_raw*` | Visual-system graph: 128 nodes, 989 edges, weight ≥ 1 | Phases 9–10 |
| `graphs/visual128/source_download/drosophila_visual_128*` | Raw neuPrint download. Its edge list equals the `raw` graph; its node table lacks the degree and `isolated` columns. Its metadata documents the neuron selection | Source of the Phase 9–10 graph |
| `graphs/visual128/controls/{rewired,ws}_visual128_raw_seed{0-4}*` | Primary null (rewired) and WS control | Phases 9–10 |
| `graphs/visual128/controls/{er,ba}_visual128_raw_seed{0-4}*` | ER and BA controls | Phase 9 only |
| `node_order/*_node_order.csv` | Mapping `node_index → bodyId` (or control slot) used by the models; one file per trained graph | Phases 8–10 |
| `splits/fixed_split/` | Slice manifest of the seed-0, 70/15/15 group-stratified split, plus metadata | Phases 8–9 |
| `splits/cv_leakage_safe/fold{1-5}/` | Per outer fold: slice manifest (train, val and test), `exclusions.csv` and metadata (per-fold `pos_weight` and normalisation statistics) | Phase 10 |
| `splits/cv_leakage_safe/{counts,exclusions_all,planned_runs}.csv`, `cv_prerun_report.json` | Fold counts before and after exclusions, all 76 exclusions, the 100 planned runs, and the leakage and OOF-coverage report | Phase 10 |
| `run_artifact_usage.csv` | One row per run (155): the graph and split files it used and their recorded SHA-256 | Phases 8–10 |

Each graph comes as three files:

- `<name>.csv`: edges as `source,target,weight`; `source` is the presynaptic neuron.
- `<name>_nodes.csv`: the node table.
- `<name>_metadata.json`: provenance, selection rule and statistics.

**Edge weights are not used by the models.** Only the presence and direction of an edge are used; there is one trainable weight per edge.

**Paths.** Paths inside the metadata files and `run_artifact_usage.csv` use the original layout (`data/connectome/…`, `data/splits/…`, `results/…`). To train, copy the graph files to `data/connectome/` and run the configs in [`configs/`](../configs/). Training also needs the MRI images, which are not included. Download them with `scripts/download_figshare.py` and prepare them with `scripts/prepare_dataset.py` and `scripts/prepare_cv_folds.py`. The manifests record the SHA-256 of every source `.mat` slice, so you can check the prepared data against them.

## Sources, licences and attribution

### Janelia MaleCNS v1.0 connectome

- **Source.** The *Drosophila* male central nervous system connectome, dataset `male-cns:v1.0`, from Janelia. It was accessed through neuPrint (https://neuprint.janelia.org) with neuprint-python 0.6.3.
  - Database uuid: `4b2087c0fbe046bfaf0d60bc970e3e5d`.
  - Downloads: EB on 2026-09-28, visual on 2026-09-29.
- **Homepage.** https://male-cns.janelia.org/
- **Preprint.** https://doi.org/10.1101/2025.10.09.680999. This package does not give an author list, because one was not verified.
- **Licence: CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The dataset homepage states "The Male CNS dataset is licensed under CC-BY" and links to CC BY 4.0. Checked on 2026-11-08.
- **Not checked.**
  - The homepage gives no required citation format.
  - neuPrint's own terms of service, as distinct from the dataset licence, were not checked.

**What was selected and changed.** Nothing was edited by hand; every step is recorded in the `*_metadata.json` files.

- **Subsets only.** The package holds two 128-neuron subsets, not the whole connectome. For each subset it keeps the neuron IDs (`bodyId`), the neuPrint annotations listed in the node tables, and the directed edge lists.
- **Edge weights.** In both subsets the edge weight is the total synapse count from source to target across the whole dataset: `fetch_adjacencies(..., omit_rois=True)`. Self-loops are dropped; there were none.
- **EB-128.**
  - Candidates are all neurons with synapses in ROI `EB`.
  - The 128 with the most `pre + post` synapses inside `EB` are kept. Ties go to the smaller `bodyId`.
  - This is a generic ROI subgraph, not a functional circuit.
  - The primary graph `w20` keeps only edges with weight ≥ 20: 3,047 of 12,899 edges.
- **Visual-128.**
  - Candidates are neurons of 11 columnar types (L1, L2, L3, Mi1, Mi4, Mi9, C3, Tm1, Tm2, Tm4, Tm9) with synapses in `ME(R)` and an assigned column. There are 9,727 candidates.
  - The centre is column (18, 20), the median of the 817 columns that contain every type.
  - Neurons are ranked by hex distance from the centre, then by column, type order and `bodyId`. The first 128 are kept; the patch reaches distance 2.
  - All 989 edges are kept (`raw`, weight ≥ 1).
- **Control graphs.** These are generated by this project (`src/flyconnectome_mri/connectome/random_graphs.py`) to match each biological graph's node and edge counts.
  - Rewired: degree-preserving swaps.
  - ER, WS-type and BA: abstract node slots `0..127`, not neurons.
  - Seeds and generator details are in each control's metadata.
  - They are derived works of the biological graphs.

### Figshare brain tumor dataset (v8)

- **Source.** Jun Cheng, *brain tumor dataset*, figshare, version 8, published 2024-12-21. https://doi.org/10.6084/m9.figshare.1512427 (record `10.6084/m9.figshare.1512427.v8`). The images were acquired at Nanfang Hospital, Guangzhou, and General Hospital, Tianjin Medical University, according to the dataset README.
- **Licence: CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). This is the licence field of the Figshare API record (https://api.figshare.com/v2/articles/1512427), downloaded 2026-09-28 and stored in `splits/*/dataset_metadata.json`.
- **Papers to cite.** The dataset README asks users to cite two papers. Their full author lists, page numbers and DOIs have not been verified here:
  - Cheng J. et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition", *PLoS ONE* 10(10), 2015.
  - Cheng J. et al., "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation", *PLoS ONE* 11(6), 2016.

**What was selected and changed.** **No images are included.** The package holds only slice-level metadata derived from the dataset:

- **Slices.** The pituitary class (930 slices) is excluded. That leaves 2,134 slices from 171 patient IDs: 1,426 glioma and 708 meningioma.
- **Manifest fields.** Each manifest row has:
  - the source `.mat` file name and its SHA-256;
  - image width and height;
  - the class and original label code;
  - the patient ID `cjdata.PID` as published in the dataset;
  - the official `cv_fold` from `cvind.mat`;
  - the split assignment made by this project.
- **`patient_link_key`.** This is this project's conservative grouping of IDs that may belong to one patient (for example `MR024780B` and `MR024780D`). It is an assumption, not a known fact. In the fixed split, linked IDs are kept in the same split; in cross-validation they are removed, as described below.
- **Fixed split.** Seed 0, 70/15/15, stratified by class, whole groups kept together.
- **Cross-validation.**
  - The test set is official fold *k*, unchanged.
  - Validation is fold *k* mod 5 + 1.
  - Training uses the other three folds.
  - Linked IDs are removed from validation and training when they are linked to the test fold, and from training when they are linked to validation. That removes 76 IDs in all, listed in `exclusions_all.csv`.
- **Image preprocessing (not part of this package).** Images were converted to grayscale and resized to 128 × 128 (bilinear). They were normalised with statistics from the training split only. The values are in each `dataset_metadata.json`.
- **Personal data.** The manifests contain no names, dates or images.

### Licence scope

- **Data.** CC BY 4.0 applies to the third-party data this package derives from, and so to the connectome-derived graphs and the Figshare-derived manifests. Reusing them requires attribution to the MaleCNS and Figshare sources above.
- **Code.** **The code in this repository is not covered by these data licences.** It includes `scripts/verify_reproducibility.py` and the generators that produced the control graphs. When this package was prepared, the repository had no licence file for its code, so no licence for the code is granted or implied here.
- **Citations.** The entries in `docs/paper/latex/references.bib` are still marked as drafts, except for the Figshare dataset entry.

## Content notes

- **No secrets or local paths.** There are no absolute local paths, tokens or credentials in this folder. The neuPrint token was only ever read from an environment variable and is not stored anywhere.
- **Code at download time.** The EB raw download records git commit `2c00739`, because the fetch was run before its code was committed in `4863ba5` (see `docs/RESEARCH_PLAN.md`).
- **Not included.** The unused weight variants were declared as sensitivity analyses only and were never trained on: EB raw as a training graph, EB w5 and w10, and visual w5. Visual w5 has a frozen checksum in `RESEARCH_PLAN.md`, but it is not in this package.
