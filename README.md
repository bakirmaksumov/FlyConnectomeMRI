# FlyConnectomeMRI

*Written with Phase 1 and filled in as later phases land.*

This is a research prototype. It asks whether a neural network whose connectivity is constrained by a graph derived from the real *Drosophila* (fruit fly) connectome classifies brain MRI images differently from the same network wired with random graphs, or from a plain CNN.

> **This is not a medical device.** Nothing in this repository is diagnostic, and no result here should be read as "the fly brain detects tumors." The accurate wording is: *a neural network whose connectivity is constrained by a Drosophila-derived graph.*

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Project skeleton, config system, reproducibility, tests | ✅ done |
| 2 | Generic connectome edge-list loader + graph statistics | ✅ done |
| 3 | Drosophila (MaleCNS v1.0) subgraph fetching | ✅ done (128-neuron EB graph) |
| 4 | Weight-threshold variants + random graph controls (ER / WS / BA / degree-preserving rewired) | ✅ done |
| 5 | MRI dataset abstraction with leak-free, patient-level splits | ✅ done (tested on synthetic data) |
| 6 | Small CNN baseline (slice- and patient-level evaluation) | ✅ pipeline validated (seed 0) |
| 7 | Connectome-constrained model | ✅ implemented + pipeline run (seed 0) |
| 8 | Controlled topology comparison: 6 models × 5 seeds | infrastructure ready; see `results/phase8_*` |
| 9–12 | Small-data, rewiring, graph size, report | not started |

## Setup (CPU is enough)

Requires Python 3.10+ (developed on 3.12).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .
```

`pip install -e .` installs everything in `requirements.txt` and then installs this project's package (`flyconnectome_mri`) in "editable" mode, so any change to `src/` takes effect immediately. The exact versions used for reported runs are recorded in `requirements-lock.txt`.

A GPU is optional. The default PyPI `torch` wheel on Windows is CPU-only. For CUDA, install torch from https://pytorch.org/get-started/locally/ first.

## Usage

```powershell
# Run the tests
.venv\Scripts\python -m pytest -v

# Validate an edge-list CSV and print graph statistics
.venv\Scripts\python scripts/inspect_connectome.py --edges tests/data/tiny_graph.csv

# Fetch a real 128-neuron MaleCNS subgraph (needs a neuPrint token, see below)
$env:NEUPRINT_TOKEN = '<your token>'      # current PowerShell session only
.venv\Scripts\python scripts/fetch_connectome.py --config configs/malecns_fetch.yaml
.venv\Scripts\python scripts/fetch_connectome.py --config configs/malecns_fetch.yaml --num-nodes 256

# Derive weight-threshold variants (raw / w5 / w10 / w20) from the raw graph
.venv\Scripts\python scripts/derive_graph_variants.py --config configs/graph_variants.yaml

# Generate control graphs matched to the primary w20 graph
.venv\Scripts\python scripts/generate_controls.py --config configs/controls.yaml

# First real dataset: Cheng et al. Figshare (original .mat), Glioma vs Meningioma
.venv\Scripts\python scripts/download_figshare.py      # public, CC BY 4.0, MD5-verified
.venv\Scripts\python scripts/extract_figshare.py       # read-only .mat files
.venv\Scripts\python scripts/prepare_dataset.py --config configs/datasets/figshare_glioma_meningioma.yaml

# MRI data: create the SYNTHETIC demo dataset (not medical), then split it
.venv\Scripts\python scripts/make_synthetic_dataset.py --output data/raw/synthetic_demo
.venv\Scripts\python scripts/prepare_dataset.py --config configs/datasets/synthetic_demo.yaml

# Phase 8: all 6 models x seeds 0-4 (30 runs, ~1.5 h on CPU), then tables + figures
.venv\Scripts\python scripts/run_phase8.py
.venv\Scripts\python scripts/summarize_phase8.py --batch results/phase8_<timestamp>

# Run an experiment from a config file
.venv\Scripts\python scripts/run_experiment.py --config configs/base.yaml
.venv\Scripts\python scripts/run_experiment.py --config configs/drosophila.yaml --seed 3
```

`run_experiment.py`:

1. loads and validates the config,
2. seeds every random number generator,
3. picks the device (CPU, or GPU if requested and available),
4. creates a run folder `results/<experiment_name>_<timestamp>/` containing:
   - `config.yaml`: the fully resolved config, so the run can be repeated exactly
   - `environment.json`: Python/PyTorch/library versions, device and seed
   - `run.log`: everything the run printed
   - `seed_fingerprint.json`: the first few random numbers drawn after seeding. Same seed → same fingerprint, which is a quick proof that seeding works.
5. trains on the **train** split, evaluates on **validation** after every epoch, keeps the best validation checkpoint (`best_model.pt`) and stops early when validation stops improving;
6. only then loads the **test** split once and writes:
   - `test_predictions_slice.csv`: per slice: image_path, patient_id, true label, probability, predicted label
   - `test_predictions_patient.csv`: per patient: mean probability over their slices, true label, predicted label
   - `metrics.json`: slice-level and patient-level ROC AUC, sensitivity, specificity, balanced accuracy, F1, precision, recall, accuracy, confusion matrices; class weighting; parameter count; training and inference time
   - `history.csv` and `training_curves.png`: per-epoch losses and validation metrics

## Drosophila data (Phase 3)

Source: **Janelia MaleCNS v1.0** (`male-cns:v1.0` on https://neuprint.janelia.org), licensed CC-BY 4.0.
Preprint: https://doi.org/10.1101/2025.10.09.680999.

**Token.** Log in at https://neuprint.janelia.org, open *Account*, and copy your API token. Set it as the environment variable `NEUPRINT_TOKEN`. The code reads it only from there, never prints or logs it, and a config file containing a `token:` key is rejected. Don't put the token in `.env` files that could be committed, and don't paste it into chats.

**Selection (`configs/malecns_fetch.yaml`).** This is a *generic, anatomical* selection: the N neurons with the most synapses inside one ROI (default `EB`, the ellipsoid body), with ties broken by smaller bodyId. The edges are all connections among those neurons, and the weight is the total synapse count between the pair. **This is not a functional circuit and not a visual-system selection.** The metadata says so explicitly (`is_visual_system_selection: false`).

**Output** in `data/connectome/`:
- `drosophila_<N>.csv`: the edge list (source,target,weight), exactly as neuPrint returned it
- `drosophila_<N>_nodes.csv`: the selected neurons with rank, ROI synapse counts and cell-type annotations
- `drosophila_<N>_metadata.json`: the dataset, extraction date, exact selection rule, counts, SHA-256 checksums and git commit

**Self-loops** (a neuron synapsing onto itself) are kept in the CSV but dropped when loading (`self_loops: drop`), because the random control graphs have none. Their number is recorded in the metadata.

**Local mode.** `--from-csv file.csv` validates any local edge list and saves it under the same naming scheme, labelled "selection method unknown".

## Graph variants and controls (Phase 4)

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the pre-declared decisions.

- **Variants:** `drosophila_128_{raw,w5,w10,w20}.csv` keep edges with at least 1, 5, 10 or 20 synapses. **`w20` is the primary graph**, fixed before any MRI training; the others are for sensitivity analysis only. The raw download is verified against its checksums first and is never modified (the raw files are also marked read-only). Every variant keeps all 128 neurons in its `_nodes.csv`, even ones that filtering leaves without connections.
- **Controls:** `{er,ws,ba,rewired}_128_w20_seed{0..4}.csv`. Each has the same node count and exactly the same edge count as `w20`, is directed and has no self-loops, and is deterministic from its seed. `rewired` keeps every neuron's exact in- and out-degree and randomises who connects to whom; it is the key null model. ER/WS/BA nodes are abstract slots `0..127`, not neurons.

## MRI datasets (Phase 5)

Nothing is downloaded automatically. Put images under `data/raw/<name>/normal/...` and `data/raw/<name>/tumor/...` (PNG/JPG/JPEG; sub-folders allowed). Copy [configs/datasets/TEMPLATE_real_dataset.yaml](configs/datasets/TEMPLATE_real_dataset.yaml), fill it in, and run `scripts/prepare_dataset.py`.

- **Labels** come only from the top-level folder name.
- **Patient IDs** come from a file-name regex, from a CSV mapping, or are declared unavailable (`none`). IDs are never invented. With `none`, the missing IDs are written as `NA` in the manifest, and a leakage warning goes into the metadata's `limitations`.
- **Splitting** is by *group*, never by image. Images with the same patient ID, or with byte-identical content, form one group, and a whole group always goes to the same split. Groups are shuffled with `split_seed` and assigned per class, so every split keeps the class balance. `check_no_leakage` verifies that no patient, group or identical file is shared between splits. `MRIDataset` repeats that check and refuses to load a leaking manifest.
- **Validation:** every image is fully decoded once, so corrupt or truncated files stop the run (or are skipped and listed, with `on_invalid_image: skip`). Unsupported files are listed.
- **Preprocessing:** grayscale (or RGB), then a bilinear resize to `image_size`², then a tensor in [0, 1], then normalization with the mean/std of the **train split only**. There is no augmentation.
- **Output** in `data/splits/<name>/`:
  - `manifest.csv`: image_path, label, label_index, patient_id, group_id, split, sha256, width, height, mode
  - `dataset_metadata.json`: name, source, license, counts, patients, split sizes and class balance per split, seed, leakage check, normalization statistics, skipped files, limitations, git commit
- **Figshare `.mat` format** (`format: figshare_mat`): label and patient ID are read from `cjdata.label` / `cjdata.PID`. Classes not listed in `classes` are excluded but counted. The official 5-fold indices (`cvind.mat`) are stored per slice when they map cleanly to patients. An optional `patient_link_regex` keeps possibly-related patient IDs in the same split. See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the decisions made on this dataset.
- `configs/datasets/synthetic_demo.yaml` uses drawn shapes (**not medical images**) purely to exercise the pipeline.

## Connectome-constrained model (Phase 7)

```
MRI 1x128x128 -> CNN encoder (same as baseline) -> 128 features -> BatchNorm1d(128)
  -> feature i = state of node i -> ConnectomeLayer x 2 -> all 128 node states -> Linear(128->1) -> logit
```
(The first version, which used mean pooling into a single number, collapsed. See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).)

- **Biology decides which connections exist; training decides how strongly each one is used.** Each `ConnectomeLayer` has exactly one trainable weight per existing directed edge, plus one bias per node. A non-existent edge has no parameter, so it can never carry signal or receive gradient. Mathematically this is `W_eff = W ⊙ adjacency_mask`, stored sparsely.
- **Direction:** a signal flows from source to target: `h_new[j] = ReLU(Σ_{i→j} w_ij h[i] + b_j)`. There's no skip connection.
- **No bypass:** the classifier sees only the final node states.
- **Parameters:** every graph with the same edge count (w20 and all its controls: 3,047 edges) has **exactly the same parameter count**: 48,527 in total (encoder 41,792, BatchNorm 256, graph block 2 × (3,047 + 128), classifier 129).
- **Validation gate:** if the selected model has collapsed (constant predictions, validation AUC exactly 0.5, or a dead last graph layer), the run stops *before* the test split is loaded.
- **Diagnostics:** every epoch logs the across-image standard deviation of the encoder features, of each graph layer and of the logits, plus the share of zero and dead nodes (`diag_*` columns in `history.csv`).
- **Configs:** `configs/drosophila.yaml` (w20), `rewired.yaml`, `er.yaml`, `ws.yaml` and `ba.yaml` (seed-0 controls). They differ from `base.yaml` only in the `graph` section; a test enforces this. `graph.type: none` means the CNN baseline.
- **Weights:** biological synapse counts are not used as weights. The graph is a binary mask.

## Configuration system

Configs are YAML files in `configs/`. A config can inherit from another with `inherit: <file>`. The child file only lists what it changes, and everything else comes from the parent.

```yaml
# configs/er.yaml
inherit: base.yaml
experiment:
  name: er
graph:
  type: er
```

The four graph configs (`drosophila`, `er`, `ws`, `ba`) differ from `base.yaml` **only** in the graph section. This is how the fair comparison is enforced later (Phase 8): the only thing allowed to change between models is the graph topology.

Configs are validated strictly. An unknown key, such as the typo `learning_rat`, is an error rather than being silently ignored, because a silently ignored setting would make an experiment quietly wrong.

## Code map

| Path | What it does |
|---|---|
| `src/flyconnectome_mri/utils/config.py` | Loads YAML configs (with inheritance), checks them strictly, and returns typed Python objects. |
| `src/flyconnectome_mri/utils/logging.py` | Sets up logging to the console and to `run.log`. |
| `src/flyconnectome_mri/training/reproducibility.py` | Seeds Python/NumPy/PyTorch, picks the device, records environment info. |
| `src/flyconnectome_mri/datasets/manifest.py` | Scans class folders, validates every image, checksums, extracts patient IDs (regex / CSV / none). |
| `src/flyconnectome_mri/datasets/splits.py` | Groups (same patient or identical content), deterministic stratified group split, leakage check. |
| `src/flyconnectome_mri/datasets/transforms.py` | Grayscale/RGB loading, resize, tensor conversion, normalization, train-only statistics. |
| `src/flyconnectome_mri/datasets/mri_dataset.py` | PyTorch `Dataset` for one split; checks missing files, leakage and optionally checksums. |
| `src/flyconnectome_mri/datasets/prepare.py` | End-to-end preparation: writes manifest.csv and dataset_metadata.json. |
| `src/flyconnectome_mri/datasets/figshare.py` | Reads original Figshare `.mat` (v7.3/HDF5) files, safe read-only extraction, official CV folds. |
| `src/flyconnectome_mri/datasets/figshare_download.py` | Resumable download of public Figshare files with MD5 verification and provenance. |
| `src/flyconnectome_mri/datasets/synthetic.py` | Synthetic (non-medical) demo/test images. |
| `src/flyconnectome_mri/connectome/loader.py` | Reads a `source,target,weight` CSV, validates it (missing values, bad IDs, non-positive weights, duplicates, self-loops) and returns the graph as a sorted node mapping, edge arrays, a NetworkX `DiGraph` and a SciPy sparse adjacency matrix (`A[source, target] = weight`). |
| `src/flyconnectome_mri/connectome/statistics.py` | Node/edge counts, density, in/out-degrees, weakly/strongly connected components, weight summary. |
| `src/flyconnectome_mri/connectome/drosophila.py` | neuPrint access (token from the environment), neuron selection, edge fetching, provenance metadata, local-CSV import. |
| `src/flyconnectome_mri/utils/provenance.py` | Git commit hash, SHA-256 checksums, UTC timestamps. |
| `scripts/fetch_connectome.py` | Fetch (or import) a subgraph, validate it with the Phase 2 loader, print its statistics and first edges. |
| `src/flyconnectome_mri/connectome/filtering.py` | Weight-threshold variants with checksum verification of the raw source. |
| `src/flyconnectome_mri/connectome/random_graphs.py` | ER, directed WS, directed BA and degree-preserving rewiring algorithms (pure functions, seeded). |
| `src/flyconnectome_mri/connectome/controls.py` | Generates, saves and validates control graphs with metadata. |
| `scripts/derive_graph_variants.py`, `scripts/generate_controls.py` | Command-line wrappers that print side-by-side statistics. |
| `scripts/inspect_connectome.py` | Command-line wrapper: load a CSV, print its statistics, optionally save them as JSON. |
| `src/flyconnectome_mri/models/` | *(Phases 6–7)* CNN baseline and connectome-constrained layers. |
| `src/flyconnectome_mri/models/encoder.py`, `cnn_baseline.py` | Small 3-conv encoder (→ 128-d features) and the CNN baseline (features → 1 logit). |
| `src/flyconnectome_mri/models/connectome_block.py`, `graph_classifier.py` | Sparse edge-constrained propagation layer; encoder → graph layers → mean pooling → linear classifier. |
| `src/flyconnectome_mri/training/train.py`, `evaluate.py`, `metrics.py`, `experiment.py`, `plots.py` | Training loop with validation selection and early stopping, prediction, slice/patient metrics, the end-to-end run, and the training-curve figure. |
| `src/flyconnectome_mri/evaluation/comparison.py`, `plots.py` | Collect runs, mean ± SD per model, paired per-seed differences (descriptive, no p-values), Phase 8 figures. |
| `scripts/run_phase8.py`, `scripts/summarize_phase8.py` | Run the 30-run comparison (one process per run; failed runs recorded, never retried) and summarise it. |
| `scripts/run_experiment.py` | Command-line entry point for one experiment run. |
| `tests/` | pytest test suite. `tests/data/tiny_graph.csv` is a hand-made toy graph (not biological) with hand-computed statistics. |
| `data/` | `raw/` (your MRI data), `processed/`, `connectome/` (graph CSVs), `splits/` (saved split indices). Contents are git-ignored. |
| `results/` | One folder per run. Git-ignored. |

## Reproducibility notes

- `set_seed()` seeds Python `random`, NumPy and PyTorch (CPU + all GPUs), and asks PyTorch to use deterministic algorithms.
- CPU runs with the same seed, code and library versions should repeat exactly.
- GPU runs can still differ slightly because some CUDA kernels have no deterministic version. PyTorch prints a warning when such a kernel is used, rather than crashing.
- Results can differ across library versions or hardware. `environment.json` records both for every run.

## Scientific rules this project follows

1. The test set is never used for tuning or model selection.
2. All graph models use the same data split, preprocessing, encoder, optimizer, epochs and seeds. Only the topology changes.
3. Every comparison uses multiple seeds and reports mean ± standard deviation.
4. A degree-preserving rewired graph is always included as a control.
5. If the biological graph does worse than the random ones, that result is reported.
6. MRI slices from the same patient never cross splits.
7. A generic MaleCNS subgraph is never called the "visual system" unless it was actually extracted using visual-system annotations.
