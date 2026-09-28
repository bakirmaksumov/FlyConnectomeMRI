# FlyConnectomeMRI

*Written with Phase 1 and filled in as later phases land.*

This is a research prototype. It asks whether a neural network whose connectivity is constrained by a graph derived from the real *Drosophila* (fruit fly) connectome classifies brain MRI images differently from the same network wired with random graphs, or from a plain CNN.

> **This is not a medical device.** Nothing in this repository is diagnostic, and no result here should be read as "the fly brain detects tumors." The accurate wording is: *a neural network whose connectivity is constrained by a Drosophila-derived graph.*

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Project skeleton, config system, reproducibility, tests | ✅ done |
| 2 | Generic connectome edge-list loader + graph statistics | ✅ done |
| 3 | Drosophila (MaleCNS v1.0) subgraph fetching | not started |
| 4 | Random graph controls (ER / WS / BA / degree-preserving rewired) | not started |
| 5 | MRI dataset abstraction with leak-free splits | not started |
| 6 | Small CNN baseline | not started |
| 7 | Connectome-constrained model | not started |
| 8–12 | Fair comparison, small-data, rewiring, graph size, report | not started |

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

# Run an experiment from a config file
.venv\Scripts\python scripts/run_experiment.py --config configs/base.yaml
.venv\Scripts\python scripts/run_experiment.py --config configs/drosophila.yaml --seed 3
```

In Phase 1, `run_experiment.py` does not train anything yet. It:

1. loads and validates the config,
2. seeds every random number generator,
3. picks the device (CPU, or GPU if requested and available),
4. creates a run folder `results/<experiment_name>_<timestamp>/` containing:
   - `config.yaml`: the fully resolved config, so the run can be repeated exactly
   - `environment.json`: Python/PyTorch/library versions, device and seed
   - `run.log`: everything the run printed
   - `seed_fingerprint.json`: the first few random numbers drawn after seeding. Same seed → same fingerprint, which is a quick proof that seeding works.

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
| `src/flyconnectome_mri/datasets/` | *(Phase 5)* MRI image loading, preprocessing, leak-free splits. |
| `src/flyconnectome_mri/connectome/loader.py` | Reads a `source,target,weight` CSV, validates it (missing values, bad IDs, non-positive weights, duplicates, self-loops) and returns the graph as a sorted node mapping, edge arrays, a NetworkX `DiGraph` and a SciPy sparse adjacency matrix (`A[source, target] = weight`). |
| `src/flyconnectome_mri/connectome/statistics.py` | Node/edge counts, density, in/out-degrees, weakly/strongly connected components, weight summary. |
| `scripts/inspect_connectome.py` | Command-line wrapper: load a CSV, print its statistics, optionally save them as JSON. |
| `src/flyconnectome_mri/models/` | *(Phases 6–7)* CNN baseline and connectome-constrained layers. |
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
