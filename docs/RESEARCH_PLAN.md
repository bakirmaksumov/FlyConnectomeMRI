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

## Rules (from the project specification)

1. The test set is never used for tuning or model selection.
2. All graph models share the same split, preprocessing, encoder, node count, optimizer, learning rate, epochs, batch size and seeds. Only the topology changes.
3. Multiple seeds, reported as mean ± std.
4. Results where the biological graph does worse than the controls are reported, not dropped.
5. MRI images from the same patient never cross splits.
6. A generic MaleCNS subgraph is never called the "visual system".
