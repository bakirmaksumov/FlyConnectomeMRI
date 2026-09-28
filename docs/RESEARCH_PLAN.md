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

## Rules (from the project specification)

1. The test set is never used for tuning or model selection.
2. All graph models share the same split, preprocessing, encoder, node count, optimizer, learning rate, epochs, batch size and seeds. Only the topology changes.
3. Multiple seeds, reported as mean ± std.
4. Results where the biological graph does worse than the controls are reported, not dropped.
5. MRI images from the same patient never cross splits.
6. A generic MaleCNS subgraph is never called the "visual system".
