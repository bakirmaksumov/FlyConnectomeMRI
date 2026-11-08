# Figure plan

No figure may introduce new results. Where an existing frozen plot can be reused, its exact file is named.

## Figure 1: Overall pipeline (new schematic; no data)
MRI slice (1×128×128) → shared CNN encoder (3 conv blocks, global average pool) → 128 features → BatchNorm1d → 128 graph nodes (feature *i* → node *i*) → connectome-constrained layer 1 → layer 2 (one trainable weight per existing directed edge; no skip path) → all 128 node states → Linear(128→1) → P(glioma). An inset shows the CNN baseline (encoder → Linear(128→1)). Annotate the parameter counts (41,921 vs 44,411).
*Source for the facts: `src/flyconnectome_mri/models/encoder.py`, `graph_classifier.py`, `connectome_block.py`.*

## Figure 2: Real Drosophila visual graph vs degree-preserving rewired control (new rendering of frozen graph files)
- **Left:** the visual graph, with nodes laid out by retinotopic column (hex1, hex2) and cell type, coloured by type and with edge width ∝ synapse count.
- **Right:** `rewired_visual128_raw_seed0.csv`, with the same node positions.
- **Inset table:** reciprocity 0.51 → 0.08 and clustering 0.38 → 0.10, with degrees identical.

*Sources: `data/connectome/drosophila_visual_128_raw.csv`, `_nodes.csv`, `rewired_visual128_raw_seed0.csv` (checksums in `docs/results/phase10_prerun/graph_checksums.txt`). This figure still needs to be drawn from these files; it is a visualisation, not a new result.*

## Figure 3: Leakage-safe five-fold evaluation design (new schematic; no data)
Five official Figshare folds, with fold *k* as test (unchanged) and fold *k*%5+1 as validation. Arrows show linked-family exclusions ("linked_to_test", "linked_to_validation") from train/val. A bottom strip shows every patient appearing in the test set exactly once, giving the pooled OOF set. Annotate the per-fold exclusions (14 / 14 / 19 / 18 / 11).
*Source for the numbers: `docs/results/phase10_prerun/counts.csv`, `exclusions_all.csv`.*

## Figure 4: Fold-level patient ROC AUC across models (reuse)
**Reuse:** `docs/results/phase10/fig2_fold_level_patient_auc.png`. It shows the fold means per model with fold labels and a bar for the mean across folds.
**Alternative or companion:** `docs/results/phase10/fig1_patient_auc_by_fold.png`, which shows every seed per fold.

## Figure 5: Paired Drosophila − rewired differences (reuse)
**Reuse:** `docs/results/phase10/fig4_paired_drosophila_rewired_by_fold.png` (dots = seeds, bar = fold mean, zero line).
**Supplement:** `docs/results/phase10/fig5_paired_drosophila_rewired_all_runs.png`, which shows all 25 pairs.

## Optional supplementary figures (reuse)
- **Pooled OOF ROC AUC by seed:** `docs/results/phase10/fig3_oof_patient_auc.png`.
- **Threshold-dependent trade-off (sensitivity/specificity):** `docs/results/phase10/fig7_sensitivity_specificity.png`.
- **Training stability and runtime:** `docs/results/phase10/fig8_training_stability.png`, `fig9_runtime.png`.
- **Earlier experiments:** `docs/results/phase8/fig4_paired_drosophila_minus_rewired.png`, `docs/results/phase9/fig4_paired_visual_drosophila_minus_rewired.png`.
- **EB vs visual (descriptive only):** `docs/results/phase9/fig6_eb_vs_visual_descriptive.png`.

**Layout note for the camera-ready version:** the existing PNGs were produced for internal reporting (150 dpi, wide aspect). For ISBI they should be re-exported from the same frozen CSVs at column width, without changing any value.
