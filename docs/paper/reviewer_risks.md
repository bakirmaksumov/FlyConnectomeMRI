# Likely reviewer criticisms and how the study addresses them

Weaknesses are stated plainly. "Addressed" means handled in the design; "Partly" or "Not addressed" means the risk remains and must be stated in the paper.

**1. Small dataset (171 patients, 21-patient test split in early phases).**
- **Status:** Partly addressed.
- **What the study does:** the main result uses leakage-safe cross-validation over the full cohort (each patient tested once, 5 seeds per fold) and reports fold-level variability and pooled OOF metrics. The early 21-patient results are reported as preliminary.
- **Remaining risk:** the effect sizes resolvable at n = 171 are limited. Small topology effects cannot be excluded, only "not demonstrated".

**2. Possible site-related confounding (every `MR`-prefixed patient is a glioma).**
- **Status:** Not resolved, but disclosed.
- **What the study does:** it is documented before training, affects all models equally (so the topology comparison is internally fair), and a numeric-ID subgroup analysis is pre-declared. The OOF files keep the ID format, so this analysis can be run without retraining.
- **Remaining risk:** absolute AUCs may partly reflect site differences. The subgroup analysis has not yet been run.

**3. Why should a fly connectome be relevant to human MRI?**
- **Status:** Addressed by framing.
- **What the study does:** the paper does not assume relevance. It tests whether a biological topology used as an inductive bias differs from matched topologies, and the negative result is the answer.
- **Remaining risk:** reviewers may see the premise as unmotivated. Emphasise the methodological contribution (how to test connectome priors properly).

**4. Graph models have more parameters than the CNN.**
- **Status:** Addressed for the key claim.
- **What the study does:** the topology comparison is between graph models with *identical* architecture and parameter count (44,411), and CNN-vs-graph is explicitly labelled architectural.
- **Remaining risk:** the CNN gain may be due to extra layers or BatchNorm. That is exactly our interpretation, but no parameter-matched MLP control was run.

**5. No topology-specific improvement.**
- **Status:** This is the finding.
- **What the study does:** pre-registered protocol and analysis code, paired comparison against a degree-preserving null, and all runs reported with none failed or dropped.
- **Remaining risk:** some venues undervalue negative results. Stress the controls and leakage safety.

**6. Is a negative result useful?**
- **Status:** Argued in the Discussion.
- **What the study does:** it shows that a connectome-wired model's apparent gains over a CNN were reproduced by random, lattice-like and hub graphs. Without matched nulls this would have been misattributed to biology.
- **Remaining risk:** the reviewer's taste.

**7. Choice of visual circuit (128 neurons, 11 types; no T4/T5).**
- **Status:** Partly addressed.
- **What the study does:** three candidates were compared on graph and biological criteria before any MRI training; the rule is deterministic and uses only annotations; missing T4/T5 is disclosed rather than inferred.
- **Remaining risk:** a different circuit or scale (for example with T4/T5, or thousands of neurons) might behave differently. It is only one circuit.

**8. Patient leakage.**
- **Status:** Addressed.
- **What the study does:** patient-level splits, a conservative link key for possibly related examinations, zero overlap verified per fold, official outer test folds preserved, every patient tested exactly once, and duplicate-image detection.
- **Remaining risk:** the link key is an assumption. It excluded 11–19 glioma patients per fold from training and validation, so the CV differs from the authors' protocol, and this is disclosed.

**9. Fixed threshold of 0.5.**
- **Status:** Deliberate.
- **What the study does:** the threshold was frozen to avoid test-set tuning. AUC (threshold-free) is primary. The resulting low specificity of the graph models is reported rather than hidden.
- **Remaining risk:** the threshold-dependent metrics favour the CNN. A validation-based threshold rule would need pre-registration in future work.

**10. No external validation.**
- **Status:** Not addressed.
- **What the study does:** it is a single dataset, and the claims are restricted to the evaluated setting.
- **Remaining risk:** generalisation is unknown. External cohorts are listed as future work.

**11. Possible overfitting or selection bias.**
- **Status:** Largely addressed.
- **What the study does:** all hyperparameters, architectures, graphs, thresholds, seeds, fold rules and aggregation code were frozen and committed before the relevant runs; model selection used validation only; nothing was tuned on test results.
- **Remaining risk:**
  - The Phase 7 architecture fix was made after seeing a collapse, before any comparison. It applies identically to all graphs, and this is disclosed.
  - **Disclose explicitly:** the collapsed seed-0 run (and its reproduction) was evaluated once on the fixed 21-patient test split, because the validation gate was introduced only afterwards. That output was constant predictions (AUC 0.5); the fix was motivated by the validation diagnostics.
  - The same fixed test split was used across Phases 6–9, each time under a pre-declared protocol. Repeated use of one small test split is a legitimate concern. The Phase 10 cross-validation, where every patient is tested once per model and seed, is the main result partly for this reason.

**12. Only five seeds; no statistical test.**
- **Status:** Acknowledged.
- **What the study does:** results are descriptive (mean ± SD, per-seed and per-fold paired values, sign counts), with no significance claims. The primary paired effect (−0.002) is an order of magnitude smaller than its SD (0.030).
- **Remaining risk:** reviewers may ask for a formal test, for example a paired test over folds or a hierarchical bootstrap. If added, it must be pre-declared and reported as a new analysis.

**13. Binary mask ignores synapse weights and neurotransmitter signs.**
- **Status:** Acknowledged.
- **What the study does:** this was a deliberate choice for a clean topology test.
- **Remaining risk:** reviewers may argue the biology was not fully used. Weighted and signed variants are future work.

**14. Arbitrary feature-to-node mapping.**
- **Status:** Acknowledged.
- **What the study does:** the mapping is identical across graphs, so it is fair for the comparison.
- **Remaining risk:** the encoder can co-adapt to any wiring, which may itself explain the null result (see Discussion). A fixed or learned mapping could be tested later.
