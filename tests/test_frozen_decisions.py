"""Guards for the research decisions frozen on 2026-09-28 (see docs/RESEARCH_PLAN.md).

If any of these tests fail, a frozen decision or a documented graph-construction
rule has changed. Do NOT update the expected values to make the test pass unless
the research plan records a deliberate, dated decision to change them BEFORE the
affected experiments are run.
"""

import hashlib
from pathlib import Path

import numpy as np

from flyconnectome_mri.connectome.controls import CONTROL_STREAM
from flyconnectome_mri.connectome.random_graphs import (
    barabasi_albert_directed,
    degree_preserving_rewire,
    erdos_renyi_directed,
    watts_strogatz_directed,
)
from flyconnectome_mri.utils.config import load_controls_config, load_graph_variants_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()


def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def test_primary_graph_and_sensitivity_variants_are_frozen() -> None:
    config = load_graph_variants_config(PROJECT_ROOT / "configs" / "graph_variants.yaml")
    assert config.primary_variant == "w20"
    assert config.variants == {"raw": 1, "w5": 5, "w10": 10, "w20": 20}
    assert config.primary_declared_on == "2026-09-28"


def test_control_set_is_frozen() -> None:
    config = load_controls_config(PROJECT_ROOT / "configs" / "controls.yaml")
    assert config.source_edges == "data/connectome/drosophila_128_w20.csv"
    assert config.controls == ["er", "ws", "ba", "rewired"]
    assert config.seeds == [0, 1, 2, 3, 4]
    assert config.ws_rewire_prob == 0.1
    assert config.rewire_swaps_per_edge == 10
    assert CONTROL_STREAM == {"er": 1, "ws": 2, "ba": 3, "rewired": 4}


# Golden outputs of each construction rule for fixed small inputs (computed 2026-09-28).


def test_er_rule_unchanged() -> None:
    assert digest(erdos_renyi_directed(30, 200, rng())) == (
        "de453eb03791f07ca151b2c5d26867c68be74f47ea73ab4259668d7015c295c1")


def test_directed_ws_rule_unchanged() -> None:
    assert digest(watts_strogatz_directed(30, 200, 0.1, rng())) == (
        "7140cad2a2ada6dd6e0d02f383ab19d3e0d0017864bb79644687b00858dae573")


def test_directed_ba_rule_unchanged() -> None:
    assert digest(barabasi_albert_directed(30, 200, rng())) == (
        "db0da399edf1e42df61e600126a7495bdda6f0d7049d7b3f2e695043c36a62dc")


def test_degree_preserving_rewire_rule_unchanged() -> None:
    base = erdos_renyi_directed(30, 200, np.random.default_rng(1))
    result = degree_preserving_rewire(base, np.arange(200), 10, rng())
    assert digest(result.edges) == "07f825564e08ac00a54249409454517b0beeb631e70389bf6b2be42506690163"
    assert digest(result.weights) == "d8fc1e913ed87601ce7ad50a5e986b8fcec986a199b59622af21644bf8f694a3"
    assert result.accepted_swaps == 1010


def test_phase6_baseline_settings_are_frozen() -> None:
    """Frozen 2026-09-28 at commit f289329 (see docs/RESEARCH_PLAN.md)."""
    from flyconnectome_mri.utils.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    assert config.data.prepared_dir == "data/splits/figshare_glioma_meningioma"
    assert config.model.encoder_channels == [16, 32]
    assert config.model.feature_dim == 128
    t = config.training
    assert (t.learning_rate, t.weight_decay, t.batch_size, t.epochs) == (1e-3, 1e-4, 32, 30)
    assert t.early_stopping_patience == 7
    assert t.selection_metric == "val_roc_auc"
    assert t.class_weighting == "pos_weight_from_train"
    assert t.decision_threshold == 0.5
    assert config.experiment.seed == 0


def test_dataset_resolution_and_grouping_are_frozen() -> None:
    from flyconnectome_mri.utils.config import load_dataset_config

    config = load_dataset_config(PROJECT_ROOT / "configs" / "datasets" / "figshare_glioma_meningioma.yaml")
    assert config.image_size == 128
    assert config.split_seed == 0
    assert config.patient_link_regex == r"^(?P<base>MR\d{6})[A-Z]?$"


def test_graph_architecture_is_frozen() -> None:
    """Option A (2026-09-28): encoder -> BatchNorm1d(128) -> 2 ConnectomeLayers -> Linear(128, 1)."""
    import torch
    from torch import nn

    from flyconnectome_mri.models.connectome_block import ConnectomeLayer
    from flyconnectome_mri.models.graph_classifier import GraphClassifier, parameter_breakdown

    model = GraphClassifier(erdos_renyi_directed(128, 3047, np.random.default_rng(0)), 128)
    children = dict(model.named_children())
    assert list(children) == ["encoder", "node_norm", "graph_block", "classifier"]
    assert isinstance(children["node_norm"], nn.BatchNorm1d) and children["node_norm"].num_features == 128
    assert len(children["graph_block"]) == 2
    assert all(isinstance(layer, ConnectomeLayer) and layer.activation for layer in children["graph_block"])
    assert isinstance(children["classifier"], nn.Linear)
    assert (children["classifier"].in_features, children["classifier"].out_features) == (128, 1)
    assert parameter_breakdown(model) == {"encoder": 41792, "node_norm": 256, "graph_block": 6350,
                                          "classifier": 129, "total": 48527}
    assert model(torch.zeros(2, 1, 128, 128)).shape == (2,)


def test_visual_graph_decisions_are_frozen() -> None:
    """Declared 2026-09-29 before any MRI training on the visual graph (docs/RESEARCH_PLAN.md)."""
    from flyconnectome_mri.utils.config import load_fetch_config

    fetch = load_fetch_config(PROJECT_ROOT / "configs" / "malecns_visual.yaml")
    assert fetch.selection_method == "visual_local_columns" and fetch.roi == "ME(R)"
    assert fetch.cell_types == ["L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "C3", "Tm1", "Tm2", "Tm4", "Tm9"]
    assert (fetch.num_nodes, fetch.min_total_weight, fetch.self_loops) == (128, 1, "drop")
    variants = load_graph_variants_config(PROJECT_ROOT / "configs" / "graph_variants_visual.yaml")
    assert variants.variants == {"raw": 1, "w5": 5} and variants.primary_variant == "raw"
    controls = load_controls_config(PROJECT_ROOT / "configs" / "controls_visual.yaml")
    assert controls.source_edges == "data/connectome/drosophila_visual_128_raw.csv"
    assert controls.controls == ["er", "ws", "ba", "rewired"] and controls.seeds == [0, 1, 2, 3, 4]
    assert (controls.ws_rewire_prob, controls.rewire_swaps_per_edge) == (0.1, 10)
