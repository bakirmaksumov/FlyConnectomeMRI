"""Tests for the connectome-constrained layer and classifier (Phase 7)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.autograd.functional import jacobian

from flyconnectome_mri.connectome.random_graphs import erdos_renyi_directed
from flyconnectome_mri.datasets.prepare import prepare_dataset
from flyconnectome_mri.datasets.synthetic import make_synthetic_dataset
from flyconnectome_mri.models.cnn_baseline import count_parameters
from flyconnectome_mri.models.connectome_block import ConnectomeLayer
from flyconnectome_mri.models.graph_classifier import GraphClassifier, parameter_breakdown
from flyconnectome_mri.training import experiment as experiment_module
from flyconnectome_mri.utils.config import (
    Config,
    ConfigError,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Tiny directed graph on 4 nodes: 0->1, 1->2, 2->0, 3->2   (no edge INTO node 3)
EDGES = np.array([[0, 1], [1, 2], [2, 0], [3, 2]])


def positive_layer(edges: np.ndarray = EDGES, n: int = 4) -> ConnectomeLayer:
    """Layer with strictly positive weights so every ReLU is active for positive inputs."""
    layer = ConnectomeLayer(edges, n)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(0.5, 1.5, len(edges)))
        layer.bias.fill_(0.1)
    return layer


# ---------------------------------------------------------------------------
# ConnectomeLayer
# ---------------------------------------------------------------------------


def test_one_trainable_weight_per_existing_edge() -> None:
    layer = ConnectomeLayer(EDGES, 4)
    assert layer.weight.shape == (4,)          # 4 edges, not 4 x 4 = 16
    assert layer.bias.shape == (4,)
    assert count_parameters(layer) == 4 + 4
    assert not layer.source.requires_grad       # topology is a fixed buffer


def test_effective_weights_are_zero_outside_the_mask() -> None:
    layer = ConnectomeLayer(EDGES, 4)
    w_eff, mask = layer.effective_weight_matrix(), layer.adjacency_mask()
    assert mask.sum() == 4
    assert torch.all(w_eff[~mask] == 0)
    assert torch.all(w_eff[mask] != 0)


def test_prohibited_edges_do_not_contribute() -> None:
    """d out[j] / d in[i] may be non-zero only if the edge i -> j exists."""
    layer = positive_layer()
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    jac = jacobian(lambda v: layer(v.unsqueeze(0)).squeeze(0), x)  # jac[j, i] = d out_j / d in_i
    allowed = layer.adjacency_mask().T                               # allowed[j, i] = edge i -> j
    assert torch.all(jac[~allowed] == 0)
    assert torch.all(jac[allowed] != 0)


def test_signal_follows_edge_direction() -> None:
    """Edge 0 -> 1 exists, 1 -> 0 does not: changing node 1 must not affect node 0 via that pair."""
    layer = positive_layer()
    base = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    bumped_node0 = base.clone(); bumped_node0[0, 0] += 1
    bumped_node1 = base.clone(); bumped_node1[0, 1] += 1
    assert layer(bumped_node0)[0, 1] != layer(base)[0, 1]   # 0 -> 1: node 1 changes
    assert layer(bumped_node1)[0, 0] == layer(base)[0, 0]   # no 1 -> 0: node 0 unchanged


def test_node_without_incoming_edges_only_has_its_bias() -> None:
    layer = positive_layer()
    out = layer(torch.randn(5, 4).abs())
    assert torch.allclose(out[:, 3], torch.full((5,), 0.1))  # node 3 has no in-edges


def test_all_permitted_edges_receive_gradients() -> None:
    layer = positive_layer()
    layer(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).sum().backward()
    assert layer.weight.grad is not None
    assert torch.all(layer.weight.grad != 0)
    # the gradient of edge i->j equals h[i] here (upstream grad 1, ReLU active)
    assert torch.allclose(layer.weight.grad, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_invalid_edges_are_rejected() -> None:
    with pytest.raises(ValueError):
        ConnectomeLayer(np.array([[0, 5]]), 4)
    with pytest.raises(ValueError):
        ConnectomeLayer(np.zeros((0, 2), dtype=int), 4)
    with pytest.raises(ValueError, match="expected input"):
        ConnectomeLayer(EDGES, 4)(torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# GraphClassifier
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_128() -> np.ndarray:
    return erdos_renyi_directed(128, 3047, np.random.default_rng(0))


def test_classifier_output_shape(graph_128: np.ndarray) -> None:
    model = GraphClassifier(graph_128, 128)
    assert model(torch.randn(3, 1, 128, 128)).shape == (3,)
    assert model.node_states(torch.randn(3, 1, 128, 128)).shape == (3, 128)


def test_parameter_count_depends_only_on_edge_count(graph_128: np.ndarray) -> None:
    other = erdos_renyi_directed(128, 3047, np.random.default_rng(99))
    a, b = GraphClassifier(graph_128, 128), GraphClassifier(other, 128)
    breakdown = parameter_breakdown(a)
    # encoder 41,792 (baseline minus its 129-parameter head); BatchNorm 2 x 128;
    # graph block 2 layers x (3047 edge weights + 128 biases); classifier 128 + 1
    assert breakdown == {"encoder": 41792, "node_norm": 256, "graph_block": 2 * (3047 + 128),
                         "classifier": 129, "total": 48527}
    assert count_parameters(a) == count_parameters(b) == 48527


def test_every_edge_of_every_layer_can_learn(graph_128: np.ndarray) -> None:
    """With positive node states and active ReLUs, every edge of both layers gets a gradient.

    (Starts from node states rather than images: an encoder channel that is 0 after
    its ReLU gives its outgoing edges zero gradient for that input - normal ReLU
    behaviour, not a masking effect.)
    """
    torch.manual_seed(0)
    model = GraphClassifier(graph_128, 128)
    with torch.no_grad():  # positive weights so the ReLUs are active and every edge is exercised
        for layer in model.graph_block:
            layer.weight.uniform_(0.01, 0.05)
            layer.bias.fill_(0.01)
    h = torch.rand(4, 128) + 0.1
    for layer in model.graph_block:
        h = layer(h)
    model.classifier(h).sum().backward()
    for layer in model.graph_block:
        assert torch.all(layer.weight.grad != 0)


def test_feature_dim_must_equal_node_count(graph_128: np.ndarray) -> None:
    with pytest.raises(ValueError, match="feature_dim == num_nodes"):
        GraphClassifier(graph_128, 128, feature_dim=64)


def test_same_seed_same_initial_weights(graph_128: np.ndarray) -> None:
    torch.manual_seed(1)
    a = GraphClassifier(graph_128, 128).state_dict()
    torch.manual_seed(1)
    b = GraphClassifier(graph_128, 128).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# Configs and end-to-end training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["drosophila", "rewired", "er", "ws", "ba"])
def test_graph_configs_point_to_primary_w20_files(name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml")
    assert config.graph.type == name
    assert "128_w20" in config.graph.edges_file
    if name != "drosophila":
        assert config.graph.edges_file.endswith("_seed0.csv")  # default seed 0
        seeded = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml", seed=3)
        assert seeded.graph.edges_file == f"data/connectome/{name}_128_w20_seed3.csv"
        assert seeded.graph.nodes_file == f"data/connectome/{name}_128_w20_seed3_nodes.csv"
    else:
        assert load_config(PROJECT_ROOT / "configs" / "drosophila.yaml", seed=3).graph.edges_file == \
            config.graph.edges_file  # the biological graph does not change with the seed


def test_graph_config_validation() -> None:
    with pytest.raises(ConfigError):
        GraphConfig(type="er")                                 # files missing
    with pytest.raises(ConfigError):
        GraphConfig(type="none", edges_file="x.csv")           # baseline must not have a graph
    with pytest.raises(ConfigError, match="feature_dim"):
        Config(experiment=ExperimentConfig(name="x", seed=0),
               graph=GraphConfig(type="er", num_nodes=64, edges_file="a", nodes_file="b"))


def test_graph_model_trains_end_to_end(tmp_path: Path) -> None:
    root = make_synthetic_dataset(tmp_path / "raw", patients_per_class=8, slices_per_patient=3, image_size=32)
    prepare_dataset(DatasetConfig(
        name="t", source="synthetic", license="n/a", root=str(root), classes={"normal": 0, "tumor": 1},
        positive_class="tumor", split_seed=0, output_dir=str(tmp_path / "prepared"),
        patient_id_source="filename_regex", patient_id_regex=r"^(?P<patient>P\d+)_s\d+", image_size=32,
        train_fraction=0.5, val_fraction=0.25, test_fraction=0.25,
    ), PROJECT_ROOT)
    edges = erdos_renyi_directed(128, 600, np.random.default_rng(3))
    pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1], "weight": 1}).to_csv(tmp_path / "g.csv", index=False)
    pd.DataFrame({"node_id": range(128)}).to_csv(tmp_path / "g_nodes.csv", index=False)

    config = Config(
        experiment=ExperimentConfig(name="t", seed=0, device="cpu"),
        graph=GraphConfig(type="er", edges_file=str(tmp_path / "g.csv"), nodes_file=str(tmp_path / "g_nodes.csv"),
                          node_column="node_id"),
        data=DataConfig(prepared_dir=str(tmp_path / "prepared")),
        training=TrainingConfig(epochs=4, batch_size=8),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = experiment_module.run_training_experiment(config, run_dir, torch.device("cpu"), logging.getLogger("t"))
    assert metrics["model"] == "graph_er"
    assert metrics["graph"]["num_edges"] == 600
    assert metrics["parameter_breakdown"]["graph_block"] == 2 * (600 + 128)
    checkpoint = torch.load(run_dir / "best_model.pt")
    assert torch.equal(checkpoint["model_state"]["graph_block.0.source"], torch.as_tensor(edges[:, 0]))
    assert pd.read_csv(run_dir / "node_order.csv")["node_id"].tolist() == list(range(128))


# ---------------------------------------------------------------------------
# Equivalence with the dense masked formula W_eff = W * mask
# ---------------------------------------------------------------------------


def test_sparse_layer_equals_dense_masked_formula_and_forbidden_gradients_are_zero() -> None:
    """Reference: out = ReLU(h @ (W * mask) + b). Forbidden entries of W get exactly zero gradient."""
    layer = positive_layer()
    mask = layer.adjacency_mask().float()
    dense_w = torch.randn(4, 4)
    with torch.no_grad():
        dense_w[layer.source, layer.target] = layer.weight   # same values on allowed edges
    dense_w.requires_grad_(True)
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])

    dense_out = torch.relu(h @ (dense_w * mask) + layer.bias.detach())
    sparse_out = layer(h)
    assert torch.allclose(sparse_out, dense_out)

    dense_out.sum().backward()
    sparse_out.sum().backward()
    forbidden = mask == 0
    assert torch.all(dense_w.grad[forbidden] == 0)                       # forbidden: zero gradient
    assert torch.all(dense_w.grad[~forbidden] != 0)                      # allowed: non-zero
    assert torch.allclose(dense_w.grad[layer.source, layer.target], layer.weight.grad)  # same as sparse


# ---------------------------------------------------------------------------
# Node ordering
# ---------------------------------------------------------------------------


def test_node_order_is_sorted_node_ids_regardless_of_file_order(tmp_path: Path) -> None:
    from flyconnectome_mri.training.experiment import load_graph

    pd.DataFrame({"rank": [1, 2, 3], "bodyId": [300, 100, 200]}).to_csv(tmp_path / "n.csv", index=False)
    pd.DataFrame({"source": [300, 100], "target": [100, 200], "weight": [5, 7]}).to_csv(tmp_path / "e.csv", index=False)
    config = Config(experiment=ExperimentConfig(name="x", seed=0),
                    graph=GraphConfig(type="drosophila", num_nodes=3, edges_file=str(tmp_path / "e.csv"),
                                      nodes_file=str(tmp_path / "n.csv")),
                    model=ModelConfig(feature_dim=3))
    graph = load_graph(config)
    assert graph.node_ids.tolist() == [100, 200, 300]            # node i = i-th smallest bodyId
    assert graph.edges.tolist() == [[0, 1], [2, 0]]              # 100->200, 300->100 in index space


REAL_EDGES = PROJECT_ROOT / "data" / "connectome" / "drosophila_128_w20.csv"


@pytest.mark.skipif(not REAL_EDGES.exists(), reason="real MaleCNS data not downloaded")
def test_real_drosophila_node_order_matches_saved_node_list() -> None:
    from flyconnectome_mri.training.experiment import build_model, load_graph

    config = load_config(PROJECT_ROOT / "configs" / "drosophila.yaml")
    graph = load_graph(config)
    saved = pd.read_csv(PROJECT_ROOT / config.graph.nodes_file, dtype={"bodyId": "int64"})
    assert graph.num_nodes == len(saved) == 128
    assert graph.node_ids.tolist() == sorted(saved["bodyId"].tolist())  # every neuron, sorted by bodyId
    assert graph.num_edges == 3047

    # The model's edge buffers are exactly the biological edges in that node order.
    model = build_model(config, graph)
    layer = model.graph_block[0]
    as_ids = {(int(graph.node_ids[s]), int(graph.node_ids[t])) for s, t in zip(layer.source, layer.target)}
    raw = pd.read_csv(PROJECT_ROOT / config.graph.edges_file)
    assert as_ids == set(zip(raw["source"].tolist(), raw["target"].tolist()))


# ---------------------------------------------------------------------------
# Option A architecture: graph dependence and no bypass
# ---------------------------------------------------------------------------


def copy_model_with_edges(model: GraphClassifier, edges: np.ndarray, weight_fn) -> GraphClassifier:
    """New model on ``edges`` sharing all non-graph parameters; graph weights from ``weight_fn(layer_index)``."""
    other = GraphClassifier(edges, 128)
    other.encoder.load_state_dict(model.encoder.state_dict())
    other.node_norm.load_state_dict(model.node_norm.state_dict())
    other.classifier.load_state_dict(model.classifier.state_dict())
    with torch.no_grad():
        for i, layer in enumerate(other.graph_block):
            layer.weight.copy_(weight_fn(i))
            layer.bias.copy_(model.graph_block[i].bias)
    return other.eval()


@pytest.fixture
def trained_like_model(graph_128: np.ndarray) -> GraphClassifier:
    torch.manual_seed(0)
    model = GraphClassifier(graph_128, 128).eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics and biases, as after training
        model.node_norm.running_mean.fill_(0.05)
        model.node_norm.running_var.fill_(0.001)
        for layer in model.graph_block:
            layer.bias.fill_(0.05)
    return model


def test_output_depends_on_graph_connectivity(trained_like_model: GraphClassifier) -> None:
    """(1) Same parameters, different wiring -> different predictions."""
    x = torch.rand(6, 1, 128, 128)
    other_edges = erdos_renyi_directed(128, 3047, np.random.default_rng(123))
    other = copy_model_with_edges(trained_like_model, other_edges,
                                  lambda i: trained_like_model.graph_block[i].weight)
    assert not torch.allclose(trained_like_model(x), other(x))


def test_removing_one_edge_changes_the_output(trained_like_model: GraphClassifier, graph_128: np.ndarray) -> None:
    """(2) Remove the single most active layer-1 edge -> the output changes."""
    x = torch.rand(6, 1, 128, 128)
    base = trained_like_model(x)
    layer1 = trained_like_model.graph_block[0]
    with torch.no_grad():
        states = trained_like_model.node_norm(trained_like_model.encoder(x))
        k = int(torch.argmax(states[:, layer1.source].abs().sum(0) * layer1.weight.abs()))
    keep = np.ones(len(graph_128), dtype=bool)
    keep[k] = False
    keep_t = torch.as_tensor(keep)
    removed = copy_model_with_edges(
        trained_like_model, graph_128[keep],
        lambda i: trained_like_model.graph_block[i].weight[keep_t])
    assert not torch.allclose(base, removed(x))


def test_no_encoder_to_classifier_bypass(trained_like_model: GraphClassifier) -> None:
    """(3) Silence the graph (all edge weights 0): the output no longer depends on the image at all."""
    x = torch.rand(6, 1, 128, 128)
    with torch.no_grad():
        for layer in trained_like_model.graph_block:
            layer.weight.zero_()
    out = trained_like_model(x)
    assert torch.allclose(out, out[0].expand_as(out))  # identical for every image
    x.requires_grad_(True)
    trained_like_model(x).sum().backward()
    assert torch.all(x.grad == 0)  # no path from the image to the output


def test_classifier_receives_exactly_the_128_final_node_states(trained_like_model: GraphClassifier) -> None:
    """(7) The classifier's only input is the (batch, 128) output of the last graph layer."""
    captured, last = {}, {}
    trained_like_model.classifier.register_forward_hook(lambda m, inp, out: captured.update(x=inp[0]))
    trained_like_model.graph_block[-1].register_forward_hook(lambda m, inp, out: last.update(x=out))
    trained_like_model(torch.rand(5, 1, 128, 128))
    assert trained_like_model.classifier.in_features == 128
    assert captured["x"].shape == (5, 128)
    assert captured["x"] is last["x"]  # the very same tensor: nothing added or concatenated


def test_model_level_forbidden_edges_get_zero_gradient(trained_like_model: GraphClassifier) -> None:
    """(4)(5)(6) at model level: the dense-equivalent gradient is zero off the mask; allowed edges learn."""
    trained_like_model.train()
    trained_like_model(torch.rand(4, 1, 128, 128)).sum().backward()
    for layer in trained_like_model.graph_block:
        dense_grad = torch.zeros(128, 128)
        dense_grad[layer.source, layer.target] = layer.weight.grad
        assert torch.all(dense_grad[~layer.adjacency_mask()] == 0)
        assert torch.count_nonzero(layer.weight.grad) > 0.5 * layer.num_edges


def test_diagnostics_report_every_stage(trained_like_model: GraphClassifier) -> None:
    stats = trained_like_model.diagnostics(torch.rand(8, 1, 128, 128))
    assert set(stats) == {"encoder_feature_std", "graph_layer1_std", "graph_layer1_zero_frac",
                          "graph_layer1_dead_nodes_frac", "graph_layer2_std", "graph_layer2_zero_frac",
                          "graph_layer2_dead_nodes_frac", "logit_std"}
    assert 0.0 <= stats["graph_layer2_zero_frac"] <= 1.0


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------


def test_collapse_check() -> None:
    from flyconnectome_mri.training.experiment import collapse_check

    healthy = {"graph_layer2_std": 0.3, "graph_layer2_dead_nodes_frac": 0.2, "logit_std": 1.0}
    assert collapse_check(np.array([0.2, 0.6, 0.9]), healthy)["passed"]
    # No performance threshold: predictions that are useless but vary still pass.
    assert collapse_check(np.array([0.9, 0.1, 0.5]), healthy)["passed"]
    assert not collapse_check(np.full(5, 0.35), healthy)["passed"]                     # constant output
    assert not collapse_check(np.array([0.2, float("nan")]), healthy)["passed"]        # non-finite
    assert not collapse_check(np.array([0.2, 0.6]), {**healthy, "logit_std": float("inf")})["passed"]
    dead = {"graph_layer2_std": 0.0, "graph_layer2_dead_nodes_frac": 1.0}
    result = collapse_check(np.array([0.2, 0.6]), dead)
    assert not result["passed"] and len(result["failures"]) == 2


def test_collapsed_model_never_touches_the_test_split(tmp_path: Path, monkeypatch) -> None:
    root = make_synthetic_dataset(tmp_path / "raw", patients_per_class=6, slices_per_patient=2, image_size=32)
    prepare_dataset(DatasetConfig(
        name="t", source="synthetic", license="n/a", root=str(root), classes={"normal": 0, "tumor": 1},
        positive_class="tumor", split_seed=0, output_dir=str(tmp_path / "prepared"),
        patient_id_source="filename_regex", patient_id_regex=r"^(?P<patient>P\d+)_s\d+", image_size=32,
        train_fraction=0.5, val_fraction=0.25, test_fraction=0.25,
    ), PROJECT_ROOT)
    loaded = []
    original = experiment_module.MRIDataset.from_prepared

    def recording(prepared_dir, split, **kwargs):
        loaded.append(split)
        return original(prepared_dir, split, **kwargs)

    monkeypatch.setattr(experiment_module.MRIDataset, "from_prepared", staticmethod(recording))
    monkeypatch.setattr(experiment_module, "collapse_check",
                        lambda *a: {"passed": False, "failures": ["forced for test"]})
    config = Config(experiment=ExperimentConfig(name="t", seed=0, device="cpu"),
                    data=DataConfig(prepared_dir=str(tmp_path / "prepared")),
                    training=TrainingConfig(epochs=1, batch_size=8))
    (tmp_path / "run").mkdir()
    metrics = experiment_module.run_training_experiment(config, tmp_path / "run", torch.device("cpu"),
                                                        logging.getLogger("t"))
    assert metrics["status"] == "collapsed_on_validation" and metrics["test"] is None
    assert "test" not in loaded
    assert not (tmp_path / "run" / "test_predictions_slice.csv").exists()
