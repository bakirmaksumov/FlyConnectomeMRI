"""Tests for generating, saving and validating control graphs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconnectome_mri.connectome.controls import generate_controls
from flyconnectome_mri.connectome.random_graphs import erdos_renyi_directed
from flyconnectome_mri.utils.config import ConfigError, ControlsConfig, load_controls_config
from flyconnectome_mri.utils.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N, M = 30, 150


@pytest.fixture
def bio_dir(tmp_path: Path) -> Path:
    """A fake biological graph with large, non-contiguous bodyIds and one isolated neuron."""
    idx = erdos_renyi_directed(N - 1, M, np.random.default_rng(42))  # neuron N-1 stays isolated
    body_ids = np.arange(N, dtype=np.int64) * 1000 + 720575940600000000
    weights = np.arange(M) % 50 + 1
    pd.DataFrame({"source": body_ids[idx[:, 0]], "target": body_ids[idx[:, 1]], "weight": weights}) \
        .to_csv(tmp_path / "bio.csv", index=False)
    pd.DataFrame({"bodyId": body_ids}).to_csv(tmp_path / "bio_nodes.csv", index=False)
    return tmp_path


def make_config(**overrides) -> ControlsConfig:
    values = dict(source_edges="bio.csv", source_nodes="bio_nodes.csv", label="t", seeds=[0, 1],
                  output_dir="out")
    values.update(overrides)
    return ControlsConfig(**values)


def degree_by_id(path: Path) -> dict[str, dict[int, int]]:
    edges = pd.read_csv(path)
    return {"out": edges["source"].value_counts().to_dict(), "in": edges["target"].value_counts().to_dict()}


def test_all_controls_match_node_and_edge_count(bio_dir: Path) -> None:
    graphs = generate_controls(make_config(), bio_dir)
    assert len(graphs) == 4 * 2
    for graph in graphs.values():
        assert graph.num_nodes == N  # the isolated neuron/slot is kept
        assert graph.num_edges == M
        assert not np.any(graph.edges[:, 0] == graph.edges[:, 1])


def test_rewired_preserves_each_neurons_degrees_and_ids(bio_dir: Path) -> None:
    generate_controls(make_config(), bio_dir)
    assert degree_by_id(bio_dir / "out" / "rewired_t_seed0.csv") == degree_by_id(bio_dir / "bio.csv")
    nodes = pd.read_csv(bio_dir / "out" / "rewired_t_seed0_nodes.csv", dtype=str)
    original = pd.read_csv(bio_dir / "bio_nodes.csv", dtype=str)
    assert nodes["node_id"].tolist() == sorted(original["bodyId"].tolist())


def test_synthetic_controls_use_abstract_slot_ids(bio_dir: Path) -> None:
    generate_controls(make_config(controls=["er"]), bio_dir)
    nodes = pd.read_csv(bio_dir / "out" / "er_t_seed0_nodes.csv")
    assert nodes["node_id"].tolist() == list(range(N))
    meta = json.loads((bio_dir / "out" / "er_t_seed0_metadata.json").read_text(encoding="utf-8"))
    assert "not neurons" in meta["node_ids"]


def test_weight_distribution_is_preserved(bio_dir: Path) -> None:
    generate_controls(make_config(), bio_dir)
    bio_weights = sorted(pd.read_csv(bio_dir / "bio.csv")["weight"])
    for control in ("er", "ws", "ba", "rewired"):
        assert sorted(pd.read_csv(bio_dir / "out" / f"{control}_t_seed0.csv")["weight"]) == bio_weights


def test_generation_is_deterministic_and_seeds_differ(bio_dir: Path) -> None:
    generate_controls(make_config(), bio_dir)
    first = {p.name: sha256_file(p) for p in (bio_dir / "out").glob("*.csv")}
    generate_controls(make_config(), bio_dir, overwrite=True)
    second = {p.name: sha256_file(p) for p in (bio_dir / "out").glob("*.csv")}
    assert first == second
    for control in ("er", "ws", "ba", "rewired"):
        assert first[f"{control}_t_seed0.csv"] != first[f"{control}_t_seed1.csv"]


def test_control_streams_are_independent(bio_dir: Path, tmp_path: Path) -> None:
    """Generating only ER must give the same ER graph as generating all controls."""
    generate_controls(make_config(), bio_dir)
    generate_controls(make_config(controls=["er"], output_dir="only_er"), bio_dir)
    assert sha256_file(bio_dir / "out" / "er_t_seed1.csv") == sha256_file(bio_dir / "only_er" / "er_t_seed1.csv")


def test_metadata_records_seed_source_and_overlap(bio_dir: Path) -> None:
    generate_controls(make_config(), bio_dir)
    meta = json.loads((bio_dir / "out" / "rewired_t_seed1_metadata.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 1
    assert meta["matched_to"]["edges"]["sha256"] == sha256_file(bio_dir / "bio.csv")
    assert meta["matched_to"]["num_edges"] == M
    assert 0.0 <= meta["edge_overlap_with_biological_graph"] < 1.0
    assert meta["generator_details"]["attempted_swaps"] == 10 * M
    assert meta["statistics"]["num_edges"] == M


def test_existing_controls_are_not_overwritten(bio_dir: Path) -> None:
    generate_controls(make_config(controls=["er"]), bio_dir)
    with pytest.raises(FileExistsError):
        generate_controls(make_config(controls=["er"]), bio_dir)


@pytest.mark.parametrize(
    "overrides",
    [{"seeds": []}, {"seeds": [1, 1]}, {"seeds": [-1]}, {"controls": ["sbm"]}, {"controls": []},
     {"ws_rewire_prob": 1.5}, {"rewire_swaps_per_edge": 0}],
)
def test_invalid_controls_configs_are_rejected(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        make_config(**overrides)


def test_shipped_controls_config_targets_primary_w20_graph() -> None:
    config = load_controls_config(PROJECT_ROOT / "configs" / "controls.yaml")
    assert config.source_edges.endswith("drosophila_128_w20.csv")
    assert config.controls == ["er", "ws", "ba", "rewired"]
