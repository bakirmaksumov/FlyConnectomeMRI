"""Tests for the YAML configuration system."""

from pathlib import Path

import pytest
import yaml

from flyconnectome_mri.utils.config import (
    Config,
    ConfigError,
    config_from_dict,
    load_config,
    save_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
GRAPH_CONFIGS = ["drosophila", "rewired", "er", "ws", "ba"]


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["base", *GRAPH_CONFIGS])
def test_shipped_configs_load(name: str) -> None:
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert isinstance(config, Config)
    assert config.experiment.name == name


@pytest.mark.parametrize("name", GRAPH_CONFIGS)
def test_graph_configs_differ_from_base_only_in_graph_and_name(name: str) -> None:
    """Fairness guard: graph configs may change only the graph section and the run name."""
    base = load_config(CONFIG_DIR / "base.yaml").to_dict()
    child = load_config(CONFIG_DIR / f"{name}.yaml").to_dict()

    assert child["graph"]["type"] == name
    assert child["graph"]["num_nodes"] == base["graph"]["num_nodes"]
    base["graph"] = child["graph"]
    base["experiment"]["name"] = child["experiment"]["name"]
    assert child == base


def test_inheritance_merges_nested_values(tmp_path: Path) -> None:
    write_yaml(tmp_path / "parent.yaml", {
        "experiment": {"name": "parent", "seed": 1, "device": "cpu"},
        "graph": {"type": "none", "num_nodes": 128},
    })
    child = write_yaml(tmp_path / "child.yaml", {
        "inherit": "parent.yaml",
        "graph": {"num_nodes": 256},
    })
    config = load_config(child)
    assert config.graph.num_nodes == 256          # overridden by child
    assert config.graph.type == "none"             # kept from parent (same section)
    assert config.experiment.device == "cpu"       # kept from parent


def test_seed_override(tmp_path: Path) -> None:
    config = load_config(CONFIG_DIR / "base.yaml", seed=42)
    assert config.experiment.seed == 42


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"experiment": {"name": "x", "seed": 0, "sed": 1}})


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown config section"):
        config_from_dict({"experiment": {"name": "x", "seed": 0}, "trainig": {}})


def test_missing_experiment_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="experiment"):
        config_from_dict({"graph": {"type": "er"}})


def test_missing_required_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="seed"):
        config_from_dict({"experiment": {"name": "x"}})


@pytest.mark.parametrize(
    "experiment",
    [
        {"name": "x", "seed": -1},             # negative seed
        {"name": "x", "seed": "0"},            # string instead of int
        {"name": "x", "seed": True},           # bool is not accepted as int
        {"name": "x", "seed": 0, "device": "tpu"},
        {"name": "", "seed": 0},
    ],
)
def test_invalid_experiment_values_are_rejected(experiment: dict) -> None:
    with pytest.raises(ConfigError):
        config_from_dict({"experiment": experiment})


@pytest.mark.parametrize("graph", [{"type": "hexagon"}, {"num_nodes": 1}, {"num_nodes": 12.5}])
def test_invalid_graph_values_are_rejected(graph: dict) -> None:
    with pytest.raises(ConfigError):
        config_from_dict({"experiment": {"name": "x", "seed": 0}, "graph": graph})


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_circular_inheritance_is_rejected(tmp_path: Path) -> None:
    write_yaml(tmp_path / "a.yaml", {"inherit": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"inherit": "a.yaml"})
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_saved_config_round_trips(tmp_path: Path) -> None:
    original = load_config(CONFIG_DIR / "drosophila.yaml", seed=7)
    save_config(original, tmp_path / "resolved.yaml")
    assert load_config(tmp_path / "resolved.yaml") == original
