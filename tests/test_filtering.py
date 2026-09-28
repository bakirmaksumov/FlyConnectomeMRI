"""Tests for weight-threshold graph variants."""

import json
from pathlib import Path

import pandas as pd
import pytest

from flyconnectome_mri.connectome.filtering import ChecksumMismatchError, derive_weight_variants
from flyconnectome_mri.utils.config import ConfigError, GraphVariantsConfig, load_graph_variants_config
from flyconnectome_mri.utils.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw graph: neuron 40 only has a weak (weight 2) connection, so w>=5 isolates it.
RAW_EDGES = "source,target,weight\n10,20,25\n20,10,4\n20,30,7\n30,10,12\n40,10,2\n"
RAW_NODES = "rank,bodyId,type\n1,10,A\n2,20,B\n3,30,C\n4,40,D\n"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A fake raw download: edges, nodes and metadata with correct checksums."""
    (tmp_path / "g.csv").write_text(RAW_EDGES, encoding="utf-8")
    (tmp_path / "g_nodes.csv").write_text(RAW_NODES, encoding="utf-8")
    meta = {
        "label": "GENERIC test graph",
        "is_visual_system_selection": False,
        "source": {"dataset": "test:v0", "database": {"uuid": "u1"}},
        "files": {
            "edges": {"name": "g.csv", "sha256": sha256_file(tmp_path / "g.csv")},
            "nodes": {"name": "g_nodes.csv", "sha256": sha256_file(tmp_path / "g_nodes.csv")},
        },
    }
    (tmp_path / "g_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def make_config(**overrides) -> GraphVariantsConfig:
    values = dict(
        source_edges="g.csv", source_nodes="g_nodes.csv", source_metadata="g_metadata.json",
        variants={"raw": 1, "w5": 5, "w20": 20}, primary_variant="w20",
        primary_declared_on="2026-09-28", output_dir="out",
    )
    values.update(overrides)
    return GraphVariantsConfig(**values)


def read_edges(path: Path) -> list[list[int]]:
    return pd.read_csv(path).values.tolist()


def test_thresholds_keep_exactly_the_heavy_edges(raw_dir: Path) -> None:
    derive_weight_variants(make_config(), raw_dir)
    out = raw_dir / "out"
    assert read_edges(out / "g_raw.csv") == [[10, 20, 25], [20, 10, 4], [20, 30, 7], [30, 10, 12], [40, 10, 2]]
    assert read_edges(out / "g_w5.csv") == [[10, 20, 25], [20, 30, 7], [30, 10, 12]]
    assert read_edges(out / "g_w20.csv") == [[10, 20, 25]]


def test_direction_and_weights_are_preserved(raw_dir: Path) -> None:
    results = derive_weight_variants(make_config(), raw_dir)
    graph, _ = results["w5"]
    assert graph.graph[30][10]["weight"] == 12  # 30 -> 10 kept with its weight
    assert not graph.graph.has_edge(10, 30)


def test_all_nodes_are_kept_even_when_isolated(raw_dir: Path) -> None:
    results = derive_weight_variants(make_config(), raw_dir)
    graph, meta = results["w5"]
    assert graph.node_ids.tolist() == [10, 20, 30, 40]
    assert meta["counts"]["isolated_nodes"] == [40]
    assert meta["statistics"]["num_isolated_nodes"] == 1

    nodes = pd.read_csv(raw_dir / "out" / "g_w5_nodes.csv")
    assert nodes["bodyId"].tolist() == [10, 20, 30, 40]
    assert nodes["type"].tolist() == ["A", "B", "C", "D"]  # annotations copied
    assert nodes["isolated"].tolist() == [False, False, False, True]
    assert nodes["out_degree"].tolist() == [1, 1, 1, 0]
    assert nodes["in_degree"].tolist() == [1, 1, 1, 0]

    graph20, meta20 = results["w20"]
    assert graph20.num_nodes == 4
    assert sorted(meta20["counts"]["isolated_nodes"]) == [30, 40]


def test_metadata_records_threshold_role_and_source_checksums(raw_dir: Path) -> None:
    derive_weight_variants(make_config(), raw_dir)
    meta = json.loads((raw_dir / "out" / "g_w20_metadata.json").read_text(encoding="utf-8"))
    assert meta["filter"]["min_weight"] == 20
    assert meta["role"] == "PRIMARY (pre-declared)"
    assert meta["primary_declared_on"] == "2026-09-28"
    assert meta["source"]["edges"]["sha256"] == sha256_file(raw_dir / "g.csv")
    assert meta["source"]["nodes"]["sha256"] == sha256_file(raw_dir / "g_nodes.csv")
    assert meta["source"]["dataset"] == "test:v0"
    assert meta["files"]["edges"]["sha256"] == sha256_file(raw_dir / "out" / "g_w20.csv")
    assert meta["counts"]["edges_removed"] == 4

    other = json.loads((raw_dir / "out" / "g_w5_metadata.json").read_text(encoding="utf-8"))
    assert other["role"] == "sensitivity analysis only"


def test_raw_files_are_not_modified(raw_dir: Path) -> None:
    before = {p.name: sha256_file(p) for p in raw_dir.glob("g*")}
    derive_weight_variants(make_config(output_dir="."), raw_dir)  # even when writing next to them
    after = {name: sha256_file(raw_dir / name) for name in before}
    assert before == after


def test_modified_raw_file_is_refused(raw_dir: Path) -> None:
    (raw_dir / "g.csv").write_text(RAW_EDGES + "30,40,99\n", encoding="utf-8")
    with pytest.raises(ChecksumMismatchError, match="changed since it was downloaded"):
        derive_weight_variants(make_config(), raw_dir)


def test_existing_variants_are_not_overwritten(raw_dir: Path) -> None:
    derive_weight_variants(make_config(), raw_dir)
    with pytest.raises(FileExistsError):
        derive_weight_variants(make_config(), raw_dir)
    derive_weight_variants(make_config(), raw_dir, overwrite=True)


def test_threshold_removing_every_edge_is_an_error(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="keeps no edges"):
        derive_weight_variants(make_config(variants={"w99": 99}, primary_variant="w99"), raw_dir)


@pytest.mark.parametrize(
    "overrides",
    [{"primary_variant": "w7"}, {"variants": {}}, {"variants": {"w0": 0}}, {"variants": {"W5": 5}},
     {"variants": {"w5": 5.5}}],
)
def test_invalid_variant_configs_are_rejected(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        make_config(**overrides)


def test_shipped_variants_config_declares_w20_primary() -> None:
    config = load_graph_variants_config(PROJECT_ROOT / "configs" / "graph_variants.yaml")
    assert config.variants == {"raw": 1, "w5": 5, "w10": 10, "w20": 20}
    assert config.primary_variant == "w20"
    assert config.primary_declared_on == "2026-09-28"
