"""Tests for the visual-system (local retinotopic column) neuron selection, with a mocked neuPrint."""

import json
from pathlib import Path

import neuprint
import pandas as pd
import pytest

from flyconnectome_mri.connectome.drosophila import (
    HEX_NEIGHBOURS,
    fetch_malecns_subgraph,
    hex_distance,
    select_visual_local_columns,
)
from flyconnectome_mri.utils.config import ConfigError, DrosophilaFetchConfig, load_fetch_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPES = ["A", "B"]


def test_hex_distance_matches_the_verified_neighbourhood() -> None:
    assert hex_distance(0, 0) == 0
    assert all(hex_distance(*offset) == 1 for offset in HEX_NEIGHBOURS)
    assert hex_distance(1, -1) == 2 and hex_distance(-1, 1) == 2     # not neighbours on this grid
    assert hex_distance(2, 1) == 2 and hex_distance(2, 2) == 2
    assert hex_distance(3, -2) == 5
    # exactly 6 columns at distance 1 and 12 at distance 2
    ring = lambda d: sum(hex_distance(a, b) == d for a in range(-4, 5) for b in range(-4, 5))
    assert (ring(1), ring(2)) == (6, 12)


class FakeClient:
    """5 x 5 grid of columns (hex 1..5); types A and B in every column except (5, 5) lacks B."""

    all_rois = ["ME(R)", "LO(R)"]
    primary_rois = ["ME(R)", "LO(R)"]
    meta: dict = {}

    def __init__(self) -> None:
        rows, body = [], 1000
        for h1 in range(1, 6):
            for h2 in range(1, 6):
                for t in TYPES:
                    if (h1, h2) == (5, 5) and t == "B":
                        continue
                    body += 1
                    rows.append({"bodyId": body, "type": t, "instance": f"{t}_R", "superclass": "ol_intrinsic",
                                 "hex1": h1, "hex2": h2, "consensusNt": "ACh"})
        self.table = pd.DataFrame(rows)
        self.queries: list[str] = []

    def fetch_custom(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        return self.table.copy()


def test_selection_is_a_local_patch_around_the_central_column() -> None:
    client = FakeClient()
    nodes, details = select_visual_local_columns(client, "ME(R)", TYPES, 14)
    assert details["central_column"]["hex1"] == 3 and details["central_column"]["hex2"] == 3
    # centre column (2 neurons) + its 6 neighbours (12 neurons) = 14
    columns = set(zip(nodes["hex1"], nodes["hex2"]))
    assert columns == {(3, 3), (3, 4), (3, 2), (4, 3), (2, 3), (4, 4), (2, 2)}
    assert nodes["hex_distance"].max() == 1
    assert nodes["rank"].tolist() == list(range(1, 15))
    assert nodes.iloc[0][["hex1", "hex2", "type"]].tolist() == [3, 3, "A"]   # centre first, type order A, B
    assert nodes.iloc[1]["type"] == "B"
    assert details["type_counts"] == {"A": 7, "B": 7}
    assert "assignedOlHex1 IS NOT NULL" in client.queries[0]                    # no invented columns


def test_partial_outer_ring_is_taken_in_documented_order() -> None:
    nodes, details = select_visual_local_columns(FakeClient(), "ME(R)", TYPES, 16)
    outer = nodes[nodes["hex_distance"] == 2]
    assert len(outer) == 2 and details["outermost_ring_neurons_selected"] == 2
    # 12 columns at distance 2 x 2 types = 24, minus the missing B in column (5, 5)
    assert details["outermost_ring_neurons_available"] == 23
    # the smallest (hex1, hex2) at distance 2 from (3,3) is (1,1): both of its neurons
    assert outer[["hex1", "hex2"]].drop_duplicates().values.tolist() == [[1, 1]]


def test_selection_is_deterministic_regardless_of_row_order() -> None:
    a = FakeClient()
    b = FakeClient()
    b.table = b.table.sample(frac=1, random_state=3).reset_index(drop=True)
    assert select_visual_local_columns(a, "ME(R)", TYPES, 20)[0]["bodyId"].tolist() == \
        select_visual_local_columns(b, "ME(R)", TYPES, 20)[0]["bodyId"].tolist()


def test_invalid_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        select_visual_local_columns(FakeClient(), "XYZ", TYPES, 10)
    with pytest.raises(ValueError, match="Only 49 candidate"):
        select_visual_local_columns(FakeClient(), "ME(R)", TYPES, 50)
    with pytest.raises(ValueError, match="No column contains all"):
        select_visual_local_columns(FakeClient(), "ME(R)", ["A", "B", "C"], 10)


def test_visual_fetch_writes_labelled_files(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()

    def fake_adjacencies(sources=None, targets=None, **kwargs):
        ids = sorted(sources)
        rows = [(ids[i], ids[i + 1], 5) for i in range(len(ids) - 1)] + [(ids[1], ids[0], 2)]
        return None, pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "weight"])

    monkeypatch.setattr(neuprint, "fetch_adjacencies", fake_adjacencies)
    config = DrosophilaFetchConfig(server="https://x.org", dataset="male-cns:v1.0", roi="ME(R)", num_nodes=14,
                                   selection_method="visual_local_columns", cell_types=TYPES,
                                   output_prefix="drosophila_visual")
    graph, meta = fetch_malecns_subgraph(config, client, tmp_path)
    assert (tmp_path / "drosophila_visual_14.csv").is_file()
    nodes = pd.read_csv(tmp_path / "drosophila_visual_14_nodes.csv")
    assert {"rank", "bodyId", "type", "hex1", "hex2", "hex_distance"} <= set(nodes.columns)
    assert meta["is_visual_system_selection"] is True
    assert meta["label"].startswith("VISUAL-SYSTEM")
    assert meta["selection"]["method"] == "visual_local_columns"
    assert meta["selection"]["central_column"] == {"hex1": 3, "hex2": 3, "rule": meta["selection"]["central_column"]["rule"]}
    assert graph.num_nodes == 14
    saved = json.loads((tmp_path / "drosophila_visual_14_metadata.json").read_text(encoding="utf-8"))
    assert saved["selection"]["cell_types"] == TYPES


def test_visual_config_validation() -> None:
    base = dict(server="https://x.org", dataset="d", roi="ME(R)", num_nodes=10)
    with pytest.raises(ConfigError, match="cell_types"):
        DrosophilaFetchConfig(**base, selection_method="visual_local_columns")
    with pytest.raises(ConfigError, match="only used by"):
        DrosophilaFetchConfig(**base, cell_types=["L1"])
    with pytest.raises(ConfigError, match="output_prefix"):
        DrosophilaFetchConfig(**base, output_prefix="bad name")


def test_shipped_visual_config() -> None:
    config = load_fetch_config(PROJECT_ROOT / "configs" / "malecns_visual.yaml")
    assert config.selection_method == "visual_local_columns"
    assert config.roi == "ME(R)"
    assert config.cell_types == ["L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "C3", "Tm1", "Tm2", "Tm4", "Tm9"]
    assert config.num_nodes == 128 and config.output_prefix == "drosophila_visual"
    # the EB config is untouched and still generic
    eb = load_fetch_config(PROJECT_ROOT / "configs" / "malecns_fetch.yaml")
    assert eb.selection_method == "roi_top_synapses" and eb.output_prefix == "drosophila"
