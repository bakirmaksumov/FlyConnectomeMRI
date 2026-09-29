"""Tests for neuPrint subgraph extraction, using mocked neuPrint responses (no network)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import neuprint
import pandas as pd
import pytest

from flyconnectome_mri.connectome import drosophila
from flyconnectome_mri.connectome.drosophila import (
    NeuprintTokenError,
    connect,
    fetch_edges_among,
    fetch_malecns_subgraph,
    import_local_edge_list,
    read_token,
    select_roi_top_neurons,
)
from flyconnectome_mri.utils.config import ConfigError, DrosophilaFetchConfig, load_fetch_config
from flyconnectome_mri.utils.provenance import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = PROJECT_ROOT / "tests" / "data" / "tiny_graph.csv"
SECRET = "SECRET-TEST-TOKEN-do-not-leak-12345"

# --- A tiny fake neuPrint database ------------------------------------------------
# Synapses inside ROI "EB" (pre + post):
#   30: 300   10: 150   20: 150 (ties with 10 -> smaller bodyId first)   50: 90   60: 80   40: 5
# Neuron 30 also has a huge count in "FB", which must be ignored.
ROI_COUNTS = pd.DataFrame(
    [
        (10, "EB", 100, 50), (20, "EB", 70, 80), (30, "EB", 200, 100), (30, "FB", 9000, 9000),
        (40, "EB", 2, 3), (50, "EB", 45, 45), (60, "EB", 40, 40),
    ],
    columns=["bodyId", "roi", "pre", "post"],
)
NEURONS = pd.DataFrame({
    "bodyId": [10, 20, 30, 40, 50, 60],
    "type": ["A", "B", "C", "D", "E", None],
    "instance": ["A_R", "B_R", "C_L", "D_R", "E_R", None],
    "status": ["Traced"] * 6,
})
# All directed connections in the fake database (bodyId_pre, bodyId_post, weight).
DB_EDGES = [(30, 10, 12), (10, 20, 3), (20, 30, 7), (20, 20, 2), (10, 40, 9), (50, 30, 4)]


class FakeClient:
    all_rois = ["EB", "FB", "CX"]
    primary_rois = ["EB", "FB"]
    meta = {"lastDatabaseEdit": "2026-01-01", "uuid": "abc123", "unrelated": "x"}


@pytest.fixture
def fake_neuprint(monkeypatch):
    """Replace network-calling neuprint functions with fakes; returns recorded calls."""
    calls: dict[str, list] = {"fetch_neurons": [], "fetch_adjacencies": []}

    def fake_fetch_neurons(criteria, client=None, **kwargs):
        calls["fetch_neurons"].append(criteria)
        return NEURONS.copy(), ROI_COUNTS.copy()

    def fake_fetch_adjacencies(sources=None, targets=None, min_total_weight=1, **kwargs):
        calls["fetch_adjacencies"].append({"sources": sources, "targets": targets,
                                           "min_total_weight": min_total_weight, **kwargs})
        rows = [e for e in DB_EDGES if e[0] in sources and e[1] in targets and e[2] >= min_total_weight]
        conn = pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "weight"])
        return NEURONS.copy(), conn

    monkeypatch.setattr(neuprint, "fetch_neurons", fake_fetch_neurons)
    monkeypatch.setattr(neuprint, "fetch_adjacencies", fake_fetch_adjacencies)
    return calls


def make_config(**overrides) -> DrosophilaFetchConfig:
    values = dict(server="https://neuprint.example.org", dataset="male-cns:v1.0", roi="EB", num_nodes=3)
    values.update(overrides)
    return DrosophilaFetchConfig(**values)


# ---------------------------------------------------------------------------
# Neuron selection
# ---------------------------------------------------------------------------


def test_selection_ranks_by_roi_synapses_with_bodyid_tiebreak(fake_neuprint) -> None:
    nodes = select_roi_top_neurons(FakeClient(), "EB", 3)
    assert nodes["bodyId"].tolist() == [30, 10, 20]
    assert nodes["rank"].tolist() == [1, 2, 3]
    assert nodes["roi_synapses"].tolist() == [300, 150, 150]  # FB counts ignored
    assert nodes["roi_pre"].tolist() == [200, 100, 70]
    assert nodes["type"].tolist() == ["C", "A", "B"]
    assert fake_neuprint["fetch_neurons"][0].rois == {"EB"}


def test_unknown_roi_is_rejected_with_available_list(fake_neuprint) -> None:
    with pytest.raises(ValueError, match=r"'XYZ' does not exist.*\['EB', 'FB'\]"):
        select_roi_top_neurons(FakeClient(), "XYZ", 3)


def test_too_few_neurons_is_rejected(fake_neuprint) -> None:
    with pytest.raises(ValueError, match="only 6 neurons; 7 were requested"):
        select_roi_top_neurons(FakeClient(), "EB", 7)


# ---------------------------------------------------------------------------
# Edge fetching
# ---------------------------------------------------------------------------


def test_edges_are_canonical_directed_and_sorted(fake_neuprint) -> None:
    edges = fetch_edges_among(FakeClient(), [30, 10, 20], min_total_weight=1)
    assert list(edges.columns) == ["source", "target", "weight"]
    assert edges.values.tolist() == [[10, 20, 3], [20, 20, 2], [20, 30, 7], [30, 10, 12]]
    call = fake_neuprint["fetch_adjacencies"][0]
    assert call["sources"] == call["targets"] == [30, 10, 20]
    assert call["omit_rois"] is True  # one row per neuron pair = total weight


def test_min_total_weight_is_passed_to_neuprint(fake_neuprint) -> None:
    edges = fetch_edges_among(FakeClient(), [30, 10, 20], min_total_weight=5)
    assert edges.values.tolist() == [[20, 30, 7], [30, 10, 12]]


def test_edges_outside_selection_are_rejected(monkeypatch) -> None:
    bad = pd.DataFrame([(10, 999, 4)], columns=["bodyId_pre", "bodyId_post", "weight"])
    monkeypatch.setattr(neuprint, "fetch_adjacencies", lambda **kw: (None, bad))
    with pytest.raises(RuntimeError, match="outside the selected neurons"):
        fetch_edges_among(FakeClient(), [10, 20], min_total_weight=1)


# ---------------------------------------------------------------------------
# End-to-end fetch: files, validation, metadata
# ---------------------------------------------------------------------------


def test_fetch_writes_canonical_csv_and_validates(fake_neuprint, tmp_path: Path) -> None:
    graph, meta = fetch_malecns_subgraph(make_config(), FakeClient(), tmp_path)

    csv_text = (tmp_path / "drosophila_3.csv").read_text(encoding="utf-8")
    assert csv_text == "source,target,weight\n10,20,3\n20,20,2\n20,30,7\n30,10,12\n"
    # The loaded graph applies self_loops="drop": raw CSV keeps the loop, graph does not.
    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert graph.report.self_loops_removed == 1
    assert graph.graph[30][10]["weight"] == 12  # direction and weight preserved
    assert not graph.graph.has_edge(10, 30)

    nodes = pd.read_csv(tmp_path / "drosophila_3_nodes.csv")
    assert nodes["bodyId"].tolist() == [30, 10, 20]


def test_metadata_describes_selection_and_is_labelled_generic(fake_neuprint, tmp_path: Path) -> None:
    fetch_malecns_subgraph(make_config(), FakeClient(), tmp_path)
    meta = json.loads((tmp_path / "drosophila_3_metadata.json").read_text(encoding="utf-8"))

    assert "GENERIC" in meta["label"]
    assert meta["is_visual_system_selection"] is False
    assert meta["source"]["dataset"] == "male-cns:v1.0"
    assert meta["source"]["database"] == {"lastDatabaseEdit": "2026-01-01", "uuid": "abc123"}
    assert meta["selection"]["method"] == "roi_top_synapses"
    assert meta["selection"]["roi"] == "EB"
    assert meta["selection"]["num_nodes_requested"] == 3
    assert meta["counts"]["edges_in_csv"] == 4
    assert meta["counts"]["self_loops_in_csv"] == 1
    assert meta["extraction_date_utc"]
    assert meta["files"]["edges"]["sha256"] == sha256_file(tmp_path / "drosophila_3.csv")
    assert meta["files"]["nodes"]["sha256"] == sha256_file(tmp_path / "drosophila_3_nodes.csv")


def test_isolated_selected_neurons_are_reported(fake_neuprint, tmp_path: Path) -> None:
    """Neuron 60 is selected but has no connection to the others: reported, and kept as an isolated node."""
    graph, meta = fetch_malecns_subgraph(make_config(num_nodes=5), FakeClient(), tmp_path)
    assert meta["counts"]["neurons_selected"] == 5
    assert meta["counts"]["isolated_selected_neurons"] == [60]
    assert meta["counts"]["neurons_in_edge_list"] == 4
    assert graph.num_nodes == 5                     # the isolated neuron is kept as a node
    assert 60 in graph.node_ids.tolist()


def test_existing_files_are_not_overwritten(fake_neuprint, tmp_path: Path) -> None:
    fetch_malecns_subgraph(make_config(), FakeClient(), tmp_path)
    with pytest.raises(FileExistsError):
        fetch_malecns_subgraph(make_config(), FakeClient(), tmp_path)
    fetch_malecns_subgraph(make_config(), FakeClient(), tmp_path, overwrite=True)


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def test_missing_token_gives_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    with pytest.raises(NeuprintTokenError, match="NEUPRINT_TOKEN is not set"):
        read_token("NEUPRINT_TOKEN")


def test_token_is_read_from_environment_and_passed_to_client(monkeypatch) -> None:
    monkeypatch.setenv("NEUPRINT_TOKEN", SECRET)
    captured = {}

    def fake_client(server, dataset=None, token=None, progress=True):
        captured.update(server=server, dataset=dataset, token=token)
        return FakeClient()

    monkeypatch.setattr(neuprint, "Client", fake_client)
    connect(make_config())
    assert captured == {"server": "https://neuprint.example.org", "dataset": "male-cns:v1.0", "token": SECRET}


def test_connection_error_does_not_leak_token(monkeypatch) -> None:
    monkeypatch.setenv("NEUPRINT_TOKEN", SECRET)

    def failing_client(*args, token=None, **kwargs):
        raise ConnectionError(f"401 Unauthorized for header Bearer {token}")

    monkeypatch.setattr(neuprint, "Client", failing_client)
    with pytest.raises(RuntimeError) as info:
        connect(make_config())
    assert SECRET not in str(info.value)
    assert "401 Unauthorized" in str(info.value)


def test_token_never_written_to_files_or_output(fake_neuprint, monkeypatch, tmp_path: Path, capsys) -> None:
    """Full script run: the token must not appear in any output file, stdout or stderr."""
    monkeypatch.setenv("NEUPRINT_TOKEN", SECRET)
    monkeypatch.setattr(neuprint, "Client", lambda *a, **k: FakeClient())
    config = tmp_path / "fetch.yaml"
    config.write_text(
        "server: https://neuprint.example.org\ndataset: male-cns:v1.0\nroi: EB\nnum_nodes: 3\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import fetch_connectome
    finally:
        sys.path.pop(0)
    fetch_connectome.main(["--config", str(config), "--output-dir", str(tmp_path / "out")])

    out = capsys.readouterr()
    assert "Graph statistics" in out.out
    assert SECRET not in out.out + out.err
    files = list((tmp_path / "out").iterdir())
    assert len(files) == 3
    for f in files:
        assert SECRET not in f.read_text(encoding="utf-8")


def test_script_without_token_fails_cleanly(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "NEUPRINT_TOKEN"}
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_connectome.py"),
         "--config", str(PROJECT_ROOT / "configs" / "malecns_fetch.yaml"), "--output-dir", str(tmp_path)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0
    assert "NEUPRINT_TOKEN is not set" in result.stderr
    assert not any(tmp_path.iterdir())  # nothing written


# ---------------------------------------------------------------------------
# Local CSV mode
# ---------------------------------------------------------------------------


def test_import_local_edge_list(tmp_path: Path) -> None:
    graph, meta = import_local_edge_list(TINY, tmp_path, self_loops="error")
    out = tmp_path / "drosophila_6.csv"
    assert out.is_file()
    assert not (tmp_path / "drosophila_6_nodes.csv").exists()
    assert meta["source"]["mode"] == "local_csv"
    assert meta["source"]["sha256"] == sha256_file(TINY)
    assert "selection method unknown" in meta["label"]
    # Same edges as the original, in canonical sorted order.
    original = pd.read_csv(TINY).sort_values(["source", "target"]).reset_index(drop=True)
    assert pd.read_csv(out).equals(original)


# ---------------------------------------------------------------------------
# Fetch config
# ---------------------------------------------------------------------------


def test_shipped_fetch_config_loads() -> None:
    config = load_fetch_config(PROJECT_ROOT / "configs" / "malecns_fetch.yaml")
    assert config.dataset == "male-cns:v1.0"
    assert config.num_nodes == 128
    assert config.token_env_var == "NEUPRINT_TOKEN"


def test_fetch_config_num_nodes_override() -> None:
    config = load_fetch_config(PROJECT_ROOT / "configs" / "malecns_fetch.yaml", num_nodes=256)
    assert config.num_nodes == 256


def test_fetch_config_rejects_token_key(tmp_path: Path) -> None:
    """A token pasted into a config file is an unknown key and is refused."""
    path = tmp_path / "bad.yaml"
    path.write_text("server: https://x.org\ndataset: d\nroi: EB\nnum_nodes: 8\ntoken: abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_fetch_config(path)


@pytest.mark.parametrize(
    "overrides",
    [{"server": "http://insecure.org"}, {"num_nodes": 1}, {"min_total_weight": 0},
     {"self_loops": "ignore"}, {"selection_method": "random"}, {"roi": ""}],
)
def test_fetch_config_rejects_invalid_values(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        make_config(**overrides)
