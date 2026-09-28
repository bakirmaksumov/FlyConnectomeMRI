"""Tests for graph statistics, using hand-computed values (see tests/data/README.md)."""

import subprocess
import sys
from pathlib import Path

import pytest

from flyconnectome_mri.connectome.loader import load_edge_list
from flyconnectome_mri.connectome.statistics import compute_graph_statistics, format_statistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY = PROJECT_ROOT / "tests" / "data" / "tiny_graph.csv"


def test_tiny_graph_statistics_match_hand_calculation() -> None:
    stats = compute_graph_statistics(load_edge_list(TINY))
    assert stats.num_nodes == 6
    assert stats.num_edges == 6
    assert stats.num_self_loops == 0
    assert stats.density == pytest.approx(6 / (6 * 5))
    assert stats.mean_in_degree == pytest.approx(1.0)
    assert stats.mean_out_degree == pytest.approx(1.0)
    assert stats.max_in_degree == 2      # 8120
    assert stats.max_out_degree == 2     # 1001
    assert stats.max_degree == 3         # 1001 and 8120
    assert stats.num_isolated_nodes == 0
    # out-degrees [2,1,1,1,0,1] and in-degrees [1,1,1,0,1,2]: mean 1, variance 2/6
    assert stats.out_degree_std == pytest.approx((2 / 6) ** 0.5)
    assert stats.in_degree_std == pytest.approx((2 / 6) ** 0.5)
    assert stats.reciprocity == 0.0      # no edge has its reverse
    assert stats.num_weakly_connected_components == 2
    assert stats.largest_weakly_connected_component == 4
    assert stats.num_strongly_connected_components == 3
    assert stats.largest_strongly_connected_component == 4
    assert stats.total_weight == pytest.approx(57.0)
    assert stats.mean_weight == pytest.approx(9.5)
    assert stats.min_weight == 2.0
    assert stats.max_weight == 22.0


def test_mean_in_and_out_degree_are_always_equal(tmp_path: Path) -> None:
    """Each edge has one sender and one receiver, so both means equal E / N."""
    path = tmp_path / "star.csv"
    path.write_text("source,target,weight\n1,2,1\n1,3,1\n1,4,1\n4,1,1\n", encoding="utf-8")
    stats = compute_graph_statistics(load_edge_list(path))
    assert stats.mean_in_degree == stats.mean_out_degree == pytest.approx(4 / 4)
    assert stats.max_out_degree == 3
    assert stats.max_in_degree == 1


def test_self_loop_statistics(tmp_path: Path) -> None:
    path = tmp_path / "loop.csv"
    path.write_text("source,target,weight\n1,1,5\n1,2,1\n", encoding="utf-8")
    stats = compute_graph_statistics(load_edge_list(path, self_loops="keep"))
    assert stats.num_self_loops == 1
    assert stats.max_out_degree == 2  # the self-loop counts as an outgoing edge
    assert stats.max_in_degree == 1
    assert stats.num_strongly_connected_components == 2  # {1} and {2}


def test_format_statistics_mentions_every_statistic() -> None:
    text = format_statistics(compute_graph_statistics(load_edge_list(TINY)))
    for label in ("Nodes", "Edges", "Density", "Mean in-degree", "Mean out-degree", "Max degree",
                  "Weakly connected components", "Strongly connected components"):
        assert label in text


def test_inspect_script_runs_on_tiny_graph(tmp_path: Path) -> None:
    out = tmp_path / "stats.json"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "inspect_connectome.py"),
         "--edges", str(TINY), "--json", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert "Nodes" in result.stdout
    assert out.is_file()



def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "g.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reciprocity_hand_example(tmp_path: Path) -> None:
    # 1<->2 is reciprocated (2 of 4 edges); 2->3 and 3->4 are not.
    path = write(tmp_path, "source,target,weight\n1,2,1\n2,1,1\n2,3,1\n3,4,1\n")
    assert compute_graph_statistics(load_edge_list(path)).reciprocity == pytest.approx(0.5)


def test_clustering_complete_graph_is_one_and_chain_is_zero(tmp_path: Path) -> None:
    rows = "".join(f"{a},{b},1\n" for a in (1, 2, 3) for b in (1, 2, 3) if a != b)
    triangle = write(tmp_path, "source,target,weight\n" + rows)
    assert compute_graph_statistics(load_edge_list(triangle)).average_clustering == pytest.approx(1.0)
    chain = write(tmp_path, "source,target,weight\n1,2,1\n2,3,1\n3,4,1\n")
    assert compute_graph_statistics(load_edge_list(chain)).average_clustering == 0.0


def test_isolated_nodes_counted_in_statistics(tmp_path: Path) -> None:
    g = load_edge_list(write(tmp_path, "source,target,weight\n1,2,1\n"), node_ids=[1, 2, 3])
    stats = compute_graph_statistics(g)
    assert stats.num_nodes == 3
    assert stats.num_isolated_nodes == 1
    assert stats.density == pytest.approx(1 / 6)
    assert stats.num_weakly_connected_components == 2


def test_format_comparison_has_one_column_per_graph() -> None:
    from flyconnectome_mri.connectome.statistics import format_comparison

    g = load_edge_list(TINY)
    header = format_comparison({"first": g, "second": g}).splitlines()[0]
    assert "first" in header and "second" in header
