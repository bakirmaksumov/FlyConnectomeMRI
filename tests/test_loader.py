"""Tests for the edge-list loader: parsing, conversion, and every validation rule."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from flyconnectome_mri.connectome.loader import EdgeListValidationError, load_edge_list

TINY = Path(__file__).parent / "data" / "tiny_graph.csv"
TINY_IDS = [1001, 1050, 3002, 5000, 5001, 8120]


def write_csv(tmp_path: Path, text: str, name: str = "edges.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Successful loading of the tiny fixture
# ---------------------------------------------------------------------------


def test_tiny_graph_node_mapping_is_sorted_ids() -> None:
    g = load_edge_list(TINY)
    assert g.node_ids.tolist() == TINY_IDS
    assert g.node_index == {node_id: i for i, node_id in enumerate(TINY_IDS)}
    assert g.num_nodes == 6
    assert g.num_edges == 6


def test_tiny_graph_edges_and_weights_are_preserved() -> None:
    g = load_edge_list(TINY)
    as_ids = {
        (int(g.node_ids[s]), int(g.node_ids[t])): float(w) for (s, t), w in zip(g.edges, g.weights)
    }
    assert as_ids == {
        (1001, 1050): 17.0, (1001, 3002): 5.0, (1050, 8120): 22.0,
        (8120, 1001): 3.0, (3002, 8120): 8.0, (5000, 5001): 2.0,
    }


def test_edge_direction_is_preserved() -> None:
    g = load_edge_list(TINY)
    a, b = g.node_index[1001], g.node_index[1050]
    assert g.adjacency[a, b] == 17  # 1001 -> 1050 exists
    assert g.adjacency[b, a] == 0   # 1050 -> 1001 does not
    assert g.graph.has_edge(1001, 1050)
    assert not g.graph.has_edge(1050, 1001)


def test_sparse_adjacency_matches_edges() -> None:
    g = load_edge_list(TINY)
    assert sp.issparse(g.adjacency)
    assert g.adjacency.shape == (6, 6)
    assert g.adjacency.nnz == 6
    for (s, t), w in zip(g.edges, g.weights):
        assert g.adjacency[s, t] == w
    assert g.adjacency.sum() == pytest.approx(57.0)


def test_networkx_graph_matches_edges() -> None:
    g = load_edge_list(TINY)
    assert isinstance(g.graph, nx.DiGraph)
    assert sorted(g.graph.nodes) == TINY_IDS
    assert g.graph.number_of_edges() == 6
    assert g.graph[1050][8120]["weight"] == 22.0


def test_edges_are_sorted_by_source_then_target() -> None:
    g = load_edge_list(TINY)
    keys = [tuple(e) for e in g.edges.tolist()]
    assert keys == sorted(keys)


def test_clean_file_report() -> None:
    report = load_edge_list(TINY).report
    assert report.rows_read == 6
    assert report.self_loops_found == 0
    assert report.duplicates_found == 0
    assert report.ignored_columns == ()


def test_mapping_is_independent_of_row_order(tmp_path: Path) -> None:
    lines = TINY.read_text(encoding="utf-8").strip().splitlines()
    header, rows = lines[0], lines[1:]
    shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    g1 = load_edge_list(TINY)
    g2 = load_edge_list(write_csv(tmp_path, "\n".join([header, *shuffled]) + "\n"))
    assert np.array_equal(g1.node_ids, g2.node_ids)
    assert np.array_equal(g1.edges, g2.edges)
    assert np.array_equal(g1.weights, g2.weights)
    assert (g1.adjacency != g2.adjacency).nnz == 0


def test_large_ids_keep_full_precision(tmp_path: Path) -> None:
    """IDs above 2**53 would be silently rounded if parsed through float64."""
    big = 720575940612345679  # 18-digit ID, the size used by FlyWire
    g = load_edge_list(write_csv(tmp_path, f"source,target,weight\n{big},{big + 1},1\n"))
    assert g.node_ids.tolist() == [big, big + 1]


def test_whitespace_around_values_is_tolerated(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, "source, target, weight\n 1 , 2 , 3 \n"))
    assert g.node_ids.tolist() == [1, 2]
    assert g.weights.tolist() == [3.0]


def test_float_weights_are_accepted(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, "source,target,weight\n1,2,0.25\n"))
    assert g.weights.tolist() == [0.25]


# ---------------------------------------------------------------------------
# File / column problems
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_edge_list(tmp_path / "nope.csv")


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match="empty"):
        load_edge_list(write_csv(tmp_path, ""))


def test_header_only(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match="no edges"):
        load_edge_list(write_csv(tmp_path, "source,target,weight\n"))


def test_missing_column(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match=r"missing column\(s\) \['weight'\]"):
        load_edge_list(write_csv(tmp_path, "source,target\n1,2\n"))


def test_extra_columns_are_ignored_and_reported(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, "source,target,weight,roi\n1,2,3,ME(R)\n"))
    assert g.report.ignored_columns == ("roi",)
    assert g.num_edges == 1


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "column"),
    [(",2,3", "source"), ("1,,3", "target"), ("1,2,", "weight"), ("1, ,3", "target")],
)
def test_missing_values_are_rejected_with_line_number(tmp_path: Path, row: str, column: str) -> None:
    path = write_csv(tmp_path, f"source,target,weight\n5,6,1\n{row}\n")
    with pytest.raises(EdgeListValidationError, match=rf"Missing value in column '{column}' on line\(s\) 3"):
        load_edge_list(path)


# ---------------------------------------------------------------------------
# Invalid node IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["abc", "1.5", "-3", "1e5", "NaN"])
def test_invalid_node_ids_are_rejected(tmp_path: Path, bad_id: str) -> None:
    path = write_csv(tmp_path, f"source,target,weight\n{bad_id},2,1\n")
    with pytest.raises(EdgeListValidationError, match="non-negative integer node IDs"):
        load_edge_list(path)


def test_node_id_overflow_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, f"source,target,weight\n{2**64},2,1\n")
    with pytest.raises(EdgeListValidationError, match="int64"):
        load_edge_list(path)


# ---------------------------------------------------------------------------
# Invalid weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", ["0", "0.0", "-5", "-0.1"])
def test_zero_or_negative_weights_are_rejected(tmp_path: Path, weight: str) -> None:
    path = write_csv(tmp_path, f"source,target,weight\n1,2,4\n2,3,{weight}\n")
    with pytest.raises(EdgeListValidationError, match=r"must be > 0.*line\(s\) 3"):
        load_edge_list(path)


def test_non_numeric_weight_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match="non-numeric"):
        load_edge_list(write_csv(tmp_path, "source,target,weight\n1,2,many\n"))


def test_infinite_weight_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match="infinite"):
        load_edge_list(write_csv(tmp_path, "source,target,weight\n1,2,inf\n"))


# ---------------------------------------------------------------------------
# Duplicate edges
# ---------------------------------------------------------------------------

DUPLICATES = "source,target,weight\n1,2,3\n2,3,1\n1,2,4\n"


def test_duplicate_edges_are_rejected_by_default(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match=r"1 duplicate edge\(s\).*line\(s\) 4"):
        load_edge_list(write_csv(tmp_path, DUPLICATES))


def test_duplicate_edges_can_be_summed(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, DUPLICATES), duplicates="sum")
    assert g.num_edges == 2
    assert g.graph[1][2]["weight"] == 7.0  # 3 + 4
    assert g.report.duplicates_found == 1
    assert g.report.duplicates_merged == 1


def test_reverse_direction_is_not_a_duplicate(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, "source,target,weight\n1,2,3\n2,1,5\n"))
    assert g.num_edges == 2
    assert g.graph[1][2]["weight"] == 3.0
    assert g.graph[2][1]["weight"] == 5.0


# ---------------------------------------------------------------------------
# Self-loops
# ---------------------------------------------------------------------------

SELF_LOOP = "source,target,weight\n1,2,3\n7,7,4\n2,1,1\n"


def test_self_loops_are_rejected_by_default(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match=r"1 self-loop\(s\).*line\(s\) 3"):
        load_edge_list(write_csv(tmp_path, SELF_LOOP))


def test_self_loops_can_be_dropped_and_are_reported(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, SELF_LOOP), self_loops="drop")
    assert g.num_edges == 2
    assert not any(s == t for s, t in g.edges)
    # Dropping removes the connection, not the neuron: node 7 still exists.
    assert g.node_ids.tolist() == [1, 2, 7]
    assert g.report.self_loops_found == 1
    assert g.report.self_loops_removed == 1


def test_self_loops_can_be_kept(tmp_path: Path) -> None:
    g = load_edge_list(write_csv(tmp_path, SELF_LOOP), self_loops="keep")
    assert g.num_edges == 3
    i = g.node_index[7]
    assert g.adjacency[i, i] == 4  # self-loops live on the diagonal
    assert g.graph.has_edge(7, 7)
    assert g.report.self_loops_found == 1
    assert g.report.self_loops_removed == 0


def test_duplicate_self_loops_are_handled_after_dropping(tmp_path: Path) -> None:
    """Dropped self-loops must not be reported as duplicates."""
    g = load_edge_list(write_csv(tmp_path, "source,target,weight\n7,7,1\n7,7,2\n1,2,1\n"), self_loops="drop")
    assert g.report.duplicates_found == 0
    assert g.num_edges == 1


def test_only_self_loops_dropped_leaves_no_edges(tmp_path: Path) -> None:
    with pytest.raises(EdgeListValidationError, match="No edges left"):
        load_edge_list(write_csv(tmp_path, "source,target,weight\n7,7,1\n"), self_loops="drop")


def test_invalid_policy_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        load_edge_list(TINY, self_loops="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        load_edge_list(TINY, duplicates="max")  # type: ignore[arg-type]
