"""Tests for the control-graph generators (pure algorithms, index space)."""

import numpy as np
import pytest

from flyconnectome_mri.connectome.random_graphs import (
    barabasi_albert_directed,
    degree_preserving_rewire,
    edge_overlap,
    erdos_renyi_directed,
    watts_strogatz_directed,
)

N, M = 40, 300


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def assert_valid_simple_digraph(edges: np.ndarray, n: int, m: int) -> None:
    """Exactly m edges, all endpoints in range, no self-loops, no duplicates, sorted."""
    assert edges.shape == (m, 2)
    assert edges.min() >= 0 and edges.max() < n
    assert not np.any(edges[:, 0] == edges[:, 1])
    assert len({tuple(e) for e in edges.tolist()}) == m
    assert edges.tolist() == sorted(edges.tolist())


def degrees(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(edges[:, 0], minlength=n), np.bincount(edges[:, 1], minlength=n)


GENERATORS = {
    "er": lambda n, m, r: erdos_renyi_directed(n, m, r),
    "ws": lambda n, m, r: watts_strogatz_directed(n, m, 0.1, r),
    "ba": lambda n, m, r: barabasi_albert_directed(n, m, r),
}


# ---------------------------------------------------------------------------
# Shared properties of ER / WS / BA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize(("n", "m"), [(N, M), (10, 1), (10, 45), (10, 50)])
def test_exact_edge_count_and_simple_graph(name: str, n: int, m: int) -> None:
    assert_valid_simple_digraph(GENERATORS[name](n, m, rng()), n, m)


@pytest.mark.parametrize("name", ["er", "ws"])
@pytest.mark.parametrize("m", [89, 90])
def test_er_and_ws_reach_complete_graphs(name: str, m: int) -> None:
    assert_valid_simple_digraph(GENERATORS[name](10, m, rng()), 10, m)


def test_ba_refuses_graphs_denser_than_its_limit() -> None:
    """At most 2 * max_k k*(n-k) = n*n/2 directed edges (50 for 10 nodes)."""
    assert_valid_simple_digraph(barabasi_albert_directed(10, 50, rng()), 10, 50)
    with pytest.raises(ValueError, match="too dense"):
        barabasi_albert_directed(10, 51, rng())


@pytest.mark.parametrize("name", GENERATORS)
def test_same_seed_same_graph_different_seed_different_graph(name: str) -> None:
    a = GENERATORS[name](N, M, rng(1))
    b = GENERATORS[name](N, M, rng(1))
    c = GENERATORS[name](N, M, rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize(("n", "m"), [(1, 0), (5, 0), (5, 21)])
def test_impossible_sizes_are_rejected(name: str, n: int, m: int) -> None:
    with pytest.raises(ValueError):
        GENERATORS[name](n, m, rng())


# ---------------------------------------------------------------------------
# Generator-specific behaviour
# ---------------------------------------------------------------------------


def test_er_degrees_are_homogeneous_compared_with_ba() -> None:
    """ER has no hubs; BA's preferential attachment creates them."""
    er_out, er_in = degrees(erdos_renyi_directed(200, 2000, rng()), 200)
    ba_out, ba_in = degrees(barabasi_albert_directed(200, 2000, rng()), 200)
    assert (ba_out + ba_in).max() > 1.5 * (er_out + er_in).max()


def test_ws_without_rewiring_is_the_ring_lattice() -> None:
    # 10 nodes, 25 edges -> out-degree 3 for nodes 0..4 and 2 for nodes 5..9,
    # targets in order i+1, i-1, i+2.
    edges = watts_strogatz_directed(10, 25, 0.0, rng())
    targets = {i: sorted(edges[edges[:, 0] == i, 1].tolist()) for i in range(10)}
    assert targets[0] == [1, 2, 9]
    assert targets[4] == [3, 5, 6]
    assert targets[5] == [4, 6]
    assert targets[9] == [0, 8]


def test_ws_rewiring_keeps_out_degrees() -> None:
    lattice_out, _ = degrees(watts_strogatz_directed(N, M, 0.0, rng()), N)
    rewired_out, _ = degrees(watts_strogatz_directed(N, M, 0.5, rng()), N)
    assert np.array_equal(lattice_out, rewired_out)


def test_ws_rewiring_probability_controls_how_much_changes() -> None:
    lattice = watts_strogatz_directed(N, M, 0.0, rng())
    little = edge_overlap(watts_strogatz_directed(N, M, 0.05, rng()), lattice)
    lots = edge_overlap(watts_strogatz_directed(N, M, 0.9, rng()), lattice)
    assert little > 0.85
    assert lots < little


def test_ba_uses_reverse_edges_only_when_needed() -> None:
    sparse = barabasi_albert_directed(20, 50, rng())
    pairs = {tuple(e) for e in sparse.tolist()}
    assert not any((t, s) in pairs for s, t in pairs)  # one direction per edge suffices
    dense = barabasi_albert_directed(20, 150, rng())   # > 20*20/4 = 100 -> needs reverse edges
    assert_valid_simple_digraph(dense, 20, 150)
    pairs = {tuple(e) for e in dense.tolist()}
    assert any((t, s) in pairs for s, t in pairs)


# ---------------------------------------------------------------------------
# Degree-preserving rewiring
# ---------------------------------------------------------------------------


@pytest.fixture
def source_graph() -> tuple[np.ndarray, np.ndarray]:
    edges = barabasi_albert_directed(N, M, rng(7))  # heterogeneous degrees: a demanding test
    weights = np.arange(1, M + 1, dtype=float)
    return edges, weights


def test_rewiring_preserves_every_in_and_out_degree(source_graph) -> None:
    edges, weights = source_graph
    result = degree_preserving_rewire(edges, weights, 10, rng())
    assert_valid_simple_digraph(result.edges, N, M)
    before_out, before_in = degrees(edges, N)
    after_out, after_in = degrees(result.edges, N)
    assert np.array_equal(before_out, after_out)
    assert np.array_equal(before_in, after_in)


def test_rewiring_randomises_connections(source_graph) -> None:
    edges, weights = source_graph
    result = degree_preserving_rewire(edges, weights, 10, rng())
    assert result.accepted_swaps > 0
    assert edge_overlap(result.edges, edges) < 0.5


def test_rewiring_keeps_weights_with_source_neuron(source_graph) -> None:
    edges, weights = source_graph
    result = degree_preserving_rewire(edges, weights, 10, rng())
    assert sorted(result.weights.tolist()) == sorted(weights.tolist())
    out_strength = lambda e, w: np.bincount(e[:, 0], weights=w, minlength=N)
    assert np.allclose(out_strength(edges, weights), out_strength(result.edges, result.weights))


def test_rewiring_is_deterministic(source_graph) -> None:
    edges, weights = source_graph
    a = degree_preserving_rewire(edges, weights, 5, rng(3))
    b = degree_preserving_rewire(edges, weights, 5, rng(3))
    assert np.array_equal(a.edges, b.edges) and np.array_equal(a.weights, b.weights)
    assert a.accepted_swaps == b.accepted_swaps


def test_complete_graph_cannot_be_rewired() -> None:
    """Every possible edge exists, so every swap would create a duplicate."""
    full = erdos_renyi_directed(5, 20, rng())
    result = degree_preserving_rewire(full, np.ones(20), 10, rng())
    assert result.accepted_swaps == 0
    assert np.array_equal(result.edges, full)


def test_rewiring_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        degree_preserving_rewire(np.array([[0, 0], [0, 1]]), np.ones(2), 1, rng())
    with pytest.raises(ValueError, match="duplicate"):
        degree_preserving_rewire(np.array([[0, 1], [0, 1]]), np.ones(2), 1, rng())
    with pytest.raises(ValueError, match="at least 2"):
        degree_preserving_rewire(np.array([[0, 1]]), np.ones(1), 1, rng())


def test_edge_overlap() -> None:
    a = np.array([[0, 1], [1, 2], [2, 0]])
    b = np.array([[0, 1], [2, 1]])
    assert edge_overlap(a, b) == pytest.approx(1 / 3)
    assert edge_overlap(a, a) == 1.0
