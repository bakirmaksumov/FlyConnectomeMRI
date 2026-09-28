"""Random control graphs ("null models") matched to a biological graph.

Why controls: if a fly-wired network does well on MRI, we must ask "better than
what?". A null model keeps some properties of the biological graph and
randomises the rest. If the biological graph beats a control, the advantage
must come from something the control destroyed.

All generators here work in *index space* (nodes 0..n-1), return a sorted
array of unique directed edges with shape (E, 2), never create self-loops, and
are fully deterministic given a ``numpy.random.Generator``.

| Control  | Keeps                                 | Randomises                        |
|----------|---------------------------------------|-----------------------------------|
| ER       | node count, exact edge count          | everything else (no hubs)         |
| WS       | node count, exact edge count          | starts from a ring lattice (high  |
|          |                                       | clustering), rewires a fraction p |
| BA       | node count, exact edge count          | grows hubs by preferential        |
|          |                                       | attachment (heavy-tailed degrees) |
| rewired  | node count, exact edge count, EVERY   | which specific neurons connect    |
|          | node's exact in- AND out-degree       | (biological identity)             |

ER, WS and BA are structurally *different* null models, not "random versions of
the fly graph": they do not preserve the fly graph's degree distribution. WS and
BA are defined for undirected graphs; the directed variants used here are
documented in each function.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class RewireResult:
    """Output of :func:`degree_preserving_rewire`.

    Attributes:
        edges: Rewired edges, shape (E, 2), sorted.
        weights: Weight of each rewired edge (travels with the edge's source).
        attempted_swaps / accepted_swaps: Swap bookkeeping.
    """

    edges: np.ndarray
    weights: np.ndarray
    attempted_swaps: int
    accepted_swaps: int


def _check_size(n: int, m: int) -> None:
    if n < 2:
        raise ValueError(f"need at least 2 nodes, got {n}")
    max_edges = n * (n - 1)
    if not 0 < m <= max_edges:
        raise ValueError(f"edge count must be in 1..{max_edges} for {n} nodes, got {m}")


def _sorted_edges(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


# ---------------------------------------------------------------------------
# Erdos-Renyi
# ---------------------------------------------------------------------------


def erdos_renyi_directed(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Directed G(n, m): exactly ``m`` edges chosen uniformly among all n*(n-1) ordered pairs.

    Every possible connection is equally likely, so degrees cluster tightly
    around the mean and there are no hubs.
    """
    _check_size(n, m)
    # Number the n*(n-1) possible non-self-loop pairs 0..n*(n-1)-1:
    # code k -> source = k // (n-1), target offset = k % (n-1), skipping the diagonal.
    codes = rng.choice(n * (n - 1), size=m, replace=False)
    source = codes // (n - 1)
    target = codes % (n - 1)
    target = target + (target >= source)  # skip the self-loop position
    return _sorted_edges(np.stack([source, target], axis=1))


# ---------------------------------------------------------------------------
# Watts-Strogatz (directed variant)
# ---------------------------------------------------------------------------


def watts_strogatz_directed(n: int, m: int, rewire_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Directed small-world graph with exactly ``m`` edges.

    Directed variant of Watts & Strogatz (1998):
      1. Ring lattice: nodes sit on a circle. Node i sends edges to its nearest
         neighbours in the order i+1, i-1, i+2, i-2, ... (mod n). Out-degrees are
         m // n, and the first m % n nodes get one extra, so the total is exactly m.
      2. Each edge, independently with probability ``rewire_prob``, keeps its
         source and gets a new target drawn uniformly from nodes that are not the
         source and not already its targets.
    Small p keeps most of the lattice (high clustering) plus a few long-range
    shortcuts - the "small world" regime.
    """
    _check_size(n, m)
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError(f"rewire_prob must be in [0, 1], got {rewire_prob}")
    base, extra = divmod(m, n)
    offsets = [step * sign for step in range(1, n) for sign in (1, -1)]

    targets: list[list[int]] = []
    for i in range(n):
        out_degree = base + (1 if i < extra else 0)
        chosen: list[int] = []
        for offset in offsets:
            if len(chosen) == out_degree:
                break
            j = (i + offset) % n
            if j != i and j not in chosen:
                chosen.append(j)
        targets.append(chosen)

    for i in range(n):
        current = set(targets[i])
        for position, old in enumerate(targets[i]):
            if rng.random() >= rewire_prob:
                continue
            free = [j for j in range(n) if j != i and j not in current]
            if not free:
                continue  # node already connects to everyone
            new = free[int(rng.integers(len(free)))]
            current.remove(old)
            current.add(new)
            targets[i][position] = new

    pairs = [(i, j) for i in range(n) for j in targets[i]]
    return _sorted_edges(np.array(pairs))


# ---------------------------------------------------------------------------
# Barabasi-Albert (directed variant)
# ---------------------------------------------------------------------------


def barabasi_albert_directed(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Directed scale-free-style graph with exactly ``m`` edges.

    Directed variant of Barabasi & Albert (1999):
      1. Build an undirected BA graph with NetworkX: nodes arrive one by one and
         each attaches ``k`` edges to existing nodes with probability proportional
         to their degree ("the rich get richer" -> hubs). It has U = k*(n-k) edges.
      2. Give every undirected edge a random direction.
      3. Match the edge count exactly:
         - if U >= m (smallest such k is used): delete U - m random edges;
         - otherwise (m too large for one direction per edge): use the smallest k
           with 2U >= m and add the reverse direction to m - U random edges.

    Limitation: U is at most n*n/4, so at most about n*n/2 directed edges can be
    matched (8192 for 128 nodes). Denser graphs raise ValueError: a "hub"
    structure is not meaningful when most possible connections exist anyway.
    """
    _check_size(n, m)
    k, reciprocal = _ba_attachment_count(n, m)
    undirected = nx.barabasi_albert_graph(n, k, seed=int(rng.integers(2**31 - 1)))
    pairs = np.array(sorted(undirected.edges()), dtype=np.int64)  # sorted for determinism

    flip = rng.random(len(pairs)) < 0.5
    directed = np.where(flip[:, None], pairs[:, ::-1], pairs)
    if not reciprocal:
        keep = rng.choice(len(directed), size=m, replace=False)
        directed = directed[np.sort(keep)]
    else:
        add = rng.choice(len(directed), size=m - len(directed), replace=False)
        directed = np.concatenate([directed, directed[np.sort(add)][:, ::-1]])
    return _sorted_edges(directed)


def _ba_attachment_count(n: int, m: int) -> tuple[int, bool]:
    """Smallest BA parameter k whose edge count can be matched; True if reverse edges are needed."""
    for k in range(1, n):
        if k * (n - k) >= m:
            return k, False
    for k in range(1, n):
        if 2 * k * (n - k) >= m:
            return k, True
    raise ValueError(f"{m} edges is too dense for a BA-style graph on {n} nodes (max {2 * (n // 2) * (n - n // 2)})")


# ---------------------------------------------------------------------------
# Degree-preserving rewiring
# ---------------------------------------------------------------------------


def degree_preserving_rewire(
    edges: np.ndarray, weights: np.ndarray, swaps_per_edge: int, rng: np.random.Generator
) -> RewireResult:
    """Randomise a directed graph while keeping every node's in- and out-degree exactly.

    Directed double-edge swap (Maslov & Sneppen 2002): pick two edges a->b and
    c->d and swap their targets to a->d and c->b. Node a still sends one edge and
    d still receives one, and so on, so every degree is unchanged. A swap is
    rejected if it would create a self-loop or a duplicate edge.

    Weights travel with the source side (a keeps a->b's weight on a->d), so each
    neuron's total outgoing synapse count is also preserved.

    Args:
        edges: Shape (E, 2) directed edges without self-loops or duplicates.
        weights: Shape (E,) edge weights.
        swaps_per_edge: Attempted swaps = swaps_per_edge * E.
        rng: Random generator (determinism).
    """
    edges = np.asarray(edges, dtype=np.int64)
    weights = np.asarray(weights)
    if len(edges) < 2:
        raise ValueError("need at least 2 edges to rewire")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("input graph must not contain self-loops")

    source = edges[:, 0].tolist()  # plain Python lists: much faster in the swap loop
    target = edges[:, 1].tolist()
    present = set(zip(source, target))
    if len(present) != len(edges):
        raise ValueError("input graph must not contain duplicate edges")

    attempts = swaps_per_edge * len(edges)
    picks = rng.integers(len(edges), size=(attempts, 2))  # drawn up front: one deterministic stream
    accepted = 0
    for i, j in picks.tolist():
        if i == j:
            continue
        a, b, c, d = source[i], target[i], source[j], target[j]
        if a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present.discard((a, b))
        present.discard((c, d))
        present.add((a, d))
        present.add((c, b))
        target[i], target[j] = d, b
        accepted += 1

    source_arr = np.array(source, dtype=np.int64)
    target_arr = np.array(target, dtype=np.int64)
    order = np.lexsort((target_arr, source_arr))
    return RewireResult(
        edges=np.stack([source_arr[order], target_arr[order]], axis=1),
        weights=weights[order],
        attempted_swaps=attempts,
        accepted_swaps=accepted,
    )


def edge_overlap(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    """Fraction of edges in ``edges_a`` that also exist in ``edges_b`` (index space)."""
    set_b = set(map(tuple, np.asarray(edges_b).tolist()))
    return sum(tuple(e) in set_b for e in np.asarray(edges_a).tolist()) / len(edges_a)
