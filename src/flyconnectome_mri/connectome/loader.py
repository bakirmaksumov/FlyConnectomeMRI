"""Load and validate a directed, weighted edge list (the project's canonical graph format).

Canonical CSV format (one row = one directed connection)::

    source,target,weight
    1001,1050,17
    1001,3002,5

- ``source``/``target``: non-negative integer node IDs (for a connectome these
  are neuron IDs; neuron 1001 sends signals to neuron 1050).
- ``weight``: strictly positive number (for a connectome, the synapse count).

Direction matters: ``1001 -> 1050`` is a different edge from ``1050 -> 1001``.

Node mapping: neural-network code needs nodes numbered 0..N-1. The mapping is
"sort the unique IDs ascending, then number them", so it depends only on
*which* IDs exist, never on the row order of the file.

Adjacency convention: ``adjacency[i, j] = weight`` of the edge from node ``i``
(row = sender) to node ``j`` (column = receiver).

Limitation: an edge list cannot express a node with no edges at all, so the
node set is every ID that appears in at least one row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconnectome_mri.utils.logging import get_logger

REQUIRED_COLUMNS = ("source", "target", "weight")
# Header is line 1 of the file, so data row i (0-based) is on line i + 2.
_FIRST_DATA_LINE = 2
# Show at most this many offending line numbers in an error message.
_MAX_LINES_IN_MESSAGE = 10

SelfLoopPolicy = Literal["error", "drop", "keep"]
DuplicatePolicy = Literal["error", "sum"]


class EdgeListValidationError(ValueError):
    """Raised when an edge-list file violates the canonical format."""


@dataclass(frozen=True)
class ValidationReport:
    """What the loader found and changed while reading the file.

    Attributes:
        rows_read: Data rows in the file (header excluded).
        self_loops_found: Rows whose source equals target.
        self_loops_removed: Self-loops removed because of ``self_loops="drop"``.
        duplicates_found: Rows repeating an earlier (source, target) pair.
        duplicates_merged: Duplicate rows merged because of ``duplicates="sum"``.
        ignored_columns: Extra CSV columns that were not used.
    """

    rows_read: int
    self_loops_found: int
    self_loops_removed: int
    duplicates_found: int
    duplicates_merged: int
    ignored_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectomeGraph:
    """A validated directed weighted graph in several interchangeable forms.

    Attributes:
        node_ids: Original node IDs; ``node_ids[i]`` is the ID of node index ``i``.
            Sorted ascending.
        node_index: Maps original ID -> node index (inverse of ``node_ids``).
        edges: Array of shape (E, 2) with ``[source_index, target_index]`` rows,
            sorted by source then target.
        weights: Array of shape (E,); ``weights[k]`` belongs to ``edges[k]``.
        graph: NetworkX DiGraph keyed by original IDs, edge attribute ``weight``.
        adjacency: SciPy CSR sparse matrix, shape (N, N), ``[source, target] = weight``.
        report: What validation found and changed.
        source_path: File the graph was loaded from.
    """

    node_ids: np.ndarray
    node_index: dict[int, int]
    edges: np.ndarray
    weights: np.ndarray
    graph: nx.DiGraph
    adjacency: sp.csr_matrix
    report: ValidationReport
    source_path: Path | None = field(default=None)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.weights)


def load_edge_list(
    path: str | Path,
    self_loops: SelfLoopPolicy = "error",
    duplicates: DuplicatePolicy = "error",
) -> ConnectomeGraph:
    """Load, validate and convert a ``source,target,weight`` CSV file.

    Args:
        path: CSV file with a header row containing source, target, weight.
        self_loops: What to do with edges from a node to itself.
            "error" (default) refuses the file, "drop" removes those edges
            (the node itself stays), "keep" keeps them (they appear on the
            adjacency diagonal). The default forces an explicit decision.
        duplicates: What to do when the same (source, target) pair appears in
            more than one row. "error" (default) refuses the file, "sum" adds
            the weights into one edge.

    Returns:
        A :class:`ConnectomeGraph`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        EdgeListValidationError: If the file violates any validation rule.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Edge list not found: {path}")
    if self_loops not in ("error", "drop", "keep"):
        raise ValueError(f"self_loops must be 'error', 'drop' or 'keep', got {self_loops!r}")
    if duplicates not in ("error", "sum"):
        raise ValueError(f"duplicates must be 'error' or 'sum', got {duplicates!r}")

    raw, ignored_columns = _read_csv(path)
    table = _parse_and_validate_values(raw)
    rows_read = len(table)

    # --- Self-loops ---------------------------------------------------------
    # The node set is taken BEFORE dropping self-loops: "drop" removes the
    # connection, not the neuron.
    all_ids = np.union1d(table["source"].to_numpy(), table["target"].to_numpy())
    is_loop = (table["source"] == table["target"]).to_numpy()
    n_loops = int(is_loop.sum())
    if n_loops and self_loops == "error":
        raise EdgeListValidationError(
            f"{n_loops} self-loop(s) (source == target) on {_lines(is_loop)}. "
            "Choose self_loops='drop' or self_loops='keep' explicitly."
        )
    if n_loops and self_loops == "drop":
        table = table[~is_loop]
        get_logger().warning("Dropped %d self-loop(s) from %s", n_loops, path.name)

    # --- Duplicates ---------------------------------------------------------
    is_dup = table.duplicated(subset=["source", "target"], keep="first").to_numpy()
    n_dups = int(is_dup.sum())
    if n_dups and duplicates == "error":
        dup_lines = _lines(is_dup, index=table.index.to_numpy())
        raise EdgeListValidationError(
            f"{n_dups} duplicate edge(s) (same source and target as an earlier row) "
            f"on {dup_lines}. Use duplicates='sum' to add their weights together."
        )
    if n_dups:  # duplicates == "sum"
        table = table.groupby(["source", "target"], as_index=False, sort=False)["weight"].sum()
        get_logger().warning("Merged %d duplicate edge row(s) in %s by summing weights", n_dups, path.name)

    if table.empty:
        raise EdgeListValidationError(f"No edges left in {path} after validation")

    report = ValidationReport(
        rows_read=rows_read,
        self_loops_found=n_loops,
        self_loops_removed=n_loops if self_loops == "drop" else 0,
        duplicates_found=n_dups,
        duplicates_merged=n_dups,  # only reachable when duplicates == "sum"
        ignored_columns=ignored_columns,
    )
    return _build_graph(table, all_ids, report, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Read the CSV as text (no type guessing) and check the columns."""
    # dtype=str: IDs are parsed ourselves, so large IDs never pass through
    # float64 (which silently rounds integers above 2**53).
    # keep_default_na=False: empty cells stay "" so we can report them ourselves.
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise EdgeListValidationError(f"{path} is empty") from exc
    except pd.errors.ParserError as exc:
        raise EdgeListValidationError(f"{path} is not a valid CSV file: {exc}") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise EdgeListValidationError(
            f"{path} is missing column(s) {missing}; found {list(raw.columns)}. "
            f"Expected header: {','.join(REQUIRED_COLUMNS)}"
        )
    ignored = tuple(c for c in raw.columns if c not in REQUIRED_COLUMNS)
    if ignored:
        get_logger().info("Ignoring extra column(s) in %s: %s", path.name, list(ignored))
    if raw.empty:
        raise EdgeListValidationError(f"{path} has a header but no edges")
    return raw[list(REQUIRED_COLUMNS)].apply(lambda col: col.str.strip()), ignored


def _parse_and_validate_values(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to int64 IDs and float64 weights, rejecting bad values."""
    for column in REQUIRED_COLUMNS:
        missing = (raw[column] == "").to_numpy()
        if missing.any():
            raise EdgeListValidationError(f"Missing value in column '{column}' on {_lines(missing)}")

    table = pd.DataFrame(index=raw.index)
    for column in ("source", "target"):
        # Node IDs must be plain non-negative integers (no "1.5", "-3", "abc", "1e5").
        bad = ~raw[column].str.fullmatch(r"\d+").to_numpy()
        if bad.any():
            raise EdgeListValidationError(
                f"Column '{column}' must contain non-negative integer node IDs; "
                f"invalid value(s) on {_lines(bad)}"
            )
        try:
            table[column] = raw[column].astype("int64")
        except OverflowError as exc:
            raise EdgeListValidationError(f"Node ID in column '{column}' exceeds the int64 range") from exc

    weight = pd.to_numeric(raw["weight"], errors="coerce").astype("float64")
    not_numeric = weight.isna().to_numpy()
    if not_numeric.any():
        raise EdgeListValidationError(f"Column 'weight' has non-numeric value(s) on {_lines(not_numeric)}")
    not_finite = ~np.isfinite(weight.to_numpy())
    if not_finite.any():
        raise EdgeListValidationError(f"Column 'weight' has infinite value(s) on {_lines(not_finite)}")
    # A connection with zero or negative synapses is not a connection. Negative
    # "inhibitory" signs, if ever needed, must be modelled separately, not
    # smuggled into the synapse count.
    not_positive = (weight <= 0).to_numpy()
    if not_positive.any():
        raise EdgeListValidationError(
            f"Column 'weight' must be > 0; zero or negative value(s) on {_lines(not_positive)}"
        )
    table["weight"] = weight
    return table


def _build_graph(
    table: pd.DataFrame, all_ids: np.ndarray, report: ValidationReport, path: Path
) -> ConnectomeGraph:
    """Create the node mapping, edge arrays, NetworkX graph and sparse adjacency."""
    node_ids = np.sort(all_ids).astype(np.int64)
    node_index = {int(node_id): i for i, node_id in enumerate(node_ids)}

    # searchsorted maps each ID to its position in the sorted ID array (= its index).
    src = np.searchsorted(node_ids, table["source"].to_numpy())
    tgt = np.searchsorted(node_ids, table["target"].to_numpy())
    weights = table["weight"].to_numpy(dtype=np.float64)

    order = np.lexsort((tgt, src))  # sort by source, then target: file order never matters
    edges = np.stack([src[order], tgt[order]], axis=1).astype(np.int64)
    weights = weights[order]

    n = len(node_ids)
    adjacency = sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))

    graph = nx.DiGraph()
    graph.add_nodes_from(int(i) for i in node_ids)
    graph.add_weighted_edges_from(
        (int(node_ids[s]), int(node_ids[t]), float(w)) for (s, t), w in zip(edges, weights)
    )
    return ConnectomeGraph(
        node_ids=node_ids,
        node_index=node_index,
        edges=edges,
        weights=weights,
        graph=graph,
        adjacency=adjacency,
        report=report,
        source_path=path,
    )


def _lines(mask: np.ndarray, index: np.ndarray | None = None) -> str:
    """Describe the CSV line numbers where ``mask`` is True, e.g. 'line(s) 3, 7'."""
    positions = np.flatnonzero(mask)
    rows = positions if index is None else index[positions]
    numbers = [str(int(r) + _FIRST_DATA_LINE) for r in rows[:_MAX_LINES_IN_MESSAGE]]
    more = f" (+{len(rows) - _MAX_LINES_IN_MESSAGE} more)" if len(rows) > _MAX_LINES_IN_MESSAGE else ""
    return f"line(s) {', '.join(numbers)}{more}"
