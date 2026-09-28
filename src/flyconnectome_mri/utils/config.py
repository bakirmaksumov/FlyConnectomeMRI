"""Configuration loading and validation.

Every experiment is described by a YAML file in ``configs/``. This module:

1. reads the YAML file,
2. resolves inheritance (a child file says ``inherit: base.yaml`` and only lists
   the values it changes),
3. validates everything strictly (unknown keys, wrong types and invalid choices
   are errors, never silently ignored),
4. returns a typed :class:`Config` object so the rest of the code gets
   autocompletion and cannot misspell a key.

Sections are added here as later phases need them (data, model, training, ...).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

INHERIT_KEY = "inherit"

# Graph topologies the project will support. "none" means "no graph", which is
# the plain CNN baseline.
GRAPH_TYPES = ("none", "drosophila", "er", "ws", "ba", "rewired")
DEVICES = ("auto", "cpu", "cuda")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """General run settings.

    Attributes:
        name: Short experiment name, used in the results folder name.
        seed: Seed for every random number generator (Python, NumPy, PyTorch).
        deterministic: Ask PyTorch to use deterministic algorithms where possible.
        device: "cpu", "cuda", or "auto" (use CUDA if available, else CPU).
        output_dir: Parent folder for run folders, relative to the project root.
    """

    name: str
    seed: int
    deterministic: bool = True
    device: str = "auto"
    output_dir: str = "results"

    def __post_init__(self) -> None:
        _require_type("experiment.name", self.name, str)
        if not self.name.strip():
            raise ConfigError("experiment.name must not be empty")
        _require_type("experiment.seed", self.seed, int)
        if self.seed < 0:
            raise ConfigError(f"experiment.seed must be >= 0, got {self.seed}")
        _require_type("experiment.deterministic", self.deterministic, bool)
        _require_choice("experiment.device", self.device, DEVICES)
        _require_type("experiment.output_dir", self.output_dir, str)


@dataclass(frozen=True)
class GraphConfig:
    """Which graph topology constrains the model.

    Attributes:
        type: One of ``GRAPH_TYPES``. "none" = conventional CNN baseline.
        num_nodes: Number of graph nodes (e.g. 128, 256, 512, 1000).
    """

    type: str = "none"
    num_nodes: int = 128

    def __post_init__(self) -> None:
        _require_choice("graph.type", self.type, GRAPH_TYPES)
        _require_type("graph.num_nodes", self.num_nodes, int)
        if self.num_nodes < 2:
            raise ConfigError(f"graph.num_nodes must be >= 2, got {self.num_nodes}")


@dataclass(frozen=True)
class Config:
    """The complete, validated experiment configuration."""

    experiment: ExperimentConfig
    graph: GraphConfig = field(default_factory=GraphConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain nested dicts (for saving to YAML/JSON)."""
        return dataclasses.asdict(self)


# Maps each top-level YAML section to the dataclass that validates it.
_SECTIONS: dict[str, type] = {
    "experiment": ExperimentConfig,
    "graph": GraphConfig,
}


def load_config(path: str | Path, seed: int | None = None) -> Config:
    """Load, resolve inheritance, and validate a YAML config file.

    Args:
        path: Path to the YAML file.
        seed: Optional override for ``experiment.seed`` (e.g. from the command line).

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If the file is missing, malformed, or contains invalid values.
    """
    raw = _load_with_inheritance(Path(path).resolve(), chain=())
    if seed is not None:
        raw.setdefault("experiment", {})["seed"] = seed
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Validate a plain nested dict and convert it into a :class:`Config`."""
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(
            f"Unknown config section(s): {sorted(unknown)}. Allowed: {sorted(_SECTIONS)}"
        )
    if "experiment" not in raw:
        raise ConfigError("Config is missing the required 'experiment' section")

    sections: dict[str, Any] = {}
    for name, section_cls in _SECTIONS.items():
        if name not in raw:
            continue  # optional sections fall back to their defaults
        values = raw[name]
        if not isinstance(values, dict):
            raise ConfigError(f"Section '{name}' must be a mapping, got {type(values).__name__}")
        sections[name] = _build_section(name, section_cls, values)
    return Config(**sections)


SELECTION_METHODS = ("roi_top_synapses",)
SELF_LOOP_POLICIES = ("error", "drop", "keep")


@dataclass(frozen=True)
class DrosophilaFetchConfig:
    """How to extract a Drosophila subgraph from neuPrint (see configs/malecns_fetch.yaml).

    Attributes:
        server: neuPrint server URL.
        dataset: neuPrint dataset name, e.g. "male-cns:v1.0".
        selection_method: How neurons are chosen. Only "roi_top_synapses" exists:
            the ``num_nodes`` neurons with the most synapses inside ``roi``.
        roi: neuPrint region-of-interest name used by the selection.
        num_nodes: How many neurons to select.
        min_total_weight: Keep only connections with at least this many synapses.
        self_loops: Loader policy for neuron->itself edges when validating
            ("error" | "drop" | "keep"). The saved CSV always contains the raw data.
        token_env_var: NAME of the environment variable holding the API token.
            The token value itself must never appear in a config file.
        output_dir: Folder for the CSV and metadata, relative to the project root.
    """

    server: str
    dataset: str
    roi: str
    num_nodes: int
    selection_method: str = "roi_top_synapses"
    min_total_weight: int = 1
    self_loops: str = "drop"
    token_env_var: str = "NEUPRINT_TOKEN"
    output_dir: str = "data/connectome"

    def __post_init__(self) -> None:
        _require_type("server", self.server, str)
        if not self.server.startswith("https://"):
            raise ConfigError(f"server must start with https://, got {self.server!r}")
        for key in ("dataset", "roi", "token_env_var", "output_dir"):
            _require_type(key, getattr(self, key), str)
            if not getattr(self, key).strip():
                raise ConfigError(f"{key} must not be empty")
        _require_choice("selection_method", self.selection_method, SELECTION_METHODS)
        _require_type("num_nodes", self.num_nodes, int)
        if self.num_nodes < 2:
            raise ConfigError(f"num_nodes must be >= 2, got {self.num_nodes}")
        _require_type("min_total_weight", self.min_total_weight, int)
        if self.min_total_weight < 1:
            raise ConfigError(f"min_total_weight must be >= 1, got {self.min_total_weight}")
        _require_choice("self_loops", self.self_loops, SELF_LOOP_POLICIES)


def load_fetch_config(path: str | Path, num_nodes: int | None = None) -> DrosophilaFetchConfig:
    """Load and strictly validate a connectome-fetch YAML file.

    Args:
        path: YAML file (supports ``inherit:`` like experiment configs).
        num_nodes: Optional override for ``num_nodes`` (e.g. from the command line).
    """
    raw = _load_with_inheritance(Path(path).resolve(), chain=())
    if num_nodes is not None:
        raw["num_nodes"] = num_nodes
    return _build_section("fetch", DrosophilaFetchConfig, raw)


@dataclass(frozen=True)
class GraphVariantsConfig:
    """Derive weight-threshold variants from a raw downloaded graph (configs/graph_variants.yaml).

    Attributes:
        source_edges / source_nodes / source_metadata: Raw files from fetch_connectome.py.
            Their checksums are verified against the metadata before use.
        variants: Variant name -> minimum synapse count kept (e.g. {"w20": 20}).
        primary_variant: The variant pre-declared for the main experiments.
        primary_declared_on: Date (YYYY-MM-DD) the primary variant was fixed,
            before any MRI training. Recorded in every variant's metadata.
        output_dir: Folder for the derived files, relative to the project root.
    """

    source_edges: str
    source_nodes: str
    source_metadata: str
    variants: dict
    primary_variant: str
    primary_declared_on: str
    output_dir: str = "data/connectome"

    def __post_init__(self) -> None:
        for key in ("source_edges", "source_nodes", "source_metadata", "primary_variant",
                    "primary_declared_on", "output_dir"):
            _require_type(key, getattr(self, key), str)
        _require_type("variants", self.variants, dict)
        if not self.variants:
            raise ConfigError("variants must not be empty")
        for name, threshold in self.variants.items():
            if not isinstance(name, str) or not name.isalnum() or not name.islower():
                raise ConfigError(f"variant name must be lowercase letters/digits, got {name!r}")
            _require_type(f"variants.{name}", threshold, int)
            if threshold < 1:
                raise ConfigError(f"variants.{name} must be >= 1, got {threshold}")
        if self.primary_variant not in self.variants:
            raise ConfigError(f"primary_variant {self.primary_variant!r} is not one of {list(self.variants)}")


CONTROL_TYPES = ("er", "ws", "ba", "rewired")


@dataclass(frozen=True)
class ControlsConfig:
    """Generate random control graphs matched to a biological graph (configs/controls.yaml).

    Attributes:
        source_edges / source_nodes: The biological graph to match (edges + full node list).
        source_node_column: Node-ID column name in ``source_nodes``.
        label: Used in output file names, e.g. "128_w20" -> er_128_w20_seed0.csv.
        controls: Which control types to generate.
        seeds: One control graph per type per seed.
        ws_rewire_prob: Watts-Strogatz probability of rewiring each lattice edge.
        rewire_swaps_per_edge: Degree-preserving rewiring attempts = this x edge count.
        output_dir: Folder for the control files, relative to the project root.
    """

    source_edges: str
    source_nodes: str
    label: str
    seeds: list
    source_node_column: str = "bodyId"
    controls: list = field(default_factory=lambda: list(CONTROL_TYPES))
    ws_rewire_prob: float = 0.1
    rewire_swaps_per_edge: int = 10
    output_dir: str = "data/connectome"

    def __post_init__(self) -> None:
        for key in ("source_edges", "source_nodes", "source_node_column", "label", "output_dir"):
            _require_type(key, getattr(self, key), str)
        _require_type("seeds", self.seeds, list)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds must be a non-empty list without repeats")
        for seed in self.seeds:
            _require_type("seeds[]", seed, int)
            if seed < 0:
                raise ConfigError(f"seeds must be >= 0, got {seed}")
        _require_type("controls", self.controls, list)
        if not self.controls:
            raise ConfigError("controls must not be empty")
        for control in self.controls:
            _require_choice("controls[]", control, CONTROL_TYPES)
        if isinstance(self.ws_rewire_prob, int) and not isinstance(self.ws_rewire_prob, bool):
            object.__setattr__(self, "ws_rewire_prob", float(self.ws_rewire_prob))
        _require_type("ws_rewire_prob", self.ws_rewire_prob, float)
        if not 0.0 <= self.ws_rewire_prob <= 1.0:
            raise ConfigError(f"ws_rewire_prob must be in [0, 1], got {self.ws_rewire_prob}")
        _require_type("rewire_swaps_per_edge", self.rewire_swaps_per_edge, int)
        if self.rewire_swaps_per_edge < 1:
            raise ConfigError(f"rewire_swaps_per_edge must be >= 1, got {self.rewire_swaps_per_edge}")


def load_graph_variants_config(path: str | Path) -> GraphVariantsConfig:
    """Load and strictly validate a graph-variants YAML file."""
    return _build_section("variants", GraphVariantsConfig, _load_with_inheritance(Path(path).resolve(), chain=()))


def load_controls_config(path: str | Path) -> ControlsConfig:
    """Load and strictly validate a control-graph YAML file."""
    return _build_section("controls", ControlsConfig, _load_with_inheritance(Path(path).resolve(), chain=()))


def save_config(config: Config, path: str | Path) -> None:
    """Write the fully resolved config (no inheritance) to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_with_inheritance(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    """Read ``path`` and recursively merge it on top of its ``inherit`` parent."""
    if path in chain:
        cycle = " -> ".join(p.name for p in (*chain, path))
        raise ConfigError(f"Circular config inheritance: {cycle}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")

    parent_name = data.pop(INHERIT_KEY, None)
    if parent_name is None:
        return data
    if not isinstance(parent_name, str):
        raise ConfigError(f"'{INHERIT_KEY}' in {path} must be a file name")
    # The parent path is relative to the child file's folder.
    parent = _load_with_inheritance((path.parent / parent_name).resolve(), (*chain, path))
    return _deep_merge(parent, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` with ``override`` merged in, recursively for nested dicts."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_section(name: str, section_cls: type, values: dict[str, Any]) -> Any:
    """Instantiate a section dataclass, rejecting unknown keys with a clear message."""
    allowed = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in section '{name}': {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )
    try:
        return section_cls(**values)
    except TypeError as exc:  # e.g. a required key is missing
        raise ConfigError(f"Invalid section '{name}': {exc}") from exc


def _require_type(key: str, value: Any, expected: type) -> None:
    # bool is a subclass of int in Python, so `True` would pass an int check.
    # Reject that explicitly: `seed: true` is almost certainly a mistake.
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer, got a boolean ({value})")
    if not isinstance(value, expected):
        raise ConfigError(f"{key} must be {expected.__name__}, got {type(value).__name__} ({value!r})")


def _require_choice(key: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ConfigError(f"{key} must be one of {list(choices)}, got {value!r}")
