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
        type: One of ``GRAPH_TYPES``. "none" = conventional CNN baseline;
            anything else = connectome-constrained model on the given files.
        num_nodes: Number of graph nodes; must equal model.feature_dim.
        edges_file: source,target,weight CSV (required unless type is "none").
        nodes_file: Full node list CSV (keeps isolated nodes; defines node order).
        node_column: Node-ID column in ``nodes_file`` ("bodyId" or "node_id").
    """

    type: str = "none"
    num_nodes: int = 128
    edges_file: str | None = None
    nodes_file: str | None = None
    node_column: str = "bodyId"

    def __post_init__(self) -> None:
        _require_choice("graph.type", self.type, GRAPH_TYPES)
        _require_type("graph.num_nodes", self.num_nodes, int)
        if self.num_nodes < 2:
            raise ConfigError(f"graph.num_nodes must be >= 2, got {self.num_nodes}")
        if self.type == "none":
            if self.edges_file is not None or self.nodes_file is not None:
                raise ConfigError("graph.type 'none' must not set edges_file / nodes_file")
        else:
            for key in ("edges_file", "nodes_file", "node_column"):
                _require_type(f"graph.{key}", getattr(self, key), str)


@dataclass(frozen=True)
class DataConfig:
    """Which prepared dataset to train on.

    Attributes:
        prepared_dir: Folder with manifest.csv + dataset_metadata.json
            (output of scripts/prepare_dataset.py), relative to the project root.
        cache_in_memory: Preprocess every image once and keep the tensors in RAM
            (faster epochs; preprocessing is deterministic, so results are identical).
    """

    prepared_dir: str = "data/splits/figshare_glioma_meningioma"
    cache_in_memory: bool = True

    def __post_init__(self) -> None:
        _require_type("data.prepared_dir", self.prepared_dir, str)
        _require_type("data.cache_in_memory", self.cache_in_memory, bool)


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture (shared by the CNN baseline and all graph models).

    The model family is decided by ``graph.type``: "none" -> CNN baseline
    (encoder -> linear), otherwise GraphClassifier (encoder -> graph layers ->
    mean pooling -> linear).

    Attributes:
        encoder_channels: Output channels of the first two conv blocks.
        feature_dim: Size of the feature vector (= number of graph nodes).
        graph_layers: Number of connectome propagation layers (graph models only).
    """

    encoder_channels: list = field(default_factory=lambda: [16, 32])
    feature_dim: int = 128
    graph_layers: int = 2

    def __post_init__(self) -> None:
        _require_type("model.encoder_channels", self.encoder_channels, list)
        if len(self.encoder_channels) != 2:
            raise ConfigError("model.encoder_channels must list exactly 2 values")
        for c in self.encoder_channels:
            _require_type("model.encoder_channels[]", c, int)
            if c < 1:
                raise ConfigError("model.encoder_channels must be positive")
        for key in ("feature_dim", "graph_layers"):
            _require_type(f"model.{key}", getattr(self, key), int)
            if getattr(self, key) < 1:
                raise ConfigError(f"model.{key} must be positive")


SELECTION_METRICS = ("val_roc_auc", "val_loss")
CLASS_WEIGHTINGS = ("pos_weight_from_train", "none")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation and model-selection settings (identical for every model).

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        learning_rate / weight_decay: Adam optimizer settings.
        early_stopping_patience: Stop after this many epochs without a new best
            validation score.
        selection_metric: Validation score used for checkpoint selection and
            early stopping: "val_roc_auc" (higher is better) or "val_loss".
        class_weighting: "pos_weight_from_train" sets BCEWithLogitsLoss
            pos_weight = (#negatives / #positives) counted on the TRAIN split only.
        decision_threshold: Probability above which a slice/patient is predicted
            positive. Fixed in advance, never tuned on the test set.
    """

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    selection_metric: str = "val_roc_auc"
    class_weighting: str = "pos_weight_from_train"
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        for key in ("epochs", "batch_size", "early_stopping_patience"):
            _require_type(f"training.{key}", getattr(self, key), int)
            if getattr(self, key) < 1:
                raise ConfigError(f"training.{key} must be >= 1")
        for key in ("learning_rate", "weight_decay", "decision_threshold"):
            value = getattr(self, key)
            if isinstance(value, int) and not isinstance(value, bool):
                object.__setattr__(self, key, float(value))
            _require_type(f"training.{key}", getattr(self, key), float)
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigError("training.learning_rate must be > 0 and weight_decay >= 0")
        if not 0 < self.decision_threshold < 1:
            raise ConfigError("training.decision_threshold must be in (0, 1)")
        _require_choice("training.selection_metric", self.selection_metric, SELECTION_METRICS)
        _require_choice("training.class_weighting", self.class_weighting, CLASS_WEIGHTINGS)


@dataclass(frozen=True)
class Config:
    """The complete, validated experiment configuration."""

    experiment: ExperimentConfig
    graph: GraphConfig = field(default_factory=GraphConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        if self.graph.type != "none" and self.graph.num_nodes != self.model.feature_dim:
            raise ConfigError(
                f"graph.num_nodes ({self.graph.num_nodes}) must equal model.feature_dim "
                f"({self.model.feature_dim}): feature i is mapped to node i"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain nested dicts (for saving to YAML/JSON)."""
        return dataclasses.asdict(self)


# Maps each top-level YAML section to the dataclass that validates it.
_SECTIONS: dict[str, type] = {
    "experiment": ExperimentConfig,
    "graph": GraphConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
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


PATIENT_ID_SOURCES = ("none", "filename_regex", "csv", "figshare_pid")
DATASET_FORMATS = ("image_folder", "figshare_mat")
INVALID_IMAGE_POLICIES = ("error", "skip")


@dataclass(frozen=True)
class DatasetConfig:
    """Describe an image folder dataset and how to split it (configs/datasets/*.yaml).

    Formats:
        image_folder: ``<root>/<class_folder>/.../<image>.png|jpg|jpeg``; label = folder.
        figshare_mat: original Cheng et al. Figshare ``.mat`` files anywhere under
            ``<root>``; label and patient ID are read from ``cjdata.label`` / ``cjdata.PID``.

    Attributes:
        name, source, license: Provenance, recorded in the dataset metadata.
            Use "unknown" for a license that is not known - never guess.
        root: Dataset folder (relative paths are relative to the project root).
        classes: Folder name -> label index. Exactly two classes (binary task).
        positive_class: The class treated as label 1 (e.g. "tumor").
        format: "image_folder" or "figshare_mat".
        figshare_label_codes: figshare_mat only: cjdata.label code -> class name
            (from the dataset README). Codes whose name is not in ``classes`` are
            EXCLUDED from this dataset (counted in the metadata, files untouched).
        cv_indices_file: figshare_mat only, optional: the official cvind.mat.
        provenance_file: Optional JSON describing where the raw data came from
            (e.g. download_provenance.json); copied into the dataset metadata.
        patient_id_source: "none" (unavailable), "filename_regex", "csv", or
            "figshare_pid" (required for figshare_mat).
        patient_id_regex: With "filename_regex": a regex with a named group
            ``(?P<patient>...)`` matched against each file name.
        patient_id_csv / patient_id_csv_image_column / patient_id_csv_patient_column:
            With "csv": a mapping file from image path (relative to root) to patient ID.
        patient_link_regex: Optional conservative linking rule: a regex with a
            named group ``(?P<base>...)``. Patient IDs that fully match it and share
            the same ``base`` are treated as possibly the same person and are always
            put in the same split. The original patient IDs are not changed.
        split_seed: Seed for the deterministic split.
        train_fraction / val_fraction / test_fraction: Target fractions (sum to 1).
        on_invalid_image: "error" (default) or "skip" (skipped files are recorded).
        image_size: Images are resized to image_size x image_size pixels.
        grayscale: Convert to 1 channel (True) or 3-channel RGB (False).
        output_dir: Where manifest.csv and dataset_metadata.json are written.
    """

    name: str
    source: str
    license: str
    root: str
    classes: dict
    positive_class: str
    split_seed: int
    output_dir: str
    format: str = "image_folder"
    figshare_label_codes: dict | None = None
    cv_indices_file: str | None = None
    provenance_file: str | None = None
    patient_id_source: str = "none"
    patient_id_regex: str | None = None
    patient_id_csv: str | None = None
    patient_id_csv_image_column: str = "image_path"
    patient_id_csv_patient_column: str = "patient_id"
    patient_link_regex: str | None = None
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    on_invalid_image: str = "error"
    image_size: int = 224
    grayscale: bool = True

    def __post_init__(self) -> None:
        for key in ("name", "source", "license", "root", "positive_class", "output_dir"):
            _require_type(key, getattr(self, key), str)
            if not getattr(self, key).strip():
                raise ConfigError(f"{key} must not be empty")
        _require_type("classes", self.classes, dict)
        if len(self.classes) != 2 or sorted(self.classes.values()) != [0, 1]:
            raise ConfigError(f"classes must map exactly two folder names to 0 and 1, got {self.classes}")
        if self.positive_class not in self.classes or self.classes[self.positive_class] != 1:
            raise ConfigError(f"positive_class {self.positive_class!r} must be a class with label 1")
        _require_type("split_seed", self.split_seed, int)
        _require_choice("format", self.format, DATASET_FORMATS)
        _require_choice("patient_id_source", self.patient_id_source, PATIENT_ID_SOURCES)
        if self.format == "figshare_mat":
            if self.patient_id_source != "figshare_pid":
                raise ConfigError("format figshare_mat requires patient_id_source: figshare_pid")
            _require_type("figshare_label_codes", self.figshare_label_codes, dict)
            for code, name in self.figshare_label_codes.items():
                _require_type("figshare_label_codes key", code, int)
                _require_type("figshare_label_codes value", name, str)
            names = list(self.figshare_label_codes.values())
            if len(set(names)) != len(names):
                raise ConfigError("figshare_label_codes must map codes to distinct class names")
            unknown = sorted(set(self.classes) - set(names))
            if unknown:
                raise ConfigError(f"classes {unknown} do not appear in figshare_label_codes")
            if not self.grayscale:
                raise ConfigError("figshare_mat images are single-channel: grayscale must be true")
            if self.cv_indices_file is not None:
                _require_type("cv_indices_file", self.cv_indices_file, str)
        else:
            if self.patient_id_source == "figshare_pid":
                raise ConfigError("patient_id_source figshare_pid is only valid with format figshare_mat")
            if self.figshare_label_codes is not None or self.cv_indices_file is not None:
                raise ConfigError("figshare_label_codes / cv_indices_file are only valid with format figshare_mat")
        if self.patient_id_source == "filename_regex":
            _require_type("patient_id_regex", self.patient_id_regex, str)
            if "(?P<patient>" not in self.patient_id_regex:
                raise ConfigError("patient_id_regex must contain a named group (?P<patient>...)")
        if self.patient_id_source == "csv":
            _require_type("patient_id_csv", self.patient_id_csv, str)
        if self.patient_link_regex is not None:
            _require_type("patient_link_regex", self.patient_link_regex, str)
            if "(?P<base>" not in self.patient_link_regex:
                raise ConfigError("patient_link_regex must contain a named group (?P<base>...)")
            if self.patient_id_source == "none":
                raise ConfigError("patient_link_regex needs patient IDs (patient_id_source is 'none')")
        fractions = []
        for key in ("train_fraction", "val_fraction", "test_fraction"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ConfigError(f"{key} must be a number strictly between 0 and 1, got {value!r}")
            fractions.append(float(value))
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ConfigError(f"train/val/test fractions must sum to 1, got {sum(fractions)}")
        _require_choice("on_invalid_image", self.on_invalid_image, INVALID_IMAGE_POLICIES)
        _require_type("image_size", self.image_size, int)
        if self.image_size < 8:
            raise ConfigError(f"image_size must be >= 8, got {self.image_size}")
        _require_type("grayscale", self.grayscale, bool)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load and strictly validate a dataset YAML file."""
    return _build_section("dataset", DatasetConfig, _load_with_inheritance(Path(path).resolve(), chain=()))


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
