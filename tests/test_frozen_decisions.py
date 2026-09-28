"""Guards for the research decisions frozen on 2026-09-28 (see docs/RESEARCH_PLAN.md).

If any of these tests fail, a frozen decision or a documented graph-construction
rule has changed. Do NOT update the expected values to make the test pass unless
the research plan records a deliberate, dated decision to change them BEFORE the
affected experiments are run.
"""

import hashlib
from pathlib import Path

import numpy as np

from flyconnectome_mri.connectome.controls import CONTROL_STREAM
from flyconnectome_mri.connectome.random_graphs import (
    barabasi_albert_directed,
    degree_preserving_rewire,
    erdos_renyi_directed,
    watts_strogatz_directed,
)
from flyconnectome_mri.utils.config import load_controls_config, load_graph_variants_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()


def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def test_primary_graph_and_sensitivity_variants_are_frozen() -> None:
    config = load_graph_variants_config(PROJECT_ROOT / "configs" / "graph_variants.yaml")
    assert config.primary_variant == "w20"
    assert config.variants == {"raw": 1, "w5": 5, "w10": 10, "w20": 20}
    assert config.primary_declared_on == "2026-09-28"


def test_control_set_is_frozen() -> None:
    config = load_controls_config(PROJECT_ROOT / "configs" / "controls.yaml")
    assert config.source_edges == "data/connectome/drosophila_128_w20.csv"
    assert config.controls == ["er", "ws", "ba", "rewired"]
    assert config.seeds == [0, 1, 2, 3, 4]
    assert config.ws_rewire_prob == 0.1
    assert config.rewire_swaps_per_edge == 10
    assert CONTROL_STREAM == {"er": 1, "ws": 2, "ba": 3, "rewired": 4}


# Golden outputs of each construction rule for fixed small inputs (computed 2026-09-28).


def test_er_rule_unchanged() -> None:
    assert digest(erdos_renyi_directed(30, 200, rng())) == (
        "de453eb03791f07ca151b2c5d26867c68be74f47ea73ab4259668d7015c295c1")


def test_directed_ws_rule_unchanged() -> None:
    assert digest(watts_strogatz_directed(30, 200, 0.1, rng())) == (
        "7140cad2a2ada6dd6e0d02f383ab19d3e0d0017864bb79644687b00858dae573")


def test_directed_ba_rule_unchanged() -> None:
    assert digest(barabasi_albert_directed(30, 200, rng())) == (
        "db0da399edf1e42df61e600126a7495bdda6f0d7049d7b3f2e695043c36a62dc")


def test_degree_preserving_rewire_rule_unchanged() -> None:
    base = erdos_renyi_directed(30, 200, np.random.default_rng(1))
    result = degree_preserving_rewire(base, np.arange(200), 10, rng())
    assert digest(result.edges) == "07f825564e08ac00a54249409454517b0beeb631e70389bf6b2be42506690163"
    assert digest(result.weights) == "d8fc1e913ed87601ce7ad50a5e986b8fcec986a199b59622af21644bf8f694a3"
    assert result.accepted_swaps == 1010
