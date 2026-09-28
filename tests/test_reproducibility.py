"""Tests for seeding and device selection."""

import random

import numpy as np
import pytest
import torch

from flyconnectome_mri.training.reproducibility import (
    collect_environment_info,
    make_generator,
    resolve_device,
    seed_fingerprint,
    set_seed,
)


def draw_all() -> tuple[float, np.ndarray, torch.Tensor]:
    return random.random(), np.random.rand(5), torch.rand(5)


def test_same_seed_gives_identical_random_numbers() -> None:
    set_seed(123)
    first = draw_all()
    set_seed(123)
    second = draw_all()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_different_seeds_give_different_random_numbers() -> None:
    set_seed(1)
    a = draw_all()
    set_seed(2)
    b = draw_all()
    assert a[0] != b[0]
    assert not np.array_equal(a[1], b[1])
    assert not torch.equal(a[2], b[2])


def test_model_initialisation_is_reproducible() -> None:
    """The use case that matters: identical seeds give identical initial weights."""
    set_seed(0)
    w1 = torch.nn.Linear(16, 4).weight.detach().clone()
    set_seed(0)
    w2 = torch.nn.Linear(16, 4).weight.detach().clone()
    assert torch.equal(w1, w2)


def test_seed_fingerprint_is_reproducible() -> None:
    set_seed(5)
    f1 = seed_fingerprint()
    set_seed(5)
    f2 = seed_fingerprint()
    assert f1 == f2


def test_generator_is_independent_of_global_state() -> None:
    perm1 = torch.randperm(10, generator=make_generator(9))
    torch.rand(100)  # consuming global random numbers must not affect the generator
    perm2 = torch.randperm(10, generator=make_generator(9))
    assert torch.equal(perm1, perm2)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_matches_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a GPU")
def test_resolve_device_cuda_without_gpu_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_environment_info_contains_versions() -> None:
    info = collect_environment_info(torch.device("cpu"))
    assert info["device"] == "cpu"
    recorded = info["packages"]["torch"]
    assert recorded is not None
    assert recorded.split("+")[0] == torch.__version__.split("+")[0]
    assert info["python"]
