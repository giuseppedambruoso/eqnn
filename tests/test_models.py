"""Checks of the statements of the paper on the four models (run with: python -m pytest tests)."""

import math

import pytest
import torch

from eqnn.circuits import circuit_gates, layer_gates
from eqnn.data import apply_d4, make_shapes
from eqnn.models import MODELS, build_model

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def images():
    x, _ = make_shapes(6, seed=0, square_sides=(8, 14), cross_spans=(9, 15), cross_thick=(1, 1), hollow=False,
                       noise=0.05)
    return x


def transformed(x, g):
    return apply_d4(x.reshape(-1, 16, 16), g).reshape(x.shape[0], -1)


def angles(n, seed=0):
    return torch.empty(n, dtype=torch.float64).uniform_(-1, 1, generator=torch.Generator().manual_seed(seed))


def test_circuit_structure():
    for eq in (True, False):
        gates = layer_gates(eq)
        assert len(gates) == 25  # five blocks of two R pairs and one four-qubit rotation
        assert sorted({k for _, k in gates}) == list(range(6))  # six angles per layer
        assert len(circuit_gates(eq, 3)) == 75
    # NonEquiv cycles the column axes: X -> Y on the single-qubit gates, Y -> Z in the four-qubit rotations
    for (we, _), (wn, _) in zip(layer_gates(True), layer_gates(False)):
        for w in we:
            col = {"X": "Y", "Y": "Z"}
            assert wn[w] == (col[we[w]] if w >= 4 else we[w])


@pytest.mark.parametrize("arch,invariant", [("config6", True), ("config10", True), ("config7", False),
                                            ("config6n", False)])
def test_invariance(images, arch, invariant):
    """Equiv and NonEquiv-Twirled are exactly D4-invariant for every value of the angles; the other two are not."""
    f, n = build_model(arch, 2)
    th = angles(n)
    m = f(images, th)
    dev = max(float((f(transformed(images, g), th) - m).abs().max()) for g in range(1, 8))
    assert (dev < 1e-12) == invariant, (MODELS[arch]["name"], dev)


def test_twirled_nosym_is_equiv(images):
    """Proposition of the paper: the twirl of Equiv-NoSym is Equiv divided by 1 + sqrt(2)."""
    f_ns, n = build_model("config6n", 2)
    f_eq, _ = build_model("config6", 2)
    th = angles(n, 3)
    twirled = torch.stack([f_ns(transformed(images, g), th) for g in range(8)]).mean(dim=0)
    assert torch.allclose(twirled, f_eq(images, th) / (1 + math.sqrt(2)), atol=1e-12)


@pytest.mark.parametrize("arch", list(MODELS))
def test_noise_heisenberg_matches_density_matrices(images, arch):
    """The Heisenberg-picture simulation of the noisy circuits agrees with PennyLane default.mixed."""
    f_h, n = build_model(arch, 1, noise_scale=2.0, chip_seed=0)
    f_m, _ = build_model(arch, 1, noise_scale=2.0, chip_seed=0, engine="mixed")
    th = angles(n, 1)
    assert torch.allclose(f_h(images[:3], th), f_m(images[:3], th), atol=1e-10)


@pytest.mark.parametrize("arch", list(MODELS))
def test_noise_scale_zero_is_noiseless(images, arch):
    f0, n = build_model(arch, 2, noise_scale=0.0)
    f, _ = build_model(arch, 2)
    th = angles(n, 2)
    assert torch.allclose(f0(images, th), f(images, th), atol=1e-12)


def test_uniform_noise_keeps_equiv_invariant_only_without_dispersion(images):
    """With identical qubits the noisy Equiv circuit keeps the transposition as an exact symmetry."""
    f, n = build_model("config6", 1, noise_scale=1.0, chip_spread=False)
    th = angles(n, 4)
    assert torch.allclose(f(transformed(images, 7), th), f(images, th), atol=1e-12)  # transposition


def test_dla_small_case():
    from eqnn.dla import P, add, dla
    assert dla([P({0: "X"}), P({1: "X"}), P({0: "Z", 1: "Z"})]) == 6
    assert dla([add(P({0: "X"}), P({1: "X"}))]) == 1
