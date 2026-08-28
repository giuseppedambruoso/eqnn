import torch

from src.ansatz_builder import build_qnn_from_spec
from src.data_encoding import embedding_unitary
from src.noise import apply_gate_noise, make_noise_rng
from src.qnn import create_qnn

DEVICE_NAME = "default.qubit"


def _sample_embedding(num_qubits: int) -> torch.Tensor:
    size = 2 ** (num_qubits // 2)
    img = torch.rand(size, size)
    img = img / torch.linalg.norm(img.reshape(-1))
    return embedding_unitary(img)


def _simple_spec() -> list[dict]:
    return [
        {"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}},
        {"gate": "RY", "wires": [1], "param": {"init": "random", "frozen": False}},
        {"gate": "CNOT", "wires": [0, 1]},
        {"gate": "RX", "wires": [2], "param": {"init": "random", "frozen": False}},
        {"gate": "CNOT", "wires": [2, 3]},
    ]


def test_make_noise_rng_none_when_disabled():
    assert make_noise_rng(42, 0.0) is None
    assert make_noise_rng(42, -1.0) is None


def test_make_noise_rng_reproducible_for_same_seed():
    rng_a = make_noise_rng(42, 0.5)
    rng_b = make_noise_rng(42, 0.5)
    draws_a = [rng_a.random() for _ in range(20)]
    draws_b = [rng_b.random() for _ in range(20)]
    assert draws_a == draws_b


def test_apply_gate_noise_is_noop_without_rng():
    # Must not raise, and (since default.qubit has no ambient state here)
    # simply doing nothing is the only way to verify — call it standalone
    # outside a QNode to confirm it doesn't touch anything when rng=None.
    apply_gate_noise([0, 1], None, 0.5)


def test_build_qnn_from_spec_noise_p_zero_matches_no_noise_kwarg():
    """Passing noise_p=0.0 explicitly must reproduce the exact same
    output as not passing the noise kwargs at all — the default must be
    a true no-op, not just "usually the same"."""
    spec = _simple_spec()
    emb = _sample_embedding(8)

    qnn_default, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec)
    out_default = qnn_default(emb, params)

    qnn_explicit_zero, _, _ = build_qnn_from_spec(
        DEVICE_NAME, 8, spec, noise_p=0.0, noise_seed=123
    )
    out_explicit_zero = qnn_explicit_zero(emb, params)

    assert torch.allclose(out_default, out_explicit_zero)


def test_build_qnn_from_spec_same_noise_seed_is_reproducible():
    spec = _simple_spec()
    emb = _sample_embedding(8)

    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec, noise_p=0.4, noise_seed=7)
    out1 = qnn(emb, params)
    out2 = qnn(emb, params)

    assert torch.allclose(out1, out2)


def test_build_qnn_from_spec_different_noise_seed_changes_output():
    spec = _simple_spec()
    emb = _sample_embedding(8)
    # A large num_qubits-independent spec with several gates and a high
    # noise_p makes a coincidental match between two different seeds
    # exceedingly unlikely, so this isn't flaky in practice.
    qnn_a, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec, noise_p=0.9, noise_seed=1)
    qnn_b, _, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec, noise_p=0.9, noise_seed=2)

    out_a = qnn_a(emb, params)
    out_b = qnn_b(emb, params)

    assert not torch.allclose(out_a, out_b)


def test_build_qnn_from_spec_noise_changes_output_vs_clean():
    spec = _simple_spec()
    emb = _sample_embedding(8)

    qnn_clean, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec)
    qnn_noisy, _, _ = build_qnn_from_spec(
        DEVICE_NAME, 8, spec, noise_p=0.9, noise_seed=1
    )

    out_clean = qnn_clean(emb, params)
    out_noisy = qnn_noisy(emb, params)

    assert not torch.allclose(out_clean, out_noisy)


def test_create_qnn_uniform_kind_supports_noise():
    """config1-config5 (the "uniform" kind, a separate code path from
    build_qnn_from_spec) must support noise_p/noise_seed too."""
    emb = _sample_embedding(8)
    params = torch.zeros(16, dtype=torch.float64) + 0.05

    qnn_clean = create_qnn(DEVICE_NAME, 8, 2, "config1", noise_p=0.0)
    qnn_noisy = create_qnn(DEVICE_NAME, 8, 2, "config1", noise_p=0.5, noise_seed=3)

    out_clean = qnn_clean(emb, params)
    out_noisy_1 = qnn_noisy(emb, params)
    out_noisy_2 = qnn_noisy(emb, params)

    assert not torch.allclose(out_clean, out_noisy_1)
    assert torch.allclose(out_noisy_1, out_noisy_2)


def test_create_qnn_paper_kind_supports_noise():
    """config6-config9 (the "paper" kind, delegating to
    build_qnn_from_spec) must support noise_p/noise_seed too."""
    emb = _sample_embedding(8)
    params = torch.zeros(6, dtype=torch.float64) + 0.05

    qnn_clean = create_qnn(DEVICE_NAME, 8, 2, "config6", noise_p=0.0)
    qnn_noisy = create_qnn(DEVICE_NAME, 8, 2, "config6", noise_p=0.5, noise_seed=3)

    out_clean = qnn_clean(emb, params)
    out_noisy_1 = qnn_noisy(emb, params)
    out_noisy_2 = qnn_noisy(emb, params)

    assert not torch.allclose(out_clean, out_noisy_1)
    assert torch.allclose(out_noisy_1, out_noisy_2)
