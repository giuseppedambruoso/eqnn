import pennylane as qml
import pytest
import torch

from src.ansatz_builder import (
    architecture_to_spec,
    build_qnn_from_spec,
    check_p4m_invariance,
    param_labels,
    validate_spec,
)
from src.data_encoding import embedding_unitary
from src.qnn import ARCHITECTURES, create_qnn

DEVICE_NAME = "default.qubit"


def _sample_embedding(num_qubits: int) -> torch.Tensor:
    size = 2 ** (num_qubits // 2)
    img = torch.rand(size, size)
    img = img / torch.linalg.norm(img.reshape(-1))
    return embedding_unitary(img)


def test_build_and_run_simple_spec():
    spec = [
        {"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}},
        {"gate": "RY", "wires": [1], "param": {"init": "random", "frozen": False}},
        {"gate": "CNOT", "wires": [0, 1]},
    ]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)

    assert params.shape == (2,)
    emb = _sample_embedding(4)
    out = qnn(emb, params, torch.tensor(0.0))
    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_qnode_attribute_draws_full_circuit():
    """Regression test: qml.draw()/draw_mpl() on qnn_forward itself silently
    truncates the diagram after the first couple of operations — everything
    from the measurement's Hadamard layer onward goes missing, with no
    error raised. Drawing must target the exposed `qnn.qnode` instead."""
    spec = [
        {"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}},
        {"gate": "CNOT", "wires": [0, 1]},
    ]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)
    emb = _sample_embedding(4)

    drawing = qml.draw(qnn.qnode)(emb, params, torch.tensor(0.0))
    assert "H" in drawing
    assert "<𝓗" in drawing  # the Hamiltonian expval marker


def test_frozen_gate_excluded_from_trainable_params():
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {"init": "custom", "value": 0.3, "frozen": True},
        },
        {"gate": "RY", "wires": [1], "param": {"init": "random", "frozen": False}},
    ]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)

    assert params.shape == (1,)  # only the non-frozen gate
    assert param_labels(spec) == ["g1_RY_w1"]


def test_resolved_spec_is_reproducible():
    """A frozen gate with init='random' must keep the exact value it was
    built with when rebuilt later from the resolved spec (e.g. reloading a
    trained checkpoint for inference) — re-running build_qnn_from_spec on
    the ORIGINAL spec would draw a different random value each time."""
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {"init": "random", "value": None, "frozen": True},
        },
    ]
    _, _, resolved_spec = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)
    assert resolved_spec[0]["param"]["init"] == "custom"

    _, _, resolved_again = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, resolved_spec)
    assert resolved_again[0]["param"]["value"] == resolved_spec[0]["param"]["value"]


def test_custom_init_value_is_used():
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {"init": "custom", "value": 0.42, "frozen": False},
        },
    ]
    _, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)
    assert params[0].item() == pytest.approx(0.42)


def test_frozen_param_does_not_receive_gradient():
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {"init": "custom", "value": 0.5, "frozen": True},
        },
        {"gate": "RY", "wires": [1], "param": {"init": "random", "frozen": False}},
        {"gate": "CNOT", "wires": [0, 1]},
    ]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)
    params = params.clone().requires_grad_()

    emb = _sample_embedding(4)
    out = qnn(emb, params, torch.tensor(0.0))
    out.backward()

    # Only one trainable param exists (the frozen RX never entered `params`).
    assert params.grad.shape == (1,)


def test_pauli_rot_gate_multi_qubit():
    spec = [
        {"gate": "RX", "wires": [i], "param": {"init": "random", "frozen": False}}
        for i in range(4)
    ] + [
        {
            "gate": "PAULIROT",
            "wires": [0, 1, 2, 3],
            "pauli_word": "YYYY",
            "param": {"init": "custom", "value": 1.5707963267948966, "frozen": True},
        }
    ]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)
    assert params.shape == (4,)

    emb = _sample_embedding(4)
    out = qnn(emb, params, torch.tensor(0.0))
    assert torch.isfinite(out)


@pytest.mark.parametrize(
    "bad_spec,match",
    [
        ([], "empty"),
        (
            [{"gate": "NOPE", "wires": [0]}],
            "unknown gate",
        ),
        (
            [{"gate": "CNOT", "wires": [0]}],
            "expected 2 wire",
        ),
        (
            [{"gate": "CNOT", "wires": [0, 0]}],
            "repeated wire",
        ),
        (
            [
                {
                    "gate": "RX",
                    "wires": [8],
                    "param": {"init": "random", "frozen": False},
                }
            ],
            "out of range",
        ),
        (
            [{"gate": "RX", "wires": [0]}],
            "no 'param'",
        ),
        (
            [{"gate": "H", "wires": [0], "param": {"init": "random", "frozen": False}}],
            "not parametric",
        ),
        (
            [
                {
                    "gate": "PAULIROT",
                    "wires": [0, 1],
                    "pauli_word": "XQ",
                    "param": {"init": "random", "frozen": False},
                }
            ],
            "pauli_word",
        ),
        (
            [
                {
                    "gate": "RX",
                    "wires": [0],
                    "param": {"init": "custom", "frozen": False},
                }
            ],
            "requires a 'value'",
        ),
    ],
)
def test_invalid_specs_raise(bad_spec, match):
    with pytest.raises(ValueError, match=match):
        validate_spec(bad_spec, num_qubits=8)


@pytest.mark.parametrize(
    "architecture",
    sorted(a for a in ARCHITECTURES if ARCHITECTURES[a]["kind"] == "uniform"),
)
def test_architecture_to_spec_matches_create_qnn(architecture):
    """Every one of config1-config5, expanded into a spec (with twirled=
    matching ARCHITECTURES[architecture]["twirled"]), must give numerically
    identical output to the real fixed architecture for the same
    parameters — not just "something plausible"."""
    num_qubits, reps = 8, 2
    spec = architecture_to_spec(architecture, num_qubits, reps)
    twirled = ARCHITECTURES[architecture]["twirled"]
    qnn_spec, initial_params, _ = build_qnn_from_spec(
        DEVICE_NAME, num_qubits, 0.0, spec, twirled=twirled
    )
    qnn_fixed = create_qnn(DEVICE_NAME, num_qubits, 0.0, reps, architecture)

    assert initial_params.shape == (num_qubits * reps,)

    params = torch.empty(num_qubits * reps).uniform_(-0.1, 0.1)
    emb = _sample_embedding(num_qubits)
    phi = torch.tensor(0.0)

    out_spec = qnn_spec(emb, params, phi)
    out_fixed = qnn_fixed(emb, params, phi)

    assert torch.allclose(out_spec, out_fixed, atol=1e-6)


@pytest.mark.parametrize("architecture", ["config6", "config7", "config8", "config9"])
def test_architecture_to_spec_rejects_paper_architectures(architecture):
    with pytest.raises(ValueError, match="paper_architecture_spec"):
        architecture_to_spec(architecture, 8, 2)


def test_tied_group_parameters_share_one_slot():
    """Two gates sharing a "group" must collapse to ONE trainable slot —
    the mechanism config6-config9's paired row/column rotations rely on
    (see src.paper_ansatzes)."""
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {
                "init": "random",
                "value": None,
                "frozen": False,
                "group": "tied",
            },
        },
        {
            "gate": "RX",
            "wires": [1],
            "param": {
                "init": "random",
                "value": None,
                "frozen": False,
                "group": "tied",
            },
        },
        {"gate": "RY", "wires": [2], "param": {"init": "random", "frozen": False}},
    ]
    _, params, resolved = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec)

    assert params.shape == (2,)  # the tied pair collapses to 1 slot + the RY
    assert param_labels(spec) == ["g0_RX_w0", "g2_RY_w2"]
    assert resolved[0]["param"]["value"] == resolved[1]["param"]["value"]


def test_readout_x0_xhalf_is_bounded():
    spec = [{"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}}]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec, readout="x0_xhalf")
    emb = _sample_embedding(4)
    out = qnn(emb, params, torch.tensor(0.0))
    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_readout_avg_x_is_bounded():
    spec = [{"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}}]
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec, readout="avg_x")
    emb = _sample_embedding(4)
    out = qnn(emb, params, torch.tensor(0.0))
    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_readout_avg_x_matches_sum_z_noiseless():
    """config1-config5's "sum_z" readout (H then measure Z) is
    mathematically the same as measuring X directly (H Z H = X) when
    there's no noise (p_err=0) — so "avg_x" must give an identical output
    for the same spec/params in that regime."""
    spec = [
        {
            "gate": "RX",
            "wires": [0],
            "param": {"init": "custom", "value": 0.3, "frozen": False},
        },
        {
            "gate": "RY",
            "wires": [1],
            "param": {"init": "custom", "value": -0.2, "frozen": False},
        },
        {"gate": "CNOT", "wires": [0, 1]},
    ]
    qnn_sum_z, params, _ = build_qnn_from_spec(
        DEVICE_NAME, 4, 0.0, spec, readout="sum_z"
    )
    qnn_avg_x, _, _ = build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec, readout="avg_x")
    emb = _sample_embedding(4)

    out_sum_z = qnn_sum_z(emb, params, torch.tensor(0.0))
    out_avg_x = qnn_avg_x(emb, params, torch.tensor(0.0))
    assert torch.allclose(out_sum_z, out_avg_x, atol=1e-8)


def test_unknown_readout_raises():
    spec = [{"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}}]
    with pytest.raises(ValueError, match="readout"):
        build_qnn_from_spec(DEVICE_NAME, 4, 0.0, spec, readout="bogus")


def test_twirled_spec_is_p4m_invariant():
    """Wrapping ANY spec in twirled=True must make it exactly
    p4m-equivariant, regardless of the spec's own content — here applied
    to config1's (normally non-equivariant) inner pattern."""
    spec = architecture_to_spec("config1", 8, 2)
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, 0.0, spec, twirled=True)
    is_invariant, deviation = check_p4m_invariance(
        qnn, params, img_size=16, n_samples=2
    )
    assert is_invariant
    assert deviation < 1e-6


def test_untwirled_spec_is_not_p4m_invariant():
    spec = architecture_to_spec("config1", 8, 2)
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, 0.0, spec, twirled=False)
    is_invariant, _deviation = check_p4m_invariance(
        qnn, params, img_size=16, n_samples=2
    )
    assert not is_invariant
