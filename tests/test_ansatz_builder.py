import pytest
import torch

from src.ansatz_builder import (
    build_qnn_from_spec,
    param_labels,
    validate_spec,
)
from src.data_encoding import embedding_unitary

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
