import pytest

from src.ansatz_builder import (
    build_qnn_from_spec,
    check_p4m_invariance,
    param_labels,
    validate_spec,
)
from src.paper_ansatzes import PAPER_ANSATZ_PARAMETER_NAMES, paper_architecture_spec

DEVICE_NAME = "default.qubit"


@pytest.mark.parametrize("paper_ansatz", ["6", "18"])
@pytest.mark.parametrize("symmetry", ["equivariant", "nonequivariant"])
def test_paper_architecture_spec_param_count(paper_ansatz, symmetry):
    """config6-config9 have a fixed, tied parameter budget: 2 * per-block
    (6 total for paper_ansatz="6", 18 for "18"), regardless of the 5-block
    schedule — see the module docstring."""
    spec = paper_architecture_spec(paper_ansatz, symmetry, num_qubits=8)
    validate_spec(spec, num_qubits=8)
    expected = 2 * len(PAPER_ANSATZ_PARAMETER_NAMES[paper_ansatz])
    assert len(param_labels(spec)) == expected


@pytest.mark.parametrize("num_qubits", [4, 6, 10])
def test_paper_architecture_spec_requires_8_qubits(num_qubits):
    with pytest.raises(ValueError, match="num_qubits"):
        paper_architecture_spec("6", "equivariant", num_qubits)


def test_unknown_paper_ansatz_raises():
    with pytest.raises(ValueError, match="paper_ansatz"):
        paper_architecture_spec("42", "equivariant", 8)


def test_unknown_symmetry_raises():
    with pytest.raises(ValueError, match="symmetry"):
        paper_architecture_spec("6", "diagonal", 8)


@pytest.mark.parametrize("readout", ["avg_x", "x0_xhalf"])
@pytest.mark.parametrize("paper_ansatz", ["6", "18"])
def test_equivariant_paper_ansatz_is_p4m_invariant(paper_ansatz, readout):
    """config6/config8: the generator-commuting design must be exactly
    p4m-equivariant WITHOUT any explicit group-twirling — "avg_x" (the
    default create_qnn now uses, matching config1-config5's effective
    measurement) and "x0_xhalf" (the alternative) must both hold."""
    spec = paper_architecture_spec(paper_ansatz, "equivariant", 8)
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec, readout=readout)
    is_invariant, deviation = check_p4m_invariance(
        qnn, params, img_size=16, n_samples=2
    )
    assert is_invariant
    assert deviation < 1e-6


@pytest.mark.parametrize("readout", ["avg_x", "x0_xhalf"])
@pytest.mark.parametrize("paper_ansatz", ["6", "18"])
def test_nonequivariant_paper_ansatz_is_not_p4m_invariant(paper_ansatz, readout):
    """config7/config9: the axis-scrambled column register must generically
    break p4m-equivariance, regardless of readout choice."""
    spec = paper_architecture_spec(paper_ansatz, "nonequivariant", 8)
    qnn, params, _ = build_qnn_from_spec(DEVICE_NAME, 8, spec, readout=readout)
    is_invariant, _deviation = check_p4m_invariance(
        qnn, params, img_size=16, n_samples=2
    )
    assert not is_invariant
