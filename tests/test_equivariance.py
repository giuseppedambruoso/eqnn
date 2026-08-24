import pytest
import torch

from src.ansatz_builder import check_p4m_invariance
from src.data_encoding import embedding_unitary
from src.qnn import ARCHITECTURES, architecture_param_names, create_qnn

DEVICE_NAME = "default.qubit"

EQUIVARIANT_CONFIGS = ["config2", "config4", "config5", "config6", "config8"]
NON_EQUIVARIANT_CONFIGS = ["config1", "config3", "config7", "config9"]


def _params_for(architecture: str, num_qubits: int, reps: int) -> torch.Tensor:
    torch.manual_seed(0)
    names = architecture_param_names(architecture, num_qubits, reps)
    return torch.empty(len(names)).uniform_(-0.1, 0.1)


@pytest.mark.parametrize("architecture", EQUIVARIANT_CONFIGS)
def test_p4m_equivariance(device_and_tensors, architecture):
    """config2/4/5 (twirled) and config6/8 (D4-generator-commuting by
    construction) must be p4m-equivariant. Uses check_p4m_invariance (max
    deviation over several random images and both flips + transpose)
    rather than a single fixed image/transform: a pointwise comparison on
    one specific (image, transform) pair can coincidentally land near zero
    even for a genuinely non-equivariant circuit, which made an earlier,
    less aggregated version of this test flaky as more architectures were
    added — see src.ansatz_builder.check_p4m_invariance's docstring for the
    ~1e-16 (invariant) vs ~1e-3+ (not) separation this relies on.
    """
    _torch_device, _params, _test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, reps, architecture)
    params = _params_for(architecture, num_qubits, reps)

    is_invariant, deviation = check_p4m_invariance(
        qnn_node, params, img_size=16, n_samples=5
    )
    assert is_invariant, f"max deviation {deviation} for {architecture}"


@pytest.mark.parametrize("architecture", NON_EQUIVARIANT_CONFIGS)
def test_not_p4m_equivariant(device_and_tensors, architecture):
    """config1/config3 have no twirling, and config7/config9 deliberately
    misalign their generators with the image symmetry (axis-scrambled
    column register) — all four must generically NOT be p4m-equivariant.
    See test_p4m_equivariance's docstring for why this uses
    check_p4m_invariance rather than a single fixed (image, transform)
    comparison.
    """
    _torch_device, _params, _test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, reps, architecture)
    params = _params_for(architecture, num_qubits, reps)

    is_invariant, deviation = check_p4m_invariance(
        qnn_node, params, img_size=16, n_samples=5
    )
    assert not is_invariant, f"max deviation {deviation} for {architecture}"


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_all_architectures_run(device_and_tensors, architecture):
    _torch_device, _params, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, reps, architecture)
    params = _params_for(architecture, num_qubits, reps)

    out = qnn_node(embedding_unitary(test_images[0]), params)

    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_invalid_architecture_raises():
    with pytest.raises(ValueError):
        create_qnn(DEVICE_NAME, 8, 2, architecture="config99")


@pytest.mark.parametrize("architecture", ["config6", "config7", "config8", "config9"])
def test_paper_architectures_require_8_qubits(architecture):
    with pytest.raises(ValueError, match="num_qubits"):
        create_qnn(DEVICE_NAME, 4, 2, architecture=architecture)


def test_readout_override_changes_paper_architecture_output():
    """readout is only meaningful for config6-config9 (see create_qnn's
    docstring) — passing "x0_xhalf" instead of the default ("avg_x") must
    actually change the measured output, proving the override takes
    effect rather than being silently ignored."""
    num_qubits, reps = 8, 2
    params = _params_for("config6", num_qubits, reps)
    emb = embedding_unitary(torch.rand(16, 16))

    qnn_default = create_qnn(DEVICE_NAME, num_qubits, reps, "config6")
    qnn_x0_xhalf = create_qnn(
        DEVICE_NAME, num_qubits, reps, "config6", readout="x0_xhalf"
    )

    out_default = qnn_default(emb, params)
    out_x0_xhalf = qnn_x0_xhalf(emb, params)
    assert not torch.allclose(out_default, out_x0_xhalf)


def test_readout_override_is_ignored_for_uniform_architectures():
    """config1-config5's measurement is hardcoded and doesn't go through
    the readout mechanism at all — passing readout= for one of them must
    not raise or change anything."""
    num_qubits, reps = 8, 2
    params = _params_for("config1", num_qubits, reps)
    emb = embedding_unitary(torch.rand(16, 16))

    qnn_default = create_qnn(DEVICE_NAME, num_qubits, reps, "config1")
    qnn_with_readout = create_qnn(
        DEVICE_NAME, num_qubits, reps, "config1", readout="x0_xhalf"
    )
    assert torch.allclose(qnn_default(emb, params), qnn_with_readout(emb, params))
