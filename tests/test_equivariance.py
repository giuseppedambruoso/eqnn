import pytest
import torch

from src.data_encoding import embedding_unitary
from src.qnn import ARCHITECTURES, create_qnn

DEVICE_NAME = "default.qubit"
P_ERR = 0.0

EQUIVARIANT_CONFIGS = ["config2", "config4", "config5"]
NON_EQUIVARIANT_CONFIGS = ["config1", "config3"]


@pytest.mark.parametrize("architecture", EQUIVARIANT_CONFIGS)
@pytest.mark.parametrize("img_idx", range(10))
def test_p4m_equivariance(device_and_tensors, architecture, img_idx):
    """config2/config4/config5 are twirled, so they must be p4m-equivariant:
    flipping the input image must not change the circuit's output."""
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, architecture)

    img = test_images[img_idx]
    img_flip = torch.flip(img, dims=[-1])

    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    assert torch.allclose(out_orig, out_flip, atol=1e-2)


@pytest.mark.parametrize("architecture", NON_EQUIVARIANT_CONFIGS)
@pytest.mark.parametrize("img_idx", range(10))
def test_not_p4m_equivariant(device_and_tensors, architecture, img_idx):
    """config1/config3 have no twirling, so generically they must NOT be
    p4m-equivariant: transposing (swapping x/y) the input image should
    change the output.

    Note: a plain horizontal/vertical flip is NOT a good probe here — RX
    commutes exactly with the X gates that a flip induces on the embedding
    (verified empirically: config3 gives an *exact* 0.0 difference under
    horizontal flip, for every fixture image), so that transform alone
    doesn't distinguish "equivariant" from "not". Transpose reliably does,
    with the smallest observed difference across all 10 fixture images
    being ~7e-4 for both config1 and config3.
    """
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, architecture)

    img = test_images[img_idx]
    img_transposed = img.transpose(-1, -2)

    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_transposed = qnn_node(embedding_unitary(img_transposed), params, phi)

    assert abs((out_orig - out_transposed).item()) > 1e-4


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_all_architectures_run(device_and_tensors, architecture):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, architecture)

    out = qnn_node(embedding_unitary(test_images[0]), params, phi)

    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_invalid_architecture_raises():
    with pytest.raises(ValueError):
        create_qnn(DEVICE_NAME, 8, P_ERR, 2, architecture="config6")
