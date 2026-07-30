import pytest
import torch

from src.data_encoding import embedding_unitary
from src.qnn import create_qnn

DEVICE_NAME = "default.qubit"
P_ERR = 0.0


@pytest.mark.parametrize("img_idx", range(10))
def test_p4m_equivariance_twirled(device_and_tensors, img_idx):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(
        DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=True, twirling=True
    )

    img = test_images[img_idx]
    img_flip = torch.flip(img, dims=[-1])

    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    assert torch.allclose(out_orig, out_flip, atol=1e-2)


@pytest.mark.parametrize("img_idx", range(10))
def test_xy_yy_unitary_compilation(device_and_tensors, img_idx):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_cnot = create_qnn(
        DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=False, twirling=False
    )
    qnn_xy_yy = create_qnn(
        DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=True, twirling=False
    )

    img = test_images[img_idx]
    out_cnot = qnn_cnot(embedding_unitary(img), params, phi)
    out_xy_yy = qnn_xy_yy(embedding_unitary(img), params, phi)

    assert torch.allclose(out_cnot, out_xy_yy, atol=1e-3)


@pytest.mark.parametrize("img_idx", range(10))
def test_cross_edge_removal_difference(device_and_tensors, img_idx):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_full = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, True, False, False)
    qnn_no_34 = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, True, False, True)

    img = test_images[img_idx]
    emb = embedding_unitary(img)

    out_full = qnn_full(emb, params, phi)
    out_no_34 = qnn_no_34(emb, params, phi)

    assert abs((out_full - out_no_34).item()) > 1e-5


@pytest.mark.parametrize("img_idx", range(10))
def test_p4m_equivariance_rx_frozen_ryy(device_and_tensors, img_idx):
    """The RX + frozen-RYY/RYYYY architecture must be p4m-equivariant too,
    same as the compiled_cnot/twirled architecture in
    test_p4m_equivariance_twirled — just with a different entangler."""
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(
        DEVICE_NAME,
        num_qubits,
        P_ERR,
        reps,
        equivariance=True,
        twirling=True,
        rotation_gate="RX",
        entangler="frozen_ryy",
    )

    img = test_images[img_idx]
    img_flip = torch.flip(img, dims=[-1])

    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    assert torch.allclose(out_orig, out_flip, atol=1e-2)


@pytest.mark.parametrize("equivariance", [False, True])
@pytest.mark.parametrize("twirling", [False, True])
@pytest.mark.parametrize("rotation_gate", ["RY", "RX"])
@pytest.mark.parametrize("entangler", ["cnot", "frozen_ryy"])
def test_all_rotation_entangler_combinations_run(
    device_and_tensors, equivariance, twirling, rotation_gate, entangler
):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(
        DEVICE_NAME,
        num_qubits,
        P_ERR,
        reps,
        equivariance=equivariance,
        twirling=twirling,
        rotation_gate=rotation_gate,
        entangler=entangler,
    )

    out = qnn_node(embedding_unitary(test_images[0]), params, phi)

    assert torch.isfinite(out)
    assert -1.0 - 1e-6 <= out.item() <= 1.0 + 1e-6


def test_invalid_rotation_gate_raises():
    with pytest.raises(ValueError):
        create_qnn(DEVICE_NAME, 8, P_ERR, 2, rotation_gate="RZ")


def test_invalid_entangler_raises():
    with pytest.raises(ValueError):
        create_qnn(DEVICE_NAME, 8, P_ERR, 2, entangler="toffoli")
