import pytest
import torch

from src.data_encoding import embedding_unitary
from src.qnn import create_qnn

DEVICE_NAME = "default.qubit"
P_ERR = 0.0

@pytest.mark.parametrize("img_idx", range(10))
def test_p4m_equivariance_twirled(device_and_tensors, img_idx):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_node = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=True, twirling=True)

    img = test_images[img_idx]
    img_flip = torch.flip(img, dims=[-1])

    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    assert torch.allclose(out_orig, out_flip, atol=1e-2)

@pytest.mark.parametrize("img_idx", range(10))
def test_xy_yy_unitary_compilation(device_and_tensors, img_idx):
    torch_device, params, phi, test_images, num_qubits, reps = device_and_tensors

    qnn_cnot = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=False, twirling=False)
    qnn_xy_yy = create_qnn(DEVICE_NAME, num_qubits, P_ERR, reps, equivariance=True, twirling=False)

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
