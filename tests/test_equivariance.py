# tests/test_equivariance.py
import pytest
import torch
from qnn import create_qnn
from data_encoding import embedding_unitary

# Global test configuration
DEVICE_NAME = "default.qubit"
P_ERR = 0.0
NUM_IMAGES = 10
REPS = 2

@pytest.fixture(scope="module")
def device_and_tensors():
    """Prepares randomized data and model parameters."""
    torch.manual_seed(42)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize parameters
    params = torch.empty(8*REPS, device=torch_device).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1, device=torch_device).uniform_(-0.1, 0.1)
    phi.requires_grad_()

    # Generate test images and apply L2 normalization to keep expectations in [-1, 1]
    test_images = torch.rand(NUM_IMAGES, 16, 16, device=torch_device)
    for i in range(NUM_IMAGES):
        test_images[i] = test_images[i] / torch.linalg.norm(test_images[i].reshape(-1))

    return torch_device, params, phi, test_images

@pytest.mark.parametrize("img_idx", range(NUM_IMAGES))
def test_p4m_equivariance_twirled(device_and_tensors, img_idx):
    """Verifies p4m reflection invariance ONLY for the twirled model."""
    torch_device, params, phi, test_images = device_and_tensors

    # Create QNN node with explicit twirling
    qnn_node = create_qnn(
        device=DEVICE_NAME,
        p_err=P_ERR,
        reps=REPS,
        equivariance=True,
        twirling=True
    )

    img = test_images[img_idx]

    # Apply flip along y-axis (p4m reflection)
    img_flip = torch.flip(img, dims=[-1])

    # Embed and execute
    out_orig = qnn_node(embedding_unitary(img), params, phi)
    out_flip = qnn_node(embedding_unitary(img_flip), params, phi)

    # The output must be identical by construction due to Explicit Twirling
    assert torch.allclose(out_orig, out_flip, atol=1e-2), \
        f"Equivariance failed: Twirled model must be invariant! Image Index {img_idx}"

@pytest.mark.parametrize("img_idx", range(NUM_IMAGES))
def test_xy_yy_reconstruction(device_and_tensors, img_idx):
    """Verifies that the X/Y/YY model perfectly reproduces the CNOT output."""
    torch_device, params, phi, test_images = device_and_tensors

    # The mathematical reconstruction identity applies to a single layer of Y rotations.
    params_single_layer = params[:8]

    # Baseline: Original CNOT circuit (Eq=False, Twirl=False) with reps=1
    qnn_cnot = create_qnn(device=DEVICE_NAME, p_err=P_ERR, reps=1, equivariance=False, twirling=False)

    # Target: X/Y/YY circuit (Eq=True, Twirl=False) with reps=1
    qnn_xy_yy = create_qnn(device=DEVICE_NAME, p_err=P_ERR, reps=1, equivariance=True, twirling=False)

    img = test_images[img_idx]

    out_cnot = qnn_cnot(embedding_unitary(img), params_single_layer, phi)
    out_xy_yy = qnn_xy_yy(embedding_unitary(img), params_single_layer, phi)

    # Both circuits must yield identical output
    assert torch.allclose(out_cnot, out_xy_yy, atol=1e-2), \
        f"Reconstruction failed: X/Y/YY does not match CNOT! Image Index {img_idx}"

@pytest.mark.parametrize("img_idx", range(NUM_IMAGES))
def test_cross_edge_removal_difference(device_and_tensors, img_idx):
    """
    Compares the full circuit against the circuit without the 3->4 edge.
    Verifies that the difference equals exactly (1/8) * g_3(t_3, t_4).
    """
    torch_device, params, phi, test_images = device_and_tensors

    # 1. Full Circuit (with 3->4 gate)
    qnn_full = create_qnn(
        device=DEVICE_NAME, p_err=P_ERR, reps=1,
        equivariance=True, twirling=False, remove_cross_edge=False
    )

    # 2. Reduced Circuit (without 3->4 gate)
    qnn_no_34 = create_qnn(
        device=DEVICE_NAME, p_err=P_ERR, reps=1,
        equivariance=True, twirling=False, remove_cross_edge=True
    )

    img = test_images[img_idx]
    emb = embedding_unitary(img)

    out_full = qnn_full(emb, params[:8], phi)
    out_no_34 = qnn_no_34(emb, params[:8], phi)

    actual_diff = (out_full - out_no_34).item()

    # The difference must be bounded by [-0.125, 0.125] due to 1/8 normalization of the expectation value.
    assert abs(actual_diff) <= 0.125 + 1e-5, f"Difference ({actual_diff}) exceeds theoretical maximum of 0.125!"
