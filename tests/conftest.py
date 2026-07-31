import pytest
import torch


@pytest.fixture(scope="module")
def device_and_tensors():
    """Prepares randomized data and model parameters to be shared across tests."""
    torch.manual_seed(42)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_qubits = 8
    reps = 2
    num_images = 10

    params = (
        torch.empty(num_qubits * reps, device=torch_device)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )

    test_images = torch.rand(num_images, 16, 16, device=torch_device)
    for i in range(num_images):
        test_images[i] = test_images[i] / torch.linalg.norm(test_images[i].reshape(-1))

    return torch_device, params, test_images, num_qubits, reps
