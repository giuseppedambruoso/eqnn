import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_encoding import embedding_unitary
from src.qnn import ARCHITECTURES, architecture_param_names, create_qnn
from src.train import train_one_epoch

DEVICE_NAME = "default.qubit"


def _train_a_few_steps(architecture: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Trains a small circuit for a few steps and returns (initial_params,
    final_params). Small on purpose: config2/4/5 are twirled (8x circuit
    evaluations per forward pass), and parameter-shift gradients need ~2
    evaluations per trainable parameter on top of that.

    config6-config9 (paper-style ansatzes) require exactly 8 qubits instead
    of 4, and shared18 (config8/config9) has 18 trainable parameters (36
    parameter-shift evaluations per image) — both make each step
    noticeably more expensive, so those 4 architectures get fewer
    images/epochs. A single step is enough to detect whether gradients
    flow at all, which is all this test checks.
    """
    torch.manual_seed(0)
    is_paper = ARCHITECTURES[architecture]["kind"] == "paper"
    num_qubits, reps = (8, 1) if is_paper else (4, 1)
    num_images = 2 if is_paper else 4
    num_steps = 1 if is_paper else 3
    img_side = 2 ** (num_qubits // 2)

    images = torch.rand(num_images, img_side, img_side)
    for i in range(num_images):
        images[i] = images[i] / torch.linalg.norm(images[i].reshape(-1))
    embeddings = torch.stack([embedding_unitary(img) for img in images])
    labels = torch.tensor([0.0, 1.0] * ((num_images + 1) // 2))[:num_images]
    loader = DataLoader(TensorDataset(embeddings, labels), batch_size=num_images)

    qnn = create_qnn(DEVICE_NAME, num_qubits, 0.0, reps, architecture)
    names = architecture_param_names(architecture, num_qubits, reps)
    params = torch.empty(len(names)).uniform_(-0.1, 0.1).requires_grad_()
    phi = torch.tensor(0.0, requires_grad=False)
    initial_params = params.detach().clone()

    opt = torch.optim.Adam([params], lr=0.1)
    for _ in range(num_steps):
        train_one_epoch(loader, qnn, opt, torch.device("cpu"), params, phi)

    return initial_params, params.detach().clone()


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_training_updates_params(architecture):
    """Params must actually move during training for every one of the 5
    architectures — a sanity check that gradients flow end-to-end
    (embedding -> QNN -> loss -> optimizer), independent of the
    equivariance checks in test_equivariance.py.
    """
    initial_params, final_params = _train_a_few_steps(architecture)

    assert not torch.allclose(final_params, initial_params, atol=1e-6)
    assert torch.isfinite(final_params).all()


def test_config3_config4_first_qubit_stays_frozen():
    """Known, expected behavior: config3/config4 use a frozen RXY entangler
    to fix the exact-zero-gradient issue CNOT/RXX had with RX rotations
    (see ARCHITECTURES' comment in src/qnn.py). RXY only breaks the
    RX/measurement commutation for qubits that play the "Y" role in the
    wires=[i, i+1] convention — the very first qubit in the chain (index 0)
    always plays the "X" role, so it never gets a gradient and its
    parameter stays exactly at its initial value, while every other
    parameter moves normally.
    """
    for architecture in ["config3", "config4"]:
        initial_params, final_params = _train_a_few_steps(architecture)

        assert torch.allclose(final_params[0], initial_params[0], atol=1e-12)
        assert not torch.allclose(final_params[1:], initial_params[1:], atol=1e-6)
