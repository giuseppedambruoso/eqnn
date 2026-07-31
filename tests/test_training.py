import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_encoding import embedding_unitary
from src.qnn import ARCHITECTURES, architecture_param_names, create_qnn
from src.train import train_one_epoch

DEVICE_NAME = "default.qubit"


def _train_a_few_steps(architecture: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Trains a circuit for a few steps and returns (initial_params,
    final_params). Uses the same num_qubits=8/reps=2 as the project's
    actual defaults (src/config/config.yaml) rather than a shrunk toy
    circuit: with diff_method="backprop" + batched execution (see
    src.train.execute_batch), even config8/config9 (18 trainable
    parameters, the most expensive) run a full step in well under a
    second, so there's no real speed reason to test a different, smaller
    circuit than what's actually used — and doing so isn't free: config4
    (twirled) at a shrunk num_qubits=4/reps=1 turned out to have an
    accidental *total* zero-gradient degeneracy (not just its documented
    frozen first qubit), an artifact of that specific tiny size, not a
    real property of the architecture.

    float64 matters here too: an analytically *exact* zero gradient (e.g.
    config3/config4's frozen first qubit per rep, see
    test_config3_config4_first_qubit_stays_frozen) only reliably rounds to
    ~1e-18 at float64 precision — at the default float32, backprop's
    specific rounding path for a near-zero gradient can land around ~1e-10
    instead, small but large enough for 3 Adam steps to move a "frozen"
    parameter well past a reasonable tolerance.
    """
    torch.manual_seed(0)
    num_qubits, reps = 8, 2
    num_images, num_steps = 4, 3
    img_side = 2 ** (num_qubits // 2)

    images = torch.rand(num_images, img_side, img_side, dtype=torch.float64)
    for i in range(num_images):
        images[i] = images[i] / torch.linalg.norm(images[i].reshape(-1))
    embeddings = torch.stack([embedding_unitary(img) for img in images])
    labels = torch.tensor([0.0, 1.0] * ((num_images + 1) // 2))[:num_images]
    loader = DataLoader(TensorDataset(embeddings, labels), batch_size=num_images)

    qnn = create_qnn(DEVICE_NAME, num_qubits, reps, architecture)
    names = architecture_param_names(architecture, num_qubits, reps)
    params = (
        torch.empty(len(names), dtype=torch.float64)
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    initial_params = params.detach().clone()

    opt = torch.optim.Adam([params], lr=0.1)
    for _ in range(num_steps):
        train_one_epoch(loader, qnn, opt, torch.device("cpu"), params)

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
