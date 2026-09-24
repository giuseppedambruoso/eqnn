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
    src.train.execute_batch), even the twirled config10 (8 circuit
    evaluations per image) runs a full step in well under a second, so
    there's no speed reason to test a smaller circuit than the one actually
    used. float64 keeps near-zero gradients from being rounded into
    spurious parameter updates.
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
    """Params must actually move during training for every one of the 3
    architectures — a sanity check that gradients flow end-to-end
    (embedding -> QNN -> loss -> optimizer), independent of the
    equivariance checks in test_equivariance.py.
    """
    initial_params, final_params = _train_a_few_steps(architecture)

    assert not torch.allclose(final_params, initial_params, atol=1e-6)
    assert torch.isfinite(final_params).all()
