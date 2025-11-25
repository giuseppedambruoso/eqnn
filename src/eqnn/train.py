# train.py
import logging
from typing import Any, Literal

import torch
from qnn import create_qnn
from torch.nn import functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def loss_function(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_vectors = torch.zeros(targets.shape[0], 2, dtype=predictions.dtype)
    target_vectors[targets == 0, 0] = 1.0
    target_vectors[targets == 1, 1] = 1.0
    predictions = predictions.squeeze()
    loss = F.binary_cross_entropy_with_logits(predictions, target_vectors)
    return loss


def create_and_execute_qnn(
    image: torch.Tensor,
    device: str,
    params: torch.Tensor,
    phi: torch.Tensor,
    non_equivariance: Literal[0, 1, 2],
) -> torch.Tensor:
    qnn = create_qnn(image, device, non_equivariance)
    output = qnn(params, phi)
    return torch.stack(output)


def execute_batch(
    batch_images: torch.Tensor,
    device: str,
    params: torch.Tensor,
    phi: torch.Tensor,
    non_equivariance: Literal[0, 1, 2],
) -> torch.Tensor:
    batch_predictions = [
        create_and_execute_qnn(image, device, params, phi, non_equivariance)
        for image in batch_images
    ]
    return torch.stack(batch_predictions)


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    non_equivariance: Literal[0, 1, 2],
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[Any], list[Any]]:
    if verbose:
        logger.info("Starting QNN training...")

    params = torch.empty(31).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1).uniform_(-0.1, 0.1)
    phi.requires_grad_()

    opt = torch.optim.Adam([params], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_history = []
    train_acc_history = []

    pbar = tqdm(range(epochs), desc="Epoch") if verbose else range(epochs)

    for epoch in pbar:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_images, batch_labels in train_loader:
            opt.zero_grad()
            batch_predictions = execute_batch(
                batch_images, device, params, phi, non_equivariance
            )
            loss = loss_function(batch_predictions, batch_labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (
                (torch.argmax(batch_predictions.squeeze(), 1) == batch_labels)
                .sum()
                .item()
            )
            total_samples += batch_labels.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        if verbose:
            pbar.set_postfix(
                {"train loss": f"{epoch_loss:.4f}", "train acc": f"{epoch_acc:.3f}"}
            )

    logger.info("Training completed.")
    return params, phi, train_loss_history, train_acc_history
