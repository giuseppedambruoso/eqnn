# train.py
import logging
import os
import time
from typing import Any, Literal

import torch
from hydra.core.hydra_config import HydraConfig
from qnn import create_qnn
from torch.nn import functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def loss_function(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_vectors = torch.zeros(
        targets.shape[0], 2, dtype=predictions.dtype, device=predictions.device
    )
    target_vectors[targets == 0, 0] = 1.0
    target_vectors[targets == 1, 1] = 1.0
    predictions = predictions.squeeze()
    loss = F.binary_cross_entropy_with_logits(predictions, target_vectors)
    return loss


def loss_function_single(prediction: torch.Tensor, target: int) -> torch.Tensor:
    """
    Computes binary cross-entropy loss for a single prediction vs single label.

    Args:
        prediction: Tensor of shape (2,) - raw logits for the two classes
        target: int - 0 or 1
    """
    # Create one-hot vector for target
    target_vector = torch.zeros(2, dtype=prediction.dtype, device=prediction.device)
    target_vector[target] = 1.0

    # Ensure prediction has correct shape
    prediction = prediction.squeeze()

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(prediction, target_vector)
    return loss


def create_and_execute_qnn(
    image: torch.Tensor,
    device: str,
    params: torch.Tensor,
    phi: torch.Tensor,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
) -> torch.Tensor:
    qnn = create_qnn(image, device, non_equivariance, p_err)
    output = qnn(params, phi)
    return torch.stack(output)


def execute_batch(
    batch_images: torch.Tensor,
    device: str,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
) -> torch.Tensor:

    batch_images = batch_images.to(dev)

    batch_predictions = [
        create_and_execute_qnn(image, device, params, phi, non_equivariance, p_err)
        for image in batch_images
    ]
    return torch.stack(batch_predictions)


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    dev: str,
    seed: int,
    N: int,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[Any], list[Any], list[Any], list[Any]]:

    dev = torch.device(dev)
    if verbose:
        logger.info(f"Using device: {dev}")
        logger.info("Starting QNN training and validation...")

    params = torch.empty(8, device=dev).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1, device=dev).uniform_(-0.1, 0.1)
    phi.requires_grad_()

    opt = torch.optim.Adam([params], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    pbar = tqdm(range(epochs), desc="Epoch") if verbose else range(epochs)

    for epoch in pbar:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_images, batch_labels in train_loader:
            batch_labels = batch_labels.to(dev)
            opt.zero_grad()
            batch_predictions = execute_batch(
                batch_images, device, dev, params, phi, non_equivariance, p_err
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

        epoch_train_loss = total_loss / (total_samples + 1e-8)
        epoch_train_acc = total_correct / (total_samples + 1e-8)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_images, batch_labels in val_loader:
            batch_labels = batch_labels.to(dev)
            opt.zero_grad()
            batch_predictions = execute_batch(
                batch_images, device, dev, params, phi, non_equivariance, p_err
            )
            loss = loss_function(batch_predictions, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (
                (torch.argmax(batch_predictions.squeeze(), 1) == batch_labels)
                .sum()
                .item()
            )
            total_samples += batch_labels.size(0)

        epoch_val_loss = total_loss / (total_samples + 1e-8)
        epoch_val_acc = total_correct / (total_samples + 1e-8)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        if verbose:
            pbar.set_postfix(
                {
                    "train loss": f"{epoch_train_loss:.4f}",
                    "train acc": f"{epoch_train_acc:.3f}",
                    "val loss": f"{epoch_val_loss:.4f}",
                    "val acc": f"{epoch_val_acc:.3f}",
                }
            )

        max_val_acc = max(val_acc_history)
        max_val_acc_idx = val_acc_history.index(max_val_acc) + 1

    logger.info(
        f"Training completed with maximum val acc equal to {max_val_acc}, found at epoch {max_val_acc_idx}"
    )

    sweep_dir = HydraConfig.get().sweep.dir
    os.makedirs(sweep_dir, exist_ok=True)
    file_path = os.path.join(sweep_dir, "test_accuracies.txt")
    with open(file_path, "a") as f:
        f.write(
            f"Seed: {seed}, Sample size: {N}, Non equivariance: {non_equivariance}, Noise: {p_err}, Test Accuracy: {max_val_acc:.4f} (at epoch {max_val_acc_idx})\n"
        )

    return (
        params,
        phi,
        train_loss_history,
        train_acc_history,
        val_loss_history,
        val_acc_history,
    )


def train_loop_in(
    image: torch.Tensor,
    label: torch.Tensor,
    learning_rate: float,
    device: str,
    dev: str,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
) -> float:
    start_time = time.time()
    dev = torch.device(dev)

    params = torch.empty(8, device=dev).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1, device=dev).uniform_(-0.1, 0.1)
    phi.requires_grad_()

    opt = torch.optim.Adam([params], lr=learning_rate, betas=(0.5, 0.99))

    opt.zero_grad()
    predictions = create_and_execute_qnn(
        image, device, params, phi, non_equivariance, p_err
    )
    loss = loss_function_single(predictions, label)
    loss.backward()

    params_grad = params.grad
    grad_norm = torch.sqrt(torch.dot(params_grad, params_grad)).item()
    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"Gradient norm after single training step: {grad_norm:.6f}, time: {duration:.6f} seconds"
    )

    return grad_norm
