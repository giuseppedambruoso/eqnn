# train.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import pennylane as qml
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
    """
    target_vector = torch.zeros(2, dtype=prediction.dtype, device=prediction.device)
    target_vector[target] = 1.0
    prediction = prediction.squeeze()
    loss = F.binary_cross_entropy_with_logits(prediction, target_vector)
    return loss


def execute_batch(
    qnn: qml.QNode,
    batch_images: torch.Tensor,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """
    Executes the pre-initialized QNN on a batch of images.
    """
    batch_images = batch_images.to(dev)

    batch_predictions = []
    for image in batch_images:
        output = qnn(image, params, phi)
        batch_predictions.append(torch.stack(output))

    return torch.stack(batch_predictions)


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    val_loader_aug: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    dev: str,
    seed: int,
    N: int,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], list[float], list[float], list[float], list[float]]:

    dev = torch.device(dev)
    if verbose:
        logger.info(f"Using device: {dev}")
        logger.info("Starting QNN training and validation...")

    # Initialize QNN
    qnn = create_qnn(device, non_equivariance, p_err)
    if verbose:
        logger.info("QNode initialized successfully.")

    # --- REPRODUCIBILITY FIX: Use a seeded generator for parameter initialization ---
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    params = torch.empty(8, device=dev).uniform_(-0.1, 0.1, generator=g)
    params.requires_grad_()
    phi = torch.empty(1, device=dev).uniform_(-0.1, 0.1, generator=g)
    phi.requires_grad_()
    # -----------------------------------------------------------------------------

    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99)) # Added phi to optimizer!

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    val_aug_loss_history = []  # --- NEW ---
    val_aug_acc_history = []   # --- NEW ---

    best_val_acc = -1.0

    pbar = tqdm(range(epochs), desc="Epoch") if verbose else range(epochs)

    for epoch in pbar:
        # --- TRAINING ---
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_images, batch_labels in train_loader:
            batch_labels = batch_labels.to(dev)
            opt.zero_grad()
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += ((torch.argmax(batch_predictions.squeeze(), 1) == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

        epoch_train_loss = total_loss / (total_samples + 1e-8)
        epoch_train_acc = total_correct / (total_samples + 1e-8)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        # --- VALIDATION (Standard) ---
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # We don't need gradients for validation
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_labels = batch_labels.to(dev)
                batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
                loss = loss_function(batch_predictions, batch_labels)
                total_loss += loss.item() * batch_labels.size(0)
                total_correct += ((torch.argmax(batch_predictions.squeeze(), 1) == batch_labels).sum().item())
                total_samples += batch_labels.size(0)

        epoch_val_loss = total_loss / (total_samples + 1e-8)
        epoch_val_acc = total_correct / (total_samples + 1e-8)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        # --- VALIDATION (Augmented) ---
        total_loss_aug = 0.0
        total_correct_aug = 0
        total_samples_aug = 0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader_aug:
                batch_labels = batch_labels.to(dev)
                batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
                loss = loss_function(batch_predictions, batch_labels)
                total_loss_aug += loss.item() * batch_labels.size(0)
                total_correct_aug += ((torch.argmax(batch_predictions.squeeze(), 1) == batch_labels).sum().item())
                total_samples_aug += batch_labels.size(0)

        epoch_val_aug_loss = total_loss_aug / (total_samples_aug + 1e-8)
        epoch_val_aug_acc = total_correct_aug / (total_samples_aug + 1e-8)
        val_aug_loss_history.append(epoch_val_aug_loss)
        val_aug_acc_history.append(epoch_val_aug_acc)

        # --- Check performance and save best model ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

            try:
                job_dir = HydraConfig.get().runtime.output_dir
                os.makedirs(job_dir, exist_ok=True)
                model_path = os.path.join(job_dir, "best_model.pt")

                torch.save({
                    'epoch': epoch,
                    'params': params.detach().cpu(),
                    'phi': phi.detach().cpu(),
                    'val_acc': best_val_acc,
                    'val_aug_acc': epoch_val_aug_acc
                }, model_path)
            except ValueError:
                # Fallback if not running strictly through Hydra sweeping
                pass

    max_val_acc = max(val_acc_history)
    max_val_acc_idx = val_acc_history.index(max_val_acc) + 1
    
    # Prendo la augmented acc corrispondente all'epoca con la miglior standard val_acc
    best_aug_acc = val_aug_acc_history[max_val_acc_idx - 1]

    if verbose:
        logger.info(
            f"Training completed. Max val acc: {max_val_acc:.4f} (Epoch {max_val_acc_idx}). Aug acc at that epoch: {best_aug_acc:.4f}"
        )

    try:
        sweep_dir = HydraConfig.get().sweep.dir
        os.makedirs(sweep_dir, exist_ok=True)
        file_path = os.path.join(sweep_dir, "test_accuracies.txt")
        with open(file_path, "a") as f:
            f.write(
                f"Seed: {seed}, Sample size: {N}, Non equivariance: {non_equivariance}, Noise: {p_err}, "
                f"Test Acc: {max_val_acc:.4f}, Aug Test Acc: {best_aug_acc:.4f} (at epoch {max_val_acc_idx})\n"
            )
    except Exception as e:
        logger.warning(f"Could not save test_accuracies.txt: {e}")

    return (
        params,
        phi,
        train_loss_history,
        train_acc_history,
        val_loss_history,
        val_acc_history,
        val_aug_loss_history,  # Added to returns
        val_aug_acc_history    # Added to returns
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
    dev = torch.device(dev)

    # Initialize QNN
    qnn = create_qnn(device, non_equivariance, p_err)

    # Note: If this function needs explicit reproducibility across runs, you should pass a seed here too.
    params = torch.empty(8, device=dev).uniform_(-0.1, 0.1)
    params.requires_grad_()
    phi = torch.empty(1, device=dev).uniform_(-0.1, 0.1)
    phi.requires_grad_()

    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99)) # Added phi here too

    opt.zero_grad()
    predictions = qnn(image, params, phi)
    loss = loss_function_single(predictions, label)
    loss.backward()

    params_grad = params.grad
    if params_grad is not None:
        grad_norm = torch.sqrt(torch.dot(params_grad, params_grad)).item()
    else:
        grad_norm = 0.0

    return grad_norm


def test_loop(
    test_loader: torch.utils.data.DataLoader,
    aug_test_loader: torch.utils.data.DataLoader,
    device: str,
    dev: str,
    params: torch.Tensor,
    phi: torch.Tensor,
    N: int,
    non_equivariance: Literal[0, 1, 2],
    p_err: float,
    verbose: bool = False,
) -> tuple[float, float]:

    dev = torch.device(dev)
    if verbose:
        logger.info(f"Using device: {dev}")
        logger.info("Starting final QNN testing...")

    qnn = create_qnn(device, non_equivariance, p_err)
    
    # Ensure params are on correct device and don't require grads for testing
    params = params.to(dev)
    phi = phi.to(dev)

    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_labels = batch_labels.to(dev)
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            total_correct += ((torch.argmax(batch_predictions.squeeze(), 1) == batch_labels).sum().item())
            total_samples += batch_labels.size(0)
    acc = total_correct / (total_samples + 1e-8)

    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_images, batch_labels in aug_test_loader:
            batch_labels = batch_labels.to(dev)
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            total_correct += ((torch.argmax(batch_predictions.squeeze(), 1) == batch_labels).sum().item())
            total_samples += batch_labels.size(0)
    aug_acc = total_correct / (total_samples + 1e-8)

    return acc, aug_acc


def load_model(file_path, device='cpu'):
    """
    Loads a PyTorch .pt file and extracts its weights.
    """
    loaded_data = torch.load(file_path, map_location=device, weights_only=False)

    # Support dictionary-based checkpoints (which we are saving in train_loop!)
    if isinstance(loaded_data, dict):
        if 'params' in loaded_data and 'phi' in loaded_data:
            return loaded_data['params'], loaded_data['phi']
        elif 'model_state_dict' in loaded_data:
            state_dict = loaded_data['model_state_dict']
        else:
            state_dict = loaded_data
    elif isinstance(loaded_data, torch.nn.Module):
        state_dict = loaded_data.state_dict()
    else:
        raise ValueError("Unrecognized file format.")

    # Fallback to index-based extraction if it's a raw state_dict
    all_tensors = list(state_dict.values())
    if len(all_tensors) < 2:
        raise ValueError("The model file must contain at least two parameter tensors.")

    return all_tensors[0], all_tensors[1]
