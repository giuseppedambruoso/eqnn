import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import matplotlib.pyplot as plt
import pennylane as qml
import torch
from hydra.core.hydra_config import HydraConfig
from qnn import create_qnn
from torch.nn import functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def loss_function(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes binary cross-entropy loss for a batch of probability predictions vs labels.
    """
    predictions = predictions.squeeze()
    loss = F.binary_cross_entropy(predictions, targets.to(predictions.dtype))
    return loss

def loss_function_single(prediction: torch.Tensor, target: int) -> torch.Tensor:
    prediction = prediction.squeeze()
    target_tensor = torch.tensor(target, dtype=prediction.dtype, device=prediction.device)
    loss = F.binary_cross_entropy(prediction, target_tensor)
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
        output = (1.0 + output)/2
        batch_predictions.append(output)

    # Impila gli scalari per ottenere un tensore 1D per il batch
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
    reps : int,
    p_err: float,
    dataset: str,  # <-- Added dataset parameter
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], list[float], list[float], list[float], list[float]]:

    dev = torch.device(dev)
    if verbose:
        logger.info(f"Using device: {dev}")
        logger.info("Starting QNN training...")

    # Initialize QNN
    qnn = create_qnn(device, non_equivariance, p_err, reps)
    if verbose:
        logger.info("QNode initialized successfully.")

    # --- REPRODUCIBILITY FIX: Use a seeded generator for parameter initialization ---
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    params = torch.empty(8*reps, device=dev).uniform_(-0.1, 0.1, generator=g)
    params.requires_grad_()
    phi = torch.empty(1, device=dev).uniform_(-0.1, 0.1, generator=g)
    phi.requires_grad_()
    # -----------------------------------------------------------------------------

    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_history = []
    train_acc_history = []
    
    # We will just append the final values to these lists to maintain your return signature
    val_loss_history = []
    val_acc_history = []
    val_aug_loss_history = [] 
    val_aug_acc_history = []   

    if verbose:
        try:
            job_idx = HydraConfig.get().job.num
        except Exception:
            job_idx = 0 

        pbar = tqdm(
            range(epochs), 
            desc=f"Job {job_idx} (Eq={non_equivariance})", 
            position=job_idx, 
            leave=True
        )
    else:
        pbar = range(epochs)

    t0 = time.time()

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
            total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

        epoch_train_loss = total_loss / (total_samples + 1e-8)
        epoch_train_acc = total_correct / (total_samples + 1e-8)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

    t1 = time.time()
    training_time = t1 - t0
    epoch_time = training_time / max(epochs, 1)

    if verbose:
        logger.info(f"Training completed in {training_time:.2f}s. Starting validation...")

    # --- VALIDATION (Standard & Augmented Simultaneously at the END of training) ---
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_loss_aug = 0.0
    total_correct_aug = 0
    total_samples_aug = 0

    with torch.no_grad():
        for (batch_images, batch_labels), (batch_images_aug, batch_labels_aug) in zip(val_loader, val_loader_aug):
            batch_labels = batch_labels.to(dev)
            batch_labels_aug = batch_labels_aug.to(dev)

            assert torch.equal(batch_labels, batch_labels_aug), "FATAL: Standard and Augmented validation labels do not match!"

            # --- 1. Standard Prediction ---
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

            # --- 2. Augmented Prediction ---
            batch_predictions_aug = execute_batch(qnn, batch_images_aug, dev, params, phi)
            loss_aug = loss_function(batch_predictions_aug, batch_labels)

            total_loss_aug += loss_aug.item() * batch_labels.size(0)
            total_correct_aug += (((batch_predictions_aug.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples_aug += batch_labels.size(0)

    final_val_loss = total_loss / (total_samples + 1e-8)
    final_val_acc = total_correct / (total_samples + 1e-8)
    val_loss_history.append(final_val_loss)
    val_acc_history.append(final_val_acc)

    final_val_aug_loss = total_loss_aug / (total_samples_aug + 1e-8)
    final_val_aug_acc = total_correct_aug / (total_samples_aug + 1e-8)
    val_aug_loss_history.append(final_val_aug_loss)
    val_aug_acc_history.append(final_val_aug_acc)

    if verbose:
        logger.info(f"Validation completed. Val Acc: {final_val_acc:.4f}, Aug Val Acc: {final_val_aug_acc:.4f}")

    # --- FILE SAVING LOGIC ---
    try:
        job_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        job_dir = os.getcwd() # Fallback if not running strictly through Hydra
    
    os.makedirs(job_dir, exist_ok=True)

    # 1. Save loss_history.csv in the job folder
    loss_csv_path = os.path.join(job_dir, "loss_history.csv")
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for e, l in enumerate(train_loss_history):
            writer.writerow([e, l])

    # 2. Save loss_history.jpg in the job folder
    loss_jpg_path = os.path.join(job_dir, "loss_history.jpg")
    plt.figure()
    plt.plot(range(epochs), train_loss_history, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_jpg_path, bbox_inches='tight')
    plt.close()

    # Save final model state
    model_path = os.path.join(job_dir, "final_model.pt")
    torch.save({
        'epochs_completed': epochs,
        'params': params.detach().cpu(),
        'phi': phi.detach().cpu(),
        'val_acc': final_val_acc,
        'val_aug_acc': final_val_aug_acc
    }, model_path)

    # 3. Accumulate results.txt in the sweep/common directory
    try:
        # If using hydra multi-run, this grabs the parent sweep directory
        sweep_dir = HydraConfig.get().sweep.dir
    except Exception:
        # Fallback to the parent directory of the current job directory
        sweep_dir = os.path.dirname(job_dir)
    
    os.makedirs(sweep_dir, exist_ok=True)
    results_path = os.path.join(sweep_dir, "results.txt")
    
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header if the file is newly created (added "dataset")
        if not file_exists:
            writer.writerow(["dataset", "N", "seed", "p_err", "non_equivariance", "reps",  "val_acc", "val_aug_acc", "training_time", "epoch_time"])
        # Append the results for this specific configuration (added dataset)
        writer.writerow([dataset, N, seed, p_err, non_equivariance, reps, final_val_acc, final_val_aug_acc, training_time, epoch_time])

    return (
        params,
        phi,
        train_loss_history,
        train_acc_history,
        val_loss_history,
        val_acc_history,
        val_aug_loss_history,
        val_aug_acc_history
    )
