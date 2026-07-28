# train.py
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
import torch
import math
from hydra.core.hydra_config import HydraConfig
from qnn import create_qnn
from torch.nn import functional as F
from tqdm import tqdm

from data_loading import load_mnist_data, load_kaggle_nwpu_data, load_aug_mnist_data

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
    qnn: Any,
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
        
        # Scale expectation [-1, 1] to probability [0, 1]
        output = (1.0 + output) / 2.0
        
        # STRICT CLAMPING: Prevents BCE loss from crashing due to 
        # floating-point arithmetic drift (e.g., 1.0000001 or -0.0000001)
        # We use 1e-7 and 1.0 - 1e-7 to also prevent log(0) NaN errors in BCE
        output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        batch_predictions.append(output)

    # Stack the scalars to obtain a 1D tensor for the batch
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
    equivariance: bool,
    reps: int,
    p_err: float,
    dataset: str, 
    twirling: bool = False,
    remove_cross_edge: bool = False,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], list[float], list[float], list[float], list[float]]:

    dev = torch.device(dev)
    if verbose:
        logger.info(f"Using device: {dev}")
        logger.info("Starting QNN training...")

    # Initialize QNN
    qnn = create_qnn(device, p_err, reps, equivariance, twirling, remove_cross_edge)
    if verbose:
        logger.info("QNode initialized successfully.")

    # Use a seeded generator for parameter initialization
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    params = torch.empty(8*reps, device=dev).uniform_(-0.1, 0.1, generator=g)
    params.requires_grad_()
    phi = torch.tensor(0.0, requires_grad=False)

    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_history = []
    train_acc_history = []
    
    val_loss_history = []
    val_acc_history = []
    val_aug_loss_history = [] 
    val_aug_acc_history = []   
    
    params_history = []

    if verbose:
        try:
            job_idx = HydraConfig.get().job.num
        except Exception:
            job_idx = 0 

        pbar = tqdm(
            range(epochs), 
            desc=f"Job {job_idx} (Eq={equivariance}, Twirl={twirling}, NoCross={remove_cross_edge})", 
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

        # Save parameters and phi at the end of the epoch
        current_params = params.detach().cpu().numpy().tolist()
        current_phi = phi.detach().cpu().item()
        params_history.append([epoch] + current_params + [current_phi])

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

            # 1. Standard Prediction
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

            # 2. Augmented Prediction
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

    # --- CALCULATE CONVERGENCE EPOCH ---
    patience = 5
    min_delta = 1e-4
    best_loss = float('inf')
    epochs_no_improve = 0
    convergence_epoch = epochs

    for epoch_idx, current_loss in enumerate(train_loss_history):
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            convergence_epoch = epoch_idx + 1
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
    
    if verbose:
        logger.info(f"Calculated convergence at epoch: {convergence_epoch}")
    # -----------------------------------

    # --- FILE SAVING LOGIC ---
    try:
        job_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        job_dir = os.getcwd() 
    
    os.makedirs(job_dir, exist_ok=True)

    # 1. Save loss_history.csv
    loss_csv_path = os.path.join(job_dir, "loss_history.csv")
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for e, l in enumerate(train_loss_history):
            writer.writerow([e, l])

    # 2. Save params_history.csv
    params_csv_path = os.path.join(job_dir, "params_history.csv")
    with open(params_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch"] + [f"param_{i}" for i in range(len(current_params))] + ["phi"]
        writer.writerow(header)
        for row in params_history:
            writer.writerow(row)

    # 3. Save loss_history.jpg
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

    # 4. Accumulate results.txt in the sweep/common directory
    try:
        sweep_dir = HydraConfig.get().sweep.dir
    except Exception:
        sweep_dir = os.path.dirname(job_dir)
    
    os.makedirs(sweep_dir, exist_ok=True)
    results_path = os.path.join(sweep_dir, "results.txt")

    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "dataset", "N", "seed", "p_err", "equivariance", "reps",  
                "val_acc", "val_aug_acc", "training_time", "epoch_time", "convergence_epoch", "remove_cross_edge"
            ])
        writer.writerow([
            dataset, N, seed, p_err, equivariance, reps, 
            final_val_acc, final_val_aug_acc, training_time, epoch_time, convergence_epoch, remove_cross_edge
        ])

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

def study_gradients(
    datasets: list[str],
    equivariances: list[bool],
    device: str,
    dev: str,
    p_err: float,
    reps: int,
    twirling: bool = False,
    remove_cross_edge: bool = False,
    num_inits: int = 100,
    verbose: bool = True
):
    """
    Studia la norma del gradiente al variare di N, per diversi dataset e 
    valori di equivariance, salvando i grafici per ogni dataset.
    """
    N_values = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    local_dev = torch.device(dev)
    
    # Struttura dati per salvare i risultati: results[dataset][eq][N] = [grad_norm_1, ...]
    results = {ds: {eq: {n: [] for n in N_values} for eq in equivariances} for ds in datasets}
    
    pbar_ds = tqdm(datasets, desc="Datasets", position=0, leave=True) if verbose else datasets
    
    for ds in pbar_ds:
        pbar_N = tqdm(N_values, desc=f"Valori di N ({ds})", position=1, leave=False) if verbose else N_values

        for N in pbar_N:
            # Carica il dato UNA sola volta per questo (dataset, N)
            if ds == "mnist":
                train_loader, _ = load_mnist_data(batch_size=N, N=N, num_workers=0, seed=42, augment_test=False)
            elif ds == "nwpu":
                train_loader, _ = load_kaggle_nwpu_data(batch_size=N, N=N, num_workers=0, seed=42, augment_test=False)
            elif ds == "aug_mnist":
                train_loader, _ = load_aug_mnist_data(batch_size=N, N=N, num_workers=0, seed=42, augment_test=False)
            else:
                raise ValueError(f"Dataset {ds} non supportato per questa analisi.")

            batch_images, batch_labels = next(iter(train_loader))
            batch_images = batch_images.to(local_dev)
            batch_labels = batch_labels.to(local_dev)

            # Analizza per i diversi valori di equivariance
            for eq in equivariances:
                qnn = create_qnn(device, p_err, reps, eq, twirling, remove_cross_edge)

                pbar_inits = tqdm(range(num_inits), desc=f"Eq={eq}", position=2, leave=False) if verbose else range(num_inits)

                for _ in pbar_inits:
                    params = torch.empty(8*reps, device=local_dev).uniform_(-0.1, 0.1)
                    params.requires_grad_()
                    phi = torch.empty(1, device=local_dev).uniform_(-0.1, 0.1)
                    phi.requires_grad_()

                    # Forward e Backward
                    batch_predictions = execute_batch(qnn, batch_images, local_dev, params, phi)
                    loss = loss_function(batch_predictions, batch_labels)
                    loss.backward()

                    # Calcolo Norma
                    grad_norm_sq = 0.0
                    if params.grad is not None:
                        grad_norm_sq += params.grad.norm(2).item() ** 2
                    if phi.grad is not None:
                        grad_norm_sq += phi.grad.norm(2).item() ** 2

                    results[ds][eq][N].append(math.sqrt(grad_norm_sq))

        # --- GENERAZIONE GRAFICI PER IL DATASET CORRENTE ---
        if verbose:
            print(f"\nSalvataggio grafici per {ds}...")

        colors = ['skyblue', 'salmon', 'lightgreen', 'plum'] 

        # Plot A: Grid di Istogrammi
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        for i, N in enumerate(N_values):
            for idx, eq in enumerate(equivariances):
                data = results[ds][eq][N]
                axes[i].hist(data, bins=50, alpha=0.6, color=colors[idx % len(colors)], 
                             edgecolor='black', label=f'Eq = {eq}', density=True)

            axes[i].set_title(f'N = {N}')
            axes[i].set_xlabel('Gradient Norm')
            axes[i].set_ylabel('Density')
            axes[i].legend()

        plt.suptitle(f"Istogrammi Norme Gradiente - Dataset: {ds}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"histo_grid_{ds}.pdf")
        plt.close()

        # Plot B: Media +- Varianza vs N
        plt.figure(figsize=(10, 6))
        for idx, eq in enumerate(equivariances):
            means = [np.mean(results[ds][eq][n]) for n in N_values]
            variances = [np.var(results[ds][eq][n]) for n in N_values]

            plt.errorbar(N_values, means, yerr=variances, fmt='-o', 
                         color=colors[idx % len(colors)], ecolor='black', 
                         capsize=5, alpha=0.8, label=f'Eq = {eq} (Media $\pm$ Var)')

        plt.xscale('log', base=2)
        plt.xticks(N_values, labels=[str(n) for n in N_values])
        plt.xlabel('N (Numero di Immagini nel batch)')
        plt.ylabel('Norma del Gradiente')
        plt.title(f'Norma del Gradiente vs N - Dataset: {ds}')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig(f"mean_vs_N_{ds}.pdf")
        plt.close()
