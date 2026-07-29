import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import current_process
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
    predictions = predictions.squeeze()
    
    is_nan = torch.isnan(predictions).any()
    out_of_bounds = (predictions < 0).any() or (predictions > 1).any()
    
    if is_nan or out_of_bounds:
        print("\n" + "!"*60)
        print("CRITICAL ERROR: INVALID PREDICTIONS DETECTED IN LOSS FUNCTION")
        print(f"Contains NaNs: {is_nan}")
        print(f"Out of [0, 1] bounds: {out_of_bounds}")
        print(f"Predictions Tensor: \n{predictions}")
        print(f"Targets Tensor: \n{targets}")
        print("!"*60 + "\n")
        
    loss = F.binary_cross_entropy(predictions, targets.to(predictions.dtype))
    return loss

def execute_batch(
    qnn: Any,
    batch_images: torch.Tensor,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    batch_images = batch_images.to(dev)
    batch_predictions = []
    
    for i, image in enumerate(batch_images):
        raw_output = qnn(image, params, phi)
        output = (1.0 + raw_output) / 2.0
        clamped_output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
        
        if torch.isnan(raw_output) or torch.isnan(clamped_output):
            print("\n" + "-"*50)
            print(f"[DEBUG] QNN Output is NaN at Image Index {i}!")
            print(f"Raw Expectation Value: {raw_output}")
            print(f"Scaled & Clamped Val : {clamped_output}")
            print(f"Current Params       : {params.detach().cpu().numpy()}")
            print("-"*50 + "\n")

        batch_predictions.append(clamped_output)
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

    qnn = create_qnn(device, p_err, reps, equivariance, twirling, remove_cross_edge)
    if verbose:
        logger.info("QNode initialized successfully.")

    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    params = torch.empty(8*reps, device=dev).uniform_(-0.1, 0.1, generator=g)
    params.requires_grad_()
    phi = torch.tensor(0.0, requires_grad=False)

    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []
    val_aug_loss_history, val_aug_acc_history = [], []   
    params_history = []

    total_batches = len(train_loader)
    total_steps = epochs * total_batches

    if verbose:
        try:
            job_idx = HydraConfig.get().job.num
        except Exception:
            job_idx = 0 
            
        # Dynamically assign a vertical position based on the CPU worker's ID
        worker_identity = current_process()._identity
        pos = (worker_identity[0] - 1) if worker_identity else 0
        
        # Bypass Joblib's output capture by writing directly to the raw stderr
        pbar = tqdm(
            total=total_steps, 
            desc=f"Job {job_idx} (Eq={equivariance}, Twirl={twirling}, NoCross={remove_cross_edge})", 
            leave=False, 
            position=pos,
            file=sys.__stderr__,
            mininterval=1.0 
        )
    else:
        pbar = None

    t0 = time.time()

    for epoch in range(epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0

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

            if pbar is not None:
                pbar.update(1)

        epoch_train_loss = total_loss / (total_samples + 1e-8)
        epoch_train_acc = total_correct / (total_samples + 1e-8)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        current_params = params.detach().cpu().numpy().tolist()
        current_phi = phi.detach().cpu().item()
        params_history.append([epoch] + current_params + [current_phi])

    if pbar is not None:
        pbar.close()

    t1 = time.time()
    training_time = t1 - t0
    epoch_time = training_time / max(epochs, 1)

    if verbose:
        logger.info(f"Training completed in {training_time:.2f}s. Starting validation...")

    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_loss_aug, total_correct_aug, total_samples_aug = 0.0, 0, 0

    with torch.no_grad():
        for (batch_images, batch_labels), (batch_images_aug, batch_labels_aug) in zip(val_loader, val_loader_aug):
            batch_labels = batch_labels.to(dev)
            batch_labels_aug = batch_labels_aug.to(dev)

            assert torch.equal(batch_labels, batch_labels_aug), "FATAL: Standard and Augmented validation labels do not match!"

            # Standard Prediction
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

            # Augmented Prediction
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
    
    try:
        job_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        job_dir = os.getcwd() 
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "loss_history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for e, l in enumerate(train_loss_history):
            writer.writerow([e, l])

    with open(os.path.join(job_dir, "params_history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch"] + [f"param_{i}" for i in range(len(current_params))] + ["phi"]
        writer.writerow(header)
        for row in params_history:
            writer.writerow(row)

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

    model_path = os.path.join(job_dir, "final_model.pt")
    torch.save({
        'epochs_completed': epochs,
        'params': params.detach().cpu(),
        'phi': phi.detach().cpu(),
        'val_acc': final_val_acc,
        'val_aug_acc': final_val_aug_acc
    }, model_path)

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
        params, phi, train_loss_history, train_acc_history,
        val_loss_history, val_acc_history, val_aug_loss_history, val_aug_acc_history
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
    N_values = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    local_dev = torch.device(dev)
    
    results = {ds: {eq: {n: [] for n in N_values} for eq in equivariances} for ds in datasets}
    pbar_ds = tqdm(datasets, desc="Datasets", position=0, leave=True) if verbose else datasets
    
    for ds in pbar_ds:
        pbar_N = tqdm(N_values, desc=f"Valori di N ({ds})", position=1, leave=False) if verbose else N_values

        for N in pbar_N:
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

            for eq in equivariances:
                qnn = create_qnn(device, p_err, reps, eq, twirling, remove_cross_edge)
                pbar_inits = tqdm(range(num_inits), desc=f"Eq={eq}", position=2, leave=False) if verbose else range(num_inits)

                for _ in pbar_inits:
                    params = torch.empty(8*reps, device=local_dev).uniform_(-0.1, 0.1)
                    params.requires_grad_()
                    phi = torch.empty(1, device=local_dev).uniform_(-0.1, 0.1)
                    phi.requires_grad_()

                    batch_predictions = execute_batch(qnn, batch_images, local_dev, params, phi)
                    loss = loss_function(batch_predictions, batch_labels)
                    loss.backward()

                    grad_norm_sq = 0.0
                    if params.grad is not None:
                        grad_norm_sq += params.grad.norm(2).item() ** 2
                    if phi.grad is not None:
                        grad_norm_sq += phi.grad.norm(2).item() ** 2

                    results[ds][eq][N].append(math.sqrt(grad_norm_sq))

        if verbose:
            print(f"\nSalvataggio grafici per {ds}...")

        colors = ['skyblue', 'salmon', 'lightgreen', 'plum'] 
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

        plt.figure(figsize=(10, 6))
        for idx, eq in enumerate(equivariances):
            means = [np.mean(results[ds][eq][n]) for n in N_values]
            variances = [np.var(results[ds][eq][n]) for n in N_values]

            plt.errorbar(N_values, means, yerr=variances, fmt='-o', 
                         color=colors[idx % len(colors)], ecolor='black', 
                         capsize=5, alpha=0.8, label=f'Eq = {eq} (Media $\\pm$ Var)')

        plt.xscale('log', base=2)
        plt.xticks(N_values, labels=[str(n) for n in N_values])
        plt.xlabel('N (Numero di Immagini nel batch)')
        plt.ylabel('Norma del Gradiente')
        plt.title(f'Norma del Gradiente vs N - Dataset: {ds}')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.savefig(f"mean_vs_N_{ds}.pdf")
        plt.close()
