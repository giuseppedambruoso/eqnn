import csv
import logging
import os
import time
from multiprocessing import current_process
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from torch.nn import functional as F
from tqdm import tqdm

from src.data_loading import load_mnist_data
from src.qnn import create_qnn

logger = logging.getLogger(__name__)

def loss_function(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.squeeze()
    is_nan = torch.isnan(predictions).any()
    out_of_bounds = (predictions < 0).any() or (predictions > 1).any()

    if is_nan or out_of_bounds:
        logger.critical("INVALID PREDICTIONS DETECTED IN LOSS FUNCTION")
        logger.error(f"Contains NaNs: {is_nan}")
        logger.error(f"Out of bounds: {out_of_bounds}")

    loss = F.binary_cross_entropy(predictions, targets.to(predictions.dtype))
    return loss

def execute_batch(qnn: Any, batch_images: torch.Tensor, dev: torch.device, params: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    batch_images = batch_images.to(dev)
    batch_predictions = []

    for i, image in enumerate(batch_images):
        raw_output = qnn(image, params, phi)
        output = (1.0 + raw_output) / 2.0
        clamped_output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)

        if torch.isnan(raw_output) or torch.isnan(clamped_output):
            logger.critical(f"QNN Output is NaN at Image Index {i}!")
            logger.error(f"Raw: {raw_output}, Clamped: {clamped_output}")
        batch_predictions.append(clamped_output)

    return torch.stack(batch_predictions)

def train_one_epoch(loader, qnn, opt, dev, params, phi, pbar=None):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_images, batch_labels in loader:
        batch_labels = batch_labels.to(dev)
        opt.zero_grad()
        batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
        loss = loss_function(batch_predictions, batch_labels)
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_labels.size(0)
        total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
        total_samples += batch_labels.size(0)
        if pbar: pbar.update(1)

    return total_loss / (total_samples + 1e-8), total_correct / (total_samples + 1e-8)

def validate(loader, qnn, dev, params, phi):
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_labels = batch_labels.to(dev)
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

    return total_loss / (total_samples + 1e-8), total_correct / (total_samples + 1e-8)

def train_loop(
    train_loader, val_loader, val_loader_aug, epochs: int, learning_rate: float,
    patience: int, min_delta: float, device: str, num_qubits: int, dev: str, seed: int,
    N: int, equivariance: bool, reps: int, p_err: float, dataset: str,
    twirling: bool = False, remove_cross_edge: bool = False, verbose: bool = False
):
    dev = torch.device(dev)
    qnn = create_qnn(device, num_qubits, p_err, reps, equivariance, twirling, remove_cross_edge)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "eqnn"),
        group=os.environ.get("WANDB_RUN_GROUP", None),
        name=f"eq={equivariance}_twirl={twirling}_crossedge={not remove_cross_edge}_N={N}_seed={seed}",
        config={
            "seed": seed,
            "N": N,
            "dataset": dataset,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "min_delta": min_delta,
            "device": device,
            "num_qubits": num_qubits,
            "reps": reps,
            "p_err": p_err,
            "equivariance": equivariance,
            "twirling": twirling,
            "remove_cross_edge": remove_cross_edge,
        },
        reinit=True,
    )

    g = torch.Generator(device=dev).manual_seed(seed)
    params = torch.empty(num_qubits*reps, device=dev).uniform_(-0.1, 0.1, generator=g)
    params.requires_grad_()
    phi = torch.tensor(0.0, requires_grad=False)
    opt = torch.optim.Adam([params, phi], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_hist, train_acc_hist, params_hist = [], [], []
    total_steps = epochs * len(train_loader)

    pos = (current_process()._identity[0] - 1) if current_process()._identity else 0
    pbar = tqdm(total=total_steps, desc=f"Job (Eq={equivariance}, Cross={remove_cross_edge})", position=pos, leave=False) if verbose else None

    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss, epoch_acc = train_one_epoch(train_loader, qnn, opt, dev, params, phi, pbar)
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)
        params_hist.append([epoch] + params.detach().cpu().numpy().tolist() + [phi.detach().cpu().item()])

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/accuracy": epoch_acc,
                "params/phi": phi.detach().cpu().item(),
            },
            step=epoch,
        )

    if pbar: pbar.close()
    training_time = time.time() - t0
    epoch_time = training_time / max(epochs, 1)

    val_loss, val_acc = validate(val_loader, qnn, dev, params, phi)
    val_aug_loss, val_aug_acc = validate(val_loader_aug, qnn, dev, params, phi)

    best_loss, epochs_no_improve, convergence_epoch = float('inf'), 0, epochs
    for idx, current_loss in enumerate(train_loss_hist):
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            convergence_epoch = idx + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: break

    wandb.log(
        {
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val_aug/loss": val_aug_loss,
            "val_aug/accuracy": val_aug_acc,
            "convergence_epoch": convergence_epoch,
            "training_time_sec": training_time,
            "epoch_time_sec": epoch_time,
        }
    )
    wandb.summary["val_accuracy"] = val_acc
    wandb.summary["convergence_epoch"] = convergence_epoch

    try: job_dir = HydraConfig.get().runtime.output_dir
    except Exception: job_dir = os.getcwd()
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "loss_history.csv"), "w", newline="") as f:
        csv.writer(f).writerows([[e, l] for e, l in enumerate(train_loss_hist)])

    plt.figure()
    plt.plot(range(epochs), train_loss_hist, label='Train Loss', color='blue')
    plt.legend(); plt.grid(True)
    loss_plot_path = os.path.join(job_dir, "loss_history.jpg")
    plt.savefig(loss_plot_path, bbox_inches='tight')
    plt.close()

    model_path = os.path.join(job_dir, "final_model.pt")
    torch.save({'params': params.detach().cpu(), 'val_acc': val_acc}, model_path)

    wandb.log({"loss_history_plot": wandb.Image(loss_plot_path)})
    artifact = wandb.Artifact(f"model-{run.id}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return params, phi, train_loss_hist, train_acc_hist, [val_loss], [val_acc], [val_aug_loss], [val_aug_acc]

def study_gradients(datasets, equivariances, num_qubits, device, dev, p_err, reps, twirling, remove_cross_edge, num_inits, verbose):
    try: out_dir = HydraConfig.get().runtime.output_dir
    except Exception: out_dir = os.getcwd()

    N_values = [20, 40] # Truncated for example brevity
    local_dev = torch.device(dev)

    for ds in datasets:
        for N in N_values:
            train_loader, _ = load_mnist_data(batch_size=N, N=N, num_workers=0, img_size=16) # Dummy implementation
            batch_images, batch_labels = next(iter(train_loader))

            for eq in equivariances:
                qnn = create_qnn(device, num_qubits, p_err, reps, eq, twirling, remove_cross_edge)
                for _ in range(num_inits):
                    params = torch.empty(num_qubits*reps, device=local_dev).uniform_(-0.1, 0.1).requires_grad_()
                    phi = torch.empty(1, device=local_dev).uniform_(-0.1, 0.1).requires_grad_()
                    preds = execute_batch(qnn, batch_images.to(local_dev), local_dev, params, phi)
                    loss = loss_function(preds, batch_labels.to(local_dev))
                    loss.backward()

        # Ensure saving goes to out_dir
        plt.figure()
        plt.savefig(os.path.join(out_dir, f"histo_grid_{ds}.pdf"))
        plt.close()
