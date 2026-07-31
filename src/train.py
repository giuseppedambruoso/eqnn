import csv
import json
import logging
import os
import time
from multiprocessing import current_process
from typing import Any

import matplotlib.pyplot as plt
import pennylane as qml
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ansatz_builder import check_p4m_invariance

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


def execute_batch(
    qnn: Any,
    batch_images: torch.Tensor,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Runs the whole batch through ONE QNode call, relying on
    default.qubit's support for a batch dimension on the embedding gate —
    dramatically faster than one Python-level call per image. Requires a
    qnn built with diff_method="backprop" (create_qnn/build_qnn_from_spec's
    default)."""
    batch_images = batch_images.to(dev)

    raw_output = qnn(batch_images, params, phi)
    output = (1.0 + raw_output) / 2.0
    clamped_output = torch.clamp(output, min=1e-7, max=1.0 - 1e-7)
    if torch.isnan(raw_output).any() or torch.isnan(clamped_output).any():
        logger.critical("QNN Output is NaN in batch execution!")
        logger.error(f"Raw: {raw_output}, Clamped: {clamped_output}")
    return clamped_output


def train_one_epoch(
    loader: DataLoader,
    qnn: Any,
    opt: torch.optim.Optimizer,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
    pbar: tqdm | None = None,
) -> tuple[float, float]:
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_images, batch_labels in loader:
        batch_labels = batch_labels.to(dev)
        opt.zero_grad()
        batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
        loss = loss_function(batch_predictions, batch_labels)
        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_labels.size(0)
        total_correct += (
            ((batch_predictions.squeeze() > 0.5).long() == batch_labels).sum().item()
        )
        total_samples += batch_labels.size(0)
        if pbar:
            pbar.update(1)

    return total_loss / (total_samples + 1e-8), total_correct / (total_samples + 1e-8)


def validate(
    loader: DataLoader,
    qnn: Any,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[float, float]:
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_labels = batch_labels.to(dev)
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            loss = loss_function(batch_predictions, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (
                ((batch_predictions.squeeze() > 0.5).long() == batch_labels)
                .sum()
                .item()
            )
            total_samples += batch_labels.size(0)

    return total_loss / (total_samples + 1e-8), total_correct / (total_samples + 1e-8)


def _collect_predictions(
    loader: DataLoader,
    qnn: Any,
    dev: torch.device,
    params: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_labels = batch_labels.to(dev)
            batch_predictions = execute_batch(qnn, batch_images, dev, params, phi)
            all_predictions.append(batch_predictions.reshape(-1))
            all_labels.append(batch_labels.reshape(-1))
    return torch.cat(all_predictions), torch.cat(all_labels)


def _plot_confusion_matrix(
    predictions: torch.Tensor, labels: torch.Tensor, destination: str
) -> None:
    """0=digit 3, 1=digit 4 (see data_loading.py's tar_transform)."""
    predicted_classes = (predictions > 0.5).long()
    matrix = torch.zeros((2, 2), dtype=torch.long)
    for true_cls, pred_cls in zip(
        labels.long().tolist(), predicted_classes.tolist(), strict=False
    ):
        matrix[true_cls, pred_cls] += 1
    matrix_np = matrix.numpy()

    fig, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix_np, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix_np[i, j]), ha="center", va="center")
    axis.set_xticks((0, 1), labels=("3", "4"))
    axis.set_yticks((0, 1), labels=("3", "4"))
    axis.set_xlabel("Cifra predetta")
    axis.set_ylabel("Cifra reale")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _export_circuit_diagram(
    qnn: Any,
    params: torch.Tensor,
    phi: torch.Tensor,
    sample_embedding: torch.Tensor,
    destination: str,
) -> bool:
    """Text export of the trained circuit. Needs `qnn.qnode` (the real
    QNode, not a wrapper function) — see src.ansatz_builder.build_qnn_from_spec's
    docstring for why drawing the wrapper silently truncates the diagram.
    Returns whether a diagram was actually written (some qnn objects, e.g.
    older callers, may not expose `.qnode`)."""
    qnode = getattr(qnn, "qnode", None)
    if qnode is None:
        return False
    drawing = qml.draw(qnode, decimals=2)(sample_embedding, params, phi)
    with open(destination, "w") as f:
        f.write(drawing)
    return True


def train_loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_loader_aug: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    min_delta: float,
    dev: str,
    seed: int,
    N: int,
    dataset: str,
    qnn: Any,
    initial_params: torch.Tensor,
    param_names: list[str],
    run_name: str,
    checkpoint_config: dict[str, Any],
    wandb_extra_config: dict[str, Any],
    verbose: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Trains an already-built QNN (from src.qnn.create_qnn or
    src.ansatz_builder.build_qnn_from_spec) and logs/checkpoints the run.

    `param_names` must line up 1:1 with `initial_params` — each entry
    becomes a wandb.log key "params/{name}" tracking that parameter's value
    every epoch. `checkpoint_config`/`wandb_extra_config` carry whatever
    circuit-specific metadata the caller wants embedded in the saved
    checkpoint / wandb run config (e.g. architecture name, or a full custom
    circuit spec) — train_loop itself is agnostic to where the circuit came
    from.

    Each batch runs through the QNN in a single vectorized call (see
    execute_batch) — this requires `qnn` to have been built with
    diff_method="backprop" (create_qnn/build_qnn_from_spec's default).
    """
    torch_dev = torch.device(dev)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "eqnn"),
        group=os.environ.get("WANDB_RUN_GROUP", None),
        name=run_name,
        config={
            "seed": seed,
            "N": N,
            "dataset": dataset,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "min_delta": min_delta,
            **wandb_extra_config,
        },
        reinit=True,
    )

    params = initial_params.clone().to(torch_dev).requires_grad_()
    phi = torch.tensor(0.0, requires_grad=False)
    opt = torch.optim.Adam([params], lr=learning_rate, betas=(0.5, 0.99))

    train_loss_hist, train_acc_hist, params_hist = [], [], []
    total_steps = epochs * len(train_loader)

    pos = (current_process()._identity[0] - 1) if current_process()._identity else 0
    pbar = (
        tqdm(
            total=total_steps,
            desc=f"Job ({run_name})",
            position=pos,
            leave=False,
        )
        if verbose
        else None
    )

    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss, epoch_acc = train_one_epoch(
            train_loader, qnn, opt, torch_dev, params, phi, pbar
        )
        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)
        params_hist.append(
            [epoch]
            + params.detach().cpu().numpy().tolist()
            + [phi.detach().cpu().item()]
        )

        params_log = {
            f"params/{name}": value
            for name, value in zip(
                param_names, params.detach().cpu().tolist(), strict=False
            )
        }

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/accuracy": epoch_acc,
                **params_log,
            },
            step=epoch,
        )

    if pbar:
        pbar.close()
    training_time = time.time() - t0
    epoch_time = training_time / max(epochs, 1)

    val_loss, val_acc = validate(val_loader, qnn, torch_dev, params, phi)
    val_aug_loss, val_aug_acc = validate(val_loader_aug, qnn, torch_dev, params, phi)

    best_loss, epochs_no_improve, convergence_epoch = float("inf"), 0, epochs
    for idx, current_loss in enumerate(train_loss_hist):
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            convergence_epoch = idx + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

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

    try:
        job_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        job_dir = os.getcwd()
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "loss_history.csv"), "w", newline="") as f:
        csv.writer(f).writerows(list(enumerate(train_loss_hist)))

    plt.figure()
    plt.plot(range(epochs), train_loss_hist, label="Train Loss", color="blue")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(job_dir, "loss_history.jpg")
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()

    model_path = os.path.join(job_dir, "final_model.pt")
    torch.save(
        {
            "params": params.detach().cpu(),
            "val_acc": val_acc,
            "config": checkpoint_config,
        },
        model_path,
    )

    # Confusion matrix on the (non-augmented) validation set.
    val_predictions, val_labels = _collect_predictions(
        val_loader, qnn, torch_dev, params, phi
    )
    confusion_path = os.path.join(job_dir, "confusion_matrix.png")
    _plot_confusion_matrix(val_predictions, val_labels, confusion_path)

    # Best-effort text export of the trained circuit's diagram.
    sample_embedding = next(iter(val_loader))[0][0].to(torch_dev)
    circuit_path = os.path.join(job_dir, "circuit.txt")
    has_circuit_diagram = _export_circuit_diagram(
        qnn, params.detach(), phi.detach(), sample_embedding, circuit_path
    )

    # Numerical p4m-equivariance check (see src.ansatz_builder.check_p4m_invariance) —
    # best-effort: an architecture this doesn't apply to shouldn't fail the run.
    p4m_info: dict[str, Any] = {
        "checked": False,
        "is_invariant": None,
        "max_deviation": None,
        "error": None,
    }
    img_size = checkpoint_config.get("img_size")
    if img_size is not None:
        try:
            is_invariant, deviation = check_p4m_invariance(
                qnn, params.detach().cpu(), img_size, n_samples=2
            )
            p4m_info = {
                "checked": True,
                "is_invariant": is_invariant,
                "max_deviation": deviation,
                "error": None,
            }
        except Exception as exc:
            logger.warning(f"p4m-invariance check failed: {exc}")
            p4m_info["error"] = str(exc)

    summary_path = os.path.join(job_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "seed": seed,
                "N": N,
                "dataset": dataset,
                "epochs_configured": epochs,
                "epochs_completed": len(train_loss_hist),
                "convergence_epoch": convergence_epoch,
                "training_time_sec": training_time,
                "epoch_time_sec": epoch_time,
                "train_loss": train_loss_hist[-1],
                "train_accuracy": train_acc_hist[-1],
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_aug_loss": val_aug_loss,
                "val_aug_accuracy": val_aug_acc,
                "param_names": param_names,
                "final_params": params.detach().cpu().tolist(),
                "p4m_equivariance": p4m_info,
                "config": checkpoint_config,
            },
            f,
            indent=2,
        )

    wandb.log({"loss_history_plot": wandb.Image(loss_plot_path)})
    wandb.log({"confusion_matrix": wandb.Image(confusion_path)})
    if p4m_info["checked"]:
        wandb.summary["p4m_is_invariant"] = p4m_info["is_invariant"]
        wandb.summary["p4m_max_deviation"] = p4m_info["max_deviation"]

    artifact = wandb.Artifact(f"model-{run.id}", type="model")
    artifact.add_file(model_path)
    artifact.add_file(confusion_path)
    artifact.add_file(summary_path)
    if has_circuit_diagram:
        artifact.add_file(circuit_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return (
        params,
        phi,
        train_loss_hist,
        train_acc_hist,
        [val_loss],
        [val_acc],
        [val_aug_loss],
        [val_aug_acc],
    )
