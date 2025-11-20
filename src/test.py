# test.py
import logging
import os
from typing import Literal

import torch
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from train import execute_batch, loss_function

logger = logging.getLogger(__name__)


def test_loop(
    test_loader: torch.utils.data.DataLoader,
    N: int,
    device: str,
    params: torch.Tensor,
    phi: torch.Tensor,
    non_equivariance: Literal[0, 1, 2],
    verbose: bool = False,
) -> tuple[float, float]:
    if verbose:
        logger.info("Testing the QNN...")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, leave=True) if verbose else test_loader
        for batch_idx, (batch_images, batch_labels) in enumerate(pbar):

            batch_predictions = execute_batch(
                batch_images, device, params, phi, non_equivariance
            )
            loss = loss_function(batch_predictions, batch_labels)

            total_loss += loss.item() * batch_labels.size(0)
            total_correct += (
                (torch.argmax(batch_predictions, 1) == batch_labels).sum().item()
            )
            total_samples += batch_labels.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples

            if verbose:
                pbar.set_postfix(
                    {"test_loss": f"{avg_loss:.4f}", "test_acc": f"{avg_acc:.3f}"}
                )

    final_loss = total_loss / total_samples
    final_accuracy = total_correct / total_samples

    logger.info(
        f"Test completed with loss {final_loss:.4f}, and accuracy {final_accuracy*100:.0f}%"
    )

    sweep_dir = HydraConfig.get().sweep.dir
    os.makedirs(sweep_dir, exist_ok=True)
    file_path = os.path.join(sweep_dir, "test_accuracies.txt")

    with open(file_path, "a") as f:
        f.write(
            f"Sample size: {N}, Non equivariance: {non_equivariance}, Test Accuracy: {final_accuracy:.4f}\n"
        )

    return final_loss, final_accuracy
