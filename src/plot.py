# plot.py
import logging
import os

import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


def plot_results(train_loss: list[float], train_acc: list[float]) -> None:
    job_dir = HydraConfig.get().runtime.output_dir
    full_save_path = os.path.join(job_dir, "plot.png")

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(len(train_loss)), train_loss, label="Train Loss")
    axes[0].set_title("Loss over Training")
    axes[0].grid(True)

    axes[1].plot(range(len(train_acc)), train_acc, label="Train Acc")
    axes[1].set_title("Accuracy over Training")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300)
    plt.close()

    logger.info(f"Training plots saved")
    logger.info("")
