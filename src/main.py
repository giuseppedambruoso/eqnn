import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data_loading import load_mnist_data
from src.train import train_loop

logger = logging.getLogger(__name__)


@hydra.main(config_path="./config/", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    SEED = cfg.GENERAL.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Parametrized configuration variables
    device = cfg.QNN.device
    num_qubits = cfg.QNN.num_qubits
    p_err = cfg.QNN.p_err
    reps = cfg.QNN.reps
    architecture = cfg.QNN.architecture
    remove_cross_edge = cfg.QNN.remove_cross_edge

    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    patience = cfg.TRAINING.patience
    min_delta = cfg.TRAINING.min_delta

    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    img_size = cfg.DATA.img_size
    data_dir = cfg.DATA.data_dir

    batch_size = int(N // 10)
    num_workers = cfg.GENERAL.num_workers
    verbose = cfg.GENERAL.verbose
    dev = cfg.GENERAL.dev

    if dataset == "mnist":
        train_loader, test_loader = load_mnist_data(
            batch_size, N, num_workers, img_size, data_dir, SEED, verbose, False
        )
        _, aug_test_loader = load_mnist_data(
            batch_size, N, num_workers, img_size, data_dir, SEED, verbose, True
        )
    else:
        raise ValueError(
            "Only 'mnist' is currently supported in this modularized example."
        )

    train_loop(
        train_loader,
        test_loader,
        aug_test_loader,
        epochs,
        learning_rate,
        patience,
        min_delta,
        device,
        num_qubits,
        dev,
        SEED,
        N,
        architecture,
        reps,
        p_err,
        dataset,
        remove_cross_edge,
        verbose,
        img_size,
    )


if __name__ == "__main__":
    main()
