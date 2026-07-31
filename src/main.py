import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data_loading import load_mnist_data
from src.qnn import ARCHITECTURES, create_qnn
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

    qnn = create_qnn(device, num_qubits, p_err, reps, architecture)
    is_equivariant = ARCHITECTURES[architecture]["twirled"]

    g = torch.Generator(device=torch.device(dev)).manual_seed(SEED)
    initial_params = torch.empty(num_qubits * reps, device=torch.device(dev)).uniform_(
        -0.1, 0.1, generator=g
    )
    param_names = [f"rep{r}_q{i}" for r in range(reps) for i in range(num_qubits)]

    train_loop(
        train_loader,
        test_loader,
        aug_test_loader,
        epochs,
        learning_rate,
        patience,
        min_delta,
        dev,
        SEED,
        N,
        dataset,
        qnn,
        initial_params,
        param_names,
        run_name=f"{architecture}_N={N}_seed={SEED}",
        checkpoint_config={
            "device": device,
            "num_qubits": num_qubits,
            "p_err": p_err,
            "reps": reps,
            "architecture": architecture,
            "img_size": img_size,
        },
        wandb_extra_config={
            "device": device,
            "num_qubits": num_qubits,
            "reps": reps,
            "p_err": p_err,
            "architecture": architecture,
            "is_equivariant": is_equivariant,
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
