import hashlib
import json
import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data_loading import load_mnist_data
from src.qnn import ARCHITECTURES, architecture_param_names, create_qnn
from src.train import train_loop

logger = logging.getLogger(__name__)


@hydra.main(config_path="./config/", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Each Hydra multirun job (hydra/launcher=joblib) is its own OS process;
    # torch/BLAS default to spawning one thread per CPU core *within* each
    # process, so running several jobs in parallel oversubscribes the
    # machine's cores many times over (observed: ~100x slowdown running just
    # 2 jobs on a 14-core machine). The intended parallelism is entirely at
    # the process level (n_jobs), so each process keeps exactly one thread.
    torch.set_num_threads(1)

    SEED = cfg.GENERAL.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Parametrized configuration variables
    device = cfg.QNN.device
    num_qubits = cfg.QNN.num_qubits
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

    # Default wandb group: every config.yaml parameter that defines the
    # experiment EXCEPT the seed (dev/verbose/num_workers are execution
    # details, not part of the experiment definition, so they're excluded
    # too). Two runs land in the same group, and so get averaged together
    # with a std-dev band on every wandb chart, iff they're identical in
    # every one of these — i.e. iff they're the same configuration run with
    # a different random initialization. An explicit -e WANDB_RUN_GROUP=...
    # still overrides this (e.g. to compare several architectures on one
    # chart instead of averaging within each).
    config_identity = {
        "architecture": architecture,
        "device": device,
        "num_qubits": num_qubits,
        "reps": reps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "min_delta": min_delta,
        "N": N,
        "dataset": dataset,
        "img_size": img_size,
    }
    config_hash = hashlib.sha1(
        json.dumps(config_identity, sort_keys=True).encode()
    ).hexdigest()[:8]
    os.environ.setdefault("WANDB_RUN_GROUP", f"{architecture}_N{N}_{config_hash}")

    qnn = create_qnn(device, num_qubits, reps, architecture)
    is_equivariant = ARCHITECTURES[architecture]["is_equivariant"]

    param_names = architecture_param_names(architecture, num_qubits, reps)
    g = torch.Generator(device=torch.device(dev)).manual_seed(SEED)
    initial_params = torch.empty(len(param_names), device=torch.device(dev)).uniform_(
        -0.1, 0.1, generator=g
    )

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
            "reps": reps,
            "architecture": architecture,
            "img_size": img_size,
        },
        wandb_extra_config={
            "device": device,
            "num_qubits": num_qubits,
            "reps": reps,
            "architecture": architecture,
            "is_equivariant": is_equivariant,
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
