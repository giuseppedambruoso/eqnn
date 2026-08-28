import hashlib
import json
import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data_loading import AERO_LABELS, load_aero_data_full, load_mnist_data_full
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
    # Only meaningful for config6-config9 — see src.qnn.create_qnn's
    # docstring. config1-config5 ignore it.
    readout = cfg.QNN.readout

    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    patience = cfg.TRAINING.patience
    min_delta = cfg.TRAINING.min_delta

    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    img_size = cfg.DATA.img_size
    data_dir = cfg.DATA.data_dir
    augment_train = cfg.DATA.augment_train
    class1 = cfg.DATA.class1
    class2 = cfg.DATA.class2

    batch_size = int(N // 10)
    num_workers = cfg.GENERAL.num_workers
    verbose = cfg.GENERAL.verbose
    dev = cfg.GENERAL.dev

    if dataset == "mnist":
        train_loader, test_loader, aug_test_loader = load_mnist_data_full(
            batch_size,
            N,
            num_workers,
            img_size,
            data_dir,
            SEED,
            verbose,
            augment_train,
            class1,
            class2,
        )
    elif dataset == "satellite":
        # Ship vs plane is fixed by the dataset itself, not a CLI-selectable
        # pair like MNIST's digit classes — override whatever class1/class2
        # config.yaml happens to have (its 3/4 default is MNIST-specific) so
        # every downstream record (checkpoint, wandb, results_def) reflects
        # the actual labels instead of a stale, meaningless digit pair.
        class1, class2 = AERO_LABELS["ship"], AERO_LABELS["plane"]
        train_loader, test_loader, aug_test_loader = load_aero_data_full(
            batch_size,
            N,
            num_workers,
            img_size,
            SEED,
            verbose,
            augment_train,
        )
    else:
        raise ValueError(
            f"Unknown DATA.dataset {dataset!r}; must be one of 'mnist', 'satellite'."
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
        "readout": readout,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "min_delta": min_delta,
        "N": N,
        "dataset": dataset,
        "img_size": img_size,
        "augment_train": augment_train,
        "class1": class1,
        "class2": class2,
    }
    config_hash = hashlib.sha1(
        json.dumps(config_identity, sort_keys=True).encode()
    ).hexdigest()[:8]
    os.environ.setdefault("WANDB_RUN_GROUP", f"{architecture}_N{N}_{config_hash}")

    qnn = create_qnn(device, num_qubits, reps, architecture, readout=readout)
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
            "readout": readout,
            "img_size": img_size,
            "augment_train": augment_train,
            "class1": class1,
            "class2": class2,
        },
        wandb_extra_config={
            "device": device,
            "num_qubits": num_qubits,
            "reps": reps,
            "architecture": architecture,
            "readout": readout,
            "is_equivariant": is_equivariant,
            "augment_train": augment_train,
            "img_size": img_size,
            "class1": class1,
            "class2": class2,
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
