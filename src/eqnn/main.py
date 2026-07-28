# main.py
import csv
import logging
import random
import time
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from data_loading import load_mnist_data, load_eurosat_data, load_kaggle_nwpu_data, load_aug_mnist_data, add_sat_data
from plot import plot_results
from train import train_loop, study_gradients

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    SEED = cfg.GENERAL.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = cfg.QNN.device
    p_err = cfg.QNN.p_err
    reps = cfg.QNN.reps
    equivariance = cfg.QNN.equivariance
    twirling = cfg.QNN.twirling
    remove_cross_edge = getattr(cfg.QNN, 'remove_cross_edge', False)

    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    augment_test = cfg.DATA.get("augment_test", False)
    batch_size = int(N // 10)
    verbose = cfg.GENERAL.verbose
    dev = cfg.GENERAL.dev

    initialization_analysis = cfg.GENERAL.initialization_analysis

    if verbose:
        logger.info(
            f"QNN pipeline initialized. p_err={p_err}, equivariance={equivariance}, twirling={twirling}, remove_cross_edge={remove_cross_edge}"
        )

    # DATA LOADING
    if dataset == "mnist":
        train_loader, test_loader = load_mnist_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False)
        aug_train_loader, aug_test_loader = load_mnist_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True)
    elif dataset == "eurosat":
        train_loader, test_loader = load_eurosat_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False)
        aug_train_loader, aug_test_loader = load_eurosat_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True)
    elif dataset == "nwpu":
        train_loader, test_loader = load_kaggle_nwpu_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False)
        aug_train_loader, aug_test_loader = load_kaggle_nwpu_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True)
    elif dataset == "aug_mnist":
        train_loader, test_loader = load_aug_mnist_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False)
        aug_train_loader, aug_test_loader = load_aug_mnist_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True)
    elif dataset == "sat_data":
        train_loader, test_loader = add_sat_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False)
        aug_train_loader, aug_test_loader = add_sat_data(batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True)
    else:
        raise ValueError("dataset must be either 'mnist' or 'eurosat' or 'nwpu' or 'aug_mnist' or 'sat_data'")

    torch.manual_seed(SEED)

    if initialization_analysis:
        logger.info("Avvio analisi di inizializzazione massiva...")
        study_gradients(
            datasets=["nwpu", "aug_mnist"],
            equivariances=[False, True],
            device=device,
            dev=dev,
            p_err=p_err,
            reps=reps,
            twirling=twirling,
            remove_cross_edge=remove_cross_edge,
            num_inits=300,
            verbose=verbose
        )
    else:
        training_output = train_loop(
            device=device,
            dev=dev,
            train_loader=train_loader,
            val_loader=test_loader,
            val_loader_aug=aug_test_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=SEED,
            N=N,
            equivariance=equivariance,
            reps=reps,
            p_err=p_err,
            dataset=dataset,
            twirling=twirling,
            remove_cross_edge=remove_cross_edge,
            verbose=verbose,
        )

if __name__ == "__main__":
    main()
