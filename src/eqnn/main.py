# main.py
import csv
import logging
import random
import time
import os
from pathlib import Path

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
    """
    Main function to run the QNN training and testing pipeline.
    """

    SEED = cfg.GENERAL.seed

    # > # --- Set Global Seeds for Absolute Reproducibility --- <
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = cfg.QNN.device
    non_equivariance = cfg.QNN.non_equivariance
    reps = cfg.QNN.reps
    p_err = cfg.QNN.p_err
    data_seed = cfg.DATA.seed
    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    augment_test = cfg.DATA.get("augment_test", False)
    batch_size = int(N // 10)
    verbose = cfg.GENERAL.verbose
    
    # MODIFICATO: Manteniamo dev come stringa per non attivare il driver CUDA nel processo padre
    dev = cfg.GENERAL.dev 
    
    initialization_analysis = cfg.GENERAL.initialization_analysis
    
    if verbose:
        logger.info(
            f"QNN training pipeline initialized with p_err={p_err} and non_equivariance={non_equivariance}"
        )

    # DATA LOADING
    if dataset == "mnist":
        # Loader normale (pulito)
        train_loader, test_loader = load_mnist_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_mnist_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=True
        )
    elif dataset == "eurosat":
        # Loader normale (pulito)
        train_loader, test_loader = load_eurosat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_eurosat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=True
        )
    elif dataset == "nwpu":
        # Loader normale (pulito)
        train_loader, test_loader = load_kaggle_nwpu_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_kaggle_nwpu_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=True
        )
    elif dataset == "aug_mnist":
        # Loader normale (pulito)
        train_loader, test_loader = load_aug_mnist_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_aug_mnist_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=True
        )
    elif dataset == "sat_data":
        # Loader normale (pulito)
        train_loader, test_loader = add_sat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = add_sat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=data_seed, verbose=verbose, augment_test=True
        )
    else:
        raise ValueError("dataset must be either 'mnist', 'eurosat', 'sat_data', 'aug_mnist' or 'nwpu'")

    if initialization_analysis:
        logger.info("Avvio analisi di inizializzazione massiva...")
        
        # Passiamo le liste direttamente alla funzione
        study_gradients(
            datasets=["nwpu", "aug_mnist"],
            non_equivariances=[3, 4],
            device=device,
            dev=dev,
            p_err=p_err,
            reps=reps,
            num_inits=300,
            verbose=verbose
        )

    else:
        # TRAINING
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
            non_equivariance=non_equivariance,
            reps = reps,
            p_err=p_err,
            dataset=dataset,
            verbose=verbose,
        )
if __name__ == "__main__":
    main()
