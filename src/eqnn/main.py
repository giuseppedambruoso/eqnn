# main.py
import csv
import logging
import random
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from data_loading import load_mnist_data, load_eurosat_data, load_kaggle_nwpu_data
from plot import plot_results
from train import train_loop

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
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_mnist_data(
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True
        )
    elif dataset == "eurosat":
        # Loader normale (pulito)
        train_loader, test_loader = load_eurosat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_eurosat_data(
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True
        )
    elif dataset == "nwpu":
        # Loader normale (pulito)
        train_loader, test_loader = load_kaggle_nwpu_data(
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=False
        )
        # Loader con augmentation (usato per calcolare aug_acc)
        aug_train_loader, aug_test_loader = load_kaggle_nwpu_data(
            batch_size=batch_size, N=N, num_workers=0, seed=42, verbose=verbose, augment_test=True
        )
    else:
        raise ValueError("dataset must be either 'mnist' or 'eurosat'")

    torch.manual_seed(SEED) # Keeping your original seed setting here
    
    if initialization_analysis:
        images, labels = next(iter(train_loader))
        # take only the first sample
        image = images[0].unsqueeze(0)  # keep batch dim if model expects it
        label = labels[0].unsqueeze(0)
        
        # Inizializziamo il tensore CUDA qui localmente se necessario per l'analisi
        local_dev = torch.device(dev)
        image = image.to(local_dev)
        label = label.to(local_dev)
        
        grad_norms = []
        pbar = tqdm(range(1, 1000), desc="progress") if verbose else range(epochs)

        for seed_val in pbar: 
            torch.manual_seed(seed_val)
            grad_norm = train_loop_in(
                image=image,
                label=labels,
                device=device,
                dev=dev,
                learning_rate=learning_rate,
                non_equivariance=non_equivariance,
                p_err=p_err,
            )
            grad_norms.append(grad_norm)

        plt.hist(grad_norms, bins=50)
        plt.savefig(f"histo_{non_equivariance}")
        plt.show()

        csv_path = f"grad_norms_{non_equivariance}.csv"

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "grad_norm"])  # header
            for idx, grad_norm in enumerate(grad_norms, start=1):
                writer.writerow([idx, grad_norm])

        logger.info(f"Saved gradient norms to {csv_path}")

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
