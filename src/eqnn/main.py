# main.py
import csv
import logging
import time

import hydra
import matplotlib.pyplot as plt
import torch
from data_loading import load_eurosat_data, load_mnist_data
from omegaconf import DictConfig
from plot import plot_results
from tqdm import tqdm
from train import train_loop, train_loop_in

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the QNN training and testing pipeline.
    """

    SEED = cfg.GENERAL.seed
    torch.manual_seed(SEED)

    device = cfg.QNN.device
    non_equivariance = cfg.QNN.non_equivariance
    p_err = cfg.QNN.p_err
    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    batch_size = N
    verbose = cfg.GENERAL.verbose
    dev = torch.device(cfg.GENERAL.dev)
    initialization_analysis = cfg.GENERAL.initialization_analysis

    if verbose:
        logger.info("QNN training pipeline initialized")

    # DATA LOADING
    if dataset == "mnist":
        train_loader, test_loader = load_mnist_data(
            batch_size=batch_size, N=N, num_workers=1, verbose=verbose
        )
    elif dataset == "eurosat":
        train_loader, test_loader = load_eurosat_data(
            batch_size=batch_size, N=N, num_workers=1, verbose=verbose
        )
    else:
        raise ValueError("dataset must be either 'mnist' or 'eurosat'")

    if initialization_analysis:
        torch.manual_seed(SEED)
        images, labels = next(iter(train_loader))
        # take only the first sample
        image = images[0].unsqueeze(0)  # keep batch dim if model expects it
        label = labels[0].unsqueeze(0)
        image = image.to(dev)
        label = label.to(dev)
        grad_norms = []
        pbar = tqdm(range(1, 10000), desc="progress") if verbose else range(epochs)
        for seed in pbar:
            torch.manual_seed(seed)
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
            for seed, grad_norm in enumerate(grad_norms, start=1):
                writer.writerow([seed, grad_norm])

        logger.info(f"Saved gradient norms to {csv_path}")
    else:
        # TRAINING
        training_output = train_loop(
            device=device,
            dev=dev,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=SEED,
            N=N,
            non_equivariance=non_equivariance,
            p_err=p_err,
            verbose=verbose,
        )

        # PLOT RESULTS
        plot_results(*training_output[2:6])


if __name__ == "__main__":
    main()
