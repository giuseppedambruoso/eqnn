# main.py
import logging
from test import test_loop

import hydra
import torch
from data_loading import load_eurosat_data, load_mnist_data
from omegaconf import DictConfig
from plot import plot_results
from train import train_loop

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
    epochs = cfg.TRAINING.epochs
    learning_rate = cfg.TRAINING.learning_rate
    N = cfg.DATA.N
    dataset = cfg.DATA.dataset
    batch_size = int(N / 10)
    verbose = cfg.GENERAL.verbose

    if verbose:
        logger.info("QNN training pipeline initialized")
        logger.info(
            f"Non-equivariance: {non_equivariance}, Sample size: {N}, Epochs: {epochs}, Learning rate: {learning_rate}"
        )

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

    # TRAINING
    training_output = train_loop(
        device=device,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=SEED,
        N=N,
        non_equivariance=non_equivariance,
        verbose=verbose,
    )

    # PLOT RESULTS
    plot_results(*training_output[2:])


if __name__ == "__main__":
    main()
