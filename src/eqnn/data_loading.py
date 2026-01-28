import logging

import torch
import torchvision
import torchvision.transforms as transforms
from data_encoding import embedding_unitary
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

logger = logging.getLogger(__name__)


class L2Normalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(tensor.view(-1), ord=2, keepdim=True)
        return tensor / (l2_norm + 1e-12)


def load_mnist_data(
    batch_size: int, N: int, num_workers: int, verbose: bool = False
) -> tuple[DataLoader, DataLoader]:
    if verbose:
        logger.info("Loading and embedding MNIST data...")

    transform = transforms.Compose(
        [
            transforms.Resize(16),
            transforms.ToTensor(),
            L2Normalize(),
            transforms.Lambda(lambda x: x.squeeze(0)),
            transforms.Lambda(lambda x: embedding_unitary(x)),
        ]
    )

    switch = {0: 3, 1: 4, 3: 0, 4: 1}
    tar_transform = lambda y: switch[y]

    train_full = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=transform,
        target_transform=tar_transform,
        # split="digits",
    )
    test_full = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=transform,
        target_transform=tar_transform,
        # split="digits",
    )

    target_labels = torch.tensor([3, 4])
    train_idx = torch.isin(train_full.targets, target_labels).nonzero(as_tuple=True)[0]
    test_idx = torch.isin(test_full.targets, target_labels).nonzero(as_tuple=True)[0]

    train_filtered = Subset(train_full, train_idx)
    test_filtered = Subset(test_full, test_idx)

    train_final = Subset(train_filtered, list(range(N)))
    test_final = Subset(test_filtered, list(range(N)))

    train_loader = DataLoader(
        train_final, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_final, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if verbose:
        logger.info("Data loading complete.")
    return train_loader, test_loader


def load_eurosat_data(
    batch_size: int, N: int, num_workers: int, verbose: bool = False
) -> tuple[DataLoader, DataLoader]:
    if verbose:
        logger.info("Loading and embedding EuroSAT data...")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            L2Normalize(),
            transforms.Lambda(lambda x: x.squeeze(0)),
            transforms.Lambda(lambda x: embedding_unitary(x)),
        ]
    )

    train_path = "data/EuroSAT_16x16/train"
    test_path = "data/EuroSAT_16x16/test"

    train_set = datasets.ImageFolder(train_path, transform=transform)
    test_set = datasets.ImageFolder(test_path, transform=transform)

    target_labels = torch.tensor([0, 6])
    train_idx = torch.isin(torch.tensor(train_set.targets), target_labels).nonzero(
        as_tuple=True
    )[0]
    test_idx = torch.isin(torch.tensor(test_set.targets), target_labels).nonzero(
        as_tuple=True
    )[0]

    train_filtered = Subset(train_set, train_idx)
    test_filtered = Subset(test_set, test_idx)

    train_final = Subset(train_filtered, list(range(20)))
    test_final = Subset(test_filtered, list(range(int(0.5 * N))))

    train_loader = DataLoader(
        train_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_final,
        batch_size=20,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
