import logging

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from data_encoding import embedding_unitary

logger = logging.getLogger(__name__)


class L2Normalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(tensor.view(-1), ord=2, keepdim=True)
        return tensor / (l2_norm + 1e-12)


def load_mnist_data(
    batch_size: int, N: int, num_workers: int, verbose: bool = False
) -> tuple[DataLoader, DataLoader]:
    if verbose:
        logger.info("Loading and embedding data...")

    transform = transforms.Compose(
        [
            transforms.Resize(16),
            transforms.ToTensor(),
            L2Normalize(),
            transforms.Lambda(lambda x: x.squeeze(0)),
            transforms.Lambda(lambda x: embedding_unitary(x)),
        ]
    )

    train_full = torchvision.datasets.EMNIST(
        root="../data", train=True, download=True, transform=transform, split="digits"
    )
    test_full = torchvision.datasets.EMNIST(
        root="../data", train=False, download=True, transform=transform, split="digits"
    )

    target_labels = torch.tensor([0, 1])
    train_idx = torch.isin(train_full.targets, target_labels).nonzero(as_tuple=True)[0]
    test_idx = torch.isin(test_full.targets, target_labels).nonzero(as_tuple=True)[0]

    train_filtered = Subset(train_full, train_idx)
    test_filtered = Subset(test_full, test_idx)

    train_final = Subset(train_filtered, list(range(N)))
    test_final = Subset(test_filtered, list(range(int(0.5 * N))))

    train_loader = DataLoader(
        train_final, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_final, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info("Data loading complete.")
    return train_loader, test_loader
