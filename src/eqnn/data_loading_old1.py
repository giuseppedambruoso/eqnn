#data_loading_new.py
import logging
import random # > # --- NEW --- <
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # > # --- NEW --- <
from data_encoding import embedding_unitary
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

logger = logging.getLogger(__name__)

class L2Normalize(object):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(tensor.view(-1), ord=2, keepdim=True)
        return tensor / (l2_norm + 1e-12)

# > # --- NEW: Augmentation Class --- <
class QuantumTestAugmentation(object):
    """
    Applies one of three transformations to 50% of the images:
    - Clockwise rotation of 90 degrees
    - Reflection around the y-axis
    - Reflection around the x-axis
    """
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        trans_type = self.rng.choice(['rot', 'flip_y', 'flip_x'])
        if trans_type == 'rot':
            return TF.rotate(img, -90) # -90 is clockwise
        elif trans_type == 'flip_y':
            return TF.hflip(img)
        elif trans_type == 'flip_x':
            return TF.vflip(img)
        return img
# > # ------------------------------- <


def load_mnist_data(
    batch_size: int, N: int, num_workers: int, seed: int, verbose: bool = False, augment_test: bool = False # > # --- MODIFIED --- <
) -> tuple[DataLoader, DataLoader]:
    if verbose:
        logger.info("Loading and embedding MNIST data...")

    # > # --- MODIFIED: Split transforms into train/test pipelines --- <
    base_transforms = [
        transforms.Resize(16),
        transforms.ToTensor(),
    ]
    
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]

    train_transform = transforms.Compose(base_transforms + post_transforms)

    test_transform_list = list(base_transforms)
    if augment_test:
        test_transform_list.append(QuantumTestAugmentation(seed=seed))
    test_transform = transforms.Compose(test_transform_list + post_transforms)
    # > # ------------------------------------------------------------- <

    switch = {0: 3, 1: 4, 3: 0, 4: 1}
    tar_transform = lambda y: switch[y]

    train_full = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=train_transform, # > # --- MODIFIED --- <
        target_transform=tar_transform,
    )
    test_full = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=test_transform, # > # --- MODIFIED --- <
        target_transform=tar_transform,
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
    batch_size: int, N: int, num_workers: int, seed: int, verbose: bool = False, augment_test: bool = False # > # --- MODIFIED --- <
) -> tuple[DataLoader, DataLoader]:
    if verbose:
        logger.info("Loading and embedding EuroSAT data...")

    # > # --- MODIFIED: Split transforms into train/test pipelines --- <
    base_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]

    train_transform = transforms.Compose(base_transforms + post_transforms)

    test_transform_list = list(base_transforms)
    if augment_test:
        test_transform_list.append(QuantumTestAugmentation(seed=seed))
    test_transform = transforms.Compose(test_transform_list + post_transforms)
    # > # ------------------------------------------------------------- <

    train_path = "data/EuroSAT_16x16/train"
    test_path = "data/EuroSAT_16x16/test"

    train_set = datasets.ImageFolder(train_path, transform=train_transform) # > # --- MODIFIED --- <
    test_set = datasets.ImageFolder(test_path, transform=test_transform) # > # --- MODIFIED --- <

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
