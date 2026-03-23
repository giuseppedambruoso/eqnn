import logging
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from data_encoding import embedding_unitary

logger = logging.getLogger(__name__)

def seed_worker(worker_id):
    """Initializes DataLoader workers with a unique seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class L2Normalize(object):
    """Normalizes tensor to unit L2 norm for quantum state embedding."""
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(tensor.reshape(-1), ord=2, keepdim=True)
        return tensor / (l2_norm + 1e-12)

class QuantumTestAugmentation(object):
    """
    Applies exact p4m (D4) group transformations.
    
    Args:
        p (float): Probability of applying a non-identity transformation.
                   Set to 0.5 to transform half of the dataset.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Step 1: Decide whether to transform or keep identity
        if random.random() > self.p:
            return img

        # Step 2: Select one of the 7 non-identity group elements
        # (1: FlipX, 2: FlipY, 3: Rot180, 4: SWAP, 5: Rot90, 6: Rot270, 7: Anti-Diag)
        g_idx = random.randint(1, 7)
        
        if g_idx == 1:   # Horizontal Reflection
            return TF.hflip(img)
        elif g_idx == 2: # Vertical Reflection
            return TF.vflip(img)
        elif g_idx == 3: # 180 Rotation
            return TF.rotate(img, 180)
        elif g_idx == 4: # Transpose (Diagonal SWAP)
            return img.transpose(-1, -2)
        elif g_idx == 5: # 90 Rotation
            return TF.rotate(img, 90)
        elif g_idx == 6: # 270 Rotation
            return TF.rotate(img, 270)
        elif g_idx == 7: # Anti-diagonal Reflection
            return img.transpose(-1, -2).flip(-1).flip(-2)
        
        return img

def load_mnist_data(
    batch_size: int, 
    N: int, 
    num_workers: int, 
    seed: int = 42, 
    verbose: bool = False, 
    augment_test: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Loads MNIST data with deterministic subset selection and 
    optional p4m symmetry augmentation on the test set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N}, Augment Test={augment_test}")

    # Transformation Pipeline
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

    # Apply p4m augmentation to 50% of the test set
    test_transform_list = list(base_transforms)
    if augment_test:
        test_transform_list.append(QuantumTestAugmentation(p=0.5))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # Class selection and filtering
    switch = {3: 0, 4: 1, 0: 3, 1: 4}
    tar_transform = lambda y: switch.get(y, y)

    train_full = torchvision.datasets.MNIST(root="../../data", train=True, download=True,
                                            transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.MNIST(root="../../data", train=False, download=True,
                                           transform=test_transform, target_transform=tar_transform)

    target_labels = torch.tensor([3, 4])
    train_idx = torch.isin(train_full.targets, target_labels).nonzero(as_tuple=True)[0]
    test_idx = torch.isin(test_full.targets, target_labels).nonzero(as_tuple=True)[0]

    # Deterministic Subsampling
    g_select = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(len(train_idx), generator=g_select).tolist()
    test_perm = torch.randperm(len(test_idx), generator=g_select).tolist()

    train_final = Subset(Subset(train_full, train_idx), train_perm[:N])
    test_final = Subset(Subset(test_full, test_idx), test_perm[:N])

    # DataLoaders
    g_loader = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g_loader)
    
    test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, worker_init_fn=seed_worker)

    return train_loader, test_loader
