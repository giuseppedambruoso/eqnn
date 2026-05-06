import logging
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from data_encoding import embedding_unitary
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

logger = logging.getLogger(__name__)

def seed_worker(worker_id):
    """Initializes DataLoader workers with a unique seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_balanced_subset_indices(targets, class1: int, class2: int, N: int, generator: torch.Generator) -> list[int]:
    """
    Helper function to deterministically sample exactly N balanced indices 
    (N//2 from class1 and N//2 from class2).
    """
    targets_tensor = torch.as_tensor(targets)
    
    # Isolate indices for each class
    idx1 = (targets_tensor == class1).nonzero(as_tuple=True)[0]
    idx2 = (targets_tensor == class2).nonzero(as_tuple=True)[0]

    n1 = N // 2
    n2 = N - n1  # Handles odd N gracefully

    if len(idx1) < n1 or len(idx2) < n2:
        logger.warning(f"Not enough samples to guarantee balance. Requesting {n1}/{n2}, found {len(idx1)}/{len(idx2)}.")
        n1 = min(n1, len(idx1))
        n2 = min(n2, len(idx2))

    # Shuffle and select subset for each class
    sel1 = idx1[torch.randperm(len(idx1), generator=generator)[:n1]]
    sel2 = idx2[torch.randperm(len(idx2), generator=generator)[:n2]]

    # Combine and shuffle the final pool
    combined = torch.cat((sel1, sel2))
    return combined[torch.randperm(len(combined), generator=generator)].tolist()


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
        g_idx = random.randint(1, 8)

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
        elif g_idx == 8:
            return img
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
    Loads MNIST data with balanced subset selection and 
    optional p4m symmetry augmentation on the test set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N} (Balanced), Augment Test={augment_test}")

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
        test_transform_list.append(QuantumTestAugmentation(p=1))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # Class selection and filtering
    switch = {3: 0, 4: 1, 0: 3, 1: 4}
    tar_transform = lambda y: switch.get(y, y)

    train_full = torchvision.datasets.MNIST(root="data", train=True, download=True,
                                            transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.MNIST(root="data", train=False, download=True,
                                           transform=test_transform, target_transform=tar_transform)

    # Deterministic Balanced Subsampling
    g_select = torch.Generator().manual_seed(seed)
    train_balanced_idx = get_balanced_subset_indices(train_full.targets, 3, 4, N, g_select)
    test_balanced_idx = get_balanced_subset_indices(test_full.targets, 3, 4, N, g_select)

    train_final = Subset(train_full, train_balanced_idx)
    test_final = Subset(test_full, test_balanced_idx)

    # DataLoaders
    g_loader = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g_loader)
    
    test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, worker_init_fn=seed_worker)

    return train_loader, test_loader

def load_eurosat_data(
    batch_size: int, 
    N: int, 
    num_workers: int, 
    seed: int = 42, 
    verbose: bool = False, 
    augment_test: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Loads EuroSAT data in grayscale (1 channel) for classes 7 (Residential) 
    and 9 (SeaLake), with manual balanced train/test split, deterministic balanced 
    subset selection, and optional p4m symmetry augmentation on the test set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N} (Balanced), Augment Test={augment_test}")

    # Transformation Pipeline
    base_transforms = [
        transforms.Resize(16),
        transforms.Grayscale(num_output_channels=1),
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
        test_transform_list.append(QuantumTestAugmentation(p=1))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # Class selection and filtering: 7 (Residential) -> 0, 9 (SeaLake) -> 1
    switch = {7: 0, 9: 1}
    tar_transform = lambda y: switch.get(y, y)

    dataset_full_train = torchvision.datasets.EuroSAT(
        root="data", download=True, transform=train_transform, target_transform=tar_transform
    )
    dataset_full_test = torchvision.datasets.EuroSAT(
        root="data", download=True, transform=test_transform, target_transform=tar_transform
    )

    all_targets = torch.as_tensor(dataset_full_train.targets)

    # Separate classes before splitting to guarantee structural balance
    idx_class1 = (all_targets == 7).nonzero(as_tuple=True)[0]
    idx_class2 = (all_targets == 9).nonzero(as_tuple=True)[0]

    g_split = torch.Generator().manual_seed(seed)
    shuffled_1 = idx_class1[torch.randperm(len(idx_class1), generator=g_split)]
    shuffled_2 = idx_class2[torch.randperm(len(idx_class2), generator=g_split)]
    
    split_1 = int(0.8 * len(shuffled_1))
    split_2 = int(0.8 * len(shuffled_2))

    # Balanced Manual Split
    train_idx_1, test_idx_1 = shuffled_1[:split_1], shuffled_1[split_1:]
    train_idx_2, test_idx_2 = shuffled_2[:split_2], shuffled_2[split_2:]

    # Balanced Subsampling 
    n1 = N // 2
    n2 = N - n1
    g_select = torch.Generator().manual_seed(seed)

    sel_train_1 = train_idx_1[torch.randperm(len(train_idx_1), generator=g_select)[:n1]]
    sel_train_2 = train_idx_2[torch.randperm(len(train_idx_2), generator=g_select)[:n2]]
    train_comb = torch.cat((sel_train_1, sel_train_2))
    final_train_idx = train_comb[torch.randperm(len(train_comb), generator=g_select)].tolist()

    sel_test_1 = test_idx_1[torch.randperm(len(test_idx_1), generator=g_select)[:n1]]
    sel_test_2 = test_idx_2[torch.randperm(len(test_idx_2), generator=g_select)[:n2]]
    test_comb = torch.cat((sel_test_1, sel_test_2))
    final_test_idx = test_comb[torch.randperm(len(test_comb), generator=g_select)].tolist()

    train_final = Subset(dataset_full_train, final_train_idx)
    test_final = Subset(dataset_full_test, final_test_idx)

    # DataLoaders
    g_loader = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(
        train_final, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, worker_init_fn=seed_worker, generator=g_loader
    )
    
    test_loader = DataLoader(
        test_final, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, worker_init_fn=seed_worker
    )

    return train_loader, test_loader

def load_kaggle_nwpu_data(
    batch_size: int, 
    N: int, 
    num_workers: int, 
    seed: int = 42, 
    verbose: bool = False, 
    augment_test: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Loads NWPU-RESISC45 data from Kaggle (pre-split train/test folders) 
    for airplane and ship classes with balanced subsampling.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_dir = "data/NWPU-RESISC45"
    train_dir = os.path.join(data_dir, "train", "train")
    test_dir = os.path.join(data_dir, "test", "test")

    if verbose:
        logger.info(f"Loading NWPU data... Subsampling N={N} (Balanced).")

    # Transformation Pipeline
    base_transforms = [
        transforms.Resize((16, 16)),
        transforms.Grayscale(num_output_channels=1),
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
        test_transform_list.append(QuantumTestAugmentation(p=1))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    temp_dataset = torchvision.datasets.ImageFolder(root=train_dir)
    airplane_idx = temp_dataset.class_to_idx.get('airplane')
    ship_idx = temp_dataset.class_to_idx.get('ship')

    if airplane_idx is None or ship_idx is None:
        raise ValueError(f"Classi 'airplane' o 'ship' non trovate in {train_dir}.")

    # Class selection and filtering
    switch = {airplane_idx: 0, ship_idx: 1}
    tar_transform = lambda y: switch.get(y, y)

    train_full = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform, target_transform=tar_transform)

    # Deterministic Balanced Subsampling
    g_select = torch.Generator().manual_seed(seed)
    final_train_idx = get_balanced_subset_indices(train_full.targets, airplane_idx, ship_idx, N, g_select)
    final_test_idx = get_balanced_subset_indices(test_full.targets, airplane_idx, ship_idx, N, g_select)

    train_final = Subset(train_full, final_train_idx)
    test_final = Subset(test_full, final_test_idx)

    # DataLoaders
    g_loader = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g_loader)
    
    test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, worker_init_fn=seed_worker)

    return train_loader, test_loader

def load_aug_mnist_data(
    batch_size: int, 
    N: int, 
    num_workers: int, 
    seed: int = 42, 
    verbose: bool = False, 
    augment_test: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Loads MNIST data with balanced subset selection, 
    rotations and zooms on 50% of the training set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N} (Balanced), Augment Training (50% Rot/Zoom), Augment Test={augment_test}")

    # --- Common Base Transformations ---
    base_transforms = [
        transforms.Resize(16),
        transforms.ToTensor(), 
    ]
    
    # --- Post Transformations (Normalization & Embedding) ---
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]

    # --- Training Specific: 50% Rotation and Zoom ---
    train_aug = transforms.RandomApply([
        transforms.RandomAffine(
            degrees=180,       
            scale=(0.5, 1.5)  
        )
    ], p=0.5)

    train_transform = transforms.Compose(
        base_transforms + [train_aug] + post_transforms
    )

    # --- Test Specific ---
    test_transform_list = list(base_transforms)
    if augment_test:
        test_transform_list.append(QuantumTestAugmentation(p=1))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # --- Class selection and filtering ---
    switch = {3: 0, 4: 1, 0: 3, 1: 4}
    tar_transform = lambda y: switch.get(y, y)

    train_full = torchvision.datasets.MNIST(root="data", train=True, download=True,
                                            transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.MNIST(root="data", train=False, download=True,
                                           transform=test_transform, target_transform=tar_transform)

    # Deterministic Balanced Subsampling
    g_select = torch.Generator().manual_seed(seed)
    train_balanced_idx = get_balanced_subset_indices(train_full.targets, 3, 4, N, g_select)
    test_balanced_idx = get_balanced_subset_indices(test_full.targets, 3, 4, N, g_select)

    train_final = Subset(train_full, train_balanced_idx)
    test_final = Subset(test_full, test_balanced_idx)

    # DataLoaders
    g_loader = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g_loader)
    
    test_loader = DataLoader(test_final, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, worker_init_fn=seed_worker)

    return train_loader, test_loader

def save_raw_dataset_samples(dataset_name: str, seed: int = 42, data_dir: str = "data/NWPU-RESISC45"):
    """
    Loads 15 sample images from the dataset and saves them as '{dataset_name}_before_after.pdf'.
    Shows a side-by-side 'Before' and 'After' transformation for each image.
    """
    # --- 1. Define 'Before' transforms (Basic formatting only) ---
    base = [transforms.Resize((16, 16))]
    
    if dataset_name in ("eurosat", "nwpu"):
        base.append(transforms.Grayscale(num_output_channels=1))

    base.append(transforms.ToTensor())
    transform_before = transforms.Compose(base)

    # --- 2. Define 'After' transforms (Including augmentations) ---
    after = list(base)
    if dataset_name == "aug_mnist":
        after.append(
            transforms.RandomAffine(degrees=180, scale=(0.5, 1.5))
        )
    transform_after = transforms.Compose(after)

    # --- 3. Load dataset WITHOUT transforms to get raw PIL images ---
    if dataset_name == "mnist" or dataset_name == "aug_mnist":
        ds = torchvision.datasets.MNIST(
            root="data", train=True, download=True, transform=None
        )
        class_map = {3: "3", 4: "4"}
        targets = ds.targets
        mask = (targets == 3) | (targets == 4)
        idxs = mask.nonzero(as_tuple=True)[0]
        labels = [class_map[int(targets[i])] for i in idxs]

    elif dataset_name == "eurosat":
        ds = torchvision.datasets.EuroSAT(
            root="data", download=True, transform=None
        )
        targets = torch.as_tensor(ds.targets)
        mask = (targets == 7) | (targets == 9)
        idxs = mask.nonzero(as_tuple=True)[0]
        class_map = {7: "Residential", 9: "SeaLake"}
        labels = [class_map[int(targets[i])] for i in idxs]

    elif dataset_name == "nwpu":
        train_dir = os.path.join(data_dir, "train", "train")
        ds = torchvision.datasets.ImageFolder(root=train_dir, transform=None)
        airplane_idx = ds.class_to_idx.get("airplane")
        ship_idx = ds.class_to_idx.get("ship")

        if airplane_idx is None or ship_idx is None:
            raise ValueError(
                f"Classes 'airplane'/'ship' not found in {train_dir}"
            )

        targets = torch.as_tensor(ds.targets)
        mask = (targets == airplane_idx) | (targets == ship_idx)
        idxs = mask.nonzero(as_tuple=True)[0]
        class_map = {airplane_idx: "Airplane", ship_idx: "Ship"}
        labels = [class_map[int(targets[i])] for i in idxs]

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # --- 4. Deterministic selection of 15 samples ---
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(idxs), generator=g)[:15]

    # --- 5. Plot 5x6 grid (15 pairs of Before/After) ---
    fig, axes = plt.subplots(5, 6, figsize=(12, 10))
    axes = axes.flatten()

    for i, idx in enumerate(perm):
        # Extract the raw PIL image
        raw_img, _ = ds[idxs[idx]]

        # Process the image through both pipelines
        img_before = transform_before(raw_img)
        img_after = transform_after(raw_img)

        # Map to adjacent axes (e.g., axes 0 & 1 for the first image pair)
        ax_b = axes[2 * i]
        ax_a = axes[2 * i + 1]

        ax_b.imshow(img_before.squeeze(), cmap="gray")
        ax_b.set_title(f"Before ({labels[idx]})", fontsize=9)
        ax_b.axis("off")

        ax_a.imshow(img_after.squeeze(), cmap="gray")
        ax_a.set_title(f"After", fontsize=9)
        ax_a.axis("off")

    # Hide any remaining subplots if we requested fewer than 15 samples
    for i in range(2 * len(perm), len(axes)):
        axes[i].axis("off")

    suptitle = f"{dataset_name.upper()} - 15 Samples (Before & After Augmentation)"
    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    pdf_path = f"{dataset_name}_before_after.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    logger.info(f"✅ Saved {pdf_path} with 15 before/after sample pairs")
