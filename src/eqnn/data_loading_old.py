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

    train_full = torchvision.datasets.MNIST(root="data", train=True, download=True,
                                            transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.MNIST(root="data", train=False, download=True,
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
    and 9 (SeaLake), with manual train/test split, deterministic subset 
    selection, and optional p4m symmetry augmentation on the test set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N}, Augment Test={augment_test}")

    # Transformation Pipeline
    base_transforms = [
        transforms.Resize(16),
        transforms.Grayscale(num_output_channels=1), # Converte in bianco e nero (1 canale)
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

    # Class selection and filtering: 7 (Residential) -> 0, 9 (SeaLake) -> 1
    switch = {7: 0, 9: 1}
    tar_transform = lambda y: switch.get(y, y)

    # Carica il dataset completo due volte per applicare le diverse trasformazioni
    # Rimosso l'argomento 'train' che causava il TypeError
    dataset_full_train = torchvision.datasets.EuroSAT(
        root="data", download=True, transform=train_transform, target_transform=tar_transform
    )
    dataset_full_test = torchvision.datasets.EuroSAT(
        root="data", download=True, transform=test_transform, target_transform=tar_transform
    )

    # Trova gli indici di tutte le immagini che appartengono alle classi 7 e 9
    target_labels = torch.tensor([7, 9])
    all_targets = torch.tensor(dataset_full_train.targets)
    valid_idx = torch.isin(all_targets, target_labels).nonzero(as_tuple=True)[0]

    # Split manuale: mescola gli indici validi e dividi 80% Train / 20% Test
    g_split = torch.Generator().manual_seed(seed)
    shuffled_idx = valid_idx[torch.randperm(len(valid_idx), generator=g_split)]
    
    split_point = int(0.8 * len(shuffled_idx))
    train_idx = shuffled_idx[:split_point]
    test_idx = shuffled_idx[split_point:]

    # Subsampling deterministico per selezionare esattamente 'N' campioni
    g_select = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(len(train_idx), generator=g_select).tolist()
    test_perm = torch.randperm(len(test_idx), generator=g_select).tolist()

    # Mappa la permutazione ristretta a 'N' agli indici originali del dataset
    final_train_idx = [train_idx[i].item() for i in train_perm[:N]]
    final_test_idx = [test_idx[i].item() for i in test_perm[:N]]

    train_final = Subset(dataset_full_train, final_train_idx)
    test_final = Subset(dataset_full_test, final_test_idx)

    # Creazione dei DataLoaders
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
    for airplane and ship classes.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Costruiamo i path esatti basandoci sulla struttura dello screenshot
    # NOTA: Controlla se sul tuo PC la cartella si chiama "Dataset" o se estrai direttamente "train" e "test"
    data_dir = "data/NWPU-RESISC45"
    train_dir = os.path.join(data_dir, "train", "train")
    test_dir = os.path.join(data_dir, "test", "test")

    if verbose:
        logger.info(f"Loading NWPU data... Train dir: {train_dir}, Test dir: {test_dir}")

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
        test_transform_list.append(QuantumTestAugmentation(p=0.5))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # Estraiamo la mappatura delle classi dalla cartella di train
    temp_dataset = torchvision.datasets.ImageFolder(root=train_dir)
    airplane_idx = temp_dataset.class_to_idx.get('airplane')
    ship_idx = temp_dataset.class_to_idx.get('ship')

    if airplane_idx is None or ship_idx is None:
        raise ValueError(f"Classi 'airplane' o 'ship' non trovate in {train_dir}. Controlla la struttura delle cartelle.")

    # Class selection and filtering: Airplane -> 0, Ship -> 1
    switch = {airplane_idx: 0, ship_idx: 1}
    tar_transform = lambda y: switch.get(y, y)

    # Carichiamo i due dataset separatamente dalle rispettive cartelle
    train_full = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform, target_transform=tar_transform)

    # Filtriamo solo le classi di interesse e peschiamo gli indici validi
    target_labels = torch.tensor([airplane_idx, ship_idx])
    
    train_valid_idx = torch.isin(torch.tensor(train_full.targets), target_labels).nonzero(as_tuple=True)[0]
    test_valid_idx = torch.isin(torch.tensor(test_full.targets), target_labels).nonzero(as_tuple=True)[0]

    # Mescoliamo gli indici validi in modo deterministico
    g_split = torch.Generator().manual_seed(seed)
    train_shuffled = train_valid_idx[torch.randperm(len(train_valid_idx), generator=g_split)]
    test_shuffled = test_valid_idx[torch.randperm(len(test_valid_idx), generator=g_split)]

    # Subsampling fino a N elementi
    train_final = Subset(train_full, train_shuffled[:N].tolist())
    test_final = Subset(test_full, test_shuffled[:N].tolist())

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
    Loads MNIST data with rotations and zooms on 50% of the training set.
    Maintains the original test set transformation logic.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.info(f"Loading data... Subsampling N={N}, Augment Training (50% Rot/Zoom), Augment Test={augment_test}")

    # --- Common Base Transformations ---
    base_transforms = [
        transforms.Resize(16),
        # ToTensor must come before geometric transforms if using RandomApply with degrees
        transforms.ToTensor(), 
    ]
    
    # --- Post Transformations (Normalization & Embedding) ---
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]

    # --- Training Specific: 50% Rotation and Zoom ---
    # RandomAffine handles both rotation and scale (zoom)
    train_aug = transforms.RandomApply([
        transforms.RandomAffine(
            degrees=180,       # Rotations between -20 and +20 degrees
            scale=(0.5, 1.5)  # Zoom between 80% and 120%
        )
    ], p=0.5)

    train_transform = transforms.Compose(
        base_transforms + [train_aug] + post_transforms
    )

    # --- Test Specific: Maintain original logic ---
    test_transform_list = list(base_transforms)
    if augment_test:
        test_transform_list.append(QuantumTestAugmentation(p=0.5))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    # --- Class selection and filtering ---
    switch = {3: 0, 4: 1, 0: 3, 1: 4}
    tar_transform = lambda y: switch.get(y, y)

    train_full = torchvision.datasets.MNIST(root="data", train=True, download=True,
                                            transform=train_transform, target_transform=tar_transform)
    test_full = torchvision.datasets.MNIST(root="data", train=False, download=True,
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
        # Applied without RandomApply(p=0.5) to guarantee the augmentation 
        # is visible in every "After" image for visualization purposes.
        after.append(
            transforms.RandomAffine(degrees=180, scale=(0.5, 1.5))
        )
    # Add other dataset-specific augmentations here if needed in the future
    transform_after = transforms.Compose(after)

    # --- 3. Load dataset WITHOUT transforms to get raw PIL images ---
    # This allows us to apply the before/after pipelines manually to the exact same base image.
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

