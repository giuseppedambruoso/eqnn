import glob
import logging
import os
import random

import kagglehub
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.data_encoding import embedding_unitary

logger = logging.getLogger(__name__)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_balanced_subset_indices(
    targets: torch.Tensor, class1: int, class2: int, N: int, generator: torch.Generator
) -> list[int]:
    targets_tensor = torch.as_tensor(targets)
    idx1 = (targets_tensor == class1).nonzero(as_tuple=True)[0]
    idx2 = (targets_tensor == class2).nonzero(as_tuple=True)[0]

    n1, n2 = N // 2, N - (N // 2)
    sel1 = idx1[torch.randperm(len(idx1), generator=generator)[:n1]]
    sel2 = idx2[torch.randperm(len(idx2), generator=generator)[:n2]]

    combined = torch.cat((sel1, sel2))
    return combined[torch.randperm(len(combined), generator=generator)].tolist()


class L2Normalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(tensor.reshape(-1), ord=2, keepdim=True)
        return tensor / (l2_norm + 1e-12)


def _materialize(dataset: Dataset) -> TensorDataset:
    """Runs every item's transform pipeline (including the expensive
    embedding_unitary encoding, ~0.2-0.3s/image) exactly once, instead of
    on every DataLoader access. torchvision's Dataset/Subset apply their
    `transform` lazily on every __getitem__ call — with no augmentation
    randomizing the *training* transform, every epoch was silently
    recomputing the identical embedding for every image from scratch
    (measured: ~127s/epoch for N=640 train images, dwarfing the actual
    circuit training cost once diff_method="backprop" made that fast).

    Only safe to call on a transform pipeline that gives the SAME output
    every access — i.e. not one that includes D4Augmentation, which must
    keep re-randomizing every epoch to actually work as augmentation.
    """
    images, labels = zip(*[dataset[i] for i in range(len(dataset))], strict=True)
    return TensorDataset(torch.stack(images), torch.tensor(labels))


AUGMENT_TRAIN_MODES = ("none", "online", "once")


class D4Augmentation:
    """Applies a random p4m group transform (or leaves the image alone,
    with probability 1-p) — used for the augmented *test* set (a fixed
    one-off, always p=1) and, optionally, for *training* (p=1) in either
    of two ways — see load_mnist_data_full's augment_train modes: "online"
    (re-randomized every epoch, dataset left un-cached) or "once" (drawn a
    single time and then cached, like the test set)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        g_idx = random.randint(1, 8)

        if g_idx == 1:
            return TF.hflip(img)
        elif g_idx == 2:
            return TF.vflip(img)
        elif g_idx == 3:
            return TF.rotate(img, 180)
        elif g_idx == 4:
            return img.transpose(-1, -2)
        elif g_idx == 5:
            return TF.rotate(img, 90)
        elif g_idx == 6:
            return TF.rotate(img, 270)
        elif g_idx == 7:
            return img.transpose(-1, -2).flip(-1).flip(-2)
        return img


def load_mnist_data(
    batch_size: int,
    N: int,
    num_workers: int,
    img_size: int = 16,
    data_dir: str = "data",
    seed: int = 42,
    verbose: bool = False,
    augment_test: bool = False,
    class1: int = 3,
    class2: int = 4,
) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)

    base_transforms = [
        transforms.Resize(img_size),
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
        test_transform_list.append(D4Augmentation(p=1))
    test_transform = transforms.Compose(test_transform_list + post_transforms)

    switch = {class1: 0, class2: 1}

    def tar_transform(y: int) -> int:
        return switch.get(y, y)

    train_full = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=tar_transform,
    )
    test_full = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
        target_transform=tar_transform,
    )

    g_select = torch.Generator().manual_seed(seed)
    train_balanced_idx = get_balanced_subset_indices(
        train_full.targets, class1, class2, N, g_select
    )
    test_balanced_idx = get_balanced_subset_indices(
        test_full.targets, class1, class2, N, g_select
    )

    train_final = _materialize(Subset(train_full, train_balanced_idx))
    test_final = _materialize(Subset(test_full, test_balanced_idx))

    g_loader = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g_loader,
    )
    test_loader = DataLoader(
        test_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_loader, test_loader


def load_mnist_data_full(
    batch_size: int,
    N: int,
    num_workers: int,
    img_size: int = 16,
    data_dir: str = "data",
    seed: int = 42,
    verbose: bool = False,
    augment_train: str = "none",
    class1: int = 3,
    class2: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Like load_mnist_data, but returns (train_loader, test_loader,
    aug_test_loader) from a single call. Calling load_mnist_data twice
    (once with augment_test=False, once with augment_test=True — the
    pattern needed to get all three loaders) materializes the *training*
    set's embeddings twice, since only the test transform differs between
    the two calls; this builds it exactly once instead.

    augment_train selects how (if at all) a random p4m transform is
    applied to training images (matching the reference d4_eqcnn training
    script's random_d4_batch), one of:
      - "none" (default): no augmentation, training set cached as usual.
      - "online": a NEW random transform drawn every epoch, not a single
        one fixed for the whole run. This is why the training set is
        deliberately left un-cached in this mode: caching would freeze
        whichever random transform got drawn first for the entire run
        instead of re-randomizing it, defeating the point of augmentation.
        Expect training to take noticeably longer per epoch (every epoch
        re-runs the expensive embedding_unitary encoding for the whole
        training set instead of reusing a cached copy).
      - "once": a single random transform is drawn per image, up front,
        then cached and reused for every epoch — like the augmented test
        set. This is NOT the same regularizer as "online": the model
        never sees more than one orientation per training image, so it
        doesn't get exposed to the diversity that makes augmentation act
        as a regularizer against overfitting to a specific orientation —
        it's closer to training on a different, but still fixed, dataset.

    class1/class2 select which two original MNIST digit classes to use
    (remapped to labels 0/1 respectively) — default 3 vs 4.
    """
    if augment_train not in AUGMENT_TRAIN_MODES:
        raise ValueError(
            f"augment_train must be one of {AUGMENT_TRAIN_MODES}, got {augment_train!r}"
        )
    torch.manual_seed(seed)

    base_transforms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ]
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]
    train_transform_list = list(base_transforms)
    if augment_train in ("online", "once"):
        train_transform_list.append(D4Augmentation(p=1))
    train_transform = transforms.Compose(train_transform_list + post_transforms)
    test_transform = transforms.Compose(base_transforms + post_transforms)
    aug_test_transform = transforms.Compose(
        base_transforms + [D4Augmentation(p=1)] + post_transforms
    )

    switch = {class1: 0, class2: 1}

    def tar_transform(y: int) -> int:
        return switch.get(y, y)

    train_full = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=tar_transform,
    )
    test_full = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
        target_transform=tar_transform,
    )
    aug_test_full = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=aug_test_transform,
        target_transform=tar_transform,
    )

    g_select = torch.Generator().manual_seed(seed)
    train_balanced_idx = get_balanced_subset_indices(
        train_full.targets, class1, class2, N, g_select
    )
    test_balanced_idx = get_balanced_subset_indices(
        test_full.targets, class1, class2, N, g_select
    )

    train_subset = Subset(train_full, train_balanced_idx)
    # "online" is deliberately left un-cached (see the augment_train
    # docstring note above: caching would freeze the augmentation instead
    # of re-randomizing it every epoch). "none" and "once" are both a
    # single fixed transform (identity or one random draw) per image, so
    # both are safe to materialize/cache like the test sets.
    train_final: Subset | TensorDataset = (
        train_subset if augment_train == "online" else _materialize(train_subset)
    )
    test_final = _materialize(Subset(test_full, test_balanced_idx))
    aug_test_final = _materialize(Subset(aug_test_full, test_balanced_idx))

    g_loader = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g_loader,
    )
    test_loader = DataLoader(
        test_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    aug_test_loader = DataLoader(
        aug_test_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_loader, test_loader, aug_test_loader


# Label convention for load_aero_data_full: 0 = ship, 1 = plane.
AERO_LABELS = {"ship": 0, "plane": 1}


class _FileListDataset(Dataset):
    """A dataset over a flat list of (image_path, label) pairs, applying
    `transform` to each PIL image on access — the ships/planesnet chips
    aren't laid out as an ImageFolder (both classes come from separate
    Kaggle datasets, pre-filtered to only the positive-class files), so
    there's no torchvision dataset class that fits directly."""

    def __init__(self, samples: list[tuple[str, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def _positive_chip_files(kaggle_slug: str) -> list[str]:
    """Downloads (or reuses the local kagglehub cache for) one of
    rhammell's chip datasets and returns only the "object present" chips
    — filenames prefixed "1_" — sorted for a deterministic file order
    (glob's own order is filesystem-dependent). The "0_" (no object)
    chips are discarded entirely: neither class in the combined
    ship-vs-plane task is "background"."""
    dataset_path = kagglehub.dataset_download(kaggle_slug)
    return sorted(glob.glob(os.path.join(dataset_path, "**", "1_*.png"), recursive=True))


def _split_pool(
    files: list[str], train_frac: float, generator: torch.Generator
) -> tuple[list[str], list[str]]:
    """One-time, deterministic (seeded) train/test split of one class's
    file list, done BEFORE any N-based sampling — so the same image can
    never land in both the train and test set, for any N."""
    perm = torch.randperm(len(files), generator=generator).tolist()
    n_train = int(len(files) * train_frac)
    train_files = [files[i] for i in perm[:n_train]]
    test_files = [files[i] for i in perm[n_train:]]
    return train_files, test_files


def _sample_labeled(
    pool: list[str], n: int, generator: torch.Generator, label: int, pool_name: str
) -> list[tuple[str, int]]:
    """Samples exactly n files from pool without replacement, pairing each
    with `label`. Raises rather than silently truncating if the pool is
    too small — sampling with replacement here would mean the same chip
    appears more than once in a single train or test set."""
    if n > len(pool):
        raise ValueError(
            f"Requested {n} images from {pool_name}, but it only has "
            f"{len(pool)} available. Lower DATA.N or raise train_frac "
            "(if the shortfall is in a test pool) so every class has "
            "enough images to sample without duplicates."
        )
    idx = torch.randperm(len(pool), generator=generator)[:n].tolist()
    return [(pool[i], label) for i in idx]


def load_aero_data_full(
    batch_size: int,
    N: int,
    num_workers: int,
    img_size: int = 16,
    seed: int = 42,
    verbose: bool = False,
    augment_train: str = "none",
    train_frac: float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Ship-vs-plane binary classification, combined from two Kaggle
    satellite-chip datasets: rhammell/ships-in-satellite-imagery (label 0)
    and rhammell/planesnet (label 1). Only each dataset's positive-class
    chips are used (see _positive_chip_files) — the "no object" chips are
    discarded, since neither class here is meant to be "background".

    Each class's positive files are split ONCE, deterministically (seeded
    by `seed`), into a train pool and a test pool (train_frac each) before
    any N-based sampling — so no image is ever in both splits. For a given
    N, N//2 images are then sampled (without replacement) from each
    class's train pool for the train set, and N//2 from each class's test
    pool for the test set — guaranteeing perfect class balance in BOTH
    sets at every N, matching load_mnist_data_full's use of a single N for
    both loaders.

    Raises ValueError (via _sample_labeled) if N//2 exceeds either class's
    train or test pool size, rather than silently sampling fewer or
    duplicate images. The bottleneck is the ships dataset (1000 positive
    chips total, vs. planesnet's 8000): with the default train_frac=0.5,
    that allows N up to 1000 (500 ships + 500 planes) in each of the
    train/test sets.
    """
    if augment_train not in AUGMENT_TRAIN_MODES:
        raise ValueError(
            f"augment_train must be one of {AUGMENT_TRAIN_MODES}, got {augment_train!r}"
        )
    torch.manual_seed(seed)

    ship_files = _positive_chip_files("rhammell/ships-in-satellite-imagery")
    plane_files = _positive_chip_files("rhammell/planesnet")
    if verbose:
        logger.info(
            f"aero dataset: {len(ship_files)} ship chips, {len(plane_files)} plane chips"
        )

    g_split = torch.Generator().manual_seed(seed)
    ship_train_pool, ship_test_pool = _split_pool(ship_files, train_frac, g_split)
    plane_train_pool, plane_test_pool = _split_pool(plane_files, train_frac, g_split)

    n_ship, n_plane = N // 2, N - (N // 2)
    g_select = torch.Generator().manual_seed(seed)
    train_samples = _sample_labeled(
        ship_train_pool, n_ship, g_select, AERO_LABELS["ship"], "ship train pool"
    ) + _sample_labeled(
        plane_train_pool, n_plane, g_select, AERO_LABELS["plane"], "plane train pool"
    )
    test_samples = _sample_labeled(
        ship_test_pool, n_ship, g_select, AERO_LABELS["ship"], "ship test pool"
    ) + _sample_labeled(
        plane_test_pool, n_plane, g_select, AERO_LABELS["plane"], "plane test pool"
    )
    train_samples = [
        train_samples[i]
        for i in torch.randperm(len(train_samples), generator=g_select).tolist()
    ]
    test_samples = [
        test_samples[i]
        for i in torch.randperm(len(test_samples), generator=g_select).tolist()
    ]

    base_transforms = [
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    post_transforms = [
        L2Normalize(),
        transforms.Lambda(lambda x: x.squeeze(0)),
        transforms.Lambda(lambda x: embedding_unitary(x)),
    ]
    train_transform_list = list(base_transforms)
    if augment_train in ("online", "once"):
        train_transform_list.append(D4Augmentation(p=1))
    train_transform = transforms.Compose(train_transform_list + post_transforms)
    test_transform = transforms.Compose(base_transforms + post_transforms)
    aug_test_transform = transforms.Compose(
        base_transforms + [D4Augmentation(p=1)] + post_transforms
    )

    train_full = _FileListDataset(train_samples, train_transform)
    test_full = _FileListDataset(test_samples, test_transform)
    aug_test_full = _FileListDataset(test_samples, aug_test_transform)

    # Same "online" caveat as load_mnist_data_full: left un-cached so the
    # random p4m transform is re-drawn every epoch instead of frozen.
    train_final: Dataset | TensorDataset = (
        train_full if augment_train == "online" else _materialize(train_full)
    )
    test_final = _materialize(test_full)
    aug_test_final = _materialize(aug_test_full)

    g_loader = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g_loader,
    )
    test_loader = DataLoader(
        test_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    aug_test_loader = DataLoader(
        aug_test_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_loader, test_loader, aug_test_loader
