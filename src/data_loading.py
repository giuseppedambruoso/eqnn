import logging
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, TensorDataset

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


def _materialize(dataset: Subset) -> TensorDataset:
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

    switch = {3: 0, 4: 1, 0: 3, 1: 4}

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
        train_full.targets, 3, 4, N, g_select
    )
    test_balanced_idx = get_balanced_subset_indices(
        test_full.targets, 3, 4, N, g_select
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

    switch = {3: 0, 4: 1, 0: 3, 1: 4}

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
        train_full.targets, 3, 4, N, g_select
    )
    test_balanced_idx = get_balanced_subset_indices(
        test_full.targets, 3, 4, N, g_select
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
