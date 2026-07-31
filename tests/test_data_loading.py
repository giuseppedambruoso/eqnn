import pytest
import torch

from src.data_loading import load_mnist_data, load_mnist_data_full


def verify_class_balance(dataloader, tolerance=0):
    class_counts = {0: 0, 1: 0}
    for _, labels in dataloader:
        for label in labels:
            lbl = int(label.item())
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

    diff = abs(class_counts.get(0, 0) - class_counts.get(1, 0))
    return diff


@pytest.mark.parametrize("augment", [False, True])
def test_dataset_balance(augment):
    train_loader, test_loader = load_mnist_data(
        batch_size=10,
        N=20,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_test=augment,
    )

    assert verify_class_balance(train_loader) <= 0, "Train dataset is unbalanced!"
    assert verify_class_balance(test_loader) <= 0, "Test dataset is unbalanced!"


def test_full_loader_matches_balance():
    train_loader, test_loader, aug_test_loader = load_mnist_data_full(
        batch_size=10, N=20, num_workers=0, img_size=16, data_dir="data", seed=42
    )
    assert verify_class_balance(train_loader) <= 0, "Train dataset is unbalanced!"
    assert verify_class_balance(test_loader) <= 0, "Test dataset is unbalanced!"
    assert verify_class_balance(aug_test_loader) <= 0, "Aug test dataset is unbalanced!"


def test_full_loader_test_and_aug_test_share_same_images():
    """The augmented test set must be built from the exact same underlying
    images as the plain test set (just with an extra random D4 transform
    applied before encoding), not a different random subset — otherwise
    they wouldn't be comparable pairwise."""
    _, test_loader, aug_test_loader = load_mnist_data_full(
        batch_size=10, N=20, num_workers=0, img_size=16, data_dir="data", seed=42
    )
    test_labels = torch.cat([labels for _, labels in test_loader])
    aug_test_labels = torch.cat([labels for _, labels in aug_test_loader])
    assert torch.equal(test_labels, aug_test_labels)
