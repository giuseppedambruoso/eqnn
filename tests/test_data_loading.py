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


def test_augment_train_online_re_randomizes_each_access():
    """augment_train="online" must apply a FRESH random p4m transform
    every time an image is accessed (i.e. every epoch) — not a single one
    fixed for the whole run — so the training set is deliberately left
    uncached in that case (see load_mnist_data_full's docstring). Checked
    over several images since any single one has a 1/8 chance of
    coincidentally drawing the same (or an identity) transform twice."""
    train_loader, _, _ = load_mnist_data_full(
        batch_size=10,
        N=20,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_train="online",
    )
    dataset = train_loader.dataset
    assert any(not torch.equal(dataset[i][0], dataset[i][0]) for i in range(5))


def test_augment_train_once_is_cached_but_transformed():
    """augment_train="once" must draw a random p4m transform per image a
    single time and then cache it — repeated access to the same image
    gives an identical embedding (unlike "online"), but the training set
    as a whole must differ from the untransformed ("none") one, since most
    images did receive some transform."""
    train_loader, _, _ = load_mnist_data_full(
        batch_size=10,
        N=20,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_train="once",
    )
    dataset = train_loader.dataset
    for i in range(5):
        assert torch.equal(dataset[i][0], dataset[i][0])

    plain_loader, _, _ = load_mnist_data_full(
        batch_size=10,
        N=20,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_train="none",
    )
    plain_dataset = plain_loader.dataset
    assert any(
        not torch.equal(dataset[i][0], plain_dataset[i][0]) for i in range(len(dataset))
    )


def test_no_augment_train_is_cached():
    """augment_train="none" (the default) must keep the fast, cached
    behavior: repeated access to the same image gives the identical
    embedding, since no randomization is applied to it."""
    train_loader, _, _ = load_mnist_data_full(
        batch_size=10,
        N=20,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_train="none",
    )
    dataset = train_loader.dataset
    for i in range(5):
        assert torch.equal(dataset[i][0], dataset[i][0])


def test_invalid_augment_train_raises():
    with pytest.raises(ValueError, match="augment_train"):
        load_mnist_data_full(
            batch_size=10,
            N=20,
            num_workers=0,
            img_size=16,
            data_dir="data",
            seed=42,
            augment_train="always",
        )
