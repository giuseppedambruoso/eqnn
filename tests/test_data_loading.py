import pytest

from src.data_loading import load_mnist_data


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
        N=100,
        num_workers=0,
        img_size=16,
        data_dir="data",
        seed=42,
        augment_test=augment,
    )

    assert verify_class_balance(train_loader) <= 0, "Train dataset is unbalanced!"
    assert verify_class_balance(test_loader) <= 0, "Test dataset is unbalanced!"
