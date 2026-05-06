import unittest
import torch
from data_loading import (
    load_mnist_data, 
    load_eurosat_data, 
    load_kaggle_nwpu_data, 
    load_aug_mnist_data
)

class TestDatasetBalance(unittest.TestCase):
    def setUp(self):
        # Set up a small subset size for quick testing
        self.N = 100 
        self.batch_size = 10
        self.seed = 42

    def verify_class_balance(self, dataloader, dataset_name, split_name, tolerance=0):
        """
        Iterates through a dataloader and counts the labels.
        Asserts that the difference between class 0 and class 1 is within the tolerance.
        """
        class_counts = {0: 0, 1: 0}
        
        for _, labels in dataloader:
            for label in labels:
                lbl = int(label.item())
                if lbl not in class_counts:
                    class_counts[lbl] = 0
                class_counts[lbl] += 1
                
        count_0 = class_counts.get(0, 0)
        count_1 = class_counts.get(1, 0)
        
        # Calculate the difference between the two classes
        diff = abs(count_0 - count_1)
        
        # Assert that the difference is within your acceptable limit
        error_msg = (f"{dataset_name} ({split_name}) is unbalanced! "
                     f"Class 0: {count_0}, Class 1: {count_1}")
        self.assertTrue(diff <= tolerance, error_msg)

    def test_mnist_balance(self):
        train_loader, test_loader = load_mnist_data(
            batch_size=self.batch_size, N=self.N, num_workers=0, seed=self.seed, verbose=False, augment_test=False
        )
        self.verify_class_balance(train_loader, "MNIST", "Train")
        self.verify_class_balance(test_loader, "MNIST", "Test")

    def test_eurosat_balance(self):
        train_loader, test_loader = load_eurosat_data(
            batch_size=self.batch_size, N=self.N, num_workers=0, seed=self.seed, verbose=False, augment_test=False
        )
        self.verify_class_balance(train_loader, "EuroSAT", "Train")
        self.verify_class_balance(test_loader, "EuroSAT", "Test")

    def test_nwpu_balance(self):
        train_loader, test_loader = load_kaggle_nwpu_data(
            batch_size=self.batch_size, N=self.N, num_workers=0, seed=self.seed, verbose=False, augment_test=False
        )
        self.verify_class_balance(train_loader, "NWPU", "Train")
        self.verify_class_balance(test_loader, "NWPU", "Test")

    def test_aug_mnist_balance(self):
        train_loader, test_loader = load_aug_mnist_data(
            batch_size=self.batch_size, N=self.N, num_workers=0, seed=self.seed, verbose=False, augment_test=True
        )
        self.verify_class_balance(train_loader, "Augmented MNIST", "Train")
        self.verify_class_balance(test_loader, "Augmented MNIST", "Test")

if __name__ == "__main__":
    unittest.main()
