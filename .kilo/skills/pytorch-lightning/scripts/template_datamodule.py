"""Complete LightningDataModule boilerplate — copy and customize.
Supports CSV, NumPy, and torchvision datasets with train/val/test splits.
"""

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


class TemplateDataModule(L.LightningDataModule):
    """Template DataModule for tabular/array data."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def prepare_data(self):
        """Download data if needed. Called once, single process."""
        # Example: download from URL
        # if not (self.data_dir / "data.csv").exists():
        #     import requests
        #     r = requests.get("https://example.com/data.csv")
        #     (self.data_dir / "data.csv").write_bytes(r.content)
        pass

    def setup(self, stage=None):
        """Create train/val/test splits. Called on every GPU."""
        # Replace with your data loading logic
        n_samples = 1000
        n_features = 20
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples,))

        dataset = TensorDataset(X, y)
        n = len(dataset)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class CSVDataModule(L.LightningDataModule):
    """DataModule for CSV files with train/val/test splits."""

    def __init__(
        self,
        csv_path: str,
        target_col: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        feature_cols = [c for c in df.columns if c != self.target_col]

        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df[self.target_col].values, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        n = len(dataset)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
