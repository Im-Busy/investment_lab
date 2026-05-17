# LightningDataModule Reference

> **Load when**: setting up data pipelines, handling multi-file datasets, or making training data reusable.

## When to Use

Use `LightningDataModule` instead of raw DataLoaders when:
- You have multiple datasets (train/val/test)
- You need transforms applied consistently
- You want to share data loading code across projects
- You need dataset preparation that runs once (download, tokenize, etc.)

**Use raw DataLoaders** for quick prototyping with a single dataset.

## The Five Methods

```python
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # 1. Download data (called once, single process)
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    # 2. Create datasets (called on every GPU in DDP)
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    # 3. Training DataLoader
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # 4. Validation DataLoader
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # 5. Test DataLoader
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

## Method Details

### `prepare_data()`
- **Called once per process** (not per GPU)
- **Use for**: downloading, tokenizing, preprocessing to disk
- **Do NOT**: assign to `self` (state won't be shared across processes)
- **Called from**: main process only

### `setup(stage=None)`
- **Called on every GPU** in DDP
- **Uses `self`** — state is per-process
- **`stage` parameter**: `"fit"` (train+val), `"validate"` (val only), `"test"`, `"predict"`
- **Use for**: creating dataset objects, applying transforms, splitting

### DataLoader Methods
Each returns a `torch.utils.data.DataLoader`.

**Key DataLoader Parameters**:
| Param | Recommended | Why |
|-------|-------------|-----|
| `batch_size` | 32-256 | Memory/GPU dependent |
| `num_workers` | 4-8 | I/O parallelism |
| `pin_memory=True` | Yes (GPU) | Faster CPU→GPU transfer |
| `shuffle=True` | Train only | Not for val/test |
| `drop_last=True` | If batches uneven | Avoid BN issues |
| `persistent_workers=True` | Yes | Avoid worker respawn |

## Usage

```python
dm = MNISTDataModule(batch_size=64)

# Option 1: Pass to Trainer
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=dm)

# Option 2: Access DataLoaders directly
dm.prepare_data()
dm.setup("fit")
train_loader = dm.train_dataloader()
```

## Common Patterns

### Custom Dataset with DataFrame
```python
class CSVDatamodule(L.LightningDataModule):
    def setup(self, stage=None):
        df = pd.read_csv("data.csv")
        dataset = TensorDataset(
            torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32),
            torch.tensor(df.iloc[:, -1].values, dtype=torch.float32),
        )
        n = len(dataset)
        self.train, self.val, self.test = random_split(
            dataset, [int(n*0.7), int(n*0.15), n - int(n*0.7) - int(n*0.15)]
        )
```

### HuggingFace Dataset Integration
```python
from datasets import load_dataset

class HFDataModule(L.LightningDataModule):
    def prepare_data(self):
        load_dataset("imdb")  # Download only

    def setup(self, stage=None):
        dataset = load_dataset("imdb")
        # Tokenize, transform, etc.
```

## Common Pitfalls

1. **Assigning state in `prepare_data()`** → `self` in prepare_data is not shared
2. **Not using `pin_memory=True` on GPU** → 2-3x slower data transfer
3. **`shuffle=True` on validation** → Metrics become order-dependent
4. **num_workers=0 on Windows** → Must guard with `if __name__ == "__main__"`
5. **Large transforms in DataLoader** → Do heavy transforms in `setup()`, not in the collate function
