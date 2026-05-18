# Best Practices

> **Load when**: starting a new project, debugging training issues, or reviewing code quality.

## Code Organization

### Do This

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()        # 1. Always first
        self.model = self._build_model()   # 2. Build model
        self.loss_fn = nn.CrossEntropyLoss()  # 3. Define loss

    def _build_model(self):
        return nn.Sequential(...)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/acc", acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]
```

### Don't Do This

```python
# ❌ No save_hyperparameters
# ❌ .cuda() calls instead of self.device
# ❌ Manual .train()/.eval() calls
# ❌ Manual zero_grad() without automatic_optimization=False
# ❌ Logging tensors with gradients
# ❌ Missing sync_dist=True on val metrics
```

## Device Management

```python
# ✅ Good: Device-agnostic
def training_step(self, batch):
    x, y = batch
    x = x.to(self.device)  # or just rely on Lightning's automatic placement

# ❌ Bad: Hardcoded device
def training_step(self, batch):
    x, y = batch
    x = x.cuda()
```

## Reproducibility

```python
import lightning as L

# At script start
L.seed_everything(42, workers=True)

# In Trainer
trainer = L.Trainer(
    deterministic=True,  # Slower but reproducible
)
```

**WARNING**: `deterministic=True` can significantly slow down training on GPUs. Only use when you need exact reproducibility.

## Debugging

### Fast Dev Run
```python
trainer = L.Trainer(fast_dev_run=True)     # 1 batch train + val
trainer = L.Trainer(fast_dev_run=10)       # 10 batches
trainer = L.Trainer(fast_dev_run=True, overfit_batches=1)  # Overfit 1 batch
```

### Debug Flags
```python
trainer = L.Trainer(
    detect_anomaly=True,     # PyTorch anomaly detection (slow)
    profiler="simple",       # "simple", "advanced", "pytorch"
    log_every_n_steps=1,     # Log every step for debugging
)
```

### Model Summary
```python
# Printed automatically on fit()
# Or manually:
from lightning.pytorch.utilities import ModelSummary
summary = ModelSummary(model, max_depth=2)
print(summary)
```

## Hyperparameter Management

```python
class MyModel(L.LightningModule):
    def __init__(self, lr=1e-3, hidden_size=256, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()  # Saves ALL __init__ args

    # Access anywhere:
    # self.hparams.lr
    # self.hparams.hidden_size
    # self.hparams.dropout
```

For nested configs:
```python
def __init__(self, model_config: dict, train_config: dict):
    super().__init__()
    self.save_hyperparameters()  # Saves both dicts

    # Access: self.hparams.model_config["layers"]
```

## Metric Logging

```python
# ✅ Good: Semantic naming with slashes
self.log("train/loss", loss)
self.log("train/accuracy", acc)
self.log("val/loss", val_loss)
self.log("val/f1", f1)

# ✅ Good: Sync validation metrics in DDP
self.log("val/accuracy", acc, sync_dist=True)

# ❌ Bad: Flat naming
self.log("loss", loss)
self.log("val_loss", val_loss)

# ❌ Bad: Logging on_step for validation
self.log("val/loss", val_loss, on_step=True)  # Only needed for training
```

## Gradient Management

### Gradient Clipping
```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    return {"optimizer": optimizer, "gradient_clip_val": 1.0}
```

Or in Trainer:
```python
trainer = L.Trainer(gradient_clip_val=1.0)
```

### Gradient Accumulation
```python
trainer = L.Trainer(accumulate_grad_batches=4)
# Effective batch size = batch_size * 4
```

## Checkpoint Management

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    dirpath="checkpoints/",
    filename="{epoch:02d}-{val_loss:.3f}",
)

trainer = L.Trainer(callbacks=[checkpoint])

# Resume from checkpoint
trainer.fit(model, ckpt_path="last")          # Resume from last
trainer.fit(model, ckpt_path="best")          # Resume from best
trainer.fit(model, ckpt_path="path/to.ckpt")  # Resume from specific
```

## Common Pitfalls Checklist

- [ ] `self.save_hyperparameters()` called in `__init__()`
- [ ] `sync_dist=True` on all validation/test `self.log()` calls
- [ ] No `.cuda()` or `.to(device)` — use `self.device`
- [ ] `pin_memory=True` in DataLoaders when using GPU
- [ ] `L.seed_everything()` at script start
- [ ] `on_step=False` for validation metrics
- [ ] `on_epoch=True` for metrics you want in checkpoint filename
- [ ] Forward pass in `forward()`, reusable for inference
- [ ] Loss defined in `__init__()`, not inside training_step
- [ ] DataModule used instead of raw DataLoaders for multi-dataset projects

## Performance Tips

1. **Use `torch.compile()`** (PyTorch 2.0+): `self.model = torch.compile(self.model)`
2. **Use `pin_memory=True`** in all DataLoaders
3. **Use `persistent_workers=True`** to avoid worker respawn
4. **Use mixed precision**: `Trainer(precision="16-mixed")`
5. **Batch size tuning**: Maximize batch size that fits in GPU memory
6. **Profile before optimizing**: `Trainer(profiler="simple")`
