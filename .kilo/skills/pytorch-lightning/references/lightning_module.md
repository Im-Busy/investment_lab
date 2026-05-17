# LightningModule Reference

> **Load when**: defining models, implementing training/validation/test steps, configuring optimizers, or understanding hooks and properties.

## When to Use

Use `LightningModule` for any neural network training. It organizes PyTorch code into six logical sections and eliminates boilerplate (`.to(device)`, `.train()`, `.eval()`, `zero_grad()`, etc.).

**Do NOT use raw `nn.Module`** when you need training automation, multi-device support, or logging callbacks. Wrap it in LightningModule.

## The Six Methods

### 1. `__init__()` — Define layers and hyperparameters

```python
import lightning as L
import torch.nn as nn

class MyModel(L.LightningModule):
    def __init__(self, in_features=128, hidden=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Saves in_features, hidden, lr
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.loss_fn = nn.MSELoss()
```

**CRITICAL**: Always call `self.save_hyperparameters()` — it auto-logs to W&B/MLflow/TensorBoard and enables checkpoint restoration without config files.

### 2. `training_step(batch, batch_idx)` — Train on one batch

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    pred = self.net(x)
    loss = self.loss_fn(pred, y)
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    return loss  # Must return loss tensor
```

### 3. `validation_step(batch, batch_idx)` — Validate on one batch

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self.net(x)
    loss = self.loss_fn(pred, y)
    self.log("val/loss", loss, on_epoch=True, prog_bar=True)
    return loss
```

### 4. `test_step(batch, batch_idx)` — Test on one batch

```python
def test_step(self, batch, batch_idx):
    x, y = batch
    pred = self.net(x)
    loss = self.loss_fn(pred, y)
    self.log("test/loss", loss, on_epoch=True)
    return loss
```

### 5. `predict_step(batch, batch_idx)` — Inference

```python
def predict_step(self, batch, batch_idx):
    x, _ = batch
    return self.net(x)  # Raw predictions, no loss
```

### 6. `configure_optimizers()` — Return optimizer (and optional scheduler)

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

## Key Hooks (Call at Specific Points)

| Hook | When Called | Use For |
|------|-------------|---------|
| `on_train_start()` | Before training begins | Set flags, log config |
| `on_train_epoch_start()` | Before each training epoch | Reset metrics |
| `on_train_batch_start(batch, batch_idx)` | Before each batch | Dynamic batch adjustment |
| `on_train_batch_end(outputs, batch, batch_idx)` | After each batch | Custom logging |
| `on_train_epoch_end()` | After each training epoch | Epoch-level metrics |
| `on_validation_epoch_start()` | Before validation | Setup validation |
| `on_validation_epoch_end()` | After validation | Aggregate metrics |
| `on_test_epoch_end()` | After test | Final results |
| `on_save_checkpoint(checkpoint)` | Before checkpoint save | Add custom state |
| `on_load_checkpoint(checkpoint)` | After checkpoint load | Restore custom state |

## Properties

| Property | Returns | Use |
|----------|---------|-----|
| `self.device` | Current device (cuda:0, cpu) | Device-agnostic ops |
| `self.trainer` | Trainer instance | Access callbacks, logger |
| `self.logger` | Logger (WandbLogger, etc.) | Manual logging |
| `self.current_epoch` | Current epoch (int) | Conditional logic |
| `self.global_step` | Global step (int) | LR scheduling |
| `self.hparams` | Saved hyperparameters | Access config |
| `self.automatic_optimization` | Bool (default True) | Manual optimization |

## self.log() API

```python
self.log(
    name,           # Metric name (str)
    value,          # Scalar tensor
    prog_bar=False, # Show in progress bar
    on_step=False,  # Log per batch
    on_epoch=True,  # Aggregate per epoch
    sync_dist=False # Average across GPUs (DDP)
)
```

## Common Patterns

### Multi-GPU Safe Logging
```python
self.log("val/accuracy", acc, on_epoch=True, sync_dist=True)
```
Always set `sync_dist=True` for validation/test metrics in DDP.

### Manual Optimization
```python
def __init__(self):
    super().__init__()
    self.automatic_optimization = False

def training_step(self, batch, batch_idx):
    opt = self.optimizers()
    opt.zero_grad()
    loss = self.compute_loss(batch)
    self.manual_backward(loss)
    opt.step()
```

### Gradient Clipping
```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    return {"optimizer": optimizer, "gradient_clip_val": 1.0}
```

## Common Pitfalls

1. **Forgetting `self.save_hyperparameters()`** → Checkpoint can't restore without config
2. **Not setting `sync_dist=True`** → Validation metrics wrong in multi-GPU
3. **Logging tensors with gradients** → Use `loss.detach()` or let Lightning handle it
4. **Using `.cuda()` explicitly** → Use `self.device` instead; Lightning handles placement
5. **Training step doesn't return loss** → Must return a loss tensor or None
