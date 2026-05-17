# Callbacks Reference

> **Load when**: adding checkpointing, early stopping, LR monitoring, or custom training logic.

## Built-in Callbacks

### ModelCheckpoint — Save best/last models

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="val/loss",         # Metric to watch
    mode="min",                 # "min" or "max"
    save_top_k=3,               # Keep top 3 checkpoints
    save_last=True,             # Always keep last checkpoint
    dirpath="checkpoints/",     # Save directory
    filename="{epoch:02d}-{val_loss:.3f}",  # Naming pattern
    every_n_epochs=1,           # Save frequency
    save_on_train_epoch_end=False,  # Save after validation (recommended)
)
```

### EarlyStopping — Stop when plateau

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val/loss",
    patience=10,                # Epochs without improvement
    mode="min",
    min_delta=0.001,            # Minimum change to count as improvement
    check_on_train_epoch_end=False,  # Check after validation
)
```

### LearningRateMonitor — Track LR changes

```python
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(
    logging_interval="epoch",   # "epoch" or "step"
)
```

### Other Built-in Callbacks

| Callback | Purpose |
|----------|---------|
| `TQDMProgressBar` | Progress bars (enabled by default) |
| `RichProgressBar` | Rich formatted progress |
| `DeviceStatsMonitor` | GPU/CPU usage tracking |
| `RichModelSummary` | Rich-formatted model summary |
| `BatchSizeFinder` | Auto-find max batch size |
| `StochasticWeightAveraging` | SWA for better generalization |
| `Timer` | Track training duration |
| `GradientAccumulationScheduler` | Dynamic gradient accumulation |

## Custom Callbacks

Create custom callbacks by subclassing `Callback`:

```python
from lightning.pytorch.callbacks import Callback

class LogPredictionsCallback(Callback):
    def __init__(self, num_samples=8):
        self.num_samples = num_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:  # Only first batch
            x, y = batch
            preds = pl_module(x)
            # Log samples to logger
            trainer.logger.experiment.add_image(
                "val/predictions", plot_predictions(x, y, preds), trainer.current_epoch
            )
```

### Available Hooks (all callbacks)

| Hook | When |
|------|------|
| `on_fit_start(trainer, pl_module)` | Fit begins |
| `on_fit_end(trainer, pl_module)` | Fit ends |
| `on_train_start(trainer, pl_module)` | Training begins |
| `on_train_epoch_start(trainer, pl_module)` | Each train epoch start |
| `on_train_batch_start(trainer, pl_module, batch, batch_idx)` | Each batch start |
| `on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)` | Each batch end |
| `on_train_epoch_end(trainer, pl_module)` | Each train epoch end |
| `on_train_end(trainer, pl_module)` | Training ends |
| `on_validation_start(trainer, pl_module)` | Validation begins |
| `on_validation_epoch_start(trainer, pl_module)` | Each val epoch start |
| `on_validation_batch_start(trainer, pl_module, batch, batch_idx)` | Each val batch start |
| `on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)` | Each val batch end |
| `on_validation_epoch_end(trainer, pl_module)` | Each val epoch end |
| `on_test_start/epoch_start/batch_start/batch_end/epoch_end/test_end` | Same for test |
| `on_save_checkpoint(trainer, pl_module, checkpoint)` | Before save |
| `on_load_checkpoint(trainer, pl_module, checkpoint)` | After load |

## Common Patterns

### Save Only Best Model
```python
checkpoint = ModelCheckpoint(
    monitor="val/accuracy",
    mode="max",
    save_top_k=1,
    save_last=False,
)
```

### Multi-Metric Saving
```python
checkpoint = ModelCheckpoint(
    monitor="val/loss",
    save_top_k=3,
    filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
)
```

### Custom Metric Aggregation
```python
class AggregateMetricsCallback(Callback):
    def __init__(self):
        self.val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        all_preds = torch.cat([o["preds"] for o in self.val_outputs])
        all_labels = torch.cat([o["labels"] for o in self.val_outputs])
        # Compute custom metric on full dataset
        custom_metric = compute_my_metric(all_preds, all_labels)
        pl_module.log("val/custom_metric", custom_metric)
        self.val_outputs.clear()
```
