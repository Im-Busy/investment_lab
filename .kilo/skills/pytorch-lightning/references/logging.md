# Logging Reference

> **Load when**: setting up experiment tracking with TensorBoard, W&B, MLflow, or CSV logging.

## Available Loggers

| Logger | Import | Best For |
|--------|--------|----------|
| TensorBoard | `lightning.pytorch.loggers.TensorBoardLogger` | Local visualization |
| W&B | `lightning.pytorch.loggers.WandbLogger` | Team collaboration, sweeps |
| MLflow | `lightning.pytorch.loggers.MLFlowLogger` | Model registry, deployment |
| CSV | `lightning.pytorch.loggers.CSVLogger` | Simple CSV output |
| Neptune | `lightning.pytorch.loggers.NeptuneLogger` | Enterprise MLOps |
| Comet | `lightning.pytorch.loggers.CometLogger` | Alternative to W&B |

## TensorBoard Logger

```python
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir="logs/",           # Root log directory
    name="my_experiment",       # Experiment name
    version=None,               # Auto-increment version
    default_hp_metric=True,     # Log hyperparams with final metric
)

trainer = L.Trainer(logger=logger)

# View: tensorboard --logdir logs/
```

## W&B Logger

```python
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(
    project="my-project",       # W&B project name
    name="experiment-1",        # Run name
    save_dir="logs/",           # Local log dir
    log_model=True,             # Log model checkpoints to W&B
    offline=False,              # Set True for no-internet mode
)

trainer = L.Trainer(logger=logger)
```

## MLflow Logger

```python
from lightning.pytorch.loggers import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="my-experiment",
    tracking_uri="http://localhost:5000",  # MLflow server
    run_name="run-1",
)

trainer = L.Trainer(logger=logger)
```

## CSV Logger

```python
from lightning.pytorch.loggers import CSVLogger

logger = CSVLogger(
    save_dir="logs/",
    name="csv_experiment",
)

# Produces: logs/csv_experiment/version_0/metrics.csv
```

## Multiple Loggers

```python
trainer = L.Trainer(
    logger=[
        TensorBoardLogger("logs/"),     # For local debugging
        WandbLogger(project="my-proj"), # For team sharing
    ]
)
```

## self.log() in LightningModule

```python
# Inside any LightningModule method
self.log("train/loss", loss)
self.log("val/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

# Log multiple at once
self.log_dict({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
})

# Log images, histograms, etc. through logger directly
if isinstance(self.logger, WandbLogger):
    self.logger.experiment.log({
        "samples": [wandb.Image(img) for img in images],
        "gradients": wandb.Histogram(grads),
    })
```

## Logging Frequency

| Scenario | `on_step` | `on_epoch` |
|----------|-----------|------------|
| Training loss | True | True |
| Training accuracy | False | True |
| Validation metrics | False | True |
| Learning rate | True | False |
| GPU utilization | True | False |

## Common Patterns

### Log Hyperparameters
```python
# In LightningModule.__init__()
self.save_hyperparameters()  # Auto-logs all __init__ args

# Or manually
logger.log_hyperparams({"lr": 1e-3, "batch_size": 32})
```

### Disable Logging
```python
trainer = L.Trainer(logger=False)  # No logging
```

### Custom Logging Directory
```python
logger = TensorBoardLogger(
    save_dir="experiments/",
    name=f"{model_name}",
    version=f"lr{lr}_bs{batch_size}",
)
```

## Common Pitfalls

1. **Forgetting `wandb.login()`** → Run `wandb login` before training
2. **Logging tensors with gradients** → Detach first: `loss.detach().item()`
3. **On-step logging in validation** → Use `on_step=False` for val metrics
4. **Logger expects scalar** → Log tensors with `.item()` for single values
5. **W&B offline mode** → Set `offline=True` and sync later with `wandb sync`
