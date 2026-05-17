# Trainer Configuration Reference

> **Load when**: configuring training runs, setting up multi-GPU, or debugging training behavior.

## Quick Trainer

```python
import lightning as L

trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",  # auto-detect GPU/TPU
    devices="auto",      # use all available
)
trainer.fit(model, datamodule=dm)
```

## Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 1000 | Max training epochs (negative = unlimited) |
| `max_steps` | int | -1 | Max training steps (overrides max_epochs) |
| `accelerator` | str | "auto" | "cpu", "gpu", "tpu", "auto" |
| `devices` | int/list | "auto" | Number of devices or [0,1] for specific GPUs |
| `strategy` | str | "auto" | "ddp", "fsdp", "deepspeed_stage_2", "auto" |
| `precision` | str | "32-true" | "16-mixed", "bf16-mixed", "32-true" |
| `gradient_clip_val` | float | None | Clip gradient norm |
| `gradient_clip_algorithm` | str | "norm" | "norm" or "value" |
| `accumulate_grad_batches` | int | 1 | Gradient accumulation steps |
| `log_every_n_steps` | int | 50 | How often to log metrics |
| `val_check_interval` | float | 1.0 | Validate every N epochs (or fraction) |
| `check_val_every_n_epoch` | int | 1 | Validate every N epochs |
| `limit_train_batches` | int/float | 1.0 | Limit train batches (for debugging) |
| `limit_val_batches` | int/float | 1.0 | Limit val batches |
| `limit_test_batches` | int/float | 1.0 | Limit test batches |
| `fast_dev_run` | bool/int | False | Run N batches only (for debugging) |
| `overfit_batches` | int/float | 0.0 | Overfit on N batches (sanity check) |
| `deterministic` | bool | False | Ensure reproducibility (slower) |
| `enable_progress_bar` | bool | True | Show TQDM progress bar |
| `enable_model_summary` | bool | True | Print model summary |
| `num_sanity_val_steps` | int | 2 | Validation steps before training starts |
| `reload_dataloaders_every_n_epochs` | int | 0 | Reload dataloaders (for dynamic data) |

## Callbacks Parameter

```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)

trainer = L.Trainer(
    callbacks=[
        ModelCheckpoint(monitor="val/loss", mode="min"),
        EarlyStopping(monitor="val/loss", patience=5),
        LearningRateMonitor(logging_interval="epoch"),
    ]
)
```

## Logger Parameter

```python
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

trainer = L.Trainer(
    logger=[
        TensorBoardLogger("logs/", name="my_model"),
        WandbLogger(project="my-project"),
    ]
)
```

## Precision Modes

| Precision | Speed | Memory | When |
|-----------|-------|--------|------|
| `"32-true"` | 1x | High | Debugging, small models |
| `"16-mixed"` | ~2x | Lower | Ampere+ GPUs (A100, RTX 3090+) |
| `"bf16-mixed"` | ~2x | Lower | Ampere+ GPUs, more stable than fp16 |

```python
# Mixed precision (recommended for Ampere+)
trainer = L.Trainer(precision="16-mixed")

# BF16 (more stable, Ampere+ only)
trainer = L.Trainer(precision="bf16-mixed")
```

## Distributed Training Quick Configs

| Scenario | Config |
|----------|--------|
| Single GPU | `devices=1, accelerator="gpu"` |
| Multi-GPU (DDP) | `devices=4, strategy="ddp"` |
| Large model (FSDP) | `devices=8, strategy="fsdp"` |
| DeepSpeed ZeRO-2 | `devices=8, strategy="deepspeed_stage_2"` |
| DeepSpeed ZeRO-3 | `devices=8, strategy="deepspeed_stage_3"` |
| CPU only | `accelerator="cpu"` |

## Common Patterns

### Debug Mode (fast)
```python
trainer = L.Trainer(fast_dev_run=True)  # Runs 1 batch train + val
trainer = L.Trainer(fast_dev_run=5)     # Runs 5 batches
```

### Overfit Sanity Check
```python
trainer = L.Trainer(overfit_batches=10, max_epochs=50)
# If loss doesn't go to near-zero, model is broken
```

### Resume Training
```python
trainer = L.Trainer(max_epochs=20)
trainer.fit(model, ckpt_path="last.ckpt")  # Restores optimizer state
```

### Gradient Accumulation (simulate larger batch)
```python
# Effective batch size = 32 * 4 = 128
trainer = L.Trainer(accumulate_grad_batches=4)
```

## Common Pitfalls

1. **`fast_dev_run=True` hides real errors** → Disable for serious debugging
2. **`accumulate_grad_batches` + batch norm** → BN stats computed on micro-batches, not accumulated
3. **`limit_train_batches` = 1.0 ≠ all batches** → Use -1 for actual unlimited
4. **`max_epochs` with `max_steps`** → Whichever hits first stops training
5. **`deterministic=True` is slow** → Only use for reproducibility tests
