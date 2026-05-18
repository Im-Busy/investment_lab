"""Common Trainer configurations — copy the one that matches your setup."""

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger


# === Quick Debug ===
debug_trainer = L.Trainer(
    fast_dev_run=True,
    enable_progress_bar=True,
)


# === Single GPU ===
single_gpu_trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=TensorBoardLogger("logs/", name="experiment"),
    log_every_n_steps=50,
)


# === Multi-GPU DDP ===
ddp_trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3, save_last=True),
        EarlyStopping(monitor="val/loss", patience=15, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=TensorBoardLogger("logs/", name="ddp_experiment"),
    log_every_n_steps=50,
    gradient_clip_val=1.0,
)


# === Large Model FSDP ===
fsdp_trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=8,
    strategy="fsdp",
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3, save_last=True),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    log_every_n_steps=50,
    gradient_clip_val=1.0,
)


# === CPU Only ===
cpu_trainer = L.Trainer(
    max_epochs=50,
    accelerator="cpu",
    callbacks=[
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
    ],
    logger=TensorBoardLogger("logs/", name="cpu_experiment"),
)


# === Gradient Accumulation (simulate large batch) ===
grad_accum_trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    accumulate_grad_batches=4,  # effective_batch = batch_size * 4
    precision="16-mixed",
)


# === Overfit Test (sanity check) ===
overfit_trainer = L.Trainer(
    max_epochs=50,
    overfit_batches=10,
    accelerator="auto",
    log_every_n_steps=1,
)
