"""
TTS-GAN: Transformer-based Time Series Generative Adversarial Network.

Generates synthetic financial OHLCV data via a transformer-based GAN
architecture. Used for data augmentation to improve forecasting model
performance when training data is scarce.

Architecture (Podobinski & Chudziak, 2024):
- Generator: noise → linear → positional encoding → transformer encoder
  blocks → linear projection → synthetic time series
- Discriminator: input → linear → positional encoding → transformer encoder
  blocks → global pooling → real/fake classification
- Simplified Gradient Penalty: applied only to real samples (Mescheder et al.)

Key paper hyperparameters:
  D_g=3, H_g=5, M_g=10, P_g=15  (generator)
  D_d=3, H_d=30, M_d=90, P_d=15 (discriminator)

Reference: Podobinski & Chudziak (2024), Warsaw University of Technology.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .gan_convergence import GANConvergenceMonitor

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TTSGenerator(nn.Module):
    """Transformer-based generator for TTS-GAN.

    Takes a latent noise vector and generates a synthetic time series
    of shape (batch, seq_len, data_dim).
    """

    def __init__(
        self,
        seq_len: int = 90,
        data_dim: int = 4,
        latent_dim: int = 32,
        embed_dim: int = 10,
        n_heads: int = 5,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            seq_len: Length of generated time series.
            data_dim: Number of features per time step (4 for OHLCV).
            latent_dim: Dimension of input noise vector.
            embed_dim: Transformer embedding dimension (M_g in paper).
            n_heads: Number of attention heads (H_g in paper).
            n_layers: Number of transformer encoder layers (D_g in paper).
            dropout: Dropout rate.
        """
        super().__init__()
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.latent_proj = nn.Linear(latent_dim, seq_len * embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, data_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        x = self.latent_proj(z).view(batch, self.seq_len, self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return torch.tanh(x)


class TTSDiscriminator(nn.Module):
    """Transformer-based discriminator for TTS-GAN.

    Classifies time series as real or fake.
    Uses higher capacity than generator per paper findings
    (D_g=3, H_g=5, M_g=10 vs D_d=3, H_d=30, M_d=90).
    """

    def __init__(
        self,
        seq_len: int = 90,
        data_dim: int = 4,
        embed_dim: int = 90,
        n_heads: int = 30,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            seq_len: Length of input time series.
            data_dim: Number of features per time step.
            embed_dim: Transformer embedding dimension (M_d in paper).
            n_heads: Number of attention heads (H_d in paper).
            n_layers: Number of transformer encoder layers (D_d in paper).
            dropout: Dropout rate.
        """
        super().__init__()
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(data_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.cls_head(cls_out).squeeze(-1)


def _compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Simplified Gradient Penalty — applied only to real samples.

    Per Mescheder et al. (2018), penalizes gradient norm on real data
    to stabilize GAN training on volatile financial time series.

    Args:
        discriminator: Discriminator model.
        real_samples: Real data batch, shape (B, T, D).
        lambda_gp: Gradient penalty weight.

    Returns:
        Gradient penalty loss.
    """
    real_samples = real_samples.detach().requires_grad_(True)
    d_real = discriminator(real_samples)
    grad_outputs = torch.ones_like(d_real)
    gradients = torch.autograd.grad(
        outputs=d_real,
        inputs=real_samples,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 0) ** 2).mean()
    return gp


class TTSGAN:
    """TTS-GAN trainer for financial time series data augmentation.

    Manages generator/discriminator training, convergence monitoring,
    and synthetic data generation.
    """

    def __init__(
        self,
        seq_len: int = 90,
        data_dim: int = 4,
        latent_dim: int = 32,
        g_embed_dim: int = 10,
        g_n_heads: int = 5,
        g_n_layers: int = 3,
        d_embed_dim: int = 90,
        d_n_heads: int = 30,
        d_n_layers: int = 3,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        gp_lambda: float = 10.0,
        device: str | None = None,
    ) -> None:
        """
        Args:
            seq_len: Length of time series sequences.
            data_dim: Number of features (4 for OHLCV).
            latent_dim: Latent noise dimension.
            g_embed_dim: Generator embedding dim.
            g_n_heads: Generator attention heads.
            g_n_layers: Generator transformer layers.
            d_embed_dim: Discriminator embedding dim.
            d_n_heads: Discriminator attention heads.
            d_n_layers: Discriminator transformer layers.
            lr_g: Generator learning rate.
            lr_d: Discriminator learning rate.
            gp_lambda: Gradient penalty weight.
            device: Torch device string.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self.generator = TTSGenerator(
            seq_len=seq_len,
            data_dim=data_dim,
            latent_dim=latent_dim,
            embed_dim=g_embed_dim,
            n_heads=g_n_heads,
            n_layers=g_n_layers,
        ).to(self.device)

        self.discriminator = TTSDiscriminator(
            seq_len=seq_len,
            data_dim=data_dim,
            embed_dim=d_embed_dim,
            n_heads=d_n_heads,
            n_layers=d_n_layers,
        ).to(self.device)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9)
        )
        self.gp_lambda = gp_lambda

        self.monitor = GANConvergenceMonitor()

    def _generate_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def train_step(
        self,
        real_batch: torch.Tensor,
        n_critic: int = 5,
    ) -> dict[str, float]:
        """Single training step.

        Trains discriminator for n_critic iterations per generator update.
        Uses Simplified Gradient Penalty on real samples only.

        Args:
            real_batch: Real data, shape (B, T, D).
            n_critic: Number of discriminator updates per generator update.

        Returns:
            Dict with loss values.
        """
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)

        d_losses = []
        for _ in range(n_critic):
            self.optimizer_d.zero_grad()
            z = self._generate_noise(batch_size)
            with torch.no_grad():
                fake = self.generator(z)
            d_real = self.discriminator(real_batch)
            d_fake = self.discriminator(fake.detach())
            gp = _compute_gradient_penalty(self.discriminator, real_batch, self.gp_lambda)
            d_loss = (
                F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                + gp
            )
            d_loss.backward()
            self.optimizer_d.step()
            d_losses.append(d_loss.item())

        self.optimizer_g.zero_grad()
        z = self._generate_noise(batch_size)
        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        g_loss.backward()
        self.optimizer_g.step()

        return {"d_loss": float(np.mean(d_losses)), "g_loss": float(g_loss.item())}

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        n_critic: int = 5,
        convergence_patience: int = 20,
        convergence_min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> dict:
        """Train TTS-GAN on real financial time series.

        Args:
            train_data: Training samples, shape (N, T, D).
            val_data: Validation samples for convergence monitoring.
            epochs: Maximum training epochs.
            batch_size: Batch size.
            n_critic: Discriminator updates per generator update.
            convergence_patience: Epochs without improvement to stop.
            convergence_min_delta: Minimum improvement threshold.
            verbose: Print progress every 10 epochs.

        Returns:
            Dict with training history.
        """
        dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history: dict[str, list[float]] = {
            "g_loss": [],
            "d_loss": [],
            "dtw_dedims": [],
            "wasserstein": [],
        }

        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            for (real_batch,) in loader:
                losses = self.train_step(real_batch, n_critic=n_critic)
                epoch_g_loss += losses["g_loss"]
                epoch_d_loss += losses["d_loss"]
                n_batches += 1

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            history["g_loss"].append(avg_g)
            history["d_loss"].append(avg_d)

            self.generator.eval()
            with torch.no_grad():
                z = self._generate_noise(len(val_data))
                synth_val = self.generator(z).cpu().numpy()
            metrics = self.monitor.record(val_data, synth_val, epoch)
            history["dtw_dedims"].append(metrics["dtw_dedims"])
            history["wasserstein"].append(metrics["wasserstein"])

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"G_loss={avg_g:.4f} | D_loss={avg_d:.4f} | "
                    f"DTW-DeD-iMs={metrics['dtw_dedims']:.4f} | "
                    f"Wasserstein={metrics['wasserstein']:.4f}"
                )

            if self.monitor.has_converged(
                patience=convergence_patience, min_improvement=convergence_min_delta
            ):
                logger.info(f"Converged at epoch {epoch + 1}")
                break

        return history

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic time series samples.

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            Array of shape (n_samples, seq_len, data_dim).
        """
        self.generator.eval()
        all_synth = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                bs = min(batch_size, n_samples - i)
                z = self._generate_noise(bs)
                synth = self.generator(z).cpu().numpy()
                all_synth.append(synth)
        return np.concatenate(all_synth, axis=0)

    def save(self, path: str) -> None:
        """Save generator and discriminator state dicts."""
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "seq_len": self.seq_len,
                "data_dim": self.data_dim,
                "latent_dim": self.latent_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "TTSGAN":
        """Load TTS-GAN from saved state dict."""
        checkpoint = torch.load(path, map_location="cpu")
        kwargs.setdefault("seq_len", checkpoint["seq_len"])
        kwargs.setdefault("data_dim", checkpoint["data_dim"])
        kwargs.setdefault("latent_dim", checkpoint["latent_dim"])
        model = cls(**kwargs)
        model.generator.load_state_dict(checkpoint["generator"])
        model.discriminator.load_state_dict(checkpoint["discriminator"])
        return model


# ═══════════════════════════════════════════════════════════════════════
# Data Preparation Utilities
# ═══════════════════════════════════════════════════════════════════════


def _ema_smooth(series: np.ndarray, span: int = 5) -> np.ndarray:
    """Exponential moving average smoothing to reduce high-frequency noise."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def prepare_gan_samples(
    df: pd.DataFrame,
    seq_len: int = 90,
    step: int = 1,
    columns: Optional[list[str]] = None,
    smooth: bool = True,
    ema_span: int = 5,
) -> np.ndarray:
    """Convert OHLCV DataFrame to overlapping sequences for GAN training.

    Each sample is independently MinMax-scaled to [0, 1].
    Optional EMA smoothing to reduce high-frequency noise (per paper).

    Args:
        df: OHLCV DataFrame.
        seq_len: Sequence length K (paper: 90 or 120).
        step: Stride between consecutive samples.
        columns: Columns to include (default: ['Open','High','Low','Close','Volume']).
        smooth: Apply EMA smoothing before sampling.
        ema_span: EMA span for smoothing.

    Returns:
        Array of shape (N_samples, seq_len, D).
    """
    if columns is None:
        available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        columns = available

    data = df[columns].copy()

    if smooth:
        for col in columns:
            data[col] = _ema_smooth(data[col].values, span=ema_span)

    raw = data.values.astype(np.float64)
    n = len(raw)
    samples = []

    for i in range(0, n - seq_len, step):
        window = raw[i : i + seq_len].copy()
        w_min = window.min(axis=0)
        w_max = window.max(axis=0)
        denom = w_max - w_min
        denom[denom == 0] = 1.0
        window = (window - w_min) / denom
        samples.append(window)

    if not samples:
        return np.empty((0, seq_len, len(columns)), dtype=np.float32)

    return np.stack(samples).astype(np.float32)


def augment_dataset(
    gan: TTSGAN,
    real_samples: np.ndarray,
    augmentation_ratio: float = 1.0,
) -> np.ndarray:
    """Generate synthetic samples and combine with real data.

    Args:
        gan: Trained TTS-GAN model.
        real_samples: Real training samples, shape (N, T, D).
        augmentation_ratio: Ratio of synthetic to real samples (1.0 = doubles).

    Returns:
        Augmented dataset, shape (N + N*aug_ratio, T, D).
    """
    n_synth = int(len(real_samples) * augmentation_ratio)
    if n_synth == 0:
        return real_samples

    synth = gan.generate(n_synth)
    return np.concatenate([real_samples, synth.astype(np.float32)], axis=0)
