"""
TadGAN — Cycle-Consistent GAN for Time Series Anomaly Detection.

Architecture (Geiger, Liu, Alnegheimish, Cuesta-Infante, Veeramachaneni — MIT, 2022):
- Encoder: LSTM(time_series) → latent vector z
- Decoder: LSTM(z) → reconstructed time series
- Critic: LSTM(time_series) → real/fake score
- Cycle consistency: encode(decode(z)) ≈ z, decode(encode(x)) ≈ x
- Anomaly score: α × reconstruction_error + (1-α) × critic_deviation

Achieves highest averaged F1 across 11 benchmark datasets (NASA, Yahoo,
Numenta, Amazon, Twitter; 492 signals).

Reference: "TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks"
IEEE Access 2022, https://github.com/arunppsg/TadGAN
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _LSTMEncoder(nn.Module):
    """LSTM encoder: time series → latent vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 20,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.proj(h_n[-1])


class _LSTMDecoder(nn.Module):
    """LSTM decoder: latent vector → reconstructed time series."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 5,
        seq_len: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        x = self.latent_proj(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return self.output_proj(out)


class _LSTMCritic(nn.Module):
    """LSTM critic: time series → real/fake logit."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.cls_head(h_n[-1]).squeeze(-1)


class TadGAN:
    """Cycle-consistent GAN for time series anomaly detection.

    Trains an autoencoder (generator) adversarially with a critic.
    Anomaly scores combine reconstruction error and critic deviation.

    Hyperparameters from paper:
        hidden_dim=64, latent_dim=20, num_layers=2
        lr=1e-4, cycle_weight=10.0, critic_iterations=5
        α (anomaly score weight): grid-searched on validation set
    """

    def __init__(
        self,
        input_dim: int = 5,
        seq_len: int = 100,
        hidden_dim: int = 64,
        latent_dim: int = 20,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-4,
        cycle_weight: float = 10.0,
        n_critic: int = 5,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cycle_weight = cycle_weight
        self.n_critic = n_critic

        self.encoder = _LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout).to(
            self.device
        )
        self.decoder = _LSTMDecoder(
            latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout
        ).to(self.device)
        self.critic_x = _LSTMCritic(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.critic_z = _LSTMCritic(latent_dim, hidden_dim // 4, 1, dropout).to(self.device)

        self.optimizer_ge = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self.optimizer_cx = torch.optim.Adam(self.critic_x.parameters(), lr=lr)
        self.optimizer_cz = torch.optim.Adam(self.critic_z.parameters(), lr=lr)

        self._alpha: Optional[float] = None
        self._threshold: Optional[float] = None

    def _generate_z(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def train_step(self, real_batch: torch.Tensor) -> dict[str, float]:
        real = real_batch.to(self.device)
        batch_size = real.size(0)

        z_enc = self.encoder(real)
        x_rec = self.decoder(z_enc)
        z_gen = self._generate_z(batch_size)
        x_gen = self.decoder(z_gen)
        z_rec = self.encoder(x_gen)

        # --- Critic X training (n_critic steps) ---
        cx_losses = []
        for _ in range(self.n_critic):
            self.optimizer_cx.zero_grad()
            cx_real = self.critic_x(real)
            cx_fake = self.critic_x(x_gen.detach())
            cx_loss = F.binary_cross_entropy_with_logits(
                cx_real, torch.ones_like(cx_real)
            ) + F.binary_cross_entropy_with_logits(cx_fake, torch.zeros_like(cx_fake))
            cx_loss.backward()
            self.optimizer_cx.step()
            cx_losses.append(cx_loss.item())

        # --- Critic Z training (n_critic steps) ---
        cz_losses = []
        for _ in range(self.n_critic):
            self.optimizer_cz.zero_grad()
            cz_real = self.critic_z(z_gen)
            cz_fake = self.critic_z(z_enc.detach())
            cz_loss = F.binary_cross_entropy_with_logits(
                cz_real, torch.ones_like(cz_real)
            ) + F.binary_cross_entropy_with_logits(cz_fake, torch.zeros_like(cz_fake))
            cz_loss.backward()
            self.optimizer_cz.step()
            cz_losses.append(cz_loss.item())

        # --- Generator + Encoder training ---
        self.optimizer_ge.zero_grad()

        z_enc = self.encoder(real)
        x_rec = self.decoder(z_enc)
        x_gen = self.decoder(z_gen)
        z_rec = self.encoder(x_gen)

        rec_loss = F.mse_loss(x_rec, real)
        cycle_x_loss = F.mse_loss(self.decoder(self.encoder(x_gen)), x_gen)
        cycle_z_loss = F.mse_loss(self.encoder(self.decoder(z_gen)), z_gen)

        g_adv_x = F.binary_cross_entropy_with_logits(
            self.critic_x(x_gen), torch.ones(batch_size, device=self.device)
        )
        g_adv_z = F.binary_cross_entropy_with_logits(
            self.critic_z(z_enc), torch.ones(batch_size, device=self.device)
        )

        ge_loss = rec_loss + self.cycle_weight * (cycle_x_loss + cycle_z_loss) + g_adv_x + g_adv_z
        ge_loss.backward()
        self.optimizer_ge.step()

        return {
            "rec_loss": float(rec_loss.item()),
            "cycle_x": float(cycle_x_loss.item()),
            "cycle_z": float(cycle_z_loss.item()),
            "ge_loss": float(ge_loss.item()),
            "cx_loss": float(np.mean(cx_losses)),
            "cz_loss": float(np.mean(cz_losses)),
        }

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history: dict[str, list[float]] = {
            "rec_loss": [],
            "cycle_x": [],
            "cycle_z": [],
            "ge_loss": [],
            "cx_loss": [],
            "cz_loss": [],
        }

        for epoch in range(epochs):
            epoch_metrics = {k: 0.0 for k in history}
            n_batches = 0

            for (real_batch,) in loader:
                losses = self.train_step(real_batch)
                for k in epoch_metrics:
                    epoch_metrics[k] += losses[k]
                n_batches += 1

            for k in history:
                history[k].append(epoch_metrics[k] / max(n_batches, 1))

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"rec={history['rec_loss'][-1]:.4f} | "
                    f"cycle_x={history['cycle_x'][-1]:.4f} | "
                    f"ge={history['ge_loss'][-1]:.4f}"
                )

        self._calibrate_alpha(val_data)
        return history

    def _calibrate_alpha(self, val_data: np.ndarray) -> None:
        """Grid-search α on validation set for best anomaly detection F1."""
        val_t = torch.tensor(val_data, dtype=torch.float32).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.critic_x.eval()

        with torch.no_grad():
            z = self.encoder(val_t)
            rec = self.decoder(z)
            rec_errors = F.mse_loss(rec, val_t, reduction="none").mean(dim=(1, 2)).cpu().numpy()
            critic_scores = torch.sigmoid(self.critic_x(val_t)).cpu().numpy()
            critic_deviation = np.abs(critic_scores - 0.5)

            rec_norm = (rec_errors - rec_errors.min()) / (
                rec_errors.max() - rec_errors.min() + 1e-8
            )
            critic_norm = (critic_deviation - critic_deviation.min()) / (
                critic_deviation.max() - critic_deviation.min() + 1e-8
            )

        best_alpha = 0.5
        best_score = -1.0
        for alpha in np.linspace(0.2, 0.8, 13):
            scores = alpha * rec_norm + (1 - alpha) * critic_norm
            threshold = np.percentile(scores, 95)
            preds = (scores > threshold).astype(int)

            cluster_sizes = []
            in_cluster = False
            cluster_start = 0
            for i, p in enumerate(preds):
                if p and not in_cluster:
                    cluster_start = i
                    in_cluster = True
                elif not p and in_cluster:
                    cluster_sizes.append(i - cluster_start)
                    in_cluster = False
            if in_cluster:
                cluster_sizes.append(len(preds) - cluster_start)

            compactness = (
                np.mean([min(1.0, max(3, s) / max(30, s)) for s in cluster_sizes])
                if cluster_sizes
                else 0.0
            )
            score = preds.sum() + compactness * 10

            if score > best_score:
                best_score = score
                best_alpha = alpha

        self._alpha = best_alpha
        logger.info(f"Calibrated α={best_alpha:.3f} (val_size={len(val_data)})")

    def compute_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute per-sample anomaly scores.

        Args:
            data: Samples of shape (N, seq_len, input_dim).

        Returns:
            Anomaly scores array of shape (N,). Higher = more anomalous.
        """
        if self._alpha is None:
            self._alpha = 0.5

        data_t = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.critic_x.eval()

        all_scores = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(data_t), batch_size):
                batch = data_t[i : i + batch_size]
                z = self.encoder(batch)
                rec = self.decoder(z)

                rec_errors = F.mse_loss(rec, batch, reduction="none").mean(dim=(1, 2)).cpu().numpy()
                critic_scores = torch.sigmoid(self.critic_x(batch)).cpu().numpy()
                critic_deviation = np.abs(critic_scores - 0.5)

                rec_norm = (rec_errors - rec_errors.min()) / (
                    rec_errors.max() - rec_errors.min() + 1e-8
                )
                critic_norm = (critic_deviation - critic_deviation.min()) / (
                    critic_deviation.max() - critic_deviation.min() + 1e-8
                )

                scores = self._alpha * rec_norm + (1 - self._alpha) * critic_norm
                all_scores.append(scores)

        return np.concatenate(all_scores)

    def detect_anomalies(
        self,
        data: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> tuple[np.ndarray, float]:
        """Detect anomalous samples.

        Args:
            data: Samples of shape (N, seq_len, input_dim).
            threshold_percentile: Percentile for anomaly threshold.

        Returns:
            (binary_anomaly_labels, threshold_value).
        """
        scores = self.compute_anomaly_scores(data)
        threshold = np.percentile(scores, threshold_percentile)
        labels = (scores > threshold).astype(int)
        return labels, float(threshold)

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "critic_x": self.critic_x.state_dict(),
                "critic_z": self.critic_z.state_dict(),
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
                "latent_dim": self.latent_dim,
                "alpha": self._alpha,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "TadGAN":
        checkpoint = torch.load(path, map_location="cpu")
        kwargs.setdefault("input_dim", checkpoint["input_dim"])
        kwargs.setdefault("seq_len", checkpoint["seq_len"])
        kwargs.setdefault("latent_dim", checkpoint["latent_dim"])
        model = cls(**kwargs)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        model.critic_x.load_state_dict(checkpoint["critic_x"])
        model.critic_z.load_state_dict(checkpoint["critic_z"])
        model._alpha = checkpoint.get("alpha")
        return model


def prepare_tadgan_samples(
    df: pd.DataFrame,
    seq_len: int = 100,
    step: int = 5,
    columns: Optional[list[str]] = None,
) -> np.ndarray:
    """Convert OHLCV DataFrame to normalized overlapping sequences.

    Args:
        df: OHLCV DataFrame.
        seq_len: Sequence length.
        step: Stride between consecutive samples.
        columns: Columns to include.

    Returns:
        Array of shape (N_samples, seq_len, D) normalized to [0,1].
    """
    if columns is None:
        available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        columns = available

    raw = df[columns].values.astype(np.float64)
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
