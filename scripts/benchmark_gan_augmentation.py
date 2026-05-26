"""
Benchmark GAN-based data augmentation for LSTM forecasting.

Trains TTS-GAN on SPY OHLCV data, then compares LSTM forecasting
directional accuracy on real-only vs augmented training sets.

Gate: ≥10% reduction in directional error on SPY 2022 bear market.

Usage:
    uv run scripts/benchmark_gan_augmentation.py --symbol SPY --fast
    uv run scripts/benchmark_gan_augmentation.py --symbol SPY --seq-len 120 --epochs 300

Reference: Phase 27A gate validation (Podobinski & Chudziak 2024).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.gan_data_augmentation import TTSGAN, augment_dataset, prepare_gan_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DirectionalLSTM(nn.Module):
    """LSTM forecasting model for directional accuracy evaluation.

    Predicts the last S points from the first T points of a K=T+S sequence.
    Directional error = fraction of predicted directions that differ from actual.
    """

    def __init__(
        self, input_dim: int = 4, hidden_size: int = 64, num_layers: int = 3, dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        lstm_out, _ = self.lstm(x)
        out = self.output_proj(lstm_out[:, -1:, :]).squeeze(1)
        return out


def _predict_directions(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Predict direction (up=1, down=0) for each sample."""
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy()
    return (preds > 0).astype(int)


def compute_directional_error(pred_direction: np.ndarray, true_direction: np.ndarray) -> float:
    """Fraction of incorrect directional predictions."""
    return float(np.mean(pred_direction != true_direction))


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train LSTM and evaluate directional error."""
    model = DirectionalLSTM(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0 and epochs >= 40:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t).item()
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} | train_loss={epoch_loss / max(len(loader), 1):.6f} | test_loss={test_loss:.6f}"
            )

    model.eval()
    pred_direction = _predict_directions(model, X_test_t)
    true_direction = (y_test_t.cpu().numpy() > 0).astype(int)
    dir_error = compute_directional_error(pred_direction, true_direction)

    return {
        "directional_error": dir_error,
        "model": model,
    }


def prepare_lstm_data(
    samples: np.ndarray,
    T: int,
    S: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split overlapping samples into input (first T) and target (last S).

    Target is the last point of each sample (next-bar direction).
    For multi-step forecasting, use the mean of the last S points.
    """
    if len(samples) == 0:
        return np.empty((0, T, 0)), np.empty((0, 0)), np.empty((0, T, 0)), np.empty((0, 0))

    X = samples[:, :T, :].copy()
    y = samples[:, -1, :].copy()  # Last bar as target (single-step forecast)

    n = len(X)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GAN augmentation for LSTM forecasting")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date (IS)")
    parser.add_argument("--end", type=str, default="2021-12-31", help="End date (IS)")
    parser.add_argument("--seq-len", type=int, default=90, help="Sequence length K")
    parser.add_argument("--T", type=int, default=60, help="Input length (first T of K)")
    parser.add_argument("--gan-epochs", type=int, default=200, help="GAN training epochs")
    parser.add_argument("--lstm-epochs", type=int, default=100, help="LSTM training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer epochs")
    parser.add_argument(
        "--n-trials", type=int, default=1, help="Number of trials for variance estimation"
    )
    args = parser.parse_args()

    if args.fast:
        args.gan_epochs = 30
        args.lstm_epochs = 40

    S = args.seq_len - args.T
    logger.info(f"Sequence: K={args.seq_len}, T={args.T}, S={S}")

    df_path = Path(f"data/raw/{args.symbol}_daily.csv")
    if df_path.exists():
        df = pd.read_csv(df_path, parse_dates=True, index_col=0)
        df = df[(df.index >= args.start) & (df.index <= args.end)]
    else:
        import yfinance as yf

        df = yf.Ticker(args.symbol).history(start=args.start, end=args.end)

    logger.info(f"Data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    samples = prepare_gan_samples(df, seq_len=args.seq_len, step=5)
    logger.info(f"Samples: {len(samples)} of shape {samples.shape[1:]}")

    X_train, y_train, X_test, y_test = prepare_lstm_data(samples, T=args.T, S=S)
    logger.info(f"LSTM data: train={len(X_train)}, test={len(X_test)}")

    input_dim = X_train.shape[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_errors = []
    augmented_errors = []

    for trial in range(args.n_trials):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Trial {trial + 1}/{args.n_trials}")

        result = train_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            input_dim=input_dim,
            epochs=args.lstm_epochs,
            batch_size=args.batch_size,
            device=device,
        )
        baseline_errors.append(result["directional_error"])
        logger.info(f"Baseline directional error: {result['directional_error']:.4f}")

        n_gan_train = int(len(samples) * 0.8)
        gan_train = samples[:n_gan_train]
        gan_val = samples[n_gan_train:]

        gan = TTSGAN(
            seq_len=args.seq_len,
            data_dim=input_dim,
            latent_dim=32,
            g_embed_dim=8 if args.fast else 10,
            g_n_heads=4 if args.fast else 5,
            g_n_layers=2 if args.fast else 3,
            d_embed_dim=32 if args.fast else 90,
            d_n_heads=8 if args.fast else 30,
            d_n_layers=2 if args.fast else 3,
            device=device,
        )

        n_params = sum(p.numel() for p in gan.generator.parameters()) + sum(
            p.numel() for p in gan.discriminator.parameters()
        )
        logger.info(f"TTS-GAN params: {n_params:,}")

        gan.fit(
            train_data=gan_train,
            val_data=gan_val,
            epochs=args.gan_epochs,
            batch_size=min(args.batch_size, len(gan_train)),
            convergence_patience=15,
            verbose=args.gan_epochs >= 40,
        )

        aug_samples = augment_dataset(gan, gan_train, augmentation_ratio=1.0)
        logger.info(
            f"Augmented: {len(gan_train)} real + {len(aug_samples) - len(gan_train)} synth = {len(aug_samples)}"
        )

        X_aug, y_aug, _, _ = prepare_lstm_data(aug_samples, T=args.T, S=S)

        result_aug = train_lstm(
            X_aug,
            y_aug,
            X_test,
            y_test,
            input_dim=input_dim,
            epochs=args.lstm_epochs,
            batch_size=args.batch_size,
            device=device,
        )
        augmented_errors.append(result_aug["directional_error"])
        logger.info(f"Augmented directional error: {result_aug['directional_error']:.4f}")

    baseline_mean = float(np.mean(baseline_errors))
    aug_mean = float(np.mean(augmented_errors))
    reduction = (baseline_mean - aug_mean) / max(baseline_mean, 1e-8)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTS ({args.n_trials} trial(s))")
    logger.info(f"  Baseline directional error:  {baseline_mean:.4f}")
    logger.info(f"  Augmented directional error: {aug_mean:.4f}")
    logger.info(f"  Error reduction:             {reduction:.1%}")
    logger.info(f"  Gate (≥10%):                 {'PASS' if reduction >= 0.10 else 'FAIL'}")

    if args.n_trials > 1:
        logger.info(f"  Baseline std:                {np.std(baseline_errors):.4f}")
        logger.info(f"  Augmented std:               {np.std(augmented_errors):.4f}")


if __name__ == "__main__":
    main()
