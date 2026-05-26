"""
Train TadGAN for regime anomaly detection on financial OHLCV data.

Learns the normal manifold of price behavior, then scores deviations
as anomalies. Designed to detect market crisis events (COVID crash,
2022 bear, tariff vol, oil shock).

Usage:
    # Train on SPY pre-COVID data
    uv run scripts/train_tadgan.py --symbol SPY --start 2010-01-01 --end 2019-12-31

    # Train with paper hyperparameters
    uv run scripts/train_tadgan.py --symbol SPY --hidden 64 --latent 20 --epochs 200

    # Fast smoke test
    uv run scripts/train_tadgan.py --symbol SPY --fast --epochs 20

Reference: Phase 27C — GAN-Based Regime Anomaly Detection.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.anomaly_detection import TadGAN, prepare_tadgan_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/anomaly")
OUTPUT_DIR = Path("outputs/anomaly")


def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, end=end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TadGAN for anomaly detection")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2019-12-31")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--step", type=int, default=5, help="Stride between samples")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden dim")
    parser.add_argument("--latent", type=int, default=20, help="Latent vector dim")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--cycle-weight", type=float, default=10.0, help="Cycle consistency loss weight"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: fewer epochs, smaller model"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Model save path (.pt)")
    args = parser.parse_args()

    if args.fast:
        args.hidden = 32
        args.latent = 10
        args.layers = 1
        args.epochs = 30

    df = load_ohlcv(args.symbol, args.start, args.end)
    logger.info(
        f"Loaded {args.symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})"
    )

    samples = prepare_tadgan_samples(df, seq_len=args.seq_len, step=args.step)
    logger.info(f"Samples: {samples.shape}")

    n_train = int(len(samples) * 0.8)
    train_data = samples[:n_train]
    val_data = samples[n_train:]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tadgan = TadGAN(
        input_dim=samples.shape[2],
        seq_len=args.seq_len,
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        num_layers=args.layers,
        lr=args.lr,
        cycle_weight=args.cycle_weight,
        device=args.device,
    )

    n_params = sum(
        p.numel()
        for p in list(tadgan.encoder.parameters())
        + list(tadgan.decoder.parameters())
        + list(tadgan.critic_x.parameters())
        + list(tadgan.critic_z.parameters())
    )
    logger.info(f"TadGAN params: {n_params:,}")

    history = tadgan.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.epochs >= 40,
    )

    logger.info(f"Final rec_loss: {history['rec_loss'][-1]:.6f}")
    logger.info(f"Calibrated α: {tadgan._alpha}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = args.output or str(MODEL_DIR / f"tadgan_{args.symbol}_{timestamp}.pt")
    tadgan.save(model_path)
    logger.info(f"Model saved to {model_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / f"tadgan_metrics_{args.symbol}_{timestamp}.json"
    metrics = {
        "symbol": args.symbol,
        "date_range": f"{args.start}_{args.end}",
        "n_train": len(train_data),
        "n_val": len(val_data),
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden,
        "latent_dim": args.latent,
        "epochs": len(history["rec_loss"]),
        "final_rec_loss": history["rec_loss"][-1],
        "alpha": tadgan._alpha,
        "model_path": model_path,
        "n_params": n_params,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
