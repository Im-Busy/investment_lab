"""
Train TTS-GAN for financial time series data augmentation.

Trains a transformer-based GAN on OHLCV price data, generates synthetic
samples, and outputs an augmented dataset for downstream ML training.

Usage:
    # Train on SPY with default params
    uv run scripts/train_tts_gan.py --symbol SPY --start 2016-01-01 --end 2021-12-31

    # Train with paper-recommended hyperparameters (longer sequence)
    uv run scripts/train_tts_gan.py --symbol SPY --seq-len 120 --epochs 300

    # Fast smoke test
    uv run scripts/train_tts_gan.py --symbol SPY --fast --epochs 20

Reference: Phase 27A — GAN Financial Data Augmentation.
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

from src.ml.gan_data_augmentation import TTSGAN, augment_dataset, prepare_gan_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/gan")
OUTPUT_DIR = Path("outputs/gan")


def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV data from CSV or yfinance."""
    path = Path(f"data/raw/{symbol}_daily.csv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df
    else:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        return df[["Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TTS-GAN for financial data augmentation")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2021-12-31", help="End date")
    parser.add_argument(
        "--seq-len", type=int, default=90, help="Sequence length K (paper: 90 or 120)"
    )
    parser.add_argument("--step", type=int, default=5, help="Stride between samples")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent noise dimension")
    parser.add_argument("--g-embed", type=int, default=10, help="Generator embedding dim (M_g)")
    parser.add_argument("--g-heads", type=int, default=5, help="Generator attention heads (H_g)")
    parser.add_argument(
        "--g-layers", type=int, default=3, help="Generator transformer layers (D_g)"
    )
    parser.add_argument("--d-embed", type=int, default=90, help="Discriminator embedding dim (M_d)")
    parser.add_argument(
        "--d-heads", type=int, default=30, help="Discriminator attention heads (H_d)"
    )
    parser.add_argument(
        "--d-layers", type=int, default=3, help="Discriminator transformer layers (D_d)"
    )
    parser.add_argument("--lr-g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--gp-lambda", type=float, default=10.0, help="Gradient penalty weight")
    parser.add_argument(
        "--aug-ratio", type=float, default=1.0, help="Synthetic/real ratio (1.0 = double)"
    )
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA smoothing")
    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: fewer epochs, smaller model"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save augmented dataset (.npz)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to save trained model (.pt)"
    )
    args = parser.parse_args()

    if args.fast:
        args.epochs = 30
        args.batch_size = 16
        args.g_embed = 8
        args.g_heads = 4
        args.g_layers = 2
        args.d_embed = 32
        args.d_heads = 8
        args.d_layers = 2

    df = load_ohlcv(args.symbol, args.start, args.end)
    logger.info(
        f"Loaded {args.symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})"
    )

    samples = prepare_gan_samples(
        df,
        seq_len=args.seq_len,
        step=args.step,
        smooth=not args.no_ema,
    )
    logger.info(f"Prepared {len(samples)} samples of shape {samples.shape[1:]}")

    n_train = int(len(samples) * 0.8)
    train_data = samples[:n_train]
    val_data = samples[n_train:]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    data_dim = samples.shape[2]

    gan = TTSGAN(
        seq_len=args.seq_len,
        data_dim=data_dim,
        latent_dim=args.latent_dim,
        g_embed_dim=args.g_embed,
        g_n_heads=args.g_heads,
        g_n_layers=args.g_layers,
        d_embed_dim=args.d_embed,
        d_n_heads=args.d_heads,
        d_n_layers=args.d_layers,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        gp_lambda=args.gp_lambda,
    )

    n_params = sum(p.numel() for p in gan.generator.parameters()) + sum(
        p.numel() for p in gan.discriminator.parameters()
    )
    logger.info(f"TTS-GAN parameters: {n_params:,}")

    history = gan.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        convergence_patience=20,
        verbose=True,
    )

    logger.info(
        f"Best DTW DeD-iMs: {gan.monitor.best_dtw_dedims:.4f} at epoch {gan.monitor.best_epoch + 1}"
    )

    augmented = augment_dataset(gan, train_data, augmentation_ratio=args.aug_ratio)
    logger.info(
        f"Augmented: {len(train_data)} real + {len(augmented) - len(train_data)} synth = {len(augmented)} total"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or str(MODEL_DIR / f"tts_gan_{args.symbol}_{timestamp}.pt")
    gan.save(model_path)
    logger.info(f"Model saved to {model_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(OUTPUT_DIR / f"augmented_{args.symbol}_{timestamp}.npz")
    np.savez_compressed(
        output_path,
        train=train_data,
        val=val_data,
        augmented=augmented,
    )
    logger.info(f"Data saved to {output_path}")

    metrics_path = OUTPUT_DIR / f"metrics_{args.symbol}_{timestamp}.json"
    metrics = {
        "symbol": args.symbol,
        "date_range": f"{args.start}_{args.end}",
        "n_real_train": len(train_data),
        "n_val": len(val_data),
        "n_augmented": len(augmented),
        "seq_len": args.seq_len,
        "data_dim": data_dim,
        "epochs": len(history["g_loss"]),
        "final_g_loss": history["g_loss"][-1],
        "final_d_loss": history["d_loss"][-1],
        "best_dtw_dedims": gan.monitor.best_dtw_dedims,
        "best_epoch": gan.monitor.best_epoch + 1,
        "final_dtw_dedims": history["dtw_dedims"][-1],
        "final_wasserstein": history["wasserstein"][-1],
        "model_path": model_path,
        "output_path": output_path,
        "n_params": n_params,
        "convergence_history": {
            "dtw_dedims": history["dtw_dedims"],
            "wasserstein": history["wasserstein"],
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
