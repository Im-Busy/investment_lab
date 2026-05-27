"""
Benchmark TadGAN anomaly detection on financial crisis events.

Trains TadGAN on pre-crisis SPY data, then evaluates whether anomaly
scores spike during known crisis windows: COVID crash (Mar 2020),
2022 bear market (Jun 2022), 2025 tariff volatility (Mar 2025),
2026 oil shock (Mar 2026).

Gate: detect ≥4/4 events at ≤5 false positives/year.

Usage:
    uv run scripts/benchmark_tadgan.py --symbol SPY
    uv run scripts/benchmark_tadgan.py --symbol SPY --fast --epochs 30

Reference: Phase 27C gate validation.
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

from src.ml.anomaly_detection import TadGAN, prepare_tadgan_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


CRISIS_EVENTS = [
    ("COVID crash", "2020-02-19", "2020-03-23", -34.0),
    ("2022 bear", "2022-01-03", "2022-10-12", -24.5),
    ("2025 tariff vol", "2025-02-19", "2025-04-08", -18.8),
    ("2026 oil shock", "2026-02-11", "2026-03-14", -9.1),
]

CALM_PERIODS = [
    ("mid-2017", "2017-04-01", "2017-09-01"),
    ("late-2018", "2018-07-01", "2018-12-01"),
    ("mid-2021", "2021-04-01", "2021-09-01"),
    ("early-2024", "2024-01-01", "2024-06-01"),
]


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


def _count_false_positives(
    anomaly_dates: set,
    crisis_windows: list[tuple[str, str]],
    total_years: float,
) -> float:
    """Count anomaly dates outside crisis windows, normalized per year."""
    fp_dates = set()
    for d in anomaly_dates:
        in_crisis = any(
            pd.Timestamp(start) <= d <= pd.Timestamp(end) for _, start, end, _ in CRISIS_EVENTS
        )
        if not in_crisis:
            fp_dates.add(d)

    return len(fp_dates) / max(total_years, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TadGAN crisis detection gate")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--threshold-pct", type=float, default=95.0, help="Anomaly threshold percentile"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.fast:
        args.epochs = 30

    # --- Train on pre-crisis data ---
    df_train = load_ohlcv(args.symbol, "2010-01-01", "2019-12-31")
    logger.info(
        f"Training: {len(df_train)} bars ({df_train.index[0].date()} to {df_train.index[-1].date()})"
    )

    train_samples = prepare_tadgan_samples(df_train, seq_len=args.seq_len, step=5)
    n_train = int(len(train_samples) * 0.8)

    tadgan = TadGAN(
        input_dim=train_samples.shape[2],
        seq_len=args.seq_len,
        hidden_dim=32 if args.fast else 64,
        latent_dim=10 if args.fast else 20,
        num_layers=1 if args.fast else 2,
        device=args.device,
    )

    tadgan.fit(
        train_data=train_samples[:n_train],
        val_data=train_samples[n_train:],
        epochs=args.epochs,
        batch_size=min(32, n_train),
        verbose=args.epochs >= 40,
    )

    # --- Test on full history ---
    df_test = load_ohlcv(args.symbol, "2010-01-01", "2026-06-01")
    logger.info(
        f"Testing: {len(df_test)} bars ({df_test.index[0].date()} to {df_test.index[-1].date()})"
    )

    test_samples = prepare_tadgan_samples(df_test, seq_len=args.seq_len, step=5)
    scores = tadgan.compute_anomaly_scores(test_samples)
    threshold = np.percentile(scores, args.threshold_pct)
    anomaly_mask = scores > threshold

    # Map sample indices back to dates
    sample_dates = []
    for i in range(0, len(df_test) - args.seq_len, 5):
        sample_dates.append(df_test.index[i + args.seq_len - 1])

    anomaly_dates = {sample_dates[i] for i in np.where(anomaly_mask)[0] if i < len(sample_dates)}

    logger.info(f"\n{'=' * 60}")
    logger.info("TadGAN Anomaly Detection Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"Threshold: {threshold:.4f} (p{args.threshold_pct})")
    logger.info(
        f"Anomaly samples: {anomaly_mask.sum()}/{len(test_samples)} ({anomaly_mask.mean():.1%})"
    )
    logger.info(f"Anomaly dates: {len(anomaly_dates)}")

    detected_count = 0
    for name, start_str, end_str, drawdown in CRISIS_EVENTS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        window_detected = any(start <= d <= end for d in anomaly_dates)
        status = "DETECTED" if window_detected else "MISSED"
        if window_detected:
            detected_count += 1

        window_scores = []
        for i, d in enumerate(sample_dates):
            if start <= d <= end and i < len(scores):
                window_scores.append(scores[i])
        max_score = max(window_scores) if window_scores else 0.0
        logger.info(f"  {name:20s} ({drawdown:+.1f}%): {status:10s} max_score={max_score:.4f}")

    total_years = (df_test.index[-1] - df_test.index[0]).days / 365.25
    fp_per_year = _count_false_positives(anomaly_dates, CRISIS_EVENTS, total_years)
    logger.info(f"\n  Events detected: {detected_count}/4")
    logger.info(f"  False positives/year: {fp_per_year:.1f}")

    gate_pass = detected_count >= 4 and fp_per_year <= 5.0
    logger.info(f"  Gate: {'PASS' if gate_pass else 'FAIL'} (≥4/4 detected, ≤5 FP/year)")

    if fp_per_year > 5.0:
        logger.info(
            f"  Try: --threshold-pct {args.threshold_pct + 2} (tighter threshold reduces FPs)"
        )


if __name__ == "__main__":
    main()
