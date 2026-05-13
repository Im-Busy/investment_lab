"""
C6 — Basket Expansion Comparison Script.

Compares walk-forward ICs of the 12-all model vs the 33-all model on all 33 tickers.
Uses the same walk_forward_per_ticker function that was used for ticker screening,
ensuring consistent methodology.

Gate criteria:
  - Expanded model mean IC > current 12-all model mean IC on at least 80% of tickers
  - No ticker's IC drops by more than 0.02
  - At least 15 of the new tickers show IC > 0.03

Usage:
    uv run scripts/c6_compare_baskets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from src.ml.walk_forward import walk_forward_per_ticker

ALL_TICKERS = [
    "SPY",
    "KODK",
    "QQQ",
    "IWM",
    "XLK",
    "JOE",
    "XLF",
    "XLE",
    "XLV",
    "EEM",
    "TLT",
    "GLD",
    "PEG",
    "D",
    "SO",
    "SRE",
    "AVB",
    "PSA",
    "SPG",
    "NSC",
    "UNP",
    "CSX",
    "KO",
    "WMT",
    "PEP",
    "CRVL",
    "HIFS",
    "VRTX",
    "BMY",
    "REGN",
    "EOG",
    "OXY",
    "NEM",
]

OLD_MODEL_PATH = "models/pattern_classifier_v3_SPY_20260511_145927.pkl"
NEW_MODEL_PATH = "models/pattern_classifier_v3_SPY_20260511_164601.pkl"

OLD_BASKET = [
    "SPY",
    "KODK",
    "QQQ",
    "IWM",
    "XLK",
    "JOE",
    "XLF",
    "XLE",
    "XLV",
    "EEM",
    "TLT",
    "GLD",
]

NEW_TICKERS = [t for t in ALL_TICKERS if t not in OLD_BASKET]


def main() -> None:
    print("=" * 100)
    print("C6 — Basket Expansion: 12-all vs 33-all Walk-Forward IC Comparison")
    print("=" * 100)

    print(f"\nLoading old model: {OLD_MODEL_PATH}")
    old_model = PatternClassifier()
    old_model.load(OLD_MODEL_PATH)
    print(f"  Features: {len(old_model.feature_names_)}, tickers: 12")

    print(f"\nLoading new model: {NEW_MODEL_PATH}")
    new_model = PatternClassifier()
    new_model.load(NEW_MODEL_PATH)
    print(f"  Features: {len(new_model.feature_names_)}, tickers: 33")

    print(
        f"\nRunning walk-forward per ticker on OLD model (12-all) across {len(ALL_TICKERS)} tickers..."
    )
    old_results = walk_forward_per_ticker(
        model=old_model,
        tickers=ALL_TICKERS,
        data_dir="data/raw",
        initial_train_days=3 * 252,
        step_days=6 * 21,
        horizon=5,
    )

    print(
        f"\nRunning walk-forward per ticker on NEW model (33-all) across {len(ALL_TICKERS)} tickers..."
    )
    new_results = walk_forward_per_ticker(
        model=new_model,
        tickers=ALL_TICKERS,
        data_dir="data/raw",
        initial_train_days=3 * 252,
        step_days=6 * 21,
        horizon=5,
    )

    # Build comparison table
    rows: list[dict] = []
    for ticker in ALL_TICKERS:
        old_ic = old_results[ticker].mean_rank_ic if ticker in old_results else None
        new_ic = new_results[ticker].mean_rank_ic if ticker in new_results else None

        if old_ic is None or new_ic is None:
            continue

        delta = new_ic - old_ic
        is_new = ticker in NEW_TICKERS
        rows.append(
            {
                "ticker": ticker,
                "is_new": is_new,
                "old_ic": round(old_ic, 4),
                "new_ic": round(new_ic, 4),
                "delta": round(delta, 4),
                "new_pass": new_ic > 0.03,
            }
        )

    df = pd.DataFrame(rows)

    # Sort by delta descending (biggest improvements first)
    df = df.sort_values("delta", ascending=False)

    print("\n" + "=" * 100)
    print("Per-Ticker Walk-Forward IC Comparison")
    print("=" * 100)
    print(f"{'Ticker':<8} {'New?':>5} {'Old IC':>8} {'New IC':>8} {'Delta':>8} {'Pass':>5}")
    print("-" * 50)
    for _, row in df.iterrows():
        new_flag = " *" if row["is_new"] else ""
        pass_flag = " P" if row["new_pass"] else ""
        delta_str = f"{row['delta']:+.4f}"
        print(
            f"{row['ticker']:<8} {new_flag:>5} {row['old_ic']:>8.4f} {row['new_ic']:>8.4f} {delta_str:>8} {pass_flag:>5}"
        )

    # Summary statistics
    old_ics = df["old_ic"].values
    new_ics = df["new_ic"].values
    deltas = df["delta"].values

    existing_mask = ~df["is_new"].values
    new_mask = df["is_new"].values

    print("\n" + "=" * 100)
    print("Summary Statistics")
    print("=" * 100)

    print("\nGroup means:")
    print(f"  OLD model mean IC: {np.mean(old_ics):.4f}")
    print(f"  NEW model mean IC: {np.mean(new_ics):.4f}")
    print(f"  Mean delta:       {np.mean(deltas):+.4f}")

    if existing_mask.sum() > 0:
        print("\n  Existing 12 tickers only:")
        print(f"    Old mean IC: {np.mean(old_ics[existing_mask]):.4f}")
        print(f"    New mean IC: {np.mean(new_ics[existing_mask]):.4f}")
        print(f"    Mean delta:  {np.mean(deltas[existing_mask]):+.4f}")

    if new_mask.sum() > 0:
        print("\n  New 21 tickers only:")
        print(f"    Mean new IC: {np.mean(new_ics[new_mask]):.4f}")
        print(
            f"    Pass rate:   {new_ics[new_mask].mean(where=new_ics[new_mask] > 0.03).sum():.0f}/{new_mask.sum()} ({new_ics[new_mask][new_ics[new_mask] > 0.03].sum():.0f})"
        )

    # Gate criteria
    print("\n" + "=" * 100)
    print("Gate Criteria")
    print("=" * 100)

    # Gate 1: expanded model mean IC > old model mean IC on at least 80% of tickers
    n_improved = (deltas > 0).sum()
    pct_improved = n_improved / len(deltas) * 100
    gate1 = pct_improved >= 80
    print(
        f"\n  Gate 1: New IC > Old IC on {pct_improved:.0f}% of tickers ({n_improved}/{len(deltas)})"
    )
    print(f"    Threshold: >= 80% — {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: no ticker's IC drops by more than 0.02
    max_drop = deltas.min()
    n_big_drops = (deltas < -0.02).sum()
    gate2 = max_drop > -0.02
    print("\n  Gate 2: No ticker IC drops by > 0.02")
    print(f"    Max drop: {max_drop:+.4f} | Tickers with drop > 0.02: {n_big_drops}")
    print(f"    Threshold: max drop > -0.02 — {'PASS' if gate2 else 'FAIL'}")
    if n_big_drops > 0:
        big_droppers = df[df["delta"] < -0.02][["ticker", "old_ic", "new_ic", "delta"]]
        for _, r in big_droppers.iterrows():
            print(f"      {r['ticker']}: {r['old_ic']:.4f} → {r['new_ic']:.4f} ({r['delta']:+.4f})")

    # Gate 3: at least 15 new tickers show IC > 0.03
    n_new_pass = (new_ics[new_mask] > 0.03).sum()
    gate3 = n_new_pass >= 15
    print("\n  Gate 3: At least 15 new tickers with IC > 0.03")
    print(f"    New tickers passing: {n_new_pass}/{new_mask.sum()}")
    print(f"    Threshold: >= 15 — {'PASS' if gate3 else 'FAIL'}")

    # Overall verdict
    all_pass = gate1 and gate2 and gate3
    print("\n" + "=" * 100)
    if all_pass:
        print("VERDICT: C6 PASSES — Expanded 33-ticker basket beats 12-ticker basket.")
        print("Proceed to C5 integration with the new model.")
    else:
        print("VERDICT: C6 FAILS — Expanded basket does not meet gate criteria.")
        print("Proceed to C5 with the 12-all model, or prune weak tickers and retry.")
    print("=" * 100)

    # Save results
    out_path = Path("experiments/c6_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
