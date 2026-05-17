"""
Factor IC/IR Analysis Pipeline -- cross-sectional validation of ML features.

Computes Information Coefficient (IC), Rank IC, IC decay, Information Ratio (IR),
and rolling IC stability for every feature column against forward returns.

This is the standard factor literature approach for validating predictive
features before they enter an ML model.

Outputs:
    1. Per-feature IC table (IC, rank IC, t-stat, p-value, hit rate)
    2. IC decay across multiple forward horizons (1d, 5d, 10d, 20d)
    3. Rolling IC stability (21d, 63d windows)
    4. Feature ranking by information content
    5. Statistical significance summary

Usage:
    # Analyze features for a single symbol
    uv run scripts/run_factor_ic_analysis.py SPY

    # With custom horizons
    uv run scripts/run_factor_ic_analysis.py SPY --horizons 1,5,10,20

    # From a saved feature parquet file
    uv run scripts/run_factor_ic_analysis.py --features-file experiments/features/SPY_features.parquet

    # Filter to top N features by |IC|
    uv run scripts/run_factor_ic_analysis.py SPY --top-n 20

    # Output JSON for downstream consumption
    uv run scripts/run_factor_ic_analysis.py SPY --json-output ic_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.features import FeatureEngineer
from src.ml.metrics import (
    compute_ic,
    compute_ic_decay,
    compute_information_ratio,
    compute_rank_ic,
    ic_summary,
)

DEFAULT_HORIZONS = (1, 5, 10, 20)
DEFAULT_ROLLING_WINDOWS = (21, 63)


def load_data(symbol: str) -> pd.DataFrame:
    """Load OHLCV data from data/raw/."""
    path = Path(project_root) / "data" / "raw" / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    return df


def load_features_for_symbol(
    symbol: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Load OHLCV data for a symbol and generate features with forward returns."""
    df = load_data(symbol)
    if df is None or len(df) < 60:
        raise ValueError(f"No data for {symbol}")

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    eng = FeatureEngineer()
    features = eng.generate_features_with_labels(df, horizons=DEFAULT_HORIZONS)
    return features


def rolling_ic_analysis(
    features: pd.DataFrame,
    forward_returns: pd.Series,
    window: int = 21,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute rolling Rank IC over time.

    At each time step, compute the Spearman rank correlation between
    each feature and forward returns using a trailing window.

    Args:
        features: Feature DataFrame.
        forward_returns: Forward return series.
        window: Rolling window size in bars.
        min_periods: Minimum bars required per window.

    Returns:
        DataFrame with rolling IC per feature (index=date, columns=features).
    """
    if len(features) < min_periods:
        return pd.DataFrame()

    rolling_ic = {}
    for col in features.columns:
        df_sub = pd.DataFrame({"feat": features[col], "fwd": forward_returns})
        rolling_ic[col] = (
            df_sub["feat"].rolling(window, min_periods=min_periods).corr(df_sub["fwd"])
        )

    result = pd.DataFrame(rolling_ic, index=features.index)
    result.columns = [
        f"rolling_ic_{w}d_{c}" for w, c in zip([window] * len(features.columns), features.columns)
    ]
    return result


def ic_stability_score(rolling_ic_series: pd.Series) -> dict[str, float]:
    """Compute stability metrics for a rolling IC series.

    Returns:
        Dict with mean_ic, std_ic, ir (annualized), fraction_positive,
        fraction_significant (|t| > 2 per period).
    """
    ic = rolling_ic_series.dropna()
    if len(ic) < 5:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "ir": 0.0,
            "frac_positive": 0.0,
            "frac_significant": 0.0,
        }

    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    ir = float(compute_information_ratio(ic, periods_per_year=252))
    frac_pos = float((ic > 0).mean())
    frac_sig = float((ic.abs() > 2 * ic.std()).mean())

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "frac_positive": frac_pos,
        "frac_significant": frac_sig,
    }


def format_feature_table(summary_df: pd.DataFrame, top_n: int = 30) -> str:
    """Format a readable feature IC ranking table."""
    lines = []
    lines.append(
        f"{'Rank':<5} {'Feature':<40} {'IC':>8} {'RankIC':>8} {'p-val':>8} {'Hit%':>7} {'Sig':>4}"
    )
    lines.append("-" * 85)

    for i, row in summary_df.head(top_n).iterrows():
        sig = "YES" if row.get("significant", False) else ""
        lines.append(
            f"{i + 1:<5} "
            f"{str(row['feature'])[:39]:<40} "
            f"{row['ic']:>8.4f} "
            f"{row['rank_ic']:>8.4f} "
            f"{row['p_value']:>8.4f} "
            f"{row['hit_rate']:>6.1%} "
            f"{sig:>4}"
        )

    return "\n".join(lines)


def format_decay_table(decay_results: dict[int, pd.DataFrame], top_features: list[str]) -> str:
    """Format IC decay over horizons for top features."""
    lines = []
    lines.append(
        f"{'Feature':<40} " + " ".join(f"{'H' + str(h):>9}" for h in sorted(decay_results.keys()))
    )
    lines.append("-" * (40 + 10 * len(decay_results)))

    for feat in top_features[:10]:
        row = [f"{feat[:38]:<40}"]
        for h in sorted(decay_results.keys()):
            df_h = decay_results[h]
            match = df_h[df_h["feature"] == feat]
            if not match.empty:
                row.append(f"{match.iloc[0]['rank_ic']:>9.4f}")
            else:
                row.append(" " * 9)
        lines.append(" ".join(row))

    return "\n".join(lines)


def run_analysis(
    features: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    top_n: int = 30,
) -> dict:
    """Run full IC/IR analysis pipeline.

    Args:
        features: DataFrame with feature columns and forward_return_Xd columns.
        horizons: Forward return horizons to analyze.
        rolling_windows: Rolling window sizes for stability analysis.
        top_n: Number of top features to report.

    Returns:
        Dict with all analysis results.
    """
    feature_cols = [
        c
        for c in features.columns
        if not any(c.startswith(p) for p in ("forward_return_", "rolling_ic_"))
    ]

    # Filter to numeric columns only
    numeric_cols = [
        c for c in feature_cols if features[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    if not numeric_cols:
        return {"error": "No numeric feature columns found"}

    X = features[numeric_cols]

    # Primary IC analysis on 5d forward returns
    fwd_col = "forward_return_5d"
    if fwd_col not in features.columns:
        available = [c for c in features.columns if c.startswith("forward_return_")]
        fwd_col = available[0] if available else None

    if fwd_col is None:
        return {"error": "No forward_return_Xd columns found"}

    y = features[fwd_col]

    # 1. Feature IC summary
    ic_summary_df = ic_summary(X, y)

    # 2. IC decay across horizons
    decay_results = {}
    for h in horizons:
        h_col = f"forward_return_{h}d"
        if h_col in features.columns:
            decay_results[h] = compute_rank_ic(X, features[h_col])
        else:
            decay_results[h] = pd.DataFrame()

    decay_results[h] = compute_rank_ic(X, y)

    # 3. Rolling IC stability
    rolling_results = {}
    stability_scores = {}
    top_features = ic_summary_df.head(min(top_n, len(ic_summary_df)))["feature"].tolist()

    for w in rolling_windows:
        roll_df = rolling_ic_analysis(X, y, window=w)
        if roll_df.empty:
            continue
        rolling_results[w] = roll_df

        for feat in top_features[:10]:
            col_name = f"rolling_ic_{w}d_{feat}"
            if col_name in roll_df.columns:
                stability_scores[f"{feat}_w{w}"] = ic_stability_score(roll_df[col_name])

    # 4. Aggregate IR for top features
    top_ir = {}
    for feat in top_features[:15]:
        series = X[feat]
        top_ir[feat] = compute_information_ratio(series, periods_per_year=252)

    # 5. Summary stats
    significant_count = int(ic_summary_df.get("significant", pd.Series(False)).sum())
    mean_abs_ic = float(ic_summary_df["abs_ic"].mean()) if not ic_summary_df.empty else 0.0
    median_abs_ic = float(ic_summary_df["abs_ic"].median()) if not ic_summary_df.empty else 0.0

    return {
        "symbol": features.attrs.get("symbol", "unknown"),
        "n_features": len(numeric_cols),
        "n_samples": len(X.dropna()),
        "forward_return_col": fwd_col,
        "horizons": list(horizons),
        "rolling_windows": list(rolling_windows),
        "summary_stats": {
            "significant_count": significant_count,
            "mean_abs_ic": mean_abs_ic,
            "median_abs_ic": median_abs_ic,
        },
        "ic_summary": ic_summary_df.to_dict(orient="records"),
        "ic_decay": {
            str(h): df.to_dict(orient="records") for h, df in decay_results.items() if not df.empty
        },
        "stability_scores": stability_scores,
        "top_ir": top_ir,
        "top_features": top_features,
    }


def print_report(results: dict) -> None:
    """Print a comprehensive IC/IR analysis report."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    W = 70

    print("=" * W)
    print(f"  FACTOR IC/IR ANALYSIS -- {results['symbol']}")
    print("=" * W)

    print(f"\n  Features analyzed:   {results['n_features']}")
    print(f"  Observations:        {results['n_samples']}")
    print(f"  Target:              {results['forward_return_col']}")
    print(
        f"  Significant (p<0.05): {results['summary_stats']['significant_count']}/{results['n_features']}"
    )
    print(f"  Mean |IC|:           {results['summary_stats']['mean_abs_ic']:.4f}")
    print(f"  Median |IC|:         {results['summary_stats']['median_abs_ic']:.4f}")

    # Feature ranking table
    ic_summary_df = pd.DataFrame(results["ic_summary"])
    if not ic_summary_df.empty:
        print(f"\n{'-' * W}")
        print("  FEATURE RANKING (by |Rank IC|)")
        print(f"{'-' * W}")
        print(format_feature_table(ic_summary_df, top_n=min(20, len(ic_summary_df))))

    # IC decay table
    decay = results.get("ic_decay", {})
    if decay and results.get("top_features"):
        print(f"\n{'-' * W}")
        print("  IC DECAY (Rank IC across horizons)")
        print(f"{'-' * W}")
        decay_dfs = {int(k): pd.DataFrame(v) for k, v in decay.items() if v}
        if decay_dfs:
            print(format_decay_table(decay_dfs, results["top_features"]))

    # Stability
    stability = results.get("stability_scores", {})
    if stability:
        print(f"\n{'-' * W}")
        print("  ROLLING IC STABILITY (top 10 features)")
        print(f"{'-' * W}")
        print(
            f"  {'Feature':<40} {'Window':>7} {'MeanIC':>8} {'StdIC':>8} {'IR':>7} {'Pos%':>6} {'Sig%':>6}"
        )
        print("  " + "-" * 66)
        for key, scores in sorted(stability.items()):
            parts = key.rsplit("_w", 1)
            feat = parts[0] if len(parts) == 2 else key
            win = parts[1] if len(parts) == 2 else ""
            if scores["ir"] != 0 and not np.isinf(abs(scores["ir"])):
                print(
                    f"  {feat[:39]:<40} "
                    f"{win + 'd':>7} "
                    f"{scores['mean_ic']:>8.4f} "
                    f"{scores['std_ic']:>8.4f} "
                    f"{scores['ir']:>7.2f} "
                    f"{scores['frac_positive']:>5.1%} "
                    f"{scores['frac_significant']:>5.1%}"
                )

    # IR ranking
    top_ir = results.get("top_ir", {})
    if top_ir:
        print(f"\n{'-' * W}")
        print("  INFORMATION RATIO (top features)")
        print(f"{'-' * W}")
        sorted_ir = sorted(top_ir.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, ir in sorted_ir[:10]:
            print(f"  {feat[:50]:<50} IR = {ir:>7.2f}")

    print(f"\n{'=' * W}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factor IC/IR Analysis -- cross-sectional feature validation"
    )
    parser.add_argument("symbol", nargs="?", help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--features-file", help="Path to pre-saved feature parquet file")
    parser.add_argument(
        "--horizons", default="1,5,10,20", help="Forward return horizons (comma-separated)"
    )
    parser.add_argument(
        "--rolling-windows", default="21,63", help="Rolling IC window sizes (comma-separated)"
    )
    parser.add_argument("--top-n", type=int, default=30, help="Number of top features to report")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--json-output", help="Path to save JSON results")
    parser.add_argument(
        "--forward-return-col", default="forward_return_5d", help="Forward return column name"
    )
    args = parser.parse_args()

    horizons = tuple(int(h.strip()) for h in args.horizons.split(","))
    rolling_windows = tuple(int(w.strip()) for w in args.rolling_windows.split(","))

    # Load features
    if args.features_file:
        features = pd.read_parquet(args.features_file)
        features.attrs["symbol"] = Path(args.features_file).stem
    elif args.symbol:
        features = load_features_for_symbol(args.symbol, start=args.start, end=args.end)
        features.attrs["symbol"] = args.symbol
    else:
        parser.error("Either symbol or --features-file is required")

    # Override forward return column if specified
    if args.forward_return_col != "forward_return_5d":
        if args.forward_return_col not in features.columns:
            parser.error(f"Column '{args.forward_return_col}' not found in features")

    results = run_analysis(
        features,
        horizons=horizons,
        rolling_windows=rolling_windows,
        top_n=args.top_n,
    )

    print_report(results)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, default=str, indent=2)
        print(f"\nResults saved to {args.json_output}")


if __name__ == "__main__":
    main()
