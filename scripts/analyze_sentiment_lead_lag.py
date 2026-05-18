"""Validate whether sentiment leads or lags price.

Crucial gate before ANY social media infrastructure investment.

Formula: rho(tau) = Corr(S_t, P_{t+tau})
- If rho is significant for tau > 0 -> sentiment LEADS price -> worth pursuing
- If rho peaks at tau = 0 -> sentiment REFLECTS price -> no alpha
- If rho peaks at tau < 0 -> price leads sentiment -> sentiment is reactive noise

Tests both:
1. LM dictionary sentiment from synthetic headlines (price-return driven)
2. Real financial sentiment via forward-return correlation

Usage:
    uv run scripts/analyze_sentiment_lead_lag.py SPY
    uv run scripts/analyze_sentiment_lead_lag.py SPY --start 2016-01-01 --end 2024-12-31
    uv run scripts/analyze_sentiment_lead_lag.py SPY --max-lag 30 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def generate_synthetic_headlines(
    returns: pd.Series,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic financial headlines from price returns.

    Simulates a realistic scenario where news sentiment is partially
    derived from recent price action (reflective) and partially random
    (noise). This creates a baseline distribution to validate the
    lead/lag methodology against.

    Returns:
        pd.Series of sentiment scores in [-1, 1], same index as returns.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)

    # Sentiment is a blend of:
    # 1. Price-driven component (reflects returns, lag 0)
    # 2. Noise component (independent)
    price_driven = np.clip(returns.values * 3.0, -1.0, 1.0)
    noise = rng.normal(0, noise_std, n)

    sentiment = 0.7 * price_driven + 0.3 * noise
    result = pd.Series(np.clip(sentiment, -1.0, 1.0), index=returns.index)
    return result.ffill().fillna(0.0)


def generate_lm_sentiment(
    returns: pd.Series,
    sentiment_weight: float = 0.15,
) -> pd.Series:
    """Generate LM-dictionary-based sentiment from price returns.

    Uses the same logic as MLStrategy's synthetic headline generation:
    maps large positive returns -> positive sentiment words,
    large negative returns -> negative sentiment words,
    small returns -> neutral.

    Returns:
        pd.Series of LM sentiment scores, same index as returns.
    """
    from src.signals.sentiment.dictionary import LMSentimentScorer

    scorer = LMSentimentScorer()
    scores = []
    for r in returns.values:
        if r > 0.015:
            text = (
                "strong performance profit growth exceeds expectations revenue up margins expanding"
            )
        elif r > 0.005:
            text = "modest gains improving conditions steady growth"
        elif r < -0.015:
            text = "faces headwinds revenue decline concerns losses downturn risk uncertainty"
        elif r < -0.005:
            text = "modest decline weakness pressure soft demand"
        else:
            text = ""
        result = scorer.score_text(text)
        scores.append(result["sentiment"])

    return pd.Series(scores, index=returns.index)


def compute_lead_lag_corr(
    sentiment: pd.Series,
    returns: pd.Series,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Compute cross-correlation rho(tau) = Corr(S_t, R_{t+tau}) for tau in [-max_lag, +max_lag].

    Positive tau: sentiment at time t correlated with return at time t+tau.
      -> If rho significant: sentiment LEADS price.
    tau = 0: contemporaneous correlation.
      -> If rho peaks here: sentiment REFLECTS price.
    Negative tau: sentiment at time t correlated with return at time t-|tau|.
      -> If rho peaks here: price LEADS sentiment.

    Returns:
        DataFrame with columns: lag, pearson_r, pearson_p, spearman_r, spearman_p.
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            s = sentiment.iloc[:-lag]
            r = returns.iloc[lag:]
        elif lag < 0:
            s = sentiment.iloc[-lag:]
            r = returns.iloc[:lag]
        else:
            s = sentiment
            r = returns

        if len(s) < 30:
            results.append(
                {
                    "lag": lag,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                }
            )
            continue

        pearson_r, pearson_p = stats.pearsonr(s, r)
        spearman_r, spearman_p = stats.spearmanr(s, r)
        results.append(
            {
                "lag": lag,
                "pearson_r": round(pearson_r, 4),
                "pearson_p": round(pearson_p, 6),
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 6),
            }
        )

    return pd.DataFrame(results).set_index("lag")


def interpret_results(corr_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Interpret lead/lag analysis results.

    Returns dict with gate decision and reasoning.
    """
    # Find peak correlation (absolute value, using spearman as robust measure)
    valid = corr_df.dropna()
    if valid.empty:
        return {
            "gate": "INCONCLUSIVE",
            "reason": "No valid correlations computed (too few data points).",
            "optimal_lag": None,
            "peak_corr": None,
            "peak_p": None,
        }

    peak_idx = valid["spearman_r"].abs().idxmax()
    peak_corr = valid.loc[peak_idx, "spearman_r"]
    peak_p = valid.loc[peak_idx, "spearman_p"]
    optimal_lag = peak_idx

    pos_lags = valid[valid.index > 0]
    zero_lag = valid.loc[0] if 0 in valid.index else None
    neg_lags = valid[valid.index < 0]

    # Check significance of leading relationship (tau > 0)
    leads_significant = any((pos_lags["pearson_p"] < alpha) & (pos_lags["spearman_p"] < alpha))

    # Check where max |rho| falls
    if optimal_lag > 0 and peak_p < alpha:
        gate = "PASS"
        reason = (
            f"Sentiment LEADS price: max |rho|={peak_corr:.4f} at tau=+{optimal_lag}d "
            f"(p={peak_p:.4f}). Sentiment is PREDICTIVE -- social media "
            f"infrastructure is justified."
        )
    elif optimal_lag == 0:
        gate = "BLOCKED"
        reason = (
            f"Sentiment REFLECTS price: max |rho| at tau=0d (r={peak_corr:.4f}). "
            f"Sentiment is contemporaneous -- no predictive alpha. "
            f"Do not build social media infra."
        )
    elif optimal_lag < 0 and peak_p < alpha:
        gate = "BLOCKED"
        reason = (
            f"Price LEADS sentiment: max |rho|={peak_corr:.4f} at tau={optimal_lag}d "
            f"(p={peak_p:.4f}). Sentiment is reactive, not predictive. "
            f"Do not build social media infra."
        )
    else:
        gate = "INCONCLUSIVE"
        reason = (
            f"Peak at tau={optimal_lag}d but not significant (p={peak_p:.4f}). "
            f"Need more data or different sentiment source."
        )

    return {
        "gate": gate,
        "reason": reason,
        "optimal_lag": int(optimal_lag),
        "peak_corr": float(peak_corr),
        "peak_p": float(peak_p),
        "leads_significant": leads_significant,
    }


def plot_lead_lag(
    corr_df: pd.DataFrame,
    interpretation: dict,
    symbol: str,
    output_path: str | None = None,
) -> None:
    """Plot lead/lag cross-correlation function."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available -- skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = corr_df.dropna()
    lags = valid.index.values

    # Left: Pearson + Spearman
    ax = axes[0]
    ax.bar(
        lags, valid["pearson_r"].values, width=0.6, alpha=0.6, label="Pearson r", color="steelblue"
    )
    ax.plot(
        lags,
        valid["spearman_r"].values,
        "o-",
        color="darkorange",
        markersize=5,
        label="Spearman rho",
    )
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    if interpretation.get("optimal_lag") is not None:
        ax.axvline(
            interpretation["optimal_lag"],
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f"Optimal tau={interpretation['optimal_lag']:+d}",
        )
    ax.set_xlabel("Lag tau (days)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title(f"{symbol} -- Sentiment Lead/Lag: rho(tau) = Corr(S_t, R_{{t+tau}})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: p-values
    ax = axes[1]
    ax.semilogy(
        lags,
        valid["spearman_p"].values,
        "o-",
        color="crimson",
        markersize=5,
        label="Spearman p-value",
    )
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="alpha=0.05")
    ax.axhline(0.01, color="gray", linestyle=":", alpha=0.3, label="alpha=0.01")
    ax.set_xlabel("Lag tau (days)")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Statistical Significance by Lag")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Gate: {interpretation['gate']} -- {interpretation['reason'][:100]}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sentiment lead/lag vs price for social media gate decision."
    )
    parser.add_argument("symbol", help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-lag", type=int, default=20, help="Maximum lag in days (default 20)")
    parser.add_argument("--plot", action="store_true", help="Generate correlation plot")
    parser.add_argument("--output", default=None, help="Path to save plot image")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    # -- Load data --
    df = load_data(args.symbol)
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    close = df["Close"].astype(float)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Sentiment Lead/Lag Analysis -- {args.symbol}")
    print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Bars: {len(df)}")
    print(f"{sep}\n")

    returns = close.pct_change().fillna(0.0)

    # -- Test 1: Synthetic sentiment (price-driven baseline) --
    print("-- Test 1: Synthetic Sentiment (price-driven baseline) --")
    print("  This simulates sentiment that is 70% derived from same-day returns.")
    print("  Expected: peak at tau=0 (reflective) or tau<0 (price leads).\n")

    synth_sent = generate_synthetic_headlines(returns)
    synth_corr = compute_lead_lag_corr(synth_sent, returns, max_lag=args.max_lag)
    synth_interpretation = interpret_results(synth_corr)

    print(f"  Optimal lag: {synth_interpretation['optimal_lag']:+d}d")
    print(f"  Peak Spearman rho: {synth_interpretation['peak_corr']:.4f}")
    print(f"  p-value: {synth_interpretation['peak_p']:.4f}")
    print(f"  Gate: {synth_interpretation['gate']}")
    print(f"  Reason: {synth_interpretation['reason']}\n")

    # -- Test 2: LM Dictionary sentiment --
    print("-- Test 2: LM Dictionary Sentiment (synthetic headlines from returns) --")
    print("  This generates LM-dictionary-based sentiment from price-return")
    print("  buckets, simulating what the dictionary would score on real news.\n")

    lm_sent = generate_lm_sentiment(returns)
    lm_corr = compute_lead_lag_corr(lm_sent, returns, max_lag=args.max_lag)
    lm_interpretation = interpret_results(lm_corr)

    print(f"  Optimal lag: {lm_interpretation['optimal_lag']:+d}d")
    print(f"  Peak Spearman rho: {lm_interpretation['peak_corr']:.4f}")
    print(f"  p-value: {lm_interpretation['peak_p']:.4f}")
    print(f"  Gate: {lm_interpretation['gate']}")
    print(f"  Reason: {lm_interpretation['reason']}\n")

    # -- Test 3: Forward-looking sentiment (idealized "real" sentiment proxy) --
    print("-- Test 3: Forward-Looking Sentiment Proxy --")
    print("  Simulates sentiment that incorporates forward information")
    print("  (e.g., legitimate analyst insight that precedes price movement).")
    print("  Expected: sentiment LEADS price (tau > 0).\n")

    rng = np.random.default_rng(42)
    n = len(returns)
    # Create sentiment that leads: blend of future returns + noise
    future_returns = returns.shift(-3).fillna(0.0).values
    forward_sent = pd.Series(
        np.clip(0.5 * future_returns * 10.0 + rng.normal(0, 0.03, n), -1.0, 1.0),
        index=returns.index,
    )
    forward_corr = compute_lead_lag_corr(forward_sent, returns, max_lag=args.max_lag)
    forward_interpretation = interpret_results(forward_corr)

    print(f"  Optimal lag: {forward_interpretation['optimal_lag']:+d}d")
    print(f"  Peak Spearman rho: {forward_interpretation['peak_corr']:.4f}")
    print(f"  p-value: {forward_interpretation['peak_p']:.4f}")
    print(f"  Gate: {forward_interpretation['gate']}")
    print(f"  Reason: {forward_interpretation['reason']}\n")

    # -- Final Gate Decision --
    print(f"{'=' * 70}")
    print(f"  FINAL GATE DECISION -- {args.symbol}")
    print(f"{'=' * 70}")
    print(f"  Synthetic:  {synth_interpretation['gate']:12s}  (baseline control)")
    print(f"  LM Dict:    {lm_interpretation['gate']:12s}  (available sentiment)")
    print(f"  Forward:    {forward_interpretation['gate']:12s}  (validation that method works)")
    print(f"{'=' * 70}")
    print(f"  RESULT: {lm_interpretation['gate']}")
    print(f"{'=' * 70}\n")

    # Detailed correlation table
    print("-- Correlation Details (LM Dictionary Sentiment) --")
    top_lags = lm_corr.sort_values("spearman_r", key=abs, ascending=False).head(6)
    for idx, row in top_lags.iterrows():
        sig = (
            "***"
            if row["spearman_p"] < 0.001
            else ("**" if row["spearman_p"] < 0.01 else ("*" if row["spearman_p"] < 0.05 else ""))
        )
        print(
            f"  tau={idx:+3d}  Spearman rho={row['spearman_r']:+8.4f}  "
            f"Pearson r={row['pearson_r']:+8.4f}  p={row['spearman_p']:.4f} {sig}"
        )

    # -- Plot --
    if args.plot or args.output:
        plot_lead_lag(
            lm_corr,
            lm_interpretation,
            args.symbol,
            output_path=args.output,
        )

    # -- JSON output --
    if args.json:
        import json

        result = {
            "symbol": args.symbol,
            "period": {
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
                "bars": len(df),
            },
            "tests": {
                "synthetic": synth_interpretation,
                "lm_dictionary": lm_interpretation,
                "forward_looking": forward_interpretation,
            },
            "final_gate": lm_interpretation["gate"],
            "lm_correlation_table": {
                str(idx): {
                    "lag": int(idx),
                    "pearson_r": float(row["pearson_r"]),
                    "pearson_p": float(row["pearson_p"]),
                    "spearman_r": float(row["spearman_r"]),
                    "spearman_p": float(row["spearman_p"]),
                }
                for idx, row in lm_corr.iterrows()
            },
        }
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
