"""Multi-factor fundamental backtest.

Phase Q: Backtest a multi-factor strategy using fundamental data from yfinance.
Ranks stocks by composite fundamental score, longs top quintile, equal-weight,
monthly rebalance. Compares against SPY buy-and-hold.

Usage:
    uv run scripts/backtest_multi_factor.py SPY,AAPL,MSFT,GOOGL,JPM
    uv run scripts/backtest_multi_factor.py SPY,QQQ,XLK,XLF,XLE,JPM --start 2020-01-01
    uv run scripts/backtest_multi_factor.py --sector tech --top-n 10
    uv run scripts/backtest_multi_factor.py SPY,AAPL,MSFT --factor-ic
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

from src.ml.fundamental_features import (
    FACTOR_DIRECTIONS,
    FACTOR_NAMES,
    FundamentalFeatureExtractor,
)


def load_price_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        try:
            df.index = pd.DatetimeIndex(df.index)
        except (ValueError, TypeError):
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    else:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        if df.empty:
            raise FileNotFoundError(f"No data for {symbol} (local or yfinance)")
        df.index = df.index.tz_localize(None)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def load_benchmark(start: str | None = None, end: str | None = None) -> pd.Series:
    df = load_price_data("SPY")
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df["Close"]


def run_multi_factor_backtest(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    top_n: int = 5,
    rebalance_freq: str = "ME",
    benchmark_series: pd.Series | None = None,
) -> dict:
    """Run multi-factor backtest.

    Strategy:
    1. Extract fundamental factors for all tickers
    2. Compute composite z-score (equal-weighted)
    3. Long top_n tickers, equal-weight
    4. Monthly rebalance
    5. Compare against benchmark

    Note: This uses current fundamentals (simplified). For rigorous backtesting,
    point-in-time fundamental data is needed.
    """
    if len(tickers) < top_n:
        top_n = len(tickers)

    # Load price data for all tickers
    price_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = load_price_data(t)
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]
            price_data[t] = df
        except FileNotFoundError:
            print(f"[WARN] No price data for {t}, skipping")

    if len(price_data) < 2:
        return {"error": f"Need at least 2 tickers with data, got {len(price_data)}"}

    tickers = list(price_data.keys())

    # Extract fundamentals
    extractor = FundamentalFeatureExtractor()
    factors = extractor.extract(tickers)
    zscored = extractor.z_score_factors(factors)
    composite = extractor.composite_score(factors)

    # Build common date index
    all_dates = sorted(set().union(*(df.index for df in price_data.values())))
    date_index = pd.DatetimeIndex(all_dates).sort_values()

    # Build price matrix
    close_prices = pd.DataFrame(index=date_index)
    for t in tickers:
        close_prices[t] = price_data[t]["Close"].reindex(date_index).ffill()

    close_prices = close_prices.dropna(how="all")

    # Alignment: composite score (one row per ticker) mapped to time series
    monthly_prices = close_prices.resample(rebalance_freq).last()

    # Select top_n tickers at each rebalance
    portfolio_returns = pd.Series(0.0, index=monthly_prices.index, dtype=float)

    for i in range(len(monthly_prices)):
        month_close = monthly_prices.iloc[i]

        # Get available tickers for this month
        available = [t for t in tickers if not pd.isna(month_close.get(t, np.nan))]
        if len(available) < 2:
            continue

        # Rank by composite score
        comp_available = composite.loc[composite.index.isin(available)]
        selected = comp_available.nlargest(min(top_n, len(comp_available))).index.tolist()

        # Equal-weight return for this month
        prev_month_idx = max(0, i - 1)
        if i == 0:
            continue

        month_returns = []
        for t in selected:
            prev_close = monthly_prices.iloc[prev_month_idx].get(t, np.nan)
            curr_close = month_close.get(t, np.nan)
            if not pd.isna(prev_close) and not pd.isna(curr_close) and prev_close > 0:
                month_returns.append((curr_close - prev_close) / prev_close)

        if month_returns:
            portfolio_returns.iloc[i] = np.mean(month_returns)

    # Cumulative return
    portfolio_cum = (1.0 + portfolio_returns).cumprod()

    # Benchmark
    if benchmark_series is not None:
        bench_monthly = benchmark_series.resample(rebalance_freq).last().pct_change()
        bench_cum = (1.0 + bench_monthly).cumprod()

        # Align
        common_idx = portfolio_cum.index.intersection(bench_cum.index)
        portfolio_aligned = portfolio_cum.loc[common_idx]
        bench_aligned = bench_cum.loc[common_idx]
    else:
        bench_aligned = None
        common_idx = portfolio_cum.index
        portfolio_aligned = portfolio_cum

    # Metrics
    rets = portfolio_returns[portfolio_returns != 0]
    total_return = (
        float((portfolio_aligned.iloc[-1] - 1.0) * 100) if len(portfolio_aligned) > 0 else 0.0
    )
    ann_return = float(rets.mean() * 12 * 100) if len(rets) > 0 else 0.0
    ann_vol = float(rets.std() * np.sqrt(12) * 100) if len(rets) > 0 else 0.0
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(12)) if len(rets) > 0 and rets.std() > 0 else 0.0
    )
    max_dd = float(compute_max_drawdown(portfolio_aligned)) if len(portfolio_aligned) > 1 else 0.0

    bench_return = None
    bench_sharpe = None
    if bench_aligned is not None and len(bench_aligned) > 0:
        bench_return = float((bench_aligned.iloc[-1] - 1.0) * 100)
        bench_rets = bench_monthly.reindex(common_idx).dropna()
        bench_sharpe = (
            float(bench_rets.mean() / bench_rets.std() * np.sqrt(12))
            if len(bench_rets) > 0 and bench_rets.std() > 0
            else 0.0
        )

    return {
        "tickers": tickers,
        "top_n": top_n,
        "n_rebalances": len(portfolio_returns),
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_return, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "benchmark_return_pct": round(bench_return, 2) if bench_return is not None else None,
        "benchmark_sharpe": round(bench_sharpe, 2) if bench_sharpe is not None else None,
        "composite_scores": composite.to_dict(),
    }


def compute_max_drawdown(cumulative: pd.Series) -> float:
    peak = cumulative.expanding().max()
    dd = (cumulative - peak) / peak
    return float(dd.min() * 100)


def compute_factor_ic(
    factors: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    forward_periods: int = 63,
) -> pd.DataFrame:
    """Compute Information Coefficient (rank correlation) for each factor.

    IC = Spearman rank correlation between factor value and forward return.
    High |IC| means the factor has predictive power.

    Args:
        factors: Raw fundamental factor DataFrame (tickers x factors).
        price_data: Dict of ticker -> price DataFrame.
        forward_periods: Forward return horizon in trading days (default 63 = ~3mo).

    Returns:
        DataFrame with IC per factor.
    """
    forward_returns: dict[str, float] = {}
    for ticker, pdf in price_data.items():
        if len(pdf) > forward_periods:
            start_price = pdf["Close"].iloc[0]
            end_price = pdf["Close"].iloc[min(forward_periods, len(pdf) - 1)]
            if start_price > 0:
                forward_returns[ticker] = (end_price - start_price) / start_price

    results = []
    for factor in FACTOR_NAMES:
        if factor not in factors.columns:
            continue
        valid_tickers = [
            t for t in factors.index if t in forward_returns and not pd.isna(factors.loc[t, factor])
        ]
        if len(valid_tickers) < 5:
            results.append(
                {
                    "factor": factor,
                    "category": FACTOR_DIRECTIONS.get(factor, {"category": "unknown"}),
                    "ic": np.nan,
                    "p_value": np.nan,
                    "n": len(valid_tickers),
                }
            )
            continue

        import scipy.stats as stats

        factor_vals = [factors.loc[t, factor] for t in valid_tickers]
        fwd_returns = [forward_returns[t] for t in valid_tickers]
        ic, p = stats.spearmanr(factor_vals, fwd_returns)
        results.append(
            {
                "factor": factor,
                "category": FACTOR_DIRECTIONS.get(factor, {"category": "unknown"}),
                "ic": round(float(ic), 4),
                "p_value": round(float(p), 4),
                "n": len(valid_tickers),
            }
        )

    return pd.DataFrame(results).sort_values("ic", key=abs, ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest multi-factor fundamental strategy.")
    parser.add_argument(
        "tickers",
        nargs="?",
        default="SPY,AAPL,MSFT,GOOGL,JPM",
        help="Comma-separated ticker list (default: SPY,AAPL,MSFT,GOOGL,JPM)",
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--top-n", type=int, default=3, help="Number of top-ranked tickers to hold (default 3)"
    )
    parser.add_argument(
        "--factor-ic", action="store_true", help="Compute factor Information Coefficients"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--sector", default=None, help="Sector filter (e.g., tech, financial)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    # Sector presets
    SECTOR_PRESETS = {
        "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD", "INTC", "CRM"],
        "financial": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
        "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO"],
        "healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABBV", "TMO", "DHR", "LLY"],
        "consumer": ["PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE"],
    }
    if args.sector and args.sector.lower() in SECTOR_PRESETS:
        tickers = SECTOR_PRESETS[args.sector.lower()]

    sep = "=" * 70
    print(f"\n{sep}")
    print("  Multi-Factor Fundamental Backtest -- Phase Q")
    print(
        f"  Tickers: {len(tickers)} ({', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''})"
    )
    print(f"{sep}\n")

    # Load benchmark
    print("-- Loading benchmark (SPY) --")
    benchmark = load_benchmark(args.start, args.end)
    print(
        f"  SPY period: {benchmark.index[0].strftime('%Y-%m-%d')} to {benchmark.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"  SPY bars: {len(benchmark)}")

    # Extract fundamentals
    print("\n-- Extracting fundamental factors --")
    extractor = FundamentalFeatureExtractor()
    factors = extractor.extract(tickers)
    zscored = extractor.z_score_factors(factors)
    composite = extractor.composite_score(factors)

    print(f"  Factors: {len(FACTOR_NAMES)} available")
    coverage = (factors.notna().sum(axis=1) > 0).sum()
    print(f"  Tickers with data: {coverage}/{len(tickers)}")

    # Top/bottom by composite
    comp_sorted = composite.dropna().sort_values(ascending=False)
    print("\n  Top 5 by composite score:")
    for t, score in comp_sorted.head(5).items():
        print(f"    {t:6s}  {score:+8.4f}")
    print("  Bottom 5 by composite score:")
    for t, score in comp_sorted.tail(5).items():
        print(f"    {t:6s}  {score:+8.4f}")

    # Factor summary
    print("\n-- Factor Statistics --")
    summary = extractor.factor_summary(factors)
    for fn in FACTOR_NAMES:
        if fn in summary.index:
            s = summary.loc[fn]
            print(
                f"  {fn:20s}  count={s['count']:3.0f}  mean={s['mean']:10.4f}  med={s['50%']:10.4f}"
            )

    # Run backtest
    print(f"\n-- Backtest (top {args.top_n}, monthly rebalance) --")
    result = run_multi_factor_backtest(
        tickers,
        args.start,
        args.end,
        top_n=args.top_n,
        benchmark_series=benchmark,
    )

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    print(f"  Total Return:   {result['total_return_pct']:+.2f}%")
    print(f"  Ann. Return:    {result['ann_return_pct']:+.2f}%")
    print(f"  Ann. Vol:       {result['ann_vol_pct']:.2f}%")
    print(f"  Sharpe:         {result['sharpe']:+.2f}")
    print(f"  Max Drawdown:   {result['max_drawdown_pct']:.2f}%")
    print(f"  Rebalances:     {result['n_rebalances']}")
    if result["benchmark_return_pct"] is not None:
        print(f"  SPY Return:     {result['benchmark_return_pct']:+.2f}%")
        print(f"  SPY Sharpe:     {result['benchmark_sharpe']:+.2f}")

    # Factor IC
    if args.factor_ic:
        print("\n-- Factor Information Coefficients (forward 63d) --")
        price_data = {}
        for t in tickers:
            try:
                pdf = load_price_data(t)
                if args.start:
                    start_dt = pd.Timestamp(args.start)
                    pdf = pdf[pdf.index >= start_dt]
                if args.end:
                    end_dt = pd.Timestamp(args.end)
                    pdf = pdf[pdf.index <= end_dt]
                price_data[t] = pdf
            except FileNotFoundError:
                pass

        ic_df = compute_factor_ic(factors, price_data)
        for _, row in ic_df.iterrows():
            sig = (
                "***"
                if row["p_value"] < 0.001
                else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
            )
            print(
                f"  {row['factor']:20s}  IC={row['ic']:+8.4f}  "
                f"p={row['p_value']:.4f} {sig}  n={row['n']:.0f}"
            )

    # JSON output
    if args.json:
        result["factor_ic"] = ic_df.to_dict(orient="records") if args.factor_ic else None
        print("\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
