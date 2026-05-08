"""Prefect workflow for the Trading Pattern Detection System.

Replaces Ploomber pipeline (pipeline.yaml) with Prefect tasks and flows.
Adds retries, caching, scheduling, and a dashboard.

Run:    uv run python pipeline/prefect_flow.py
Serve:  uv run prefect server start
        uv run python pipeline/prefect_flow.py serve

The existing Ploomber pipeline is preserved at pipeline.yaml and pipeline/pipeline.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.tasks import task_input_hash


@task(
    name="fetch-data",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_policy=INPUTS,
    cache_expiration=86400,  # 24h
)
def fetch_data(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    output_path: str = "output/data_fetched.csv",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    import yfinance as yf

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df.to_csv(path, index=False)
    return df


@task(name="compute-indicators", retries=2, retry_delay_seconds=10)
def compute_indicators(
    data: pd.DataFrame,
    output_path: str = "output/data/indicators.parquet",
) -> pd.DataFrame:
    """Compute technical indicators on raw OHLCV data."""
    if "Date" in data.columns:
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)

    try:
        from src.indicators.technical import compute_all_indicators

        indicators = compute_all_indicators(data)
    except Exception:
        indicators = data.copy()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(path)
    return indicators


@task(name="detect-patterns", retries=2, retry_delay_seconds=10)
def detect_patterns(
    indicators: pd.DataFrame,
    output_path: str = "output/data/patterns.parquet",
) -> pd.DataFrame:
    """Detect chart patterns and produce signals."""
    from src.main import PatternDetectionSystem, get_all_patterns

    patterns = get_all_patterns()
    system = PatternDetectionSystem(patterns=patterns)
    signals = system.scan_patterns(indicators)
    signals_df = pd.DataFrame(signals)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(path)
    return signals_df


@task(name="generate-signals", retries=1)
def generate_signals(
    patterns_df: pd.DataFrame,
    output_path: str = "output/data/signals.parquet",
) -> pd.DataFrame:
    """Score and filter trading signals with confluence counts."""
    if not patterns_df.empty and "timestamp" in patterns_df.columns:
        patterns_df["timestamp"] = pd.to_datetime(patterns_df["timestamp"])
        score_counts = patterns_df.groupby("timestamp").size().reset_index(name="confluence_count")
        patterns_df = patterns_df.merge(score_counts, on="timestamp", how="left")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    patterns_df.to_parquet(path)
    return patterns_df


@task(name="run-backtest", retries=1, retry_delay_seconds=10)
def run_backtest(
    data: pd.DataFrame,
    patterns_df: pd.DataFrame,
    output_path: str = "output/data/backtest.parquet",
    cash: float = 100_000,
    commission: float = 0.001,
) -> dict[str, Any]:
    """Run backtest using the backtesting.py library."""
    from backtesting import Backtest, Strategy

    if "Date" in data.columns:
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)

    class SignalStrategy(Strategy):
        def init(self) -> None:
            pass

        def next(self) -> None:
            current_time = self.data.index[-1]
            if not hasattr(self, "signal_set"):
                self.signal_set = set(
                    pd.to_datetime(patterns_df["timestamp"])
                    if "timestamp" in patterns_df.columns
                    else []
                )
            if current_time in self.signal_set:
                matching = patterns_df[pd.to_datetime(patterns_df["timestamp"]) == current_time]
                if self.position:
                    return
                for _, sig in matching.iterrows():
                    direction = str(sig.get("direction", ""))
                    price = self.data.Close[-1]
                    sl = sig.get("stop_loss", None)
                    tp = sig.get("take_profit_1", None)
                    if sl is not None and tp is not None:
                        if "LONG" in direction.upper() and not (sl < price < tp):
                            sl, tp = None, None
                        elif "SHORT" in direction.upper() and not (tp < price < sl):
                            sl, tp = None, None
                    if "LONG" in direction.upper():
                        self.buy(sl=sl, tp=tp)
                    elif "SHORT" in direction.upper():
                        self.sell(sl=sl, tp=tp)

    bt = Backtest(data, SignalStrategy, cash=cash, commission=commission)
    stats = bt.run()

    metrics = {
        "total_trades": stats.get("# Trades", 0),
        "win_rate": (stats.get("Win Rate [%]", 0) / 100 if stats.get("Win Rate [%]") else 0),
        "total_return": stats.get("Return [%]", 0),
        "max_drawdown_pct": stats.get("Max. Drawdown [%]", 0),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_parquet(path)
    return metrics


@task(name="generate-report")
def generate_report(
    data: pd.DataFrame,
    bt_metrics: dict[str, Any],
    output_path: str = "output/reports/tearsheet.html",
) -> str:
    """Generate HTML report with backtest metrics and charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "Date" in data.columns:
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Pipeline Report</title>
<style>body{{font-family:sans-serif;margin:40px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:8px}}</style>
</head>
<body>
<h1>Trading Pattern Detection Pipeline Report</h1>
<h2>Backtest Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Trades</td><td>{bt_metrics.get("total_trades", "N/A")}</td></tr>
<tr><td>Win Rate</td><td>{bt_metrics.get("win_rate", 0):,.2%}</td></tr>
<tr><td>Total Return</td><td>{bt_metrics.get("total_return", 0):.2f}%</td></tr>
<tr><td>Max Drawdown</td><td>{bt_metrics.get("max_drawdown_pct", 0):.2f}%</td></tr>
</table>
<h2>Data Summary</h2>
<p>{len(data)} bars from {data.index[0]} to {data.index[-1]}</p>
</body>
</html>"""
    Path(output_path).write_text(html_content)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(data.index, data["Close"], linewidth=1)
    axes[0].set_title("SPY Close Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)
    axes[1].bar(range(len(data)), data["Volume"], width=1)
    axes[1].set_title("Volume")
    axes[1].grid(True)
    plt.tight_layout()
    chart_path = Path(output_path).with_suffix(".png")
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)

    return f"Report + chart saved to {output_path}"


@flow(
    name="trading-pipeline",
    description="End-to-end trading pattern detection pipeline",
    log_prints=True,
)
def trading_pipeline(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    cash: float = 100_000,
    commission: float = 0.001,
) -> dict[str, Any]:
    """Run the full trading pattern detection pipeline.

    Steps:
        1. Fetch OHLCV data
        2. Compute technical indicators
        3. Detect chart patterns
        4. Generate and filter signals
        5. Run backtest
        6. Generate HTML report + chart
    """
    data = fetch_data(ticker=ticker, start=start, end=end)
    indicators = compute_indicators(data)
    patterns = detect_patterns(indicators)
    signals = generate_signals(patterns)
    metrics = run_backtest(data, patterns, cash=cash, commission=commission)
    generate_report(data, metrics)

    print(f"Pipeline complete. Metrics: {metrics}")
    return metrics


def serve() -> None:
    """Register the flow for scheduling via Prefect server.

    Run after starting the server:
        uv run prefect server start
        uv run python pipeline/prefect_flow.py serve
    """
    trading_pipeline.serve(
        name="trading-pipeline-deployment",
        cron="0 22 * * 1-5",  # Weekdays at 10 PM
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()
    else:
        result = trading_pipeline()
        print(f"\nFinal metrics: {result}")
