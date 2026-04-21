"""Ploomber pipeline for Trading Pattern Detection System.

Each function is an entry point for a pipeline task.
Function tasks receive:
  product: output path (str), or dict if multiple outputs
  upstream: dict of task_name -> output path (str)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def fetch_data(
    product: str,
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and save."""
    import yfinance as yf

    output_path = Path(product)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df.to_csv(output_path, index=False)
    return df


def compute_indicators(
    product: str,
    upstream: Dict[str, str],
) -> pd.DataFrame:
    """Compute technical indicators on raw OHLCV data."""
    data = pd.read_csv(upstream["fetch_data"], parse_dates=["Date"], index_col="Date")
    data.index = pd.to_datetime(data.index)

    try:
        from src.indicators.technical import compute_all_indicators

        indicators = compute_all_indicators(data)
    except Exception:
        indicators = data.copy()

    output_path = Path(product)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(output_path)
    return indicators


def detect_patterns(
    product: str,
    upstream: Dict[str, str],
) -> pd.DataFrame:
    """Detect chart patterns and produce signals."""
    data = pd.read_parquet(upstream["compute_indicators"])

    from src.main import PatternDetectionSystem, get_all_patterns

    patterns = get_all_patterns()
    system = PatternDetectionSystem(patterns=patterns)
    signals = system.scan_patterns(data)
    signals_df = pd.DataFrame(signals)

    output_path = Path(product)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_path)
    return signals_df


def generate_signals(
    product: str,
    upstream: Dict[str, str],
) -> pd.DataFrame:
    """Score and filter trading signals."""
    signals_df = pd.read_parquet(upstream["detect_patterns"])

    if not signals_df.empty and "timestamp" in signals_df.columns:
        signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
        score_counts = signals_df.groupby("timestamp").size().reset_index(name="confluence_count")
        signals_df = signals_df.merge(score_counts, on="timestamp", how="left")

    output_path = Path(product)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_path)
    return signals_df


def backtest(
    product: str,
    upstream: Dict[str, str],
) -> Dict[str, Any]:
    """Run backtest using detected signals."""
    data_path = Path(upstream["fetch_data"])
    if not data_path.exists():
        raise FileNotFoundError(f"No data file at {data_path}")

    data = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")
    data.index = pd.to_datetime(data.index)

    from backtesting import Backtest, Strategy

    signals_df = pd.read_parquet(Path(upstream["detect_patterns"]))

    class SignalStrategy(Strategy):
        """Simple backtesting strategy using detected signals."""

        def init(self):
            pass

        def next(self):
            current_time = self.data.index[-1]
            if not hasattr(self, "signal_set"):
                self.signal_set = set(
                    pd.to_datetime(signals_df["timestamp"])
                    if "timestamp" in signals_df.columns
                    else []
                )
            if current_time in self.signal_set:
                matching = signals_df[pd.to_datetime(signals_df["timestamp"]) == current_time]
                for _, sig in matching.iterrows():
                    direction = sig.get("direction", "")
                    if self.position:
                        continue

                    price = self.data.Close[-1]
                    sl = sig.get("stop_loss", None)
                    tp = sig.get("take_profit_1", None)

                    # Validate SL/TP for backtesting.py constraints
                    if sl is not None and tp is not None:
                        if "LONG" in direction.upper() and not (sl < price < tp):
                            sl, tp = None, None
                        elif "SHORT" in direction.upper() and not (tp < price < sl):
                            sl, tp = None, None

                    if "LONG" in direction.upper():
                        self.buy(sl=sl, tp=tp)
                    elif "SHORT" in direction.upper():
                        self.sell(sl=sl, tp=tp)

    bt = Backtest(data, SignalStrategy, cash=100_000, commission=0.001)
    stats = bt.run()

    output_path = Path(product)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "total_trades": stats.get("# Trades", 0),
        "win_rate": (stats.get("Win Rate [%]", 0) / 100 if stats.get("Win Rate [%]") else 0),
        "total_return": stats.get("Return [%]", 0),
        "max_drawdown_pct": stats.get("Max. Drawdown [%]", 0),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_parquet(output_path)
    return metrics


def generate_report(
    product: str,
    upstream: Dict[str, str],
) -> str:
    """Generate visualization reports and charts."""
    data_path = Path(upstream["fetch_data"])
    data = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")
    data.index = pd.to_datetime(data.index)

    output_dir = Path(product).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read backtest metrics
    bt_metrics = pd.read_parquet(upstream["backtest"])

    print(f"Backtest metrics: {bt_metrics.to_dict()}")

    # Generate a simple HTML report
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
<tr><td>Total Trades</td><td>{bt_metrics.iloc[0].get("total_trades", "N/A")}</td></tr>
<tr><td>Win Rate</td><td>{bt_metrics.iloc[0].get("win_rate", "N/A"):,.2%}</td></tr>
<tr><td>Total Return</td><td>{bt_metrics.iloc[0].get("total_return", "N/A"):.2f}%</td></tr>
<tr><td>Max Drawdown</td><td>{bt_metrics.iloc[0].get("max_drawdown_pct", "N/A"):.2f}%</td></tr>
</table>
<h2>Data Summary</h2>
<p>{len(data)} bars of data from {data.index[0]} to {data.index[-1]}</p>
</body>
</html>"""

    output_path = Path(product)
    output_path.write_text(html_content)

    # Generate equity curve chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(data.index, data["Close"], linewidth=1)
        axes[0].set_title("SPY Close Price")
        axes[0].set_ylabel("Price")
        axes[0].grid(True)
        axes[1].bar(range(len(data)), data["Volume"], width=1)
        axes[1].set_title("Volume")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Volume")
        axes[1].grid(True)
        plt.tight_layout()
        chart_path = Path(product).with_suffix(".png")
        fig.savefig(str(chart_path), dpi=150)
        plt.close(fig)
        print(f"Chart saved to {chart_path}")
    except Exception as e:
        print(f"Chart generation issue: {e}")

    return "Report generation complete"
