"""Backtest simple technical baselines vs ML strategy on SPY (2020-2026).

Strategies:
  - RSI(14): Buy when RSI < 30, sell when RSI > 70
  - MA Crossover: Buy on golden cross (50 > 200), sell on death cross (50 < 200)
  - Buy & Hold
  - ML reference row (from sweep results)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

SYMBOL = "SPY"
START = "2020-01-01"
END = "2026-05-14"
CASH = 100_000
COMMISSION = 0.001


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0).dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    df = df[df.index >= START]
    if END:
        df = df[df.index <= END]
    return df


# ──────────────────────────────────────────────────────────────────────
# Indicator helpers
# ──────────────────────────────────────────────────────────────────────


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


# ──────────────────────────────────────────────────────────────────────
# Strategies
# ──────────────────────────────────────────────────────────────────────


class RSIStrategy(Strategy):
    """Buy when RSI(14) < 30 (oversold), sell when RSI(14) > 70 (overbought)."""

    rsi_period = 14
    oversold = 30
    overbought = 70

    def init(self) -> None:
        close = self.data.Close.s
        self.rsi = self.I(_rsi, close, period=self.rsi_period, name="RSI")

    def next(self) -> None:
        rsi_val = self.rsi[-1]

        if self.position:
            if self.position.is_long and rsi_val > self.overbought:
                self.position.close()
            return

        if rsi_val < self.oversold:
            self.buy()


class MACrossoverStrategy(Strategy):
    """Buy when 50-day SMA crosses above 200-day SMA (golden cross).
    Sell when 50-day SMA crosses below 200-day SMA (death cross)."""

    fast_period = 50
    slow_period = 200

    def init(self) -> None:
        close = self.data.Close.s
        self.sma_fast = self.I(SMA, close, self.fast_period, name="SMA50")
        self.sma_slow = self.I(SMA, close, self.slow_period, name="SMA200")

    def next(self) -> None:
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def run_strategy(name: str, strategy_class: type[Strategy], df: pd.DataFrame) -> dict:
    bt = Backtest(df, strategy_class, cash=CASH, commission=COMMISSION)
    stats = bt.run()
    return {
        "strategy": name,
        "return_pct": round(stats["Return [%]"], 2),
        "sharpe": round(stats["Sharpe Ratio"], 2),
        "max_drawdown": round(stats["Max. Drawdown [%]"], 2),
        "win_rate": round(stats["Win Rate [%]"], 2),
        "profit_factor": round(stats["Profit Factor"], 2),
        "num_trades": stats["# Trades"],
    }


def main() -> None:
    df = load_data(SYMBOL)
    print(
        f"Data: {SYMBOL} {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}  ({len(df)} bars)"
    )
    print(f"Cash: ${CASH:,.0f}  Commission: {COMMISSION:.3f}\n")

    results: list[dict] = []

    print("Running RSI(14)...")
    results.append(run_strategy("RSI(14) <30/>70", RSIStrategy, df))

    print("Running MA Crossover (50/200)...")
    results.append(run_strategy("MA Cross 50/200", MACrossoverStrategy, df))

    # Buy & Hold via backtesting.py (run RSI but capture stats for B&H)
    print("Evaluating Buy & Hold...")
    bt = Backtest(df, RSIStrategy, cash=CASH, commission=COMMISSION)
    stats = bt.run()
    results.append(
        {
            "strategy": "Buy & Hold",
            "return_pct": round(stats["Buy & Hold Return [%]"], 2),
            "sharpe": round(
                stats["Buy & Hold Return [%]"] / (stats["Volatility (Ann.) [%]"] or 1), 2
            )
            if stats["Volatility (Ann.) [%]"]
            else 0,
            "max_drawdown": round(stats["Max. Drawdown [%]"], 2),
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "num_trades": 0,
        }
    )
    from math import sqrt

    daily_ret = df["Close"].pct_change().dropna()
    avg_daily = daily_ret.mean()
    std_daily = daily_ret.std()
    if std_daily > 0:
        results[-1]["sharpe"] = round(sqrt(252) * avg_daily / std_daily, 2)

    # ML reference rows — from sweep_entry_thresholds.py sweep on retrained model
    results.append(
        {
            "strategy": "ML (et=0.45 trail)*",
            "return_pct": 66.1,
            "sharpe": 0.50,
            "max_drawdown": 24.4,
            "win_rate": 37.5,
            "profit_factor": 1.34,
            "num_trades": 72,
        }
    )
    results.append(
        {
            "strategy": "ML (et=0.35 trail)\u2020",
            "return_pct": 87.3,
            "sharpe": 0.60,
            "max_drawdown": 25.7,
            "win_rate": 36.6,
            "profit_factor": 1.66,
            "num_trades": 71,
        }
    )

    # ── Print comparison table ──
    print("\n" + "=" * 90)
    print(
        f"SPY {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')} -- Strategy Comparison"
    )
    print("=" * 90)
    header = f"{'Strategy':<24} {'Return%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinRate%':>9} {'PF':>6} {'#Trades':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        wr = (
            f"{r['win_rate']:.1f}"
            if not (isinstance(r["win_rate"], float) and r["win_rate"] != r["win_rate"])
            else "N/A"
        )
        pf = (
            f"{r['profit_factor']:.2f}"
            if not (
                isinstance(r["profit_factor"], float) and r["profit_factor"] != r["profit_factor"]
            )
            else "N/A"
        )
        print(
            f"{r['strategy']:<24} {r['return_pct']:>7.1f}% {r['sharpe']:>6.2f} "
            f"{r['max_drawdown']:>7.1f}% {wr:>9} {pf:>6} {r['num_trades']:>8}"
        )

    print("-" * 90)
    print("* ML results from sweep_entry_thresholds.py (SPY, approx). Actual values vary by run.")
    print()


if __name__ == "__main__":
    main()
