"""Daily Paper Trading Harness (Phase 12c P3-1).

Generates daily trade signals using the production RulesFirstStrategy.
Loads the latest market data, computes pattern signals, and logs
hypothetical trades to `reports/paper_trading/daily/`.

Zero capital at risk — purely for OOS signal accumulation.

Usage:
    uv run scripts/paper_trade_daily.py --symbol SPY
    uv run scripts/paper_trade_daily.py --symbol SPY --model models/pattern_classifier_v3_SPY.pkl
    uv run scripts/paper_trade_daily.py --basket SPY,QQQ,XLK,IWM,GLD
    uv run scripts/paper_trade_daily.py --symbol SPY --days 90  # generate 90 days of paper signals
    uv run scripts/paper_trade_daily.py --status  # show recent paper trading log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/paper_trading/daily")


@dataclass
class PaperTradeSignal:
    """A single paper trading signal record."""

    symbol: str
    timestamp: str
    signal: str
    probability: float
    price: float
    atr: float
    trail_stop: float
    regime: str = "UNKNOWN"
    signal_strength: float = 0.0
    active_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["active_patterns"] = ",".join(self.active_patterns)
        return d


class PaperTradingHarness:
    """Daily paper trading signal generator.

    Loads RulesFirstStrategy with production config, fetches latest data,
    computes signals, and logs to structured JSON files.

    Attributes:
        symbol: Ticker symbol to trade.
        entry_threshold: Minimum signal strength to enter.
        min_reliability: Minimum pattern reliability weight.
        use_multi_tp: Enable multi-TP exit (default True).
        use_quality_registry: Apply gate-based quality multipliers.
        quality_registry_path: Path to pattern gate sweep JSON.
        use_ir_weights: Use rolling IR-weighted pattern synthesis.
    """

    OUTPUT_DIR = REPORTS_DIR

    def __init__(
        self,
        symbol: str = "SPY",
        model_path: Optional[str] = None,
        entry_threshold: float = 0.55,
        min_reliability: float = 0.70,
        use_multi_tp: bool = True,
        use_quality_registry: bool = True,
        quality_registry_path: str = "reports/pattern_gate/all_patterns.json",
        use_ir_weights: bool = False,
    ) -> None:
        self.symbol = symbol
        self.model_path = model_path
        self.entry_threshold = entry_threshold
        self.min_reliability = min_reliability
        self.use_multi_tp = use_multi_tp
        self.use_quality_registry = use_quality_registry
        self.quality_registry_path = quality_registry_path
        self.use_ir_weights = use_ir_weights
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """Fetch recent OHLCV data for the symbol.

        Args:
            lookback_days: Number of calendar days to fetch.
        """
        import yfinance as yf

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        symbol_map = {
            "BTC_USD": "BTC-USD",
            "GOLD": "GC=F",
        }
        yf_symbol = symbol_map.get(self.symbol, self.symbol)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())

        if df.empty:
            raise ValueError(f"No data returned for {yf_symbol} ({start_date} to {end_date})")

        df.columns = [c.title() for c in df.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        logger.info(
            f"Fetched {len(df)} bars for {self.symbol} ({df.index[0].date()} to {df.index[-1].date()})"
        )
        return df

    def compute_rules_signal(self, df: pd.DataFrame) -> PaperTradeSignal:
        """Compute RulesFirstStrategy signal on the last bar using aggregate score.

        Returns:
            PaperTradeSignal with signal, strength, active patterns.
        """
        from src.strategies.rules_first_strategy import RulesFirstStrategy
        from backtesting import Backtest

        _strategy_ref: list = []

        bt = Backtest(df, RulesFirstStrategy, cash=100_000, commission=0.001)
        bt.run(
            entry_threshold=self.entry_threshold if hasattr(self, "entry_threshold") else 0.55,
            min_reliability=self.min_reliability,
            use_multi_tp=self.use_multi_tp,
            use_quality_registry=self.use_quality_registry,
            quality_registry_path=self.quality_registry_path,
            use_ir_weights=self.use_ir_weights,
            _strategy_ref=_strategy_ref,
        )

        if not _strategy_ref:
            raise RuntimeError("Strategy reference not captured during backtest")

        strategy = _strategy_ref[0]
        last_idx = len(df) - 1

        signal_strength = strategy._compute_score(last_idx)
        prob = float(np.tanh(signal_strength))

        atr_arr = getattr(strategy, "_atr", None)
        atr_val = float(atr_arr[last_idx]) if atr_arr is not None else 0.0
        price = float(df["Close"].iloc[-1])
        trail_stop = price - 2 * atr_val if atr_val > 0 else price * 0.95

        active_patterns: List[str] = []
        for name, signals in strategy._signals_cache.items():
            if last_idx < len(signals) and signals[last_idx] != 0:
                active_patterns.append(name)

        signal = np.tanh(signal_strength)
        if signal > self.entry_threshold:
            action = "BUY"
        elif signal < -self.entry_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return PaperTradeSignal(
            symbol=self.symbol,
            timestamp=df.index[-1].isoformat(),
            signal=action,
            probability=prob,
            price=price,
            atr=atr_val,
            trail_stop=trail_stop,
            signal_strength=signal,
            active_patterns=active_patterns,
        )

    def log_signal(self, signal: PaperTradeSignal) -> Path:
        """Append signal to the daily log file.

        Returns:
            Path to the log file.
        """
        today = date.today().isoformat()
        log_path = self.OUTPUT_DIR / f"{self.symbol}_{today}.json"

        records: List[Dict[str, Any]] = []
        if log_path.exists():
            with open(log_path) as f:
                records = json.load(f)

        records.append(signal.to_dict())

        with open(log_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

        logger.info(f"Logged {signal.signal} signal ({signal.signal_strength:+.3f}) to {log_path}")
        return log_path

    def run_backfill(self, days: int = 90) -> List[PaperTradeSignal]:
        """Generate paper signals for the last N days.

        Simulates what the paper trading system would have produced
        on each historical day.

        Args:
            days: Number of days to backfill.

        Returns:
            List of PaperTradeSignal records.
        """
        df = self.fetch_data(lookback_days=days + 252)
        signals: List[PaperTradeSignal] = []

        from src.strategies.rules_first_strategy import RulesFirstStrategy
        from backtesting import Backtest

        bt = Backtest(df, RulesFirstStrategy, cash=100_000, commission=0.001)
        stats = bt.run(
            entry_threshold=self.entry_threshold,
            min_reliability=self.min_reliability,
            use_multi_tp=self.use_multi_tp,
            use_quality_registry=self.use_quality_registry,
            quality_registry_path=self.quality_registry_path,
            use_ir_weights=self.use_ir_weights,
        )

        trades = getattr(stats, "_trades", [])
        if trades is not None:
            for t in trades:
                signals.append(
                    PaperTradeSignal(
                        symbol=self.symbol,
                        timestamp=str(t.entry_time) if hasattr(t, "entry_time") else "",
                        signal="BUY",
                        probability=0.6,
                        price=float(t.entry_price) if hasattr(t, "entry_price") else 0.0,
                        atr=0.0,
                        trail_stop=0.0,
                    )
                )

        logger.info(f"Backfill generated {len(signals)} signals over {days} days")
        return signals

    def run(self) -> PaperTradeSignal:
        """Run paper signal generation for today.

        Returns:
            PaperTradeSignal for the latest bar.
        """
        df = self.fetch_data()
        signal = self.compute_rules_signal(df)
        self.log_signal(signal)
        return signal


def show_status(symbols: Optional[List[str]] = None, days: int = 7) -> None:
    """Show recent paper trading activity.

    Args:
        symbols: Filter by symbols (None = all).
        days: Show activity from last N days.
    """
    if not REPORTS_DIR.exists():
        print("No paper trading logs found.")
        return

    cutoff = date.today() - timedelta(days=days)
    files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)

    print(f"\n--- Paper Trading Log (last {days} days) ---")
    total_buy = 0
    total_sell = 0
    total_hold = 0

    for f in files:
        sym = f.stem.split("_")[0]
        if symbols and sym not in symbols:
            continue

        with open(f) as fh:
            records = json.load(fh)
            for r in records:
                ts = r.get("timestamp", "")[:10]
                if ts >= cutoff.isoformat():
                    print(
                        f"  {ts} | {r['symbol']:5s} | {r['signal']:4s} | "
                        f"str={r.get('signal_strength', 0):+.3f} | "
                        f"prob={r.get('probability', 0):.3f} | "
                        f"price={r.get('price', 0):.2f}"
                    )
                    if r.get("signal") == "BUY":
                        total_buy += 1
                    elif r.get("signal") == "SELL":
                        total_sell += 1
                    else:
                        total_hold += 1

    print(f"\nSummary: BUY={total_buy}, SELL={total_sell}, HOLD={total_hold}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily Paper Trading Harness (Phase 07 Production)"
    )
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--basket", type=str, help="Comma-separated tickers for multi-symbol")
    parser.add_argument(
        "--days", type=int, default=0, help="Backfill N days of signals (0 = today only)"
    )
    parser.add_argument("--entry-threshold", type=float, default=0.55)
    parser.add_argument(
        "--min-reliability", type=float, default=0.70, help="Min pattern reliability"
    )
    parser.add_argument("--no-multi-tp", action="store_true", help="Disable multi-TP exit")
    parser.add_argument(
        "--no-quality-registry", action="store_true", help="Disable quality registry"
    )
    parser.add_argument("--quality-registry-path", default="reports/pattern_gate/all_patterns.json")
    parser.add_argument(
        "--ir-weights", action="store_true", help="Use IR-weighted pattern synthesis"
    )
    parser.add_argument("--status", action="store_true", help="Show recent paper trading log")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    symbols = [s.strip() for s in args.basket.split(",")] if args.basket else [args.symbol]

    for sym in symbols:
        harness = PaperTradingHarness(
            symbol=sym,
            entry_threshold=args.entry_threshold,
            min_reliability=args.min_reliability,
            use_multi_tp=not args.no_multi_tp,
            use_quality_registry=not args.no_quality_registry,
            quality_registry_path=args.quality_registry_path,
            use_ir_weights=args.ir_weights,
        )

        if args.days > 0:
            harness.run_backfill(days=args.days)
        else:
            try:
                signal = harness.run()
                print(
                    f"\n{sym}: {signal.signal} | strength={signal.signal_strength:+.3f} | "
                    f"prob={signal.probability:.3f} | price={signal.price:.2f} | "
                    f"patterns={len(signal.active_patterns)}"
                )
            except Exception as exc:
                logger.error(f"{sym} failed: {exc}")


if __name__ == "__main__":
    main()
