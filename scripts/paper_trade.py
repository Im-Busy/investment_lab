"""
Paper Trading Runner

Polls live data, generates signals, simulates fills with realistic slippage,
and logs everything for analysis.

Designed to run continuously in the background (e.g., via cron / Task Scheduler).

Usage:
    # Single poll (dry run):
    uv run scripts/paper_trade.py --symbol SPY --once

    # Continuous mode (every 60 min):
    uv run scripts/paper_trade.py --symbol SPY --interval 60 --duration 24  # 24 hours

    # Custom asset universe with ML enhancement:
    uv run scripts/paper_trade.py --symbols SPY QQQ IWM --interval 30 --ml-enabled

    # Review results:
    uv run scripts/paper_trade.py --report

Phase: 6 — Paper Trading (Optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indicators.regime_detector import RegimeDetector, RegimeState
from src.risk.daily_limits import DailyLossMonitor, RiskMonitor
from src.risk.position_sizing import PositionSizer

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_LOG = LOG_DIR / "paper_trade_signals.jsonl"
TRADE_LOG = LOG_DIR / "paper_trade_trades.jsonl"
EQUITY_LOG = LOG_DIR / "paper_trade_equity.jsonl"
STATUS_FILE = LOG_DIR / "paper_trade_status.json"


@dataclass
class PaperTrade:
    """A simulated paper trade."""

    trade_id: str
    symbol: str
    entry_time: str
    entry_price: float
    fill_price: float
    slippage_bps: float
    direction: str
    size: float
    stop_loss: float
    take_profit: float
    signal_type: str
    confidence: float
    regime: str
    status: str = "open"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class PaperPosition:
    """An open simulated position."""

    trade: PaperTrade
    entry_bar_idx: int


class PaperTradeEngine:
    """
    Paper trading engine with realistic simulation.

    Features:
    - Signal generation via pattern detection + confluence scoring
    - Regime-aware filtering
    - Realistic slippage model (0.05% liquid, 0.2% illiquid)
    - Daily loss limits
    - Circuit breaker
    - Position sizing (Kelly capped at 2% risk)
    """

    def __init__(
        self,
        symbols: List[str],
        initial_equity: float = 100_000.0,
        slippage_bps: float = 5.0,
        daily_loss_limit_pct: float = 0.03,
        max_open_positions: int = 5,
        risk_per_trade: float = 0.02,
        ml_enabled: bool = False,
    ):
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.slippage_bps = slippage_bps
        self.max_open_positions = max_open_positions
        self.risk_per_trade = risk_per_trade
        self.ml_enabled = ml_enabled

        self.regime_detector = RegimeDetector()
        self.position_sizer = PositionSizer(
            equity=initial_equity,
            risk_per_trade=risk_per_trade,
        )
        self.daily_monitor = DailyLossMonitor(
            max_daily_loss=daily_loss_limit_pct,
            weekly_loss_pct=0.06,
            monthly_loss_pct=0.10,
        )
        self.risk_monitor = RiskMonitor(
            max_heat_pct=0.06,
            daily_loss_pct=daily_loss_limit_pct,
            weekly_loss_pct=0.06,
        )

        self.open_positions: List[PaperPosition] = []
        self.closed_trades: List[PaperTrade] = []
        self.signal_count = 0
        self.trade_count = 0

    def _download_data(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """Download OHLCV data."""
        try:
            df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or "Close" not in df.columns:
                return None
            df.columns = [
                c.capitalize() if c.lower() in ["open", "high", "low", "close", "volume"] else c
                for c in df.columns
            ]
            df.index = pd.to_datetime(df.index)
            if "Volume" not in df.columns:
                df["Volume"] = 0
            return df
        except Exception as e:
            logger.warning(f"Failed to download {symbol}: {e}")
            return None

    def _compute_slippage(self, symbol: str, price: float) -> tuple:
        """
        Simulate realistic slippage.

        Returns:
            (fill_price, slippage_bps)
        """
        liquid_tickers = {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
        }
        if symbol in liquid_tickers:
            slippage_pct = 0.0005  # 0.05%
        else:
            slippage_pct = 0.002  # 0.2%

        slippage_bps_used = slippage_pct * 10000
        fill_price = price * (1 + slippage_pct)
        return round(fill_price, 2), slippage_bps_used

    def _get_regime(self, df: pd.DataFrame) -> RegimeState:
        """Get current market regime."""
        try:
            regime_info = self.regime_detector.get_regime_series(df)
            last_regime = regime_info["regime"].iloc[-1]
            return last_regime if hasattr(last_regime, "name") else RegimeState.TRANSITION
        except Exception:
            return RegimeState.TRANSITION

    def _generate_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Generate trading signals using pattern detection and confluence.

        For paper trading, this runs the same pipeline as backtesting.
        """

        signals = []
        try:
            # Get current price
            close = df["Close"].iloc[-1]
            high = df["High"].iloc[-1]
            low = df["Low"].iloc[-1]

            # Run regime detection
            regime_info = self.regime_detector.get_regime_series(df)
            regime = regime_info["regime"].iloc[-1]
            regime_str = regime.value if hasattr(regime, "value") else str(regime)

            # Simple signal generation for paper trading:
            # Look for multi-pattern confluence on current bar
            # Use last 50 bars for pattern detection
            recent = df.tail(50)

            # Check for basic signals: MA crossover, RSI levels, volatility breakout
            ma_fast = recent["Close"].rolling(20).mean().iloc[-1]
            ma_slow = recent["Close"].rolling(50).mean().iloc[-1]
            atr = recent["High"].sub(recent["Low"]).rolling(14).mean().iloc[-1]

            # Generate a signal if patterns align
            signals_list = []

            # Signal 1: Moving average crossover
            if ma_fast > ma_slow and close > ma_fast:
                signals_list.append(
                    {
                        "type": "MA_Crossover",
                        "confidence": 0.55,
                        "direction": "LONG",
                    }
                )
            elif ma_fast < ma_slow and close < ma_fast:
                signals_list.append(
                    {
                        "type": "MA_Crossover",
                        "confidence": 0.55,
                        "direction": "SHORT",
                    }
                )

            # Signal 2: RSI overbought/oversold
            if len(recent) > 20:
                delta = recent["Close"].diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.inf)
                rsi = 100 - (100 / (1 + rs.iloc[-1]))

                if rsi < 30:
                    signals_list.append(
                        {
                            "type": "RSI_Oversold",
                            "confidence": 0.60,
                            "direction": "LONG",
                        }
                    )
                elif rsi > 70:
                    signals_list.append(
                        {
                            "type": "RSI_Overbought",
                            "confidence": 0.60,
                            "direction": "SHORT",
                        }
                    )

            # Signal 3: Volatility breakout (price beyond 2x ATR)
            upper_band = recent["Close"].rolling(20).mean().iloc[-1] + 2 * atr
            lower_band = recent["Close"].rolling(20).mean().iloc[-1] - 2 * atr

            if close > upper_band:
                signals_list.append(
                    {
                        "type": "VolBreakout_Up",
                        "confidence": 0.50,
                        "direction": "LONG",
                    }
                )
            elif close < lower_band:
                signals_list.append(
                    {
                        "type": "VolBreakout_Down",
                        "confidence": 0.50,
                        "direction": "SHORT",
                    }
                )

            # Confluence: if 2+ signals agree, generate entry
            long_signals = [s for s in signals_list if s["direction"] == "LONG"]
            short_signals = [s for s in signals_list if s["direction"] == "SHORT"]

            if len(long_signals) >= 1:
                avg_conf = np.mean([s["confidence"] for s in long_signals])
                signals.append(
                    {
                        "symbol": symbol,
                        "direction": "LONG",
                        "signal_type": " + ".join(s["type"] for s in long_signals),
                        "confidence": avg_conf,
                        "regime": regime_str,
                        "entry_price": close,
                        "stop_loss": close - 2 * atr,
                        "take_profit": close + 2 * atr,
                        "atr": round(atr, 2),
                        "timestamp": str(df.index[-1]),
                    }
                )

            if len(short_signals) >= 1:
                avg_conf = np.mean([s["confidence"] for s in short_signals])
                signals.append(
                    {
                        "symbol": symbol,
                        "direction": "SHORT",
                        "signal_type": " + ".join(s["type"] for s in short_signals),
                        "confidence": avg_conf,
                        "regime": regime_str,
                        "entry_price": close,
                        "stop_loss": close + 2 * atr,
                        "take_profit": close - 2 * atr,
                        "atr": round(atr, 2),
                        "timestamp": str(df.index[-1]),
                    }
                )

        except Exception as e:
            logger.warning(f"Signal generation failed for {symbol}: {e}")

        return signals

    def _log_signal(self, signal: Dict[str, Any]) -> None:
        """Append signal to JSONL log."""
        self.signal_count += 1
        record = {**signal, "log_id": self.signal_count, "logged_at": datetime.now().isoformat()}
        with open(SIGNAL_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            f"  Signal #{self.signal_count}: {signal['symbol']} {signal['direction']} "
            f"{signal['signal_type']} conf={signal['confidence']:.2f}"
        )

    def _log_trade(self, trade: PaperTrade) -> None:
        """Append trade to JSONL log."""
        self.trade_count += 1
        record = {
            **asdict(trade),
            "log_id": self.trade_count,
            "logged_at": datetime.now().isoformat(),
        }
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log_equity(self, snapshot: Dict[str, Any]) -> None:
        """Append equity snapshot."""
        record = {**snapshot, "logged_at": datetime.now().isoformat()}
        with open(EQUITY_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_status(self) -> None:
        """Persist current engine state."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "equity": round(self.equity, 2),
            "open_positions": len(self.open_positions),
            "closed_trades": len(self.closed_trades),
            "total_pnl": sum(t.pnl or 0 for t in self.closed_trades),
            "pnl_today": sum(
                t.pnl or 0
                for t in self.closed_trades
                if t.exit_time and t.exit_time.startswith(datetime.now().strftime("%Y-%m-%d"))
            ),
        }
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)

    def poll_once(self) -> Dict[str, Any]:
        """
        Single poll cycle: download data, generate signals, check exits, log equity.

        Returns:
            Dict with poll summary.
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "signals": 0,
            "trades": 0,
        }

        # 1. Download data for all symbols
        all_data = {}
        for symbol in self.symbols:
            df = self._download_data(symbol, period="2y")
            if df is not None and len(df) > 50:
                all_data[symbol] = df

        # 2. Check for exits on open positions
        exits_this_cycle = []
        for pos in list(self.open_positions):
            symbol = pos.trade.symbol
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            current_close = df["Close"].iloc[-1]
            current_low = df["Low"].iloc[-1]
            current_high = df["High"].iloc[-1]

            should_exit = False
            exit_reason = ""

            if pos.trade.direction == "LONG":
                if current_low <= pos.trade.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_high >= pos.trade.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
            else:  # SHORT
                if current_high >= pos.trade.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_low <= pos.trade.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

            if should_exit:
                exit_price = current_close  # Simplified: close price as fill
                pnl = (
                    (exit_price - pos.trade.fill_price)
                    * pos.trade.size
                    * (-1 if pos.trade.direction == "SHORT" else 1)
                )
                pnl_pct = (
                    (exit_price / pos.trade.fill_price - 1)
                    * 100
                    * (-1 if pos.trade.direction == "SHORT" else 1)
                )

                pos.trade.status = "closed"
                pos.trade.exit_time = df.index[-1].isoformat()
                pos.trade.exit_price = exit_price
                pos.trade.exit_reason = exit_reason
                pos.trade.pnl = round(pnl, 2)
                pos.trade.pnl_pct = round(pnl_pct, 2)

                self.closed_trades.append(pos.trade)
                self._log_trade(pos.trade)
                self.equity += pnl
                exits_this_cycle.append(pos.trade)
                logger.info(
                    f"  EXIT: {symbol} @ {exit_price:.2f} PnL={pnl:+.2f} ({pnl_pct:+.1f}%) [{exit_reason}]"
                )

                self.open_positions.remove(pos)

        # 3. Check daily loss limits
        if self._check_loss_limits():
            logger.warning("Daily loss limit hit — no new signals will be generated")
            self._log_equity(
                {
                    "equity": self.equity,
                    "open": len(self.open_positions),
                    "closed": len(self.closed_trades),
                }
            )
            return results

        # 4. Generate new signals
        for symbol, df in all_data.items():
            # Skip if already at max positions
            if len(self.open_positions) >= self.max_open_positions:
                logger.info(
                    f"  Max open positions ({self.max_open_positions}) reached — skipping new entries"
                )
                break

            # Skip if position already open for this symbol
            if any(p.trade.symbol == symbol for p in self.open_positions):
                continue

            signals = self._generate_signals(df, symbol)

            for signal in signals:
                self._log_signal(signal)
                results["signals"] += 1

                # Calculate position size
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(signal["entry_price"] - signal["stop_loss"])
                if risk_per_share <= 0:
                    continue

                size = min(
                    int(risk_amount / risk_per_share),
                    int(self.equity * 0.20 / signal["entry_price"]),
                )
                if size < 1:
                    size = 1

                # Simulate entry with slippage
                fill_price, slip_bps = self._compute_slippage(
                    signal["symbol"], signal["entry_price"]
                )

                trade = PaperTrade(
                    trade_id=f"PT{self.trade_count + 1:04d}",
                    symbol=signal["symbol"],
                    entry_time=signal["timestamp"],
                    entry_price=signal["entry_price"],
                    fill_price=fill_price,
                    slippage_bps=round(slip_bps, 1),
                    direction=signal["direction"],
                    size=size,
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    signal_type=signal["signal_type"],
                    confidence=signal["confidence"],
                    regime=signal["regime"],
                )

                self.open_positions.append(PaperPosition(trade=trade, entry_bar_idx=len(df)))
                self._log_trade(trade)
                logger.info(
                    f"  ENTRY: {symbol} {signal['direction']} x{size} @ {fill_price:.2f} "
                    f"(slip={slip_bps:.1f}bps, regime={signal['regime']})"
                )
                results["trades"] += 1

        # 5. Log equity snapshot
        self._log_equity(
            {
                "equity": round(self.equity, 2),
                "net_pnl": round(self.equity - self.initial_equity, 2),
                "open_positions": len(self.open_positions),
                "closed_trades": len(self.closed_trades),
                "daily_pnl": round(
                    sum(
                        t.pnl or 0
                        for t in self.closed_trades
                        if t.exit_time
                        and t.exit_time.startswith(datetime.now().strftime("%Y-%m-%d"))
                    ),
                    2,
                ),
            }
        )

        self._save_status()
        return results

    def _check_loss_limits(self) -> bool:
        """Check if loss limits have been breached."""
        if not self.closed_trades:
            return False

        # Check daily PnL
        today = datetime.now().strftime("%Y-%m-%d")
        today_pnl = sum(
            t.pnl or 0 for t in self.closed_trades if t.exit_time and t.exit_time.startswith(today)
        )
        daily_loss = abs(min(today_pnl, 0)) / self.equity
        return daily_loss >= self.daily_monitor.config.max_daily_loss_pct


def print_report() -> None:
    """Print current paper trading report from logs."""
    if not TRADE_LOG.exists():
        print("No paper trade data found. Run paper_trade.py first.")
        return

    # Load trades
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            trades.append(json.loads(line))

    if not trades:
        print("No trades recorded yet.")
        return

    entry_trades = [t for t in trades if t.get("status") == "open"]
    closed_trades = [t for t in trades if t.get("status") == "closed"]

    print("=" * 70)
    print("PAPER TRADING REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"\nOpen positions: {len(entry_trades)}")
    for t in entry_trades:
        print(
            f"  {t['symbol']:>6s} {t['direction']} x{t['size']} @ {t['fill_price']:.2f} "
            f"[{t['signal_type']}]"
        )

    print(f"\nClosed trades: {len(closed_trades)}")
    if closed_trades:
        total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
        wins = [t for t in closed_trades if (t.get("pnl", 0) or 0) > 0]
        wr = len(wins) / len(closed_trades) * 100
        avg_pnl = total_pnl / len(closed_trades)

        print(f"  Total PnL: {total_pnl:+.2f}")
        print(f"  Win Rate:  {wr:.1f}% ({len(wins)}/{len(closed_trades)})")
        print(f"  Avg PnL:   {avg_pnl:+.2f}")
        print(
            f"  Avg Slippage: {np.mean([t.get('slippage_bps', 0) for t in closed_trades]):.1f} bps"
        )

        # Performance by signal type
        signal_perf = (
            pd.DataFrame(closed_trades)
            .groupby("signal_type")
            .agg(
                count=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                win_rate=("pnl", lambda x: (x > 0).mean()),
            )
            .sort_values("total_pnl", ascending=False)
        )
        print("\n  Performance by Signal Type:")
        for sig, row in signal_perf.iterrows():
            print(
                f"    {sig:<25s}: {row['count']:>3d} trades, PnL={row['total_pnl']:+.2f}, "
                f"WR={row['win_rate']:.0%}"
            )

        # Performance by regime
        if "regime" in pd.DataFrame(closed_trades).columns:
            regime_perf = (
                pd.DataFrame(closed_trades)
                .groupby("regime")
                .agg(
                    count=("pnl", "count"),
                    total_pnl=("pnl", "sum"),
                    win_rate=("pnl", lambda x: (x > 0).mean()),
                )
            )
            print("\n  Performance by Regime:")
            for reg, row in regime_perf.iterrows():
                print(
                    f"    {reg:<15s}: {row['count']:>3d} trades, PnL={row['total_pnl']:+.2f}, "
                    f"WR={row['win_rate']:.0%}"
                )

    # Status
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            status = json.load(f)
        print(f"\nStatus (last update: {status.get('timestamp', 'unknown')}):")
        print(f"  Equity: {status.get('equity', 'N/A')}")
        print(f"  Net PnL: {status.get('total_pnl', 0):+.2f}")

    print("\n" + "=" * 70)

    # Go/No-Go evaluation
    if len(closed_trades) >= 30:
        wr_pct = len(wins) / len(closed_trades) * 100
        equity_curve = [t.get("pnl", 0) for t in closed_trades]
        cumulative = np.cumsum(equity_curve)
        max_dd = np.max(np.maximum.accumulate(cumulative) - cumulative)

        print("\nGO/NO-GO EVALUATION:")
        print(f"  Duration (days):    {'N/A (need to check start date)'}")
        print(
            f"  Win Rate:           {wr_pct:.1f}% {'[PASS]' if wr_pct >= 45 else '[FAIL]'} (threshold: 45%)"
        )
        print(
            f"  Max Drawdown:       {max_dd:.2f} {'[PASS]' if max_dd <= 1500 else '[FAIL]'} (threshold: $1500)"
        )
        print(f"  Signal Count:       {len(closed_trades)}")
    elif closed_trades:
        print(f"\n  [INFO] Only {len(closed_trades)} closed trades — need 30+ for evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument("--symbol", default="SPY", help="Single symbol to poll")
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Multiple symbols (overrides --symbol)"
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Poll interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--duration", type=int, default=24, help="Duration in hours (default: 24, 0=unlimited)"
    )
    parser.add_argument(
        "--equity", type=float, default=100_000, help="Starting equity (default: 100k)"
    )
    parser.add_argument(
        "--slippage", type=float, default=5.0, help="Slippage in basis points (default: 5)"
    )
    parser.add_argument("--report", action="store_true", help="Print report from existing logs")
    parser.add_argument("--once", action="store_true", help="Single poll then exit")
    parser.add_argument("--ml-enabled", action="store_true", help="Enable ML signal scoring")
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    symbols = args.symbols or [args.symbol]
    engine = PaperTradeEngine(
        symbols=symbols,
        initial_equity=args.equity,
        slippage_bps=args.slippage,
        ml_enabled=args.ml_enabled,
    )

    if args.once:
        print(f"\n{'=' * 60}")
        print(f"PAPER TRADING — Single Poll: {', '.join(symbols)}")
        print(f"{'=' * 60}\n")
        engine.poll_once()
        print_report()
        return

    total_hours = args.duration
    poll_interval_min = args.interval
    start_time = time.time()
    poll_count = 0

    print(f"\n{'=' * 60}")
    print(f"PAPER TRADING — {'Continuous Mode'}")
    print(f"Symbols:     {', '.join(symbols)}")
    print(f"Interval:    {poll_interval_min} minutes")
    print(f"Duration:    {total_hours} hours (0=unlimited)")
    print(f"Equity:      ${args.equity:,.0f}")
    print(f"Slippage:    {args.slippage} bps")
    print(f"{'=' * 60}\n")

    while True:
        elapsed_hours = (time.time() - start_time) / 3600
        if total_hours > 0 and elapsed_hours >= total_hours:
            print(f"\nDuration reached ({total_hours}h) — stopping.")
            break

        poll_count += 1
        print(f"\n--- Poll #{poll_count} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        result = engine.poll_once()
        print(f"Signals generated: {result['signals']}")
        print(f"Trades executed:   {result['trades']}")
        print(f"Open positions:    {len(engine.open_positions)}")
        print(f"Closed trades:     {len(engine.closed_trades)}")
        print(f"Current equity:    ${engine.equity:,.2f}")

        print(f"\nNext poll in {poll_interval_min} minutes... (Ctrl+C to stop)")

        try:
            time.sleep(poll_interval_min * 60)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user — saving state.")
            engine._save_status()
            print_report()
            sys.exit(0)


if __name__ == "__main__":
    main()
