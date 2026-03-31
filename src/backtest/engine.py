"""
Backtest Engine

Runs historical backtests on pattern detection strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..patterns.base import BasePattern, SignalDirection
from ..signals.position_manager import PositionManager, PositionStatus
from ..signals.signal_generator import SignalGenerator
from .metrics import PerformanceMetrics


@dataclass
class BacktestConfig:
    """
    Backtest configuration settings.

    Attributes:
        initial_equity: Starting equity amount
        commission_per_trade: Commission per trade (fixed amount)
        commission_pct: Commission as percentage of trade value
        slippage_pct: Slippage percentage
        position_sizing_method: Method for position sizing
        risk_per_trade: Risk per trade as fraction of equity
        max_open_positions: Maximum concurrent positions
        use_take_profit_1: Use first take profit level
        use_take_profit_2: Use second take profit level
        use_take_profit_3: Use third take profit level
        min_confidence: Minimum signal confidence to trade
    """

    initial_equity: float = 100000.0
    commission_per_trade: float = 0.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    position_sizing_method: str = "fixed_fractional"
    risk_per_trade: float = 0.02  # 2%
    max_open_positions: int = 5
    use_take_profit_1: bool = True
    use_take_profit_2: bool = False
    use_take_profit_3: bool = False
    min_confidence: float = 0.5


@dataclass
class BacktestResult:
    """
    Result of a backtest run.

    Attributes:
        trades: DataFrame of all trades
        equity_curve: DataFrame with equity over time
        metrics: Performance metrics
        config: Backtest configuration used
        signals: List of all signals generated
    """

    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[BacktestConfig] = None
    signals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "trades": self.trades.to_dict("records") if len(self.trades) > 0 else [],
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else None,
            "total_signals": len(self.signals),
        }


class BacktestEngine:
    """
    Backtest Engine

    Runs historical backtests on pattern detection strategies.
    """

    def __init__(self, patterns: List[BasePattern], config: Optional[BacktestConfig] = None):
        """
        Initialize Backtest Engine.

        Args:
            patterns: List of pattern detectors to use
            config: Backtest configuration
        """
        self.patterns = patterns
        self.config = config or BacktestConfig()

        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            patterns=patterns,
            min_confidence=self.config.min_confidence,
            max_signals_per_bar=1,
            combine_same_direction=True,
            conflict_resolution="highest_confidence",
        )

        # Initialize position manager
        from ..signals.position_manager import PositionSizer

        position_sizer = PositionSizer(
            method=self.config.position_sizing_method,
            risk_per_trade=self.config.risk_per_trade,
            max_position_size=0.20,
        )

        self.position_manager = PositionManager(
            initial_equity=self.config.initial_equity,
            position_sizer=position_sizer,
            max_open_positions=self.config.max_open_positions,
        )

        # Results storage
        self.result = BacktestResult(config=self.config)

    def run(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            progress_callback: Callback function for progress updates

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Reset position manager
        self.position_manager = PositionManager(initial_equity=self.config.initial_equity)

        # Calculate min bars required
        min_bars = max(p.min_bars_required for p in self.patterns)

        # Storage for results
        equity_curve = []
        all_signals = []

        # Iterate through data
        total_bars = len(df)

        for i in range(min_bars, total_bars):
            current_time = df.index[i]
            current_close = float(df.iloc[i]["Close"])
            current_high = float(df.iloc[i]["High"])
            current_low = float(df.iloc[i]["Low"])

            # Check exits for open positions
            self._check_exits(i, df)

            # Generate signals
            signals = self.signal_generator.generate_signals(df, i)

            # Process signals
            for signal in signals:
                # Open new position
                position = self.position_manager.open_position(
                    signal=signal, timestamp=current_time
                )

                if position:
                    signal_dict = signal.to_dict()
                    signal_dict["position_id"] = position.id
                    all_signals.append(signal_dict)

            # Record equity
            equity = self._calculate_equity(current_close)
            equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": equity,
                    "open_positions": len(self.position_manager.open_positions),
                }
            )

            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_bars)

        # Close any remaining open positions
        self._close_all_positions(df)

        # Compile results
        self.result.trades = self._compile_trades()
        self.result.signals = all_signals
        self.result.equity_curve = pd.DataFrame(equity_curve)
        self.result.metrics = PerformanceMetrics.calculate(
            trades=self.result.trades,
            equity_curve=self.result.equity_curve,
            initial_equity=self.config.initial_equity,
        )

        return self.result

    def _check_exits(self, i: int, df: pd.DataFrame):
        """Check and execute exits for open positions."""
        current_close = float(df.iloc[i]["Close"])
        current_high = float(df.iloc[i]["High"])
        current_low = float(df.iloc[i]["Low"])
        current_time = df.index[i]

        for position in self.position_manager.get_open_positions():
            should_exit, exit_reason, exit_price = self.position_manager.check_exit(
                position=position,
                current_price=current_close,
                current_high=current_high,
                current_low=current_low,
                timestamp=current_time,
            )

            if should_exit:
                # Apply slippage
                if position.direction == SignalDirection.LONG:
                    exit_price = exit_price * (1 - self.config.slippage_pct)
                else:
                    exit_price = exit_price * (1 + self.config.slippage_pct)

                self.position_manager.close_position(
                    position_id=position.id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    timestamp=current_time,
                )

    def _close_all_positions(self, df: pd.DataFrame):
        """Close all remaining open positions at last price."""
        last_close = float(df.iloc[-1]["Close"])
        last_time = df.index[-1]

        for position in self.position_manager.get_open_positions():
            self.position_manager.close_position(
                position_id=position.id,
                exit_price=last_close,
                exit_reason="End of Backtest",
                timestamp=last_time,
            )

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open positions."""
        equity = self.position_manager.equity

        for position in self.position_manager.get_open_positions():
            if position.direction == SignalDirection.LONG:
                unrealized = (current_price - position.entry_price) * position.size
            else:
                unrealized = (position.entry_price - current_price) * position.size
            equity += unrealized

        return equity

    def _compile_trades(self) -> pd.DataFrame:
        """Compile trades as DataFrame."""
        trades = []

        for position in self.position_manager.positions.values():
            if position.status == PositionStatus.CLOSED:
                trade = {
                    "id": position.id,
                    "pattern": position.pattern_name,
                    "direction": position.direction.value,
                    "entry_time": str(position.entry_time) if position.entry_time else None,
                    "entry_price": position.entry_price,
                    "exit_time": str(position.exit_time) if position.exit_time else None,
                    "exit_price": position.exit_price,
                    "size": position.size,
                    "pnl": position.pnl,
                    "pnl_pct": position.pnl_pct,
                    "exit_reason": position.metadata.get("exit_reason", "Unknown"),
                    "stop_loss": position.stop_loss,
                    "take_profit_1": position.take_profit_1,
                    "take_profit_2": position.take_profit_2,
                    "take_profit_3": position.take_profit_3,
                }
                trades.append(trade)

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        in_sample_periods: int = 252,
        out_sample_periods: int = 63,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[BacktestResult]:
        """
        Run walk-forward optimization.

        Args:
            df: DataFrame with OHLCV data
            in_sample_periods: Number of periods for in-sample (training)
            out_sample_periods: Number of periods for out-of-sample (testing)
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            List of BacktestResult for each out-of-sample period
        """
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        results = []
        total_periods = len(df)

        # Calculate walk-forward windows
        start_idx = 0
        while start_idx + in_sample_periods + out_sample_periods <= total_periods:
            in_sample_end = start_idx + in_sample_periods
            out_sample_end = in_sample_end + out_sample_periods

            # Run backtest on out-of-sample period
            out_sample_df = df.iloc[in_sample_end:out_sample_end]
            result = self.run(out_sample_df)
            results.append(result)

            # Move to next window
            start_idx += out_sample_periods

        return results

    def get_pattern_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics broken down by pattern.

        Returns:
            Dictionary with performance metrics per pattern
        """
        pattern_stats = {}

        trades_df = self.result.trades
        if trades_df is None or len(trades_df) == 0:
            return pattern_stats

        for _, trade in trades_df.iterrows():
            pattern = trade["pattern"]
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "pnl_list": [],
                }

            pattern_stats[pattern]["trades"] += 1
            pnl_val = trade["pnl"] if pd.notna(trade["pnl"]) else 0
            pattern_stats[pattern]["total_pnl"] += pnl_val
            pattern_stats[pattern]["pnl_list"].append(pnl_val)

            if pnl_val > 0:
                pattern_stats[pattern]["wins"] += 1
            else:
                pattern_stats[pattern]["losses"] += 1

        # Calculate metrics for each pattern
        for pattern, stats in pattern_stats.items():
            if stats["trades"] > 0:
                stats["win_rate"] = stats["wins"] / stats["trades"]
                stats["avg_pnl"] = stats["total_pnl"] / stats["trades"]

                wins = [p for p in stats["pnl_list"] if p > 0]
                losses = [abs(p) for p in stats["pnl_list"] if p < 0]

                stats["avg_win"] = sum(wins) / len(wins) if wins else 0
                stats["avg_loss"] = sum(losses) / len(losses) if losses else 0

                total_wins = sum(wins)
                total_losses = sum(losses)
                stats["profit_factor"] = (
                    total_wins / total_losses if total_losses > 0 else float("inf")
                )

        return pattern_stats

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get backtest results formatted for visualization.

        Returns:
            Dictionary with equity curve, trades, signals, and metrics
            ready for use with quantstats and mplfinance visualization.
        """
        return {
            "equity_curve": self.result.equity_curve,
            "trades": self.result.trades,
            "signals": self.result.signals,
            "metrics": self.result.metrics,
            "returns": self.result.equity_curve["equity"].pct_change().dropna()
            if self.result.equity_curve is not None
            else None,
            "pattern_performance": self.get_pattern_performance(),
        }

    def export_for_visualization(
        self, output_dir: str = "reports", title: str = "Backtest Results"
    ) -> Dict[str, str]:
        """
        Export backtest results for visualization.

        Args:
            output_dir: Directory to save exported files
            title: Title for the report

        Returns:
            Dictionary with paths to exported files
        """
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {}
        base_name = title.lower().replace(" ", "_")

        # Export equity curve to CSV
        if self.result.equity_curve is not None:
            equity_path = output_path / f"{base_name}_equity.csv"
            self.result.equity_curve.to_csv(equity_path, index=False)
            files["equity_curve"] = str(equity_path)

        # Export trades to CSV
        if self.result.trades is not None and len(self.result.trades) > 0:
            trades_path = output_path / f"{base_name}_trades.csv"
            self.result.trades.to_csv(trades_path, index=False)
            files["trades"] = str(trades_path)

        # Export signals to CSV
        if self.result.signals:
            signals_path = output_path / f"{base_name}_signals.csv"
            pd.DataFrame(self.result.signals).to_csv(signals_path, index=False)
            files["signals"] = str(signals_path)

        # Export metrics to JSON
        if self.result.metrics:
            metrics_path = output_path / f"{base_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.result.metrics, f, indent=2, default=str)
            files["metrics"] = str(metrics_path)

        return files
