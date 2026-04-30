"""
Backtest Engine

Runs historical backtests on pattern detection strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict

import pandas as pd

from ..patterns.base import BasePattern, SignalDirection
from ..signals.position_manager import PositionManager, PositionStatus
from ..signals.signal_generator import SignalGenerator, AggregatedSignal
from .metrics import PerformanceMetrics
from ..risk.turnover_penalty import TurnoverPenalty, TurnoverPenaltyConfig
from ..risk.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from ..risk.position_probability import PositionRiskModel, PositionRiskConfig, RegimeState
from ..risk.dynamic_rebalancing import DynamicRebalancer, DynamicRebalanceConfig
from ..indicators.regime_detector import RegimeDetector, RegimeState as RuleRegimeState
from .friction_scoring import FrictionScorer, FrictionConfig


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

        # Phase 6 Tier 1: Risk Management Components
        enable_circuit_breaker: Enable portfolio drawdown circuit breaker
        max_drawdown_pct: Maximum portfolio drawdown before halting (circuit breaker)
        circuit_breaker_cooldown: Number of bars to stay in cooldown

        enable_turnover_penalty: Enable turnover penalty tracking
        max_allowed_turnover_pct: Maximum annualized turnover percentage

        enable_position_probability: Enable per-position risk estimation
        base_success_prob: Base probability of successful trade (from historical win rate)

        enable_dynamic_re: Enable dynamic rebalancing frequency
        min_signal_decay_rate: Minimum signal decay rate to consider daily rebalancing
        tx_cost_pct: Transaction cost as % of portfolio value

        enable_regime_detection: Enable rule-based regime detection for position probability
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

    enable_circuit_breaker: bool = True
    max_drawdown_pct: float = 20.0
    circuit_breaker_cooldown: int = 20

    enable_turnover_penalty: bool = True
    max_allowed_turnover_pct: float = 2000.0

    enable_position_probability: bool = True
    base_success_prob: float = 0.55

    enable_dynamic_re: bool = True
    min_signal_decay_rate: float = 0.02
    tx_cost_pct: float = 0.001

    enable_regime_detection: bool = True

    enable_friction_scoring: bool = True
    friction_spread_bps: float = 5.0
    friction_slippage_bps: float = 2.0


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
            use_take_profit_1=self.config.use_take_profit_1,
            use_take_profit_2=self.config.use_take_profit_2,
            use_take_profit_3=self.config.use_take_profit_3,
        )

        # Phase 6 Tier 1: Initialize risk management components

        # R1: Turnover Penalty
        if self.config.enable_turnover_penalty:
            self.turnover_penalty = TurnoverPenalty(
                config=TurnoverPenaltyConfig(
                    max_allowed_turnover_pct=self.config.max_allowed_turnover_pct
                )
            )
        else:
            self.turnover_penalty = None

        # R3: Circuit Breaker
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                config=CircuitBreakerConfig(
                    max_drawdown_pct=self.config.max_drawdown_pct,
                    cooldown_bars=self.config.circuit_breaker_cooldown,
                )
            )
        else:
            self.circuit_breaker = None

        # R2: Position Probability
        if self.config.enable_position_probability:
            self.position_risk_model = PositionRiskModel(
                config=PositionRiskConfig(base_success_prob=self.config.base_success_prob)
            )
        else:
            self.position_risk_model = None

        # R5: Dynamic Rebalancing
        if self.config.enable_dynamic_re:
            self.dynamic_rebalancer = DynamicRebalancer(
                config=DynamicRebalanceConfig(
                    min_signal_decay_rate=self.config.min_signal_decay_rate,
                    tx_cost_pct=self.config.tx_cost_pct,
                )
            )
        else:
            self.dynamic_rebalancer = None

        # R4: Regime Detection (for position probability)
        if self.config.enable_regime_detection:
            self.regime_detector = RegimeDetector()
        else:
            self.regime_detector = None

        # R10: Friction Scoring
        if self.config.enable_friction_scoring:
            self.friction_scorer = FrictionScorer(
                config=FrictionConfig(
                    spread_bps=self.config.friction_spread_bps,
                    slippage_bps=self.config.friction_slippage_bps,
                    commission_bps=self.config.commission_pct * 100,
                )
            )
        else:
            self.friction_scorer = None

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

        # Reset position manager and risk components
        self.position_manager = PositionManager(initial_equity=self.config.initial_equity)

        if self.circuit_breaker:
            self.circuit_breaker.reset()
        if self.dynamic_rebalancer:
            self.dynamic_rebalancer.reset()

        # Calculate min bars required
        min_bars = max(p.min_bars_required for p in self.patterns)

        # Storage for results and tracking
        equity_curve = []
        all_signals = []
        peak_equity = self.config.initial_equity
        current_regime = RegimeState.TRANSITION

        # R4: Pre-calculate regimes if enabled
        if self.regime_detector:
            regime_series = self.regime_detector.classify(df)
        else:
            regime_series = None

        # Iterate through data
        total_bars = len(df)

        for i in range(min_bars, total_bars):
            current_time = df.index[i]
            current_close = float(df.iloc[i]["Close"])
            current_high = float(df.iloc[i]["High"])
            current_low = float(df.iloc[i]["Low"])

            # Update peak equity for circuit breaker
            current_equity = self._calculate_equity(current_close)
            if current_equity > peak_equity:
                peak_equity = current_equity

            # R3: Check circuit breaker before any actions
            trading_allowed = True
            if self.circuit_breaker:
                halt, msg, dd = self.circuit_breaker.check_circuit(
                    current_price=current_equity, peak_price=peak_equity
                )
                if halt:
                    trading_allowed = False

            # Check exits for open positions (always allowed for risk management)
            self._check_exits(i, df, current_high=current_high, current_low=current_low)

            # Update regime if enabled
            if regime_series is not None:
                current_regime_val = regime_series.iloc[i]
                current_regime = RegimeState(current_regime_val.value)

            # R5: Check dynamic rebalancing before generating signals
            should_rebalance = True
            rebalance_reason = "Always rebalance"
            if self.dynamic_rebalancer:
                signal_strength = 0.0
                if self.position_manager.open_positions:
                    avg_confidence = 0.0
                    for pos in self.position_manager.get_open_positions():
                        avg_confidence += getattr(pos, "confidence", 0.5)
                    signal_strength = avg_confidence / len(self.position_manager.open_positions)

                should_rebalance, rebalance_reason, _ = self.dynamic_rebalancer.should_rebalance(
                    current_day=i, signal_decay_rate=signal_strength
                )

                # Update signal history for decay tracking
                if should_rebalance:
                    self.dynamic_rebalancer.update_signal_history(signal_strength)

            # Generate signals only if trading allowed
            signals = []
            if trading_allowed and should_rebalance:
                signals = self.signal_generator.generate_signals(df, i)

                # R2: Apply position probability and risk-adjusted sizing
                if self.position_risk_model and signals:
                    signals = self._apply_position_probability(signals, current_regime)

                # Process signals
                for signal in signals:
                    # Check circuit breaker again before opening
                    if self.circuit_breaker:
                        halt, _, _ = self.circuit_breaker.check_circuit(
                            current_price=current_equity, peak_price=peak_equity
                        )
                        if halt:
                            break

                    # Open new position
                    position = self.position_manager.open_position(
                        signal=signal, timestamp=current_time
                    )

                    if position:
                        signal_dict = signal.to_dict()
                        signal_dict["position_id"] = position.id
                        signal_dict["regime"] = current_regime.value
                        all_signals.append(signal_dict)

            # Record equity
            equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": current_equity,
                    "open_positions": len(self.position_manager.open_positions),
                    "trading_allowed": trading_allowed,
                    "rebalance_triggered": should_rebalance,
                    "regime": current_regime.value,
                }
            )

            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_bars)

        # Close any remaining open positions
        self._close_all_positions(df)

        # R1: Apply turnover penalty to results
        if self.turnover_penalty:
            self._apply_turnover_penalty(equity_curve)

        # Compile results
        self.result.trades = self._compile_trades()
        self.result.signals = all_signals
        self.result.equity_curve = pd.DataFrame(equity_curve)
        self.result.metrics = PerformanceMetrics.calculate(
            trades=self.result.trades,
            equity_curve=self.result.equity_curve,
            initial_equity=self.config.initial_equity,
        )

        # Add risk management info to metrics
        if self.result.metrics:
            self.result.metrics["risk_management"] = {
                "circuit_breaker_enabled": self.config.enable_circuit_breaker,
                "turnover_penalty_enabled": self.config.enable_turnover_penalty,
                "position_probability_enabled": self.config.enable_position_probability,
                "dynamic_rebalancing_enabled": self.config.enable_dynamic_re,
                "regime_detection_enabled": self.config.enable_regime_detection,
                "friction_scoring_enabled": self.config.enable_friction_scoring,
            }

        # R10: Apply friction-adjusted metrics
        if self.friction_scorer and len(self.result.trades) > 0:
            friction_metrics = self._calculate_friction_metrics()
            self.result.metrics["friction_adjusted"] = friction_metrics

        return self.result

    def _check_exits(
        self,
        i: int,
        df: pd.DataFrame,
        current_high: Optional[float] = None,
        current_low: Optional[float] = None,
    ):
        """Check and execute exits for open positions."""
        current_close = float(df.iloc[i]["Close"])
        if current_high is None:
            current_high = float(df.iloc[i]["High"])
        if current_low is None:
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

    def _apply_position_probability(
        self, signals: List[AggregatedSignal], current_regime: RegimeState
    ) -> List[AggregatedSignal]:
        """
        Apply position probability estimates and risk-adjusted sizing.

        Args:
            signals: List of signals to adjust
            current_regime: Current market regime

        Returns:
            Filtered list of signals with adjusted confidence
        """
        if not self.position_risk_model:
            return signals

        filtered_signals = []

        for signal in signals:
            # Estimate success probability
            success_prob = self.position_risk_model.estimate_success_prob(
                signal_confidence=signal.confidence,
                regime=current_regime,
                base_prob=self.config.base_success_prob,
            )

            # Skip low-probability signals (less than 40% chance)
            if success_prob < 0.40:
                continue

            # Adjust confidence based on probability
            adjusted_confidence = (signal.confidence + success_prob) / 2.0

            # Add risk metadata
            signal.metadata["success_probability"] = success_prob
            signal.metadata["failure_probability"] = 1.0 - success_prob
            signal.metadata["confidence_adjusted"] = True
            signal.metadata["original_confidence"] = signal.confidence

            signal.confidence = adjusted_confidence
            filtered_signals.append(signal)

        return filtered_signals

    def _apply_turnover_penalty(self, equity_curve: List[Dict[str, Any]]) -> None:
        """
        Apply turnover penalty to backtest results.

        Args:
            equity_curve: List of equity curve entries
        """
        if not self.turnover_penalty:
            return

        # Calculate statistics
        n_trades = len(self.position_manager.positions)
        n_days = len(equity_curve)

        if n_days == 0:
            return

        # Check turnover constraint
        exceeded, turnover_pct, message = self.turnover_penalty.check_constraint(
            n_trades=n_trades, n_days=n_days, portfolio_value=self.config.initial_equity
        )

        # Add penalty info to results
        if not hasattr(self.result, "turnover_info"):
            self.result.turnover_info = {}

        self.result.turnover_info = {
            "n_trades": n_trades,
            "n_days": n_days,
            "annualized_turnover_pct": turnover_pct,
            "constraint_exceeded": exceeded,
            "message": message,
            "penalty_applied": False,
        }

        # Calculate and apply penalty if exceeded
        if exceeded:
            penalty = self.turnover_penalty.calculate_penalty(
                n_trades=n_trades, n_days=n_days, portfolio_value=self.config.initial_equity
            )

            self.result.turnover_info["penalty_factor"] = penalty
            self.result.turnover_info["penalty_applied"] = True

            # Adjust final equity
            if equity_curve:
                final_equity = equity_curve[-1]["equity"]
                adjusted_equity = final_equity * (1.0 - penalty)

                # Log penalty in equity curve
                equity_curve[-1]["turnover_penalty_applied"] = True
                equity_curve[-1]["equity_before_penalty"] = final_equity
                equity_curve[-1]["equity"] = adjusted_equity
                equity_curve[-1]["penalty_factor"] = penalty

    def _calculate_friction_metrics(self) -> Dict[str, Any]:
        """Calculate friction-adjusted performance metrics."""
        trades_df = self.position_manager.positions
        total_trades = self.position_manager.total_trades
        n_days = len(self.result.equity_curve) if self.result.equity_curve is not None else 1

        total_trade_value = sum(
            abs(p.pnl or 0) + p.entry_price * p.size
            for p in trades_df.values()
            if p.status == PositionStatus.CLOSED
        )

        cost_per_trade_bps = (
            self.config.friction_spread_bps + self.config.friction_slippage_bps + self.config.commission_pct * 100
        )
        total_friction_cost = total_trade_value * (cost_per_trade_bps / 10000.0)

        avg_aum = self.config.initial_equity
        turnover_ratio = (total_trade_value / avg_aum) * (252 / max(n_days, 1)) if avg_aum > 0 else 0.0
        friction_drag = turnover_ratio * (cost_per_trade_bps / 10000.0)

        gross_return = self.result.metrics.get("total_return_pct", 0) if self.result.metrics else 0
        net_return = gross_return - friction_drag

        gross_sharpe = self.result.metrics.get("sharpe_ratio", 0) if self.result.metrics else 0
        net_sharpe = max(0, gross_sharpe - friction_drag * 2)

        return {
            "total_trade_value": total_trade_value,
            "total_friction_cost": total_friction_cost,
            "cost_per_trade_bps": cost_per_trade_bps,
            "avg_cost_per_trade": total_friction_cost / max(total_trades, 1),
            "turnover_ratio": turnover_ratio,
            "friction_drag": friction_drag,
            "gross_return_pct": gross_return,
            "net_return_pct": net_return,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "n_trades": total_trades,
            "n_days": n_days,
        }

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
        # Use type: ignore to avoid mypy nested dict issues
        pattern_stats: Dict = {}  # type: ignore[assignment]

        trades_df = self.result.trades
        if trades_df is None or len(trades_df) == 0:
            return pattern_stats

        for _, trade in trades_df.iterrows():
            pattern = str(trade.get("pattern", "Unknown"))
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "pnl_list": [],
                }

            pattern_stats[pattern]["trades"] += 1
            pnl_val = float(trade.get("pnl", 0)) if pd.notna(trade.get("pnl")) else 0.0
            pattern_stats[pattern]["total_pnl"] += pnl_val
            pattern_stats[pattern]["pnl_list"].append(pnl_val)

            if pnl_val > 0:
                pattern_stats[pattern]["wins"] += 1
            else:
                pattern_stats[pattern]["losses"] += 1

        # Calculate metrics for each pattern
        for pattern, stats in pattern_stats.items():
            if stats["trades"] > 0:
                stats["win_rate"] = float(stats["wins"]) / float(stats["trades"])
                stats["avg_pnl"] = float(stats["total_pnl"]) / float(stats["trades"])

                wins = [p for p in stats["pnl_list"] if p > 0]
                losses = [abs(p) for p in stats["pnl_list"] if p < 0]

                stats["avg_win"] = float(sum(wins)) / len(wins) if wins else 0.0
                stats["avg_loss"] = float(sum(losses)) / len(losses) if losses else 0.0

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
