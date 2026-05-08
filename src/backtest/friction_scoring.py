# -*- coding: utf-8 -*-
"""
Friction-Adjusted Backtest Scoring (R10)

Implements transaction cost modeling as first-class constraint in backtest evaluation.
Based on research findings that friction can destroy alpha in high-turnover strategies.

Key Research:
- OOM-RL paper: 0.08% slippage x 6700% turnover = alpha destruction
- High turnover strategies must overcome friction drag to be viable

Components:
1. Friction Cost Model: Spread + Slippage + Commission
2. Friction-Adjusted Metrics: Net Sharpe, Net Returns
3. Turnover Budget Enforcement: Flag excessive turnover

Integration:
- Modifies src/backtest/engine.py to call friction scoring
- Outputs friction-adjusted returns in backtest reports
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class AssetClass(Enum):
    """Asset classes with different friction characteristics."""

    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"


@dataclass
class FrictionConfig:
    """
    Transaction cost configuration.

    Attributes:
        spread_bps: Bid-ask spread in basis points
        slippage_bps: Market impact/slippage in basis points
        commission_bps: Commission/fee in basis points
        min_commission: Minimum commission per trade (dollars)
        asset_class: Asset class for default friction values
    """

    spread_bps: float = 5.0
    slippage_bps: float = 3.0
    commission_bps: float = 1.0
    min_commission: float = 1.0
    asset_class: AssetClass = AssetClass.EQUITY

    @classmethod
    def for_asset_class(cls, asset_class: AssetClass) -> "FrictionConfig":
        """Get default friction config for asset class."""
        defaults = {
            AssetClass.EQUITY: cls(
                spread_bps=5.0,
                slippage_bps=2.0,
                commission_bps=1.0,
                min_commission=1.0,
            ),
            AssetClass.ETF: cls(
                spread_bps=3.0,
                slippage_bps=1.0,
                commission_bps=0.5,
                min_commission=0.0,
            ),
            AssetClass.CRYPTO: cls(
                spread_bps=10.0,
                slippage_bps=5.0,
                commission_bps=10.0,
                min_commission=0.0,
            ),
            AssetClass.FOREX: cls(
                spread_bps=2.0,
                slippage_bps=1.0,
                commission_bps=0.0,
                min_commission=0.0,
            ),
            AssetClass.FUTURES: cls(
                spread_bps=1.0,
                slippage_bps=0.5,
                commission_bps=0.5,
                min_commission=2.5,
            ),
        }
        config = defaults.get(asset_class, defaults[AssetClass.EQUITY])
        config.asset_class = asset_class
        return config


@dataclass
class FrictionCosts:
    """
    Result of friction cost calculation for a single trade.

    Attributes:
        spread_cost: Cost from bid-ask spread
        slippage_cost: Cost from market impact
        commission_cost: Commission/fee cost
        total_cost: Total friction cost
        cost_bps: Total cost in basis points
        cost_percent: Total cost as percentage
    """

    spread_cost: float
    slippage_cost: float
    commission_cost: float
    total_cost: float
    cost_bps: float
    cost_percent: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "commission_cost": self.commission_cost,
            "total_cost": self.total_cost,
            "cost_bps": self.cost_bps,
            "cost_percent": self.cost_percent,
        }


@dataclass
class FrictionSummary:
    """
    Aggregate friction statistics for a backtest period.

    Attributes:
        total_friction: Total friction costs (dollars)
        total_turnover: Total turnover (dollars)
        avg_cost_per_trade: Average friction per trade
        avg_cost_bps: Average cost in basis points
        friction_drag: Annualized friction drag on returns
        net_sharpe: Sharpe ratio after friction
        gross_sharpe: Sharpe ratio before friction
        turnover_ratio: Annualized turnover / AUM
    """

    total_friction: float
    total_turnover: float
    avg_cost_per_trade: float
    avg_cost_bps: float
    friction_drag: float
    net_sharpe: float
    gross_sharpe: float
    turnover_ratio: float
    bonus_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_friction": self.total_friction,
            "total_turnover": self.total_turnover,
            "avg_cost_per_trade": self.avg_cost_per_trade,
            "avg_cost_bps": self.avg_cost_bps,
            "friction_drag": self.friction_drag,
            "net_sharpe": self.net_sharpe,
            "gross_sharpe": self.gross_sharpe,
            "turnover_ratio": self.turnover_ratio,
            **self.bonus_stats,
        }


class FrictionScorer:
    """
    Calculate friction-adjusted backtest metrics.

    Key Formula:
        Net Return = Gross Return - Friction Drag
        Friction Drag = (Total Turnover x Cost per Trade) / Avg Position
        Net Sharpe = Gross Sharpe - Friction Drag / Volatility

    Usage:
        scorer = FrictionScorer(config)
        costs = scorer.calculate_trade_cost(trade_value, price)
        summary = scorer.calculate_summary(trades, returns, equity_curve)
    """

    def __init__(self, config: Optional[FrictionConfig] = None):
        """
        Initialize friction scorer.

        Args:
            config: Friction configuration
        """
        self.config = config or FrictionConfig()

    def calculate_trade_cost(
        self,
        trade_value: float,
        price: float,
        quantity: float,
    ) -> FrictionCosts:
        """
        Calculate friction costs for a single trade.

        Args:
            trade_value: Total trade value (dollars)
            price: Trade price
            quantity: Number of shares/contracts

        Returns:
            FrictionCosts with detailed breakdown
        """
        # Spread cost: half spread x trade value
        spread_cost = trade_value * (self.config.spread_bps / 10000.0)

        # Slippage cost: market impact estimate
        # Simple model: slippage increases with trade size
        base_slippage = self.config.slippage_bps / 10000.0
        slippage_cost = trade_value * base_slippage

        # Commission: max of percentage and minimum
        commission_cost = max(
            trade_value * (self.config.commission_bps / 10000.0),
            self.config.min_commission,
        )

        # Total cost
        total_cost = spread_cost + slippage_cost + commission_cost

        # Express in bps and percent
        cost_bps = (total_cost / trade_value) * 10000.0 if trade_value > 0 else 0.0
        cost_percent = total_cost / trade_value if trade_value > 0 else 0.0

        return FrictionCosts(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission_cost=commission_cost,
            total_cost=total_cost,
            cost_bps=cost_bps,
            cost_percent=cost_percent,
        )

    def calculate_turnover(
        self,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> float:
        """
        Calculate portfolio turnover over a period.

        Turnover = Sum of absolute position changes / Average AUM

        Args:
            positions: DataFrame of position values over time
            trades: DataFrame of trades (with value column)

        Returns:
            Turnover as fraction of AUM
        """
        if len(trades) == 0:
            return 0.0

        total_turnover = float(trades["value"].abs().sum())
        avg_aum = positions.values.mean() if len(positions) > 0 else 1.0

        return total_turnover / avg_aum if avg_aum > 0 else 0.0

    def annualize_turnover(
        self,
        turnover: float,
        n_trading_days: int,
        lookback_days: int,
    ) -> float:
        """
        Annualize turnover estimate.

        Args:
            turnover: Turnover over lookback period
            n_trading_days: Number of trading days in data
            lookback_days: Length of lookback period

        Returns:
            Annualized turnover ratio
        """
        if lookback_days == 0:
            return 0.0

        daily_turnover = turnover / lookback_days
        annual_turnover = daily_turnover * n_trading_days

        return annual_turnover

    def calculate_friction_drag(
        self,
        turnover_ratio: float,
        avg_cost_bps: float,
    ) -> float:
        """
        Calculate friction drag on annualized returns.

        Friction Drag = Turnover Ratio x Average Cost (bps) / 10000

        Args:
            turnover_ratio: Annualized turnover / AUM
            avg_cost_bps: Average cost per trade in basis points

        Returns:
            Friction drag as decimal (e.g., 0.02 = 2% drag)
        """
        return turnover_ratio * (avg_cost_bps / 10000.0)

    def calculate_net_sharpe(
        self,
        gross_sharpe: float,
        gross_returns: np.ndarray,
        friction_costs: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate Sharpe ratio after friction costs.

        Args:
            gross_sharpe: Sharpe ratio before friction
            gross_returns: Array of gross returns
            friction_costs: Array of friction costs per period
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            Tuple of (net_sharpe, net_returns)
        """
        # Net returns = Gross returns - friction costs
        net_returns = gross_returns - friction_costs

        # Calculate net Sharpe
        if len(net_returns) < 2:
            return (0.0, net_returns)

        excess_returns = net_returns - risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_returns = np.std(net_returns, ddof=1)

        if std_returns < 1e-10:
            net_sharpe = 0.0
        else:
            # Annualize (assuming daily returns)
            net_sharpe = (mean_excess / std_returns) * np.sqrt(252)

        return (net_sharpe, net_returns)

    def calculate_summary(
        self,
        trades: pd.DataFrame,
        gross_returns: np.ndarray,
        equity_curve: pd.Series,
        gross_sharpe: float,
        benchmark_return: float = 0.0,
    ) -> FrictionSummary:
        """
        Calculate comprehensive friction summary.

        Args:
            trades: DataFrame with columns: date, value, price, quantity
            gross_returns: Array of gross returns
            equity_curve: Equity curve series
            gross_sharpe: Sharpe ratio before friction
            benchmark_return: Optional benchmark return for comparison

        Returns:
            FrictionSummary with all metrics
        """
        if len(trades) == 0:
            return FrictionSummary(
                total_friction=0.0,
                total_turnover=0.0,
                avg_cost_per_trade=0.0,
                avg_cost_bps=0.0,
                friction_drag=0.0,
                net_sharpe=gross_sharpe,
                gross_sharpe=gross_sharpe,
                turnover_ratio=0.0,
                bonus_stats={"n_trades": 0},
            )

        # Calculate costs for each trade
        trade_costs = []
        for _, trade in trades.iterrows():
            cost = self.calculate_trade_cost(
                trade_value=trade["value"],
                price=trade["price"],
                quantity=trade["quantity"],
            )
            trade_costs.append(cost.total_cost)

        total_friction = sum(trade_costs)
        total_turnover = float(trades["value"].abs().sum())
        avg_cost_per_trade = np.mean(trade_costs) if trade_costs else 0.0

        # Average cost in bps
        avg_cost_bps = (total_friction / total_turnover) * 10000.0 if total_turnover > 0 else 0.0

        # Turnover ratio (annualized)
        n_days = len(equity_curve)
        avg_aum = equity_curve.mean()
        turnover_ratio = (total_turnover / avg_aum) * (252 / max(n_days, 1)) if avg_aum > 0 else 0.0

        # Friction drag
        friction_drag = self.calculate_friction_drag(turnover_ratio, avg_cost_bps)

        # Calculate friction costs per period (for Sharpe adjustment)
        friction_per_period = np.zeros_like(gross_returns)
        if len(trades) > 0 and "date" in trades.columns:
            # Aggregate costs by period
            for i, ret in enumerate(gross_returns):
                period_costs = trades[trades.index == i]["value"].sum()
                period_costs = abs(period_costs)
                if period_costs > 0:
                    friction_per_period[i] = period_costs * (avg_cost_bps / 10000.0)

        # Net Sharpe
        net_sharpe, _ = self.calculate_net_sharpe(gross_sharpe, gross_returns, friction_per_period)

        # Bonus statistics
        bonus = {
            "n_trades": len(trades),
            "total_spread_cost": sum(
                self.calculate_trade_cost(t["value"], t["price"], t["quantity"]).spread_cost
                for _, t in trades.iterrows()
            ),
            "total_slippage_cost": sum(
                self.calculate_trade_cost(t["value"], t["price"], t["quantity"]).slippage_cost
                for _, t in trades.iterrows()
            ),
            "total_commission_cost": sum(
                self.calculate_trade_cost(t["value"], t["price"], t["quantity"]).commission_cost
                for _, t in trades.iterrows()
            ),
            "friction_as_pct_return": (
                total_friction / (equity_curve.iloc[-1] - equity_curve.iloc[0])
            )
            * 100
            if len(equity_curve) > 1 and equity_curve.iloc[-1] != equity_curve.iloc[0]
            else 0.0,
        }

        return FrictionSummary(
            total_friction=total_friction,
            total_turnover=total_turnover,
            avg_cost_per_trade=avg_cost_per_trade,
            avg_cost_bps=avg_cost_bps,
            friction_drag=friction_drag,
            net_sharpe=net_sharpe,
            gross_sharpe=gross_sharpe,
            turnover_ratio=turnover_ratio,
            bonus_stats=bonus,
        )


class TurnoverBudgetEnforcer:
    """
    Enforce turnover budgets to prevent alpha destruction.

    Research insight: High turnover + friction = negative expected value
    Rule: Flag strategies exceeding annualized turnover threshold
    """

    def __init__(
        self,
        max_annual_turnover: float = 4.0,  # 400% annual turnover max
        friction_threshold_bps: float = 50.0,  # 50 bps friction threshold
    ):
        """
        Initialize turnover budget enforcer.

        Args:
            max_annual_turnover: Maximum acceptable annualized turnover
            friction_threshold_bps: Maximum acceptable friction cost
        """
        self.max_annual_turnover = max_annual_turnover
        self.friction_threshold_bps = friction_threshold_bps

    def check_budget(
        self,
        friction_summary: FrictionSummary,
        trading_days: int = 252,
    ) -> Dict[str, any]:
        """
        Check if turnover is within budget.

        Args:
            friction_summary: Friction summary from backtest
            trading_days: Number of trading days per year

        Returns:
            Dictionary with budget status and recommendations
        """
        # Check turnover budget
        turnover_ok = friction_summary.turnover_ratio <= self.max_annual_turnover

        # Check friction threshold
        friction_ok = friction_summary.avg_cost_bps <= self.friction_threshold_bps

        # Calculate turnover utilization
        turnover_utilization = friction_summary.turnover_ratio / self.max_annual_turnover

        # Determine action
        if not turnover_ok and not friction_ok:
            action = "REJECT"
            reason = "Exceeds both turnover and friction thresholds"
        elif not turnover_ok:
            action = "FLAG"
            reason = f"Turnover {friction_summary.turnover_ratio:.1f}x exceeds {self.max_annual_turnover}x limit"
        elif not friction_ok:
            action = "FLAG"
            reason = f"Friction {friction_summary.avg_cost_bps:.1f}bps exceeds {self.friction_threshold_bps}bps limit"
        else:
            action = "PASS"
            reason = "Within budget constraints"

        # Calculate alpha threshold
        # Strategy needs Sharpe > friction drag to be viable
        min_required_sharpe = friction_summary.friction_drag * 2  # 2x friction drag

        return {
            "action": action,
            "reason": reason,
            "turnover_ok": turnover_ok,
            "friction_ok": friction_ok,
            "turnover_utilization": turnover_utilization,
            "min_required_sharpe": min_required_sharpe,
            "net_sharpe": friction_summary.net_sharpe,
            "recommendation": self._get_recommendation(action, friction_summary),
        }

    def _get_recommendation(
        self,
        action: str,
        summary: FrictionSummary,
    ) -> str:
        """Generate actionable recommendation."""
        if action == "PASS":
            return "Strategy is viable. Monitor turnover for drift."
        elif action == "REJECT":
            return (
                f"Strategy not viable: "
                f"friction drag ({summary.friction_drag:.2%}) exceeds returns. "
                f"Reduce turnover or improve gross alpha."
            )
        else:  # FLAG
            if summary.turnover_ratio > self.max_annual_turnover:
                return (
                    f"Reduce turnover from {summary.turnover_ratio:.1f}x to "
                    f"<{self.max_annual_turnover}x. Consider longer holding periods."
                )
            else:
                return (
                    f"Negotiate lower commissions or trade more liquid instruments "
                    f"to reduce friction from {summary.avg_cost_bps:.1f}bps."
                )
