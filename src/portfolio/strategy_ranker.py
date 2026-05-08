"""
Strategy Ranking and Selection System

This module provides tools for ranking strategies by performance,
detecting regime changes, and selecting the best strategies for
portfolio inclusion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


class RankingMethod(Enum):
    """Methods for ranking strategies."""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    WIN_RATE = "win_rate"
    COMPOSITE = "composite"


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""

    name: str
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    volatility: float
    returns: np.ndarray

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "volatility": self.volatility,
        }


@dataclass
class RegimeState:
    """Market regime detection results."""

    regime: str
    confidence: float
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@jit(nopython=True, cache=True)
def compute_rolling_sharpe_numba(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """Compute rolling Sharpe ratio."""
    n = len(returns)
    rolling_sharpe = np.zeros(n)

    for i in range(window, n):
        window_returns = returns[i - window : i]
        mean = np.mean(window_returns)
        std = np.std(window_returns)

        if std > 1e-10:
            rolling_sharpe[i] = mean / std * np.sqrt(252)
        else:
            rolling_sharpe[i] = 0.0

    return rolling_sharpe


@jit(nopython=True, cache=True)
def compute_rolling_sortino_numba(returns: np.ndarray, window: int = 63) -> np.ndarray:
    """Compute rolling Sortino ratio (downside deviation)."""
    n = len(returns)
    rolling_sortino = np.zeros(n)

    for i in range(window, n):
        window_returns = returns[i - window : i]
        mean = np.mean(window_returns)

        # Downside deviation
        downside = window_returns[window_returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            if downside_std > 1e-10:
                rolling_sortino[i] = mean / downside_std * np.sqrt(1008)
            else:
                rolling_sortino[i] = 0.0
        else:
            rolling_sortino[i] = 0.0

    return rolling_sortino


class StrategyRanker:
    """Ranks strategies by performance metrics."""

    def __init__(
        self,
        method: RankingMethod = RankingMethod.SHARPE_RATIO,
        min_trades: int = 30,
        min_sharpe: float = 0.3,
        max_drawdown_threshold: float = -0.3,
    ):
        self.method = method
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.logger = logging.getLogger(__name__)

    def compute_performance(self, name: str, returns: np.ndarray) -> StrategyPerformance:
        """Compute performance metrics for a strategy."""
        total_return = np.sum(returns)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe
        if volatility > 1e-10:
            sharpe = np.mean(returns) / volatility * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            if downside_std > 1e-10:
                sortino = np.mean(returns) / downside_std * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Calmar (return / max drawdown)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        if abs(max_drawdown) > 1e-10:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = 0.0

        # Win rate
        trades = returns[returns != 0]
        if len(trades) > 0:
            win_rate = np.sum(trades > 0) / len(trades)
        else:
            win_rate = 0.0

        # Profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        if len(losing_trades) > 0 and np.sum(np.abs(losing_trades)) > 1e-10:
            profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))
        else:
            profit_factor = 0.0

        return StrategyPerformance(
            name=name,
            total_return=float(total_return),
            sharpe=float(sharpe),
            sortino=float(sortino),
            calmar=float(calmar),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            num_trades=int(np.sum(returns != 0)),
            volatility=float(volatility),
            returns=returns,
        )

    def rank_strategies(self, strategy_returns: Dict[str, np.ndarray]) -> List[StrategyPerformance]:
        """Rank strategies by selected metric."""
        performances = []

        for name, returns in strategy_returns.items():
            perf = self.compute_performance(name, returns)

            # Apply filters
            if perf.num_trades < self.min_trades:
                continue

            if perf.sharpe < self.min_sharpe:
                continue

            if perf.max_drawdown < self.max_drawdown_threshold:
                continue

            performances.append(perf)

        # Sort by ranking method
        if self.method == RankingMethod.SHARPE_RATIO:
            performances.sort(key=lambda x: x.sharpe, reverse=True)
        elif self.method == RankingMethod.SORTINO_RATIO:
            performances.sort(key=lambda x: x.sortino, reverse=True)
        elif self.method == RankingMethod.PROFIT_FACTOR:
            performances.sort(key=lambda x: x.profit_factor, reverse=True)
        elif self.method == RankingMethod.CALMAR_RATIO:
            performances.sort(key=lambda x: x.calmar, reverse=True)
        elif self.method == RankingMethod.WIN_RATE:
            performances.sort(key=lambda x: x.win_rate, reverse=True)

        return performances

    def get_top_strategies(
        self, strategy_returns: Dict[str, np.ndarray], n: int = 5
    ) -> List[StrategyPerformance]:
        """Get top N strategies."""
        ranked = self.rank_strategies(strategy_returns)
        return ranked[:n]

    def compute_rolling_performance(self, returns: np.ndarray, window: int = 63) -> pd.DataFrame:
        """Compute rolling performance metrics."""
        rolling_sharpe = compute_rolling_sharpe_numba(returns, window)
        rolling_sortino = compute_rolling_sortino_numba(returns, window)

        return pd.DataFrame({"sharpe": rolling_sharpe, "sortino": rolling_sortino})


class RegimeDetector:
    """Detects market regimes and regime changes."""

    def __init__(self, lookback_window: int = 63, regime_threshold: float = 1.0):
        self.lookback_window = lookback_window
        self.regime_threshold = regime_threshold
        self.current_regime = "NEUTRAL"
        self.logger = logging.getLogger(__name__)

    def detect_regime(
        self, returns: np.ndarray, volatility: Optional[np.ndarray] = None
    ) -> RegimeState:
        """Detect current market regime."""
        if len(returns) < self.lookback_window:
            return RegimeState(regime="NEUTRAL", confidence=0.0)

        recent_returns = returns[-self.lookback_window :]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        # Classify regime based on returns and volatility
        if std_return > 1e-10:
            sharpe_like = mean_return / std_return
        else:
            sharpe_like = 0.0

        if sharpe_like > self.regime_threshold:
            regime = "BULL"
            confidence = min(sharpe_like / self.regime_threshold, 1.0)
        elif sharpe_like < -self.regime_threshold:
            regime = "BEAR"
            confidence = min(abs(sharpe_like) / self.regime_threshold, 1.0)
        else:
            regime = "NEUTRAL"
            confidence = 1.0 - abs(sharpe_like) / self.regime_threshold

        self.current_regime = regime

        return RegimeState(regime=regime, confidence=confidence)

    def detect_regime_change(
        self, returns: np.ndarray, volatility: Optional[np.ndarray] = None
    ) -> RegimeState:
        """Detect regime change."""
        current_state = self.detect_regime(returns, volatility)

        # Check if regime changed
        regime_changed = current_state.regime != self.current_regime

        return RegimeState(
            regime=current_state.regime,
            confidence=current_state.confidence if regime_changed else 0.0,
        )


class AdaptiveStrategySelector:
    """Adaptively selects strategies based on market conditions."""

    def __init__(self, ranker: StrategyRanker, regime_detector: RegimeDetector):
        self.ranker = ranker
        self.regime_detector = regime_detector
        self.regime_performance: Dict[str, Dict[str, List[float]]] = {}
        self.logger = logging.getLogger(__name__)

    def select_strategies_for_regime(
        self, regime: str, strategy_returns: Dict[str, np.ndarray], n: int = 5
    ) -> List[str]:
        """Select top strategies for specific regime."""
        # Filter strategies by regime performance
        strategy_scores = {}

        for name, returns in strategy_returns.items():
            perf = self.ranker.compute_performance(name, returns)

            # Get regime-specific score
            if regime in self.regime_performance and name in self.regime_performance[regime]:
                regime_sharpe = np.mean(self.regime_performance[regime][name])
                combined_score = perf.sharpe * 0.5 + regime_sharpe * 0.5
            else:
                combined_score = perf.sharpe

            strategy_scores[name] = combined_score

        # Sort and return top N
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)

        return [name for name, score in sorted_strategies[:n]]

    def update_regime_performance(self, regime: str, strategy_returns: Dict[str, np.ndarray]):
        """Update regime-specific performance tracking."""
        for name, returns in strategy_returns.items():
            perf = self.ranker.compute_performance(name, returns)

            if regime not in self.regime_performance:
                self.regime_performance[regime] = {}

            if name not in self.regime_performance[regime]:
                self.regime_performance[regime][name] = []

            self.regime_performance[regime][name].append(perf.sharpe)

    def get_adaptive_weights(
        self, strategy_returns: Dict[str, np.ndarray], regime: Optional[str] = None
    ) -> np.ndarray:
        """Get adaptive weights based on regime."""
        if regime is None:
            regime = self.regime_detector.current_regime

        # Select strategies for regime
        selected = self.select_strategies_for_regime(
            regime, strategy_returns, n=len(strategy_returns)
        )

        # Compute weights (higher weight for better performers)
        weights = np.zeros(len(strategy_returns))
        strategy_names = list(strategy_returns.keys())

        for i, name in enumerate(strategy_names):
            if name in selected:
                idx = selected.index(name)
                weights[i] = 1.0 / (idx + 1)  # Decay weight by rank

        # Normalize
        if np.sum(weights) > 1e-10:
            weights = weights / np.sum(weights)

        return weights
