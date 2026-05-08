"""
Adaptive Strategy Router

Enables/disables strategies based on detected market regime.
Maps strategies to appropriate regimes per the documented strategy rules.

Regime-Strategy Mapping (from trading_strategy.md):
| Regime  | Enabled Strategies                           | Disabled Strategies                |
|---------|---------------------------------------------|-----------------------------------|
| Trending| EMA Ribbon, SMA Crossover, ADX, Parabolic SAR| RSI Divergence, Williams %R, Stoch |
| Ranging | RSI Divergence, Williams %R, Stoch RSI, CCI  | EMA Ribbon, SMA Crossover, ADX     |
| Volatile| Chandelier Exit, Bollinger-based, VWAP Bounce| All trend-following                |
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from src.indicators.regime_detector import RegimeDetector, RegimeState


class StrategyCategory(Enum):
    """Strategy categories for routing."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM = "momentum"


# Strategy-to-regime mapping
STRATEGY_REGIME_MAP: Dict[str, List[RegimeState]] = {
    # Trend-following strategies (work well in trending regimes)
    "EMA Ribbon": [RegimeState.TRENDING],
    "SMA Crossover": [RegimeState.TRENDING],
    "ADX Trend Strength": [RegimeState.TRENDING],
    "Parabolic SAR": [RegimeState.TRENDING],
    # Mean reversion strategies (work well in ranging regimes)
    "RSI Divergence": [RegimeState.RANGING, RegimeState.TRANSITION],
    "Williams %R": [RegimeState.RANGING, RegimeState.TRANSITION],
    "Stoch RSI Crossover": [RegimeState.RANGING, RegimeState.TRANSITION],
    "CCI": [RegimeState.RANGING, RegimeState.TRANSITION],
    "MFI": [RegimeState.RANGING],
    # Volatility-based strategies (work well in volatile regimes)
    "Chandelier Exit": [RegimeState.VOLATILE, RegimeState.TRENDING],
    "Keltner Channel": [RegimeState.VOLATILE, RegimeState.RANGING],
    "Bollinger Bands": [RegimeState.VOLATILE, RegimeState.RANGING],
    "VWAP Bounce": [RegimeState.VOLATILE],
    "Donchian Channel": [RegimeState.TRENDING, RegimeState.VOLATILE],
}


class AdaptiveRouter:
    """
    Routes strategy execution based on current market regime.

    Usage:
        detector = RegimeDetector()
        router = AdaptiveRouter()

        # Get active strategies for current regime
        regimes = detector.get_regime_series(data)
        active_strategies = router.get_active_strategies(
            all_strategies,
            regimes.iloc[-1]["regime"]
        )
    """

    def __init__(
        self,
        strategy_regime_map: Optional[Dict[str, List[RegimeState]]] = None,
        fallback_regime: RegimeState = RegimeState.TRANSITION,
    ):
        """
        Initialize AdaptiveRouter.

        Args:
            strategy_regime_map: Dict[strategy_name -> list of suitable RegimeStates]
            fallback_regime: Default regime when classification is uncertain
        """
        self.strategy_regime_map = strategy_regime_map or STRATEGY_REGIME_MAP
        self.fallback_regime = fallback_regime
        self._strategy_categories = self._load_categories()

    def _load_categories(self) -> Dict[str, StrategyCategory]:
        """Load strategy categories."""
        return {
            "EMA Ribbon": StrategyCategory.TREND_FOLLOWING,
            "SMA Crossover": StrategyCategory.TREND_FOLLOWING,
            "ADX Trend Strength": StrategyCategory.TREND_FOLLOWING,
            "Parabolic SAR": StrategyCategory.TREND_FOLLOWING,
            "RSI Divergence": StrategyCategory.MEAN_REVERSION,
            "Williams %R": StrategyCategory.MEAN_REVERSION,
            "Stoch RSI Crossover": StrategyCategory.MEAN_REVERSION,
            "CCI": StrategyCategory.MEAN_REVERSION,
            "MFI": StrategyCategory.MEAN_REVERSION,
            "Chandelier Exit": StrategyCategory.VOLATILITY_BASED,
            "Keltner Channel": StrategyCategory.VOLATILITY_BASED,
            "Bollinger Bands": StrategyCategory.VOLATILITY_BASED,
            "VWAP Bounce": StrategyCategory.VOLATILITY_BASED,
            "Donchian Channel": StrategyCategory.MOMENTUM,
        }

    def is_strategy_active(
        self,
        strategy_name: str,
        regime: RegimeState,
    ) -> bool:
        """
        Check if a strategy should be active in the current regime.

        Args:
            strategy_name: Name of the strategy
            regime: Current market regime

        Returns:
            True if strategy should be active
        """
        suitable_regimes = self.strategy_regime_map.get(strategy_name, [])

        # If no mapping found, default to active
        if not suitable_regimes:
            return True

        return regime in suitable_regimes

    def get_active_strategies(
        self,
        all_strategies: List[str],
        regime: RegimeState,
    ) -> List[str]:
        """
        Filter strategies to only those active in the current regime.

        Args:
            all_strategies: List of all strategy names
            regime: Current market regime

        Returns:
            List of active strategy names
        """
        return [name for name in all_strategies if self.is_strategy_active(name, regime)]

    def get_regime_recommendations(
        self,
        regime: RegimeState,
    ) -> Dict[str, Any]:
        """
        Get recommendations for the current regime.

        Args:
            regime: Current market regime

        Returns:
            Dict with enabled strategies, disabled strategies, and advice
        """
        all_strategies = list(self.strategy_regime_map.keys())
        enabled = self.get_active_strategies(all_strategies, regime)
        disabled = [s for s in all_strategies if s not in enabled]

        advice = {
            RegimeState.TRENDING: "Focus on trend-following strategies. Avoid mean-reversion.",
            RegimeState.RANGING: "Focus on mean-reversion strategies. Avoid trend-following.",
            RegimeState.VOLATILE: "Focus on volatility-based strategies. Use wider stops.",
            RegimeState.TRANSITION: "Regime unclear. Maintain previous strategy allocation.",
        }

        return {
            "regime": regime.value,
            "enabled": enabled,
            "disabled": disabled,
            "advice": advice.get(regime, "No specific advice."),
        }

    def apply_regime_filter(
        self,
        signals: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Filter signals based on regime-appropriate strategies.

        Args:
            signals: DataFrame with signal data, must have 'strategy' and 'timestamp' columns
            regimes: Series of RegimeState indexed by timestamp

        Returns:
            Filtered DataFrame with only regime-appropriate signals
        """
        if len(signals) == 0:
            return signals

        filtered_rows = []

        for idx, row in signals.iterrows():
            timestamp = row.get("timestamp", idx)
            strategy = row.get("strategy", "")

            # Get current regime at this timestamp
            current_regime = regimes.get(timestamp, self.fallback_regime)
            if isinstance(current_regime, pd.Series):
                current_regime = self.fallback_regime

            if self.is_strategy_active(strategy, current_regime):
                filtered_rows.append(row)

        return pd.DataFrame(filtered_rows, index=range(len(filtered_rows)))

    def get_regime_time_series(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get full time series of regime classification with active strategy count.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with regime, active strategy count, and active strategy list
        """
        detector = RegimeDetector()
        regime_info = detector.get_regime_series(data)

        active_counts = []
        active_lists = []
        all_strategies = list(self.strategy_regime_map.keys())

        for _, row in regime_info.iterrows():
            regime = row["regime"]
            active = self.get_active_strategies(all_strategies, regime)
            active_counts.append(len(active))
            active_lists.append(",".join(active))

        regime_info["active_strategy_count"] = active_counts
        regime_info["active_strategies"] = active_lists

        return regime_info
