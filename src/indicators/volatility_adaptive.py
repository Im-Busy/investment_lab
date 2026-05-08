# -*- coding: utf-8 -*-
"""
Volatility-Adaptive Thresholds - R21 Implementation

Dynamic z-score adjustment based on market volatility regime.

From handover document:
- R21: Volatility-adaptive thresholds (dynamic z-scores)
- P5: Survey of Statistical Arbitrage Pair Trading
- Target: 2026-07-31
- Priority: 🟠 High

Key idea: Adjust signal detection thresholds based on volatility
- High volatility → Wider thresholds (avoid false signals)
- Low volatility → Tighter thresholds (catch more opportunities)

Usage:
    uv run python src/indicators/volatility_adaptive.py --data data/symbols/*.csv
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for volatility-adaptive thresholds."""

    # Baseline thresholds (standard z-scores)
    baseline_threshold: float = 2.0

    # Volatility adjustment method
    adjustment_method: str = "regime_based"

    # Regime classification percentiles
    regime_percentiles: Tuple[float, float, float, float] = (0.20, 0.40, 0.60, 0.80)

    # Threshold scaling factors per regime
    regime_multipliers: Dict[str, float] = None

    # Lookback for volatility estimation
    volatility_window: int = 60

    # Minimum/maximum threshold bounds
    min_threshold: float = 1.0
    max_threshold: float = 4.0

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                VolatilityRegime.VERY_LOW.value: 0.7,
                VolatilityRegime.LOW.value: 0.85,
                VolatilityRegime.NORMAL.value: 1.0,
                VolatilityRegime.HIGH.value: 1.3,
                VolatilityRegime.VERY_HIGH.value: 1.6,
            }


@dataclass
class ThresholdResult:
    """Result for one symbol."""

    symbol: str
    current_regime: str
    current_volatility: float
    baseline_threshold: float
    adaptive_threshold: float
    adjustment_factor: float
    signal_triggered: bool
    signal_value: float


class VolatilityRegressor:
    """
    Calculate adaptive thresholds based on volatility.

    Implements R21 from P5: dynamic z-score adjustment.
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None):
        """
        Args:
            config: Configuration
        """
        self.config = config or AdaptiveThresholdConfig()

    def calculate_regime_thresholds(
        self,
        volatility_series: pd.Series,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate volatility regime thresholds from historical data.

        Args:
            volatility_series: Historical volatility series

        Returns:
            Dict mapping regime → (lower_bound, upper_bound)
        """
        percentiles = self.config.regime_percentiles
        quantiles = np.percentile(volatility_series.dropna(), [0, *percentiles, 100])

        regimes = {
            VolatilityRegime.VERY_LOW.value: (quantiles[0], quantiles[1]),
            VolatilityRegime.LOW.value: (quantiles[1], quantiles[2]),
            VolatilityRegime.NORMAL.value: (quantiles[2], quantiles[3]),
            VolatilityRegime.HIGH.value: (quantiles[3], quantiles[4]),
            VolatilityRegime.VERY_HIGH.value: (quantiles[4], quantiles[5]),
        }

        return regimes

    def classify_regime(
        self,
        current_volatility: float,
        regime_thresholds: Dict[str, Tuple[float, float]],
    ) -> VolatilityRegime:
        """
        Classify current volatility regime.

        Args:
            current_volatility: Current volatility value
            regime_thresholds: Regime threshold dictionary

        Returns:
            VolatilityRegime enum value
        """
        for regime_name, (lower, upper) in regime_thresholds.items():
            if lower <= current_volatility < upper:
                return VolatilityRegime(regime_name)

        # Handle edge cases
        if current_volatility < list(regime_thresholds.values())[0][0]:
            return VolatilityRegime.VERY_LOW
        else:
            return VolatilityRegime.VERY_HIGH

    def get_adaptive_threshold(
        self,
        volatility: float,
        regime_thresholds: Dict[str, Tuple[float, float]],
        base_threshold: Optional[float] = None,
    ) -> Tuple[float, VolatilityRegime, float]:
        """
        Calculate adaptive threshold for current volatility.

        Args:
            volatility: Current volatility
            regime_thresholds: Regime threshold dictionary
            base_threshold: Base threshold (uses config default if None)

        Returns:
            Tuple of (adaptive_threshold, regime, adjustment_factor)
        """
        if base_threshold is None:
            base_threshold = self.config.baseline_threshold

        regime = self.classify_regime(volatility, regime_thresholds)

        # Get multiplier for this regime
        multiplier = self.config.regime_multipliers.get(regime.value, 1.0)

        # Apply scaling
        adaptive = base_threshold * multiplier

        # Apply bounds
        adaptive = np.clip(adaptive, self.config.min_threshold, self.config.max_threshold)

        adjustment_factor = adaptive / base_threshold

        return (adaptive, regime, adjustment_factor)

    def analyze(
        self,
        prices: pd.DataFrame,
        pattern_signals: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
    ) -> ThresholdResult:
        """
        Analyze adaptive threshold for one symbol.

        Args:
            prices: OHLCV DataFrame with 'close' column
            pattern_signals: Optional pattern signal series
            symbol: Stock symbol

        Returns:
            ThresholdResult with analysis
        """
        # Calculate volatility
        close_col = "Close" if "Close" in prices.columns else "close"
        returns = prices[close_col].pct_change()
        volatility = returns.rolling(self.config.volatility_window).std() * np.sqrt(252)

        current_vol = volatility.iloc[-1]
        historical_vol = volatility.dropna()

        # Calculate regime thresholds
        regime_thresholds = self.calculate_regime_thresholds(historical_vol)

        # Get adaptive threshold
        adaptive, regime, factor = self.get_adaptive_threshold(current_vol, regime_thresholds)

        # Check if signal triggered (if pattern signals provided)
        signal_triggered = False
        signal_value = 0.0

        if pattern_signals is not None:
            signal_value = float(pattern_signals.iloc[-1])
            signal_triggered = abs(signal_value) >= adaptive

        return ThresholdResult(
            symbol=symbol,
            current_regime=regime.value,
            current_volatility=float(current_vol),
            baseline_threshold=self.config.baseline_threshold,
            adaptive_threshold=adaptive,
            adjustment_factor=factor,
            signal_triggered=signal_triggered,
            signal_value=signal_value,
        )


class AdaptiveSignalDetector:
    """
    Pattern detection with volatility-adaptive thresholds.

    Wraps existing pattern detectors to add volatility adaptation.
    """

    def __init__(
        self,
        pattern_detector,
        volatility_regressor: Optional[VolatilityRegressor] = None,
        use_adaptive: bool = True,
    ):
        """
        Args:
            pattern_detector: Pattern detector instance
            volatility_regressor: Volatility regressor
            use_adaptive: Whether to use adaptive thresholds
        """
        self.pattern_detector = pattern_detector
        self.volatility_regressor = volatility_regressor or VolatilityRegressor()
        self.use_adaptive = use_adaptive

    def detect(
        self,
        data: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict:
        """
        Detect pattern with adaptive threshold.

        Args:
            data: OHLCV data
            prices: Price data for volatility calculation

        Returns:
            Detection result with adaptive threshold
        """
        # Get baseline signal
        baseline_result = self.pattern_detector.detect(data)
        signal_value = baseline_result.get("signal", 0.0)

        if not self.use_adaptive:
            # Use fixed threshold
            threshold = self.pattern_detector.threshold
            triggered = abs(signal_value) >= threshold

            return {
                **baseline_result,
                "threshold_used": threshold,
                "adaptive": False,
                "triggered": triggered,
            }

        # Calculate adaptive threshold
        prices_for_vol = prices.copy()
        prices_for_vol["close"] = data["close"]

        threshold_result = self.volatility_regressor.analyze(
            prices_for_vol,
            pd.Series([signal_value]),
            symbol=getattr(self.pattern_detector, "name", "UNKNOWN"),
        )

        triggered = abs(signal_value) >= threshold_result.adaptive_threshold

        return {
            **baseline_result,
            "signal": signal_value,
            "threshold_used": threshold_result.adaptive_threshold,
            "volatility_regime": threshold_result.current_regime,
            "adjustment_factor": threshold_result.adjustment_factor,
            "adaptive": True,
            "triggered": triggered,
        }


def analyze_portfolio_volatility(
    data_dir: str,
    output_dir: str,
) -> Dict:
    """
    Analyze volatility regimes across portfolio.

    Args:
        data_dir: Directory with stock data CSVs
        output_dir: Output directory

    Returns:
        Analysis results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    regressor = VolatilityRegressor()
    results = []

    data_files = list(Path(data_dir).glob("*.csv"))
    print(f"Analyzing {len(data_files)} symbols...")

    for filepath in data_files:
        try:
            prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if "close" not in prices.columns:
                continue

            symbol = filepath.stem
            result = regressor.analyze(prices, symbol=symbol)
            results.append(asdict(result))
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    results_df = pd.DataFrame(results)

    # Summary statistics
    regime_counts = results_df["current_regime"].value_counts().to_dict()
    mean_threshold = results_df["adaptive_threshold"].mean()
    mean_adjustment = results_df["adjustment_factor"].mean()

    print("\n=== VOLATILITY-ADAPTIVE THRESHOLD ANALYSIS ===\n")
    print(f"Symbols analyzed: {len(results)}")
    print("\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(results) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print("\nThreshold statistics:")
    print(f"  Baseline: {results_df['baseline_threshold'].mean():.2f}")
    print(f"  Adaptive mean: {mean_threshold:.2f}")
    print(f"  Mean adjustment: {mean_adjustment:.2f}x")

    # Save results
    results_df.to_csv(os.path.join(output_dir, "volatility_analysis.csv"), index=False)

    summary = {
        "total_symbols": len(results),
        "regime_distribution": regime_counts,
        "mean_adaptive_threshold": float(mean_threshold),
        "mean_adjustment_factor": float(mean_adjustment),
    }

    json_path = os.path.join(output_dir, "volatility_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Volatility-Adaptive Thresholds (R21)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with stock data CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/volatility_adaptive",
        help="Output directory",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=2.0,
        help="Baseline z-score threshold",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    config = AdaptiveThresholdConfig(baseline_threshold=args.baseline_threshold)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = analyze_portfolio_volatility(args.data_dir, args.output_dir)

    print("\n✅ R21 Volatility-adaptive threshold analysis complete!")


if __name__ == "__main__":
    main()
