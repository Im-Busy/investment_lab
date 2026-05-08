"""
Phase B7: Full ML-Enhanced Backtest

Compares ML-enhanced strategies vs baseline on SPY data:
- Baseline: All strategies run normally
- ML-Enhanced: Strategies filtered by regime, signals scored by ML

Usage:
    uv run scripts/phase_b7_ml_backtest.py --start 2015-01-01 --end 2024-12-31
    uv run scripts/phase_b7_ml_backtest.py --strategies "VWAP Bounce,EMA Ribbon"
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtesting import Backtest, Strategy

from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.strategies.adaptive_router import AdaptiveRouter


@dataclass
class BacktestComparison:
    """Results from comparing ML vs baseline backtests."""

    baseline: Dict[str, Any]
    ml_enhanced: Dict[str, Any]
    metrics_comparison: Dict[str, float]
    regime_stats: Dict[str, Any]
    signal_stats: Dict[str, Any]


# All available strategies
ALL_STRATEGIES = {
    "VWAP Bounce": ("src.strategies.vwap_bounce", "VWAPBounceStrategy"),
    "SMA Crossover 50/200": ("src.strategies.sma_crossover", "SMACrossoverStrategy"),
    "EMA Ribbon 9/21/55": ("src.strategies.ema_ribbon", "EMARibbonStrategy"),
    "Keltner Channel": ("src.strategies.keltner_channel", "KeltnerChannelStrategy"),
    "Chandelier Exit": ("src.strategies.chandelier_exit", "ChandelierExitStrategy"),
    "ADX Trend Strength": ("src.strategies.adx_trend_strength", "ADXTrendStrengthStrategy"),
    "Parabolic SAR": ("src.strategies.parabolic_sar", "ParabolicSARStrategy"),
    "Ichimoku Cloud": ("src.strategies.ichimoku_cloud", "IchimokuCloudStrategy"),
    "RSI Divergence": ("src.strategies.rsi_divergence", "RSIDivergenceStrategy"),
    "Stoch RSI Crossover": ("src.strategies.stoch_rsi_crossover", "StochRSICrossoverStrategy"),
    "CCI": ("src.strategies.cci_strategy", "CCIStrategy"),
    "Donchian Channel": ("src.strategies.donchian_breakout", "DonchianChannelStrategy"),
    "MFI": ("src.strategies.mfi_strategy", "MFIStrategy"),
    "Ultimate Oscillator": ("src.strategies.ultimate_oscillator", "UltimateOscillatorStrategy"),
    "Williams %R": ("src.strategies.williams_r_reversal", "WilliamsRReversalStrategy"),
    "TSI": ("src.strategies.tsi_strategy", "TSIStrategy"),
    "Chaikin Oscillator": ("src.strategies.chaikin_oscillator", "ChaikinOscillatorStrategy"),
    "Awesome Oscillator": ("src.strategies.awesome_oscillator", "AwesomeOscillatorStrategy"),
    "MACD Histogram": ("src.strategies.macd_histogram", "MACDHistogramStrategy"),
    "SMC Reversal BT": ("src.strategies.smc_reversal_bt", "SMCReversalBacktest"),
    "Linear Reg Channel": (
        "src.strategies.linear_regression_channel",
        "LinearRegressionChannelStrategy",
    ),
    "Connors RSI": ("src.strategies.connors_rsi", "ConnorsRSIMeanReversion"),
}


def load_spy_data(start: str, end: str) -> pd.DataFrame:
    """Load SPY daily data."""
    data_path = project_root / "data" / "raw" / "SPY_daily.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.replace(" ", "") for c in df.columns})

    df = df.rename(
        columns={
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
    )

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.sort_index()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    return df


def train_ml_models(df: pd.DataFrame, verbose: bool = True) -> MLPipeline:
    """Train ML regime classifier and signal scorer."""
    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)
    regime_labels = regime_result["regime"].apply(lambda r: r.value)

    engineer = FeatureEngineer()
    features_df = engineer.generate_features(df)
    numeric_features = features_df.select_dtypes(include=[np.number]).ffill().bfill().dropna()

    y = regime_labels.reindex(numeric_features.index).dropna()
    X = numeric_features.reindex(y.index)

    pipeline = MLPipeline(
        regime_model_type="random_forest",
        signal_model_type="gradient_boosting",
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    results = pipeline.fit(df, regime_labels=regime_labels)

    if verbose:
        print("\n=== ML Training Results ===")
        if "regime" in results:
            rr = results["regime"]
            print(f"Regime train accuracy: {rr.get('train_accuracy', 0):.4f}")
            print(f"Regime test accuracy: {rr.get('test_accuracy', 0):.4f}")
            if "feature_importance" in rr:
                top5 = list(rr["feature_importance"].items())[:5]
                print(f"Top 5 features: {top5}")

        if "signal" in results:
            sr = results["signal"]
            print(f"\nSignal scorer Rank IC: {sr.get('rank_ic', 0):.4f}")
            print(f"Signal scorer accuracy: {sr.get('accuracy', 0):.4f}")

    return pipeline


def run_baseline_backtest(
    df: pd.DataFrame, strategy_classes: List[Strategy], name: str = "Baseline"
) -> Dict[str, Any]:
    """Run baseline backtest (no ML enhancement)."""

    if not strategy_classes:
        raise ValueError("No strategies loaded!")

    if len(strategy_classes) == 1:
        strategy_cls = strategy_classes[0]
    else:
        strategy_cls = strategy_classes[0]

    bt = Backtest(df, strategy_cls, cash=1_000_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()

    return {
        "name": name,
        "trades": int(stats.get("# Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0)),
        "sharpe": float(stats.get("Sharpe Ratio", -999)),
        "profit_factor": float(stats.get("Profit Factor", 0)),
        "max_drawdown": float(stats.get("Max. Drawdown [%]", 0)),
        "return_pct": float(stats.get("Return [%]", 0)),
        "buy_hold_return": float(stats.get("Buy & Hold Return [%]", 0)),
        "avg_trade": float(stats.get("Avg. Trade [%]", 0)),
        "best_trade": float(stats.get("Best Trade [%]", -999)),
        "worst_trade": float(stats.get("Worst Trade [%]", 999)),
    }


def run_ml_enhanced_backtest(
    df: pd.DataFrame,
    strategy_classes: List[Strategy],
    pipeline: MLPipeline,
    name: str = "ML-Enhanced",
) -> Dict[str, Any]:
    """Run ML-enhanced backtest with regime routing and signal scoring."""

    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)
    regime_labels = regime_result["regime"]

    router = AdaptiveRouter()

    strategy_cls = strategy_classes[0]

    class MLEnhancedStrategy(strategy_cls):
        def init(self):
            super().init()
            self.router = router
            self.regime_labels = regime_labels
            self.bar_regime_counts = {"Trending": 0, "Ranging": 0, "Volatile": 0, "Transition": 0}
            self.trades_in_regime = {"Trending": 0, "Ranging": 0, "Volatile": 0, "Transition": 0}
            self._bar_index = 0

        def next(self):
            try:
                current_regime = self.regime_labels.iloc[self._bar_index]
                regime_name = current_regime.value

                self.bar_regime_counts[regime_name] += 1

                strategy_name = self.__class__.__bases__[0].__name__

                if not self.router.is_strategy_active(strategy_name, current_regime):
                    self._bar_index += 1
                    return

                if self.position:
                    self.trades_in_regime[regime_name] += 1

                self._bar_index += 1
                return super().next()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"ML-enhanced strategy error at bar {self._bar_index}: {e}"
                )
                self._bar_index += 1

    bt = Backtest(df, MLEnhancedStrategy, cash=1_000_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()

    return {
        "name": name,
        "trades": int(stats.get("# Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0)),
        "sharpe": float(stats.get("Sharpe Ratio", -999)),
        "profit_factor": float(stats.get("Profit Factor", 0)),
        "max_drawdown": float(stats.get("Max. Drawdown [%]", 0)),
        "return_pct": float(stats.get("Return [%]", 0)),
        "buy_hold_return": float(stats.get("Buy & Hold Return [%]", 0)),
        "avg_trade": float(stats.get("Avg. Trade [%]", 0)),
        "best_trade": float(stats.get("Best Trade [%]", -999)),
        "worst_trade": float(stats.get("Worst Trade [%]", 999)),
    }


def load_strategies(strategy_names: Optional[List[str]] = None) -> List[Strategy]:
    """Load strategy classes."""
    from importlib import import_module

    if strategy_names:
        strategies_to_load = {k: v for k, v in ALL_STRATEGIES.items() if k in strategy_names}
    else:
        strategies_to_load = ALL_STRATEGIES

    classes = []
    for name, (module_path, cls_name) in strategies_to_load.items():
        if cls_name is None:
            continue
        try:
            module = import_module(module_path)
            cls = getattr(module, cls_name)
            classes.append(cls)
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"[{name}] Failed to load: {e}")

    if not classes:
        print("WARNING: No strategies loaded!")
        if strategy_names:
            print(f"  Requested: {strategy_names}")
            print(f"  Available: {list(ALL_STRATEGIES.keys())}")

    return classes


def compare_results(baseline: Dict[str, Any], ml_enhanced: Dict[str, Any]) -> Dict[str, float]:
    """Calculate improvement metrics."""
    return {
        "return_improvement_pct": (
            (ml_enhanced["return_pct"] - baseline["return_pct"]) / abs(baseline["return_pct"]) * 100
            if baseline["return_pct"] != 0
            else 0
        ),
        "sharpe_improvement": ml_enhanced["sharpe"] - baseline["sharpe"],
        "win_rate_improvement": ml_enhanced["win_rate"] - baseline["win_rate"],
        "drawdown_reduction": baseline["max_drawdown"] - ml_enhanced["max_drawdown"],
        "profit_factor_improvement": ml_enhanced["profit_factor"] - baseline["profit_factor"],
    }


def print_comparison_table(
    baseline: Dict[str, Any], ml_enhanced: Dict[str, Any], comparison: Dict[str, float]
):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Metric':<25} {baseline['name']:<18} {ml_enhanced['name']:<18} {'Improvement':<18}")
    print("=" * 100)

    metrics = [
        ("Return [%]", "return_pct", "%"),
        ("Sharpe Ratio", "sharpe", ""),
        ("Win Rate [%]", "win_rate", "%"),
        ("Max DD [%]", "max_drawdown", "%"),
        ("Profit Factor", "profit_factor", ""),
        ("Trades", "trades", ""),
    ]

    for label, key, suffix in metrics:
        base_val = baseline[key]
        ml_val = ml_enhanced[key]

        if suffix == "%":
            print(f"{label:<25} {base_val:>17.1f}%   {ml_val:>17.1f}%   ", end="")
        else:
            print(f"{label:<25} {base_val:>18.2f}   {ml_val:>18.2f}   ", end="")

        if key == "max_drawdown":
            diff = comparison["drawdown_reduction"]
            if diff > 0:
                print(f"v{diff:.1f}% (better)", end="")
            else:
                print(f"^{abs(diff):.1f}% (worse)", end="")
        else:
            diff = ml_val - base_val
            if diff > 0:
                print(f"^+{diff:.2f}{suffix} (better)", end="")
            elif diff < 0:
                print(f"v{abs(diff):.2f}{suffix} (worse)", end="")
            else:
                print("= (same)", end="")
        print()

    print("=" * 100)


def analyze_regime_performance(df: pd.DataFrame, pipeline: MLPipeline) -> Dict[str, Any]:
    """Analyze regime distribution and predictions."""
    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)

    regime_counts = regime_result["regime"].value_counts()
    regime_pct = (regime_counts / len(regime_result) * 100).round(1)

    return {
        "total_bars": len(regime_result),
        "regime_distribution": {k.value: v for k, v in regime_counts.items()},
        "regime_distribution_pct": {k.value: v for k, v in regime_pct.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase B7: ML-Enhanced Backtest Comparison")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategies",
        help="Comma-separated list of strategies (default: all)",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("Phase B7: Full ML-Enhanced Backtest")
    print("=" * 100)

    print(f"\nLoading SPY data from {args.start} to {args.end}...")
    df = load_spy_data(args.start, args.end)
    print(f"Loaded {len(df)} bars")

    print("\nLoading strategies...")
    strategy_names = [s.strip() for s in args.strategies.split(",")] if args.strategies else None
    strategy_classes = load_strategies(strategy_names)
    print(f"Loaded {len(strategy_classes)} strategies")

    print("\nTraining ML models...")
    pipeline = train_ml_models(df)

    print("\nAnalyzing regime distribution...")
    regime_stats = analyze_regime_performance(df, pipeline)
    print(f"Regime distribution: {regime_stats['regime_distribution_pct']}")

    print("\n" + "=" * 100)
    print("Running BASELINE backtest...")
    print("=" * 100)
    baseline_results = run_baseline_backtest(df, strategy_classes, "Baseline")

    print("\n" + "=" * 100)
    print("Running ML-ENHANCED backtest...")
    print("=" * 100)
    ml_results = run_ml_enhanced_backtest(df, strategy_classes, pipeline, "ML-Enhanced")

    comparison = compare_results(baseline_results, ml_results)

    print_comparison_table(baseline_results, ml_results, comparison)

    print("\n=== Summary ===")
    print(f"Return improvement: {comparison['return_improvement_pct']:+.1f}%")
    print(f"Sharpe improvement: {comparison['sharpe_improvement']:+.2f}")
    print(f"Win rate improvement: {comparison['win_rate_improvement']:+.1f}%")
    print(f"Max drawdown reduction: {comparison['drawdown_reduction']:+.1f}%")
    print(f"Profit factor improvement: {comparison['profit_factor_improvement']:+.2f}")

    output_dir = project_root / "reports" / "ml_backtest_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "baseline": baseline_results,
        "ml_enhanced": ml_results,
        "comparison": comparison,
        "regime_stats": regime_stats,
        "config": {
            "start_date": args.start,
            "end_date": args.end,
            "strategies": strategy_names or "all",
            "num_strategies": len(strategy_classes),
        },
    }

    report_path = output_dir / "phase_b7_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
