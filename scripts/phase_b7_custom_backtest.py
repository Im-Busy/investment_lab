"""
Phase B7: Full ML-Enhanced Backtest (Custom Engine Version)

Compares ML-enhanced strategies vs baseline using the custom backtest engine
that actually integrates with ML models properly.

Usage:
    uv run scripts/phase_b7_custom_backtest.py --start 2015-01-01 --end 2024-12-31
    uv run scripts/phase_b7_custom_backtest.py --strategies "VWAP Bounce,EMA Ribbon"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.indicators.regime_detector import RegimeDetector
from src.ml.features import FeatureEngineer
from src.ml.pipeline import MLPipeline
from src.patterns.base import BasePattern, PatternResult
from src.strategies.adaptive_router import AdaptiveRouter


@dataclass
class BacktestComparison:
    """Results from comparing ML vs baseline backtests."""

    baseline: Dict[str, Any]
    ml_enhanced: Dict[str, Any]
    metrics_comparison: Dict[str, float]
    regime_stats: Dict[str, Any]


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
    df: pd.DataFrame, patterns: List[BasePattern], name: str = "Baseline"
) -> Dict[str, Any]:
    """Run baseline backtest (no ML enhancement)."""
    config = BacktestConfig(
        initial_equity=1_000_000,
        commission_pct=0.001,
        min_confidence=0.5,
    )

    engine = BacktestEngine(patterns=patterns, config=config)
    result = engine.run(df)

    return {
        "name": name,
        "trades": len(result.trades),
        "win_rate": result.metrics.get("win_rate", 0) * 100,
        "sharpe": result.metrics.get("sharpe_ratio", -999),
        "profit_factor": result.metrics.get("profit_factor", 0),
        "max_drawdown": result.metrics.get("max_drawdown_pct", 0) * 100,
        "return_pct": result.metrics.get("total_return_pct", 0) * 100,
        "avg_trade": result.metrics.get("avg_return_pct", 0) * 100,
    }


def run_ml_enhanced_backtest(
    df: pd.DataFrame, patterns: List[BasePattern], pipeline: MLPipeline, name: str = "ML-Enhanced"
) -> Dict[str, Any]:
    """Run ML-enhanced backtest with regime-aware filtering."""

    detector = RegimeDetector()
    regime_result = detector.get_regime_series(df)
    regime_labels = regime_result["regime"]
    regime_proba = pipeline.get_regime_probabilities(df)

    router = AdaptiveRouter()

    class MLFilteredPattern:
        """Wrapper that filters signals based on ML regime predictions."""

        def __init__(self, wrapped_pattern: BasePattern, router: AdaptiveRouter):
            self.wrapped = wrapped_pattern
            self.name = wrapped_pattern.name
            self.router = router
            self.min_bars_required = wrapped_pattern.min_bars_required

        def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
            result = self.wrapped.detect(df, i)

            if not result.detected or result.signal is None:
                return result

            try:
                current_regime = regime_labels.iloc[i]

                pattern_name = self.wrapped.name
                if not self.router.is_strategy_active(pattern_name, current_regime):
                    result.detected = False
                    return result

                regime_conf = max(regime_proba.iloc[i].values)
                result.signal.confidence = result.signal.confidence * 0.7 + regime_conf * 0.3

            except Exception:
                pass

            return result

    ml_patterns = [MLFilteredPattern(p, router) for p in patterns]

    config = BacktestConfig(
        initial_equity=1_000_000,
        commission_pct=0.001,
        min_confidence=0.4,
    )

    engine = BacktestEngine(patterns=ml_patterns, config=config)
    result = engine.run(df)

    return {
        "name": name,
        "trades": len(result.trades),
        "win_rate": result.metrics.get("win_rate", 0) * 100,
        "sharpe": result.metrics.get("sharpe_ratio", -999),
        "profit_factor": result.metrics.get("profit_factor", 0),
        "max_drawdown": result.metrics.get("max_drawdown_pct", 0) * 100,
        "return_pct": result.metrics.get("total_return_pct", 0) * 100,
        "avg_trade": result.metrics.get("avg_return_pct", 0) * 100,
    }


def load_patterns() -> List[BasePattern]:
    """Load all available pattern detectors."""
    from src.patterns.classic.double_top import DoubleTop
    from src.patterns.classic.double_bottom import DoubleBottom
    from src.patterns.classic.triple_top import TripleTop
    from src.patterns.classic.triple_bottom import TripleBottom
    from src.patterns.classic.ascending_triangle import AscendingTriangle
    from src.patterns.classic.descending_triangle import DescendingTriangle
    from src.patterns.classic.rectangle import Rectangle
    from src.patterns.breakout.donchian import DonchianChannelBreakout
    from src.patterns.breakout.gap import GapPattern
    from src.patterns.candlestick.hammer import Hammer
    from src.patterns.candlestick.engulfing import Engulfing
    from src.patterns.candlestick.doji import Doji
    from src.patterns.basic.two_bar_reversal import TwoBarReversal
    from src.patterns.basic.n_bar_decline import NBarDecline, NBarRally
    from src.patterns.basic.floor_pivot import FloorPivotBreakout

    patterns = [
        DoubleTop(),
        DoubleBottom(),
        TripleTop(),
        TripleBottom(),
        AscendingTriangle(),
        DescendingTriangle(),
        Rectangle(),
        DonchianChannelBreakout(),
        GapPattern(),
        Hammer(),
        Engulfing(),
        Doji(),
        TwoBarReversal(),
        NBarDecline(),
        NBarRally(),
        FloorPivotBreakout(),
    ]

    print(f"Loaded {len(patterns)} pattern detectors")
    return patterns


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
    args = parser.parse_args()

    print("=" * 100)
    print("Phase B7: Full ML-Enhanced Backtest (Custom Engine)")
    print("=" * 100)

    print(f"\nLoading SPY data from {args.start} to {args.end}...")
    df = load_spy_data(args.start, args.end)
    print(f"Loaded {len(df)} bars")

    print("\nLoading pattern detectors...")
    patterns = load_patterns()

    print("\nTraining ML models...")
    pipeline = train_ml_models(df)

    print("\nAnalyzing regime distribution...")
    regime_stats = analyze_regime_performance(df, pipeline)
    print(f"Regime distribution: {regime_stats['regime_distribution_pct']}")

    print("\n" + "=" * 100)
    print("Running BASELINE backtest...")
    print("=" * 100)
    baseline_results = run_baseline_backtest(df, patterns, "Baseline")

    print("\n" + "=" * 100)
    print("Running ML-ENHANCED backtest...")
    print("=" * 100)
    ml_results = run_ml_enhanced_backtest(df, patterns, pipeline, "ML-Enhanced")

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
            "num_patterns": len(patterns),
        },
    }

    report_path = output_dir / "phase_b7_custom_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
