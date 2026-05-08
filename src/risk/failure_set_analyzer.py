# -*- coding: utf-8 -*-
"""
Failure-Set Analyzer - R9 Implementation Stub

Structured framework for detecting strategy failure sets.

From handover document:
- R9: Failure-set analyzers (time-reversal, counter-trend, fat-tail tests)
- P4: Against a Universal Trading Strategy
- Target: 2026-07-31
- Priority: 🟠 High
- Related: Q4, H3

Key concept from P4:
- Every strategy has a failure set (market regimes/conditions where it loses)
- Goal: Map failure sets to avoid catastrophic losses
- Three test types: time-reversal, counter-trend, fat-tail

Usage:
    uv run python src/risk/failure_set_analyzer.py --data data/symbols/*.csv
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats


class FailureTestType(Enum):
    """Types of failure set tests."""

    TIME_REVERSAL = "time_reversal"
    COUNTER_TREND = "counter_trend"
    FAT_TAIL = "fat_tail"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_DROUGHT = "liquidity_drought"


@dataclass
class FailureTestConfig:
    """Configuration for failure set tests."""

    # Test enablement
    enable_time_reversal: bool = True
    enable_counter_trend: bool = True
    enable_fat_tail: bool = True
    enable_volatility_spike: bool = True
    enable_liquidity_drought: bool = True

    # Time-reversal test parameters
    time_reversal_window: int = 20
    time_reversal_threshold: float = -0.02

    # Counter-trend parameters
    counter_trend_lookback: int = 60
    counter_trend_threshold: float = 0.5

    # Fat-tail parameters
    fat_tail_confidence: float = 0.99
    fat_tail_min_skew: float = -1.0

    # Volatility spike parameters
    volatility_spike_multiplier: float = 2.0
    volatility_window: int = 60

    # Liquidity drought parameters
    liquidity_drop_threshold: float = 0.5
    liquidity_window: int = 20


@dataclass
class FailureTestResult:
    """Result for one failure test."""

    test_type: str
    passed: bool
    metric_value: float
    threshold: float
    failure_reason: str = ""


@dataclass
class FailureSetDiagnosis:
    """Complete failure set diagnosis for one strategy/symbol."""

    symbol: str
    strategy_name: str
    overall_status: str
    tests_passed: int
    tests_failed: int
    failure_tests: List[str]
    risk_level: str
    recommendation: str
    test_results: List[Dict]
    metadata: Dict


class FailureSetAnalyzer:
    """
    Detect strategy failure sets using multiple test types.

    Implements R9 from P4: structured failure-set validation.
    """

    def __init__(self, config: Optional[FailureTestConfig] = None):
        """
        Args:
            config: Test configuration
        """
        self.config = config or FailureTestConfig()

    def time_reversal_test(
        self,
        returns: pd.Series,
    ) -> FailureTestResult:
        """
        Test strategy performance under time-reversed data.

        Rationale: If strategy works on reversed data, it may be curve-fit.

        Args:
            returns: Strategy returns series

        Returns:
            FailureTestResult
        """
        reversed_returns = returns[::-1].reset_index(drop=True)

        if len(reversed_returns) < self.config.time_reversal_window:
            return FailureTestResult(
                test_type=FailureTestType.TIME_REVERSAL.value,
                passed=True,
                metric_value=0.0,
                threshold=self.config.time_reversal_threshold,
                failure_reason="Insufficient data",
            )

        reversed_cumulative = (1 + reversed_returns).cumprod()
        recent_performance = (
            reversed_cumulative.iloc[-self.config.time_reversal_window :]
            / reversed_cumulative.iloc[-self.config.time_reversal_window - 1]
            - 1
        )

        perf_value = (
            recent_performance.iloc[-1]
            if hasattr(recent_performance, "iloc")
            else recent_performance
        )
        passed = perf_value >= self.config.time_reversal_threshold

        return FailureTestResult(
            test_type=FailureTestType.TIME_REVERSAL.value,
            passed=passed,
            metric_value=float(perf_value),
            threshold=self.config.time_reversal_threshold,
            failure_reason="" if passed else "Poor performance on reversed data",
        )

    def counter_trend_test(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> FailureTestResult:
        """
        Test strategy performance in counter-trend regimes.

        Rationale: Strategies often fail when market trends reverse.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns

        Returns:
            FailureTestResult
        """
        if len(returns) < self.config.counter_trend_lookback:
            return FailureTestResult(
                test_type=FailureTestType.COUNTER_TREND.value,
                passed=True,
                metric_value=0.0,
                threshold=self.config.counter_trend_threshold,
                failure_reason="Insufficient data",
            )

        rolling_corr = returns.rolling(self.config.counter_trend_lookback).corr(
            benchmark_returns if benchmark_returns is not None else returns
        )

        recent_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0.0

        passed = abs(recent_corr) <= self.config.counter_trend_threshold

        return FailureTestResult(
            test_type=FailureTestType.COUNTER_TREND.value,
            passed=passed,
            metric_value=float(recent_corr),
            threshold=self.config.counter_trend_threshold,
            failure_reason="" if passed else "High correlation with trend",
        )

    def fat_tail_test(
        self,
        returns: pd.Series,
    ) -> FailureTestResult:
        """
        Test for fat-tail risk (negative skewness, excess kurtosis).

        Rationale: Fat tails indicate crash risk.

        Args:
            returns: Returns series

        Returns:
            FailureTestResult
        """
        if len(returns) < 30:
            return FailureTestResult(
                test_type=FailureTestType.FAT_TAIL.value,
                passed=True,
                metric_value=0.0,
                threshold=self.config.fat_tail_min_skew,
                failure_reason="Insufficient data",
            )

        skewness = stats.skew(returns.dropna())
        kurtosis = stats.kurtosis(returns.dropna())

        passed = skewness >= self.config.fat_tail_min_skew

        return FailureTestResult(
            test_type=FailureTestType.FAT_TAIL.value,
            passed=passed,
            metric_value=float(skewness),
            threshold=self.config.fat_tail_min_skew,
            failure_reason=""
            if passed
            else f"Negative skew ({skewness:.2f}), excess kurtosis ({kurtosis:.2f})",
        )

    def volatility_spike_test(
        self,
        returns: pd.Series,
    ) -> FailureTestResult:
        """
        Test for volatility spike risk.

        Args:
            returns: Returns series

        Returns:
            FailureTestResult
        """
        if len(returns) < self.config.volatility_window + 1:
            return FailureTestResult(
                test_type=FailureTestType.VOLATILITY_SPIKE.value,
                passed=True,
                metric_value=0.0,
                threshold=self.config.volatility_spike_multiplier,
                failure_reason="Insufficient data",
            )

        rolling_vol = returns.rolling(self.config.volatility_window).std()
        current_vol = rolling_vol.iloc[-1]
        average_vol = rolling_vol.mean()

        vol_ratio = current_vol / average_vol if average_vol > 0 else 0.0

        passed = vol_ratio <= self.config.volatility_spike_multiplier

        return FailureTestResult(
            test_type=FailureTestType.VOLATILITY_SPIKE.value,
            passed=passed,
            metric_value=float(vol_ratio),
            threshold=self.config.volatility_spike_multiplier,
            failure_reason="" if passed else f"Volatility spike ({vol_ratio:.2f}x average)",
        )

    def diagnose(
        self,
        returns: pd.Series,
        strategy_name: str,
        symbol: str = "UNKNOWN",
        benchmark_returns: Optional[pd.Series] = None,
    ) -> FailureSetDiagnosis:
        """
        Run complete failure-set diagnosis.

        Args:
            returns: Strategy returns
            strategy_name: Name of strategy
            symbol: Stock/symbol identifier
            benchmark_returns: Optional benchmark returns

        Returns:
            FailureSetDiagnosis with all test results
        """
        test_results = []

        if self.config.enable_time_reversal:
            test_results.append(self.time_reversal_test(returns))

        if self.config.enable_counter_trend:
            test_results.append(self.counter_trend_test(returns, benchmark_returns))

        if self.config.enable_fat_tail:
            test_results.append(self.fat_tail_test(returns))

        if self.config.enable_volatility_spike:
            test_results.append(self.volatility_spike_test(returns))

        tests_passed = sum(1 for t in test_results if t.passed)
        tests_failed = len(test_results) - tests_passed

        failure_types = [t.test_type for t in test_results if not t.passed]

        if tests_failed == 0:
            overall_status = "PASS"
            risk_level = "LOW"
            recommendation = "Strategy clear for trading"
        elif tests_failed == 1:
            overall_status = "WARNING"
            risk_level = "MEDIUM"
            recommendation = f"Monitor: Failed {', '.join(failure_types)}"
        else:
            overall_status = "FAIL"
            risk_level = "HIGH"
            recommendation = f"AVOID: Multiple failures ({', '.join(failure_types)})"

        return FailureSetDiagnosis(
            symbol=symbol,
            strategy_name=strategy_name,
            overall_status=overall_status,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            failure_tests=failure_types,
            risk_level=risk_level,
            recommendation=recommendation,
            test_results=[asdict(t) for t in test_results],
            metadata={
                "total_tests": len(test_results),
                "pass_rate": tests_passed / len(test_results) if test_results else 0.0,
            },
        )


class StrategyFailureMonitor:
    """
    Monitor strategies for failure-set activation.

    Implements H3: "Failure-set analyzers prevent cascade losses"
    """

    def __init__(
        self,
        analyzer: Optional[FailureSetAnalyzer] = None,
        check_frequency: int = 20,
    ):
        """
        Args:
            analyzer: Failure-set analyzer
            check_frequency: How often to run checks (in bars)
        """
        self.analyzer = analyzer or FailureSetAnalyzer()
        self.check_frequency = check_frequency

    def check_portfolio(
        self,
        strategy_returns: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, FailureSetDiagnosis]:
        """
        Check all strategies for failure-set activation.

        Args:
            strategy_returns: Dict[strategy_name -> returns]
            benchmark_returns: Optional benchmark returns

        Returns:
            Dict[strategy_name -> diagnosis]
        """
        diagnoses = {}

        for strategy_name, returns in strategy_returns.items():
            diagnosis = self.analyzer.diagnose(
                returns=returns,
                strategy_name=strategy_name,
                symbol=strategy_name,
                benchmark_returns=benchmark_returns,
            )
            diagnoses[strategy_name] = diagnosis

        return diagnoses

    def get_at_risk_strategies(
        self,
        diagnoses: Dict[str, FailureSetDiagnosis],
    ) -> List[str]:
        """
        Get strategies at risk (any failed tests).

        Args:
            diagnoses: Diagnosis results

        Returns:
            List of strategy names at risk
        """
        at_risk = []
        for name, diagnosis in diagnoses.items():
            if diagnosis.tests_failed > 0:
                at_risk.append(name)
        return at_risk


def analyze_strategies(
    data_dir: str,
    output_dir: str,
) -> Dict:
    """
    Analyze strategies for failure-set risks.

    Args:
        data_dir: Directory with strategy returns CSVs
        output_dir: Output directory

    Returns:
        Analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    analyzer = FailureSetAnalyzer()
    results = []

    data_files = list(Path(data_dir).glob("*.csv"))
    print(f"Analyzing {len(data_files)} strategies...")

    for filepath in data_files:
        try:
            returns = pd.read_csv(filepath, index_col=0, parse_dates=True, squeeze=True)
            symbol = filepath.stem

            diagnosis = analyzer.diagnose(
                returns=returns,
                strategy_name=symbol,
                symbol=symbol,
            )
            results.append(asdict(diagnosis))
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    results_df = pd.DataFrame(results)

    # Summary
    pass_count = len(results_df[results_df["overall_status"] == "PASS"])
    warning_count = len(results_df[results_df["overall_status"] == "WARNING"])
    fail_count = len(results_df[results_df["overall_status"] == "FAIL"])

    print("\n=== FAILURE-SET ANALYSIS ===\n")
    print(f"Strategies analyzed: {len(results)}")
    print(f"  PASS: {pass_count} ({pass_count / len(results) * 100:.1f}%)")
    print(f"  WARNING: {warning_count} ({warning_count / len(results) * 100:.1f}%)")
    print(f"  FAIL: {fail_count} ({fail_count / len(results) * 100:.1f}%)")

    if fail_count > 0:
        print("\nFailed strategies:")
        failed = results_df[results_df["overall_status"] == "FAIL"]
        for _, row in failed.iterrows():
            print(f"  {row['symbol']}: {', '.join(row['failure_tests'])}")

    # Save results
    results_df.to_csv(os.path.join(output_dir, "failure_set_diagnosis.csv"), index=False)

    summary = {
        "total_strategies": len(results),
        "pass_count": pass_count,
        "warning_count": warning_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / len(results) if results else 0.0,
    }

    json_path = os.path.join(output_dir, "failure_set_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Failure-Set Analyzer (R9)")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with strategy returns CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/failure_set",
        help="Output directory",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    summary = analyze_strategies(args.data_dir, args.output_dir)

    print("\n✅ R9 Failure-set analysis complete!")
    print(f"\nKey Finding: {summary['fail_count']} strategies in failure zone")


if __name__ == "__main__":
    main()
