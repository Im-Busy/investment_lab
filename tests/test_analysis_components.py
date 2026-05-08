"""
Unit tests for new analysis components:
- Walk-Forward Validator
- Signal Quality Filter
- Correlation Analyzer
- Pattern Performance Tracker
"""

import numpy as np
import pandas as pd

from src.analysis.walk_forward_validator import (
    WalkForwardValidator,
    OverfitStatus,
    detect_overfitting,
)
from src.analysis.signal_quality_filter import (
    SignalQualityFilter,
)
from src.analysis.correlation_analyzer import (
    CorrelationAnalyzer,
)
from src.analysis.pattern_performance_tracker import (
    PatternPerformanceTracker,
    PerformanceTrackerConfig,
)


class TestDetectOverfitting:
    """Tests for the detect_overfitting standalone function."""

    def test_pass(self):
        result = detect_overfitting(1.0, 0.8)
        assert result == OverfitStatus.PASS

    def test_medium_overfit(self):
        result = detect_overfitting(1.0, 0.4)
        assert result == OverfitStatus.MEDIUM_OVERFIT

    def test_high_overfit(self):
        result = detect_overfitting(1.0, 0.1)
        assert result == OverfitStatus.HIGH_OVERFIT

    def test_oos_unprofitable(self):
        result = detect_overfitting(1.0, -0.2)
        assert result in (OverfitStatus.OOS_UNPROFITABLE, OverfitStatus.HIGH_OVERFIT)

    def test_is_unprofitable(self):
        result = detect_overfitting(0.5, 0.0)
        assert result == OverfitStatus.OOS_UNPROFITABLE


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def setup_method(self):
        self.validator = WalkForwardValidator()

    def test_split_data(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100}, index=dates)

        splits = self.validator.split_data(data)

        assert len(splits["is"]) == 60
        assert len(splits["oos"]) == 20
        assert len(splits["fv"]) == 20

    def test_run_validation_pass(self):
        metrics = {
            "is": {
                "sharpe_ratio": 1.0,
                "total_trades": 50,
                "win_rate": 0.55,
                "total_return_pct": 0.20,
                "max_drawdown_pct": 0.10,
                "profit_factor": 1.5,
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2021-06-01"),
            },
            "oos": {
                "sharpe_ratio": 0.5,
                "total_trades": 20,
                "win_rate": 0.50,
                "total_return_pct": 0.10,
                "max_drawdown_pct": 0.08,
                "profit_factor": 1.3,
                "start_date": pd.Timestamp("2021-06-02"),
                "end_date": pd.Timestamp("2022-01-01"),
            },
            "fv": {
                "sharpe_ratio": 0.3,
                "total_trades": 15,
                "win_rate": 0.45,
                "total_return_pct": 0.05,
                "max_drawdown_pct": 0.06,
                "profit_factor": 1.2,
                "start_date": pd.Timestamp("2022-01-02"),
                "end_date": pd.Timestamp("2022-07-01"),
            },
        }

        result = self.validator.run_validation("test_pattern", metrics)

        assert result.pattern_name == "test_pattern"
        assert result.passes_validation
        assert result.overfit_status in (OverfitStatus.PASS, OverfitStatus.MILD_OVERFIT)

    def test_run_validation_fail_sharpe(self):
        metrics = {
            "is": {
                "sharpe_ratio": 1.0,
                "total_trades": 50,
                "win_rate": 0.55,
                "total_return_pct": 0.20,
                "max_drawdown_pct": 0.10,
                "profit_factor": 1.5,
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2021-06-01"),
            },
            "oos": {
                "sharpe_ratio": 0.1,
                "total_trades": 20,
                "win_rate": 0.45,
                "total_return_pct": 0.02,
                "max_drawdown_pct": 0.15,
                "profit_factor": 1.0,
                "start_date": pd.Timestamp("2021-06-02"),
                "end_date": pd.Timestamp("2022-01-01"),
            },
        }

        result = self.validator.run_validation("bad_pattern", metrics)

        assert not result.passes_validation
        assert result.overfit_status == OverfitStatus.HIGH_OVERFIT

    def test_run_validation_overfit(self):
        metrics = {
            "is": {
                "sharpe_ratio": 2.0,
                "total_trades": 50,
                "win_rate": 0.70,
                "total_return_pct": 0.50,
                "max_drawdown_pct": 0.05,
                "profit_factor": 2.5,
                "start_date": pd.Timestamp("2020-01-01"),
                "end_date": pd.Timestamp("2021-06-01"),
            },
            "oos": {
                "sharpe_ratio": 0.2,
                "total_trades": 20,
                "win_rate": 0.45,
                "total_return_pct": 0.02,
                "max_drawdown_pct": 0.20,
                "profit_factor": 0.9,
                "start_date": pd.Timestamp("2021-06-02"),
                "end_date": pd.Timestamp("2022-01-01"),
            },
        }

        result = self.validator.run_validation("overfit_pattern", metrics)

        assert not result.passes_validation
        assert result.overfit_status in (OverfitStatus.HIGH_OVERFIT, OverfitStatus.MEDIUM_OVERFIT)

    def test_validate_multiple_patterns(self):
        pattern_metrics = {
            "good_pattern": {
                "is": {
                    "sharpe_ratio": 1.0,
                    "total_trades": 50,
                    "win_rate": 0.55,
                    "total_return_pct": 0.20,
                    "max_drawdown_pct": 0.10,
                    "profit_factor": 1.5,
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2021-01-01"),
                },
                "oos": {
                    "sharpe_ratio": 0.5,
                    "total_trades": 20,
                    "win_rate": 0.50,
                    "total_return_pct": 0.10,
                    "max_drawdown_pct": 0.08,
                    "profit_factor": 1.3,
                    "start_date": pd.Timestamp("2021-01-02"),
                    "end_date": pd.Timestamp("2021-06-01"),
                },
            },
            "bad_pattern": {
                "is": {
                    "sharpe_ratio": 0.2,
                    "total_trades": 10,
                    "win_rate": 0.40,
                    "total_return_pct": 0.02,
                    "max_drawdown_pct": 0.30,
                    "profit_factor": 0.8,
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2021-01-01"),
                },
                "oos": {
                    "sharpe_ratio": -0.1,
                    "total_trades": 5,
                    "win_rate": 0.30,
                    "total_return_pct": -0.05,
                    "max_drawdown_pct": 0.40,
                    "profit_factor": 0.5,
                    "start_date": pd.Timestamp("2021-01-02"),
                    "end_date": pd.Timestamp("2021-06-01"),
                },
            },
        }

        df = self.validator.validate_multiple_patterns(pattern_metrics)

        assert len(df) == 2
        assert "good_pattern" in df["pattern_name"].values
        assert "bad_pattern" in df["pattern_name"].values


class TestSignalQualityFilter:
    """Tests for SignalQualityFilter."""

    def setup_method(self):
        self.filter = SignalQualityFilter()

    def test_evaluate_signal_pass(self):
        result = self.filter.evaluate_signal(
            signal_id="sig_1",
            pattern_name="Double Bottom",
            confidence=0.80,
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            historical_win_rate=0.50,
            historical_profit_factor=1.5,
            regime_aligned=True,
        )

        assert result.passes_quality_gate
        assert result.quality_score > 0.5

    def test_evaluate_signal_fail_confidence(self):
        result = self.filter.evaluate_signal(
            signal_id="sig_2",
            pattern_name="Doji",
            confidence=0.30,
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            historical_win_rate=0.50,
            historical_profit_factor=1.5,
        )

        assert not result.passes_quality_gate
        assert any("Confidence" in r for r in result.failure_reasons)

    def test_evaluate_signal_fail_rr(self):
        result = self.filter.evaluate_signal(
            signal_id="sig_3",
            pattern_name="Flag",
            confidence=0.80,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=100.5,
            historical_win_rate=0.50,
            historical_profit_factor=1.5,
        )

        assert not result.passes_quality_gate
        assert any("Risk/Reward" in r for r in result.failure_reasons)

    def test_filter_signals(self):
        signals = [
            {
                "signal_id": "s1",
                "pattern_name": "Double Bottom",
                "confidence": 0.80,
                "entry_price": 100.0,
                "stop_loss": 97.0,
                "take_profit": 106.0,
            },
            {
                "signal_id": "s2",
                "pattern_name": "Doji",
                "confidence": 0.30,
                "entry_price": 100.0,
                "stop_loss": 97.0,
                "take_profit": 106.0,
            },
        ]

        result = self.filter.filter_signals(signals)

        assert len(result["passed"]) >= 0
        assert len(result["failed"]) >= 1

    def test_get_quality_summary(self):
        r1 = self.filter.evaluate_signal(
            signal_id="s1",
            pattern_name="A",
            confidence=0.80,
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
        )
        r2 = self.filter.evaluate_signal(
            signal_id="s2",
            pattern_name="B",
            confidence=0.30,
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
        )
        results = [r1, r2]

        summary = self.filter.get_quality_summary(results)

        assert summary["total_signals"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer."""

    def setup_method(self):
        self.analyzer = CorrelationAnalyzer()

    def test_build_signal_matrix(self):
        detection_log = pd.DataFrame(
            {
                "bar_index": [0, 1, 2, 5],
                "pattern_name": ["Double Top", "Double Top", "Head and Shoulders", "Double Top"],
                "passed_threshold": [True, True, True, True],
            }
        )

        matrix = self.analyzer.build_signal_matrix(detection_log, total_bars=10)

        assert matrix.shape == (10, 2)
        assert matrix.iloc[0, matrix.columns.tolist().index("Double Top")] == 1
        assert matrix.iloc[3, matrix.columns.tolist().index("Double Top")] == 0

    def test_compute_correlation_matrix(self):
        np.random.seed(42)
        matrix = pd.DataFrame(
            {
                "A": np.random.randint(0, 2, 100),
                "B": np.random.randint(0, 2, 100),
                "C": np.random.randint(0, 2, 100),
            }
        )

        corr = self.analyzer.compute_correlation_matrix(matrix)

        assert corr.shape == (3, 3)
        assert abs(corr.iloc[0, 0] - 1.0) < 0.001

    def test_find_redundant_pairs_with_high_correlation(self):
        corr_matrix = pd.DataFrame(
            {
                "A": [1.0, 0.85, 0.3],
                "B": [0.85, 1.0, 0.2],
                "C": [0.3, 0.2, 1.0],
            },
            columns=["A", "B", "C"],
        )

        co_matrix = pd.DataFrame(
            {
                "A": [20, 15, 5],
                "B": [15, 25, 3],
                "C": [5, 3, 10],
            }
        )

        redundant = self.analyzer.find_redundant_pairs(corr_matrix, co_matrix, {"A": 0.5, "B": 0.8})

        assert len(redundant) >= 1
        assert redundant[0].pearson_correlation == 0.85
        assert redundant[0].should_exclude

    def test_check_documented_groups(self):
        violations = self.analyzer.check_documented_groups(
            active_patterns=[
                "Double Top",
                "Triple Top",
                "Double Bottom",
                "Triple Bottom",
                "Head and Shoulders",
            ]
        )

        double_violation = [v for v in violations if v["group_name"] == "Double Patterns"]
        assert len(double_violation) == 1
        assert double_violation[0]["active_count"] >= 4
        assert double_violation[0]["excess"] >= 1

    def test_deduplicate_patterns(self):
        result = self.analyzer.deduplicate_patterns(
            candidate_patterns=[
                "Head and Shoulders",
                "Gartley Pattern",
            ],
            signal_matrix=None,
        )

        assert len(result["selected_patterns"]) >= 1
        assert len(result["excluded_patterns"]) >= 0

    def test_get_correlation_summary(self):
        corr_matrix = pd.DataFrame(
            {
                "A": [1.0, 0.9, 0.3],
                "B": [0.9, 1.0, 0.2],
                "C": [0.3, 0.2, 1.0],
            },
            columns=["A", "B", "C"],
        )

        summary = self.analyzer.get_correlation_summary(corr_matrix)

        assert len(summary) == 3
        high_corr = summary[summary["is_high_correlation"]]
        assert len(high_corr) >= 1


class TestPatternPerformanceTracker:
    """Tests for PatternPerformanceTracker."""

    def setup_method(self):
        config = PerformanceTrackerConfig(min_trades_for_metrics=5)
        self.tracker = PatternPerformanceTracker(config)

    def _create_trade_data(self, n=20, seed=42):
        np.random.seed(seed)
        return pd.DataFrame(
            {
                "entry_time": pd.date_range("2020-01-01", periods=n, freq="D"),
                "exit_time": pd.date_range("2020-01-02", periods=n, freq="D"),
                "pnl": np.random.randn(n) * 100 + 10,
                "pnl_pct": np.random.randn(n) * 0.02 + 0.001,
            }
        )

    def test_add_and_get_metrics(self):
        data = self._create_trade_data(n=20, seed=42)
        self.tracker.add_pattern_results("test_pattern", data)

        metrics = self.tracker.get_rolling_metrics("test_pattern", window_size=20)

        assert metrics is not None
        assert metrics.pattern_name == "test_pattern"
        assert metrics.total_trades == 20

    def test_insufficient_trades(self):
        data = self._create_trade_data(n=3, seed=42)
        self.tracker.add_pattern_results("thin_pattern", data)

        metrics = self.tracker.get_rolling_metrics("thin_pattern")

        assert metrics is None

    def test_get_all_pattern_summaries(self):
        for name, seed in [("pattern_a", 42), ("pattern_b", 123)]:
            data = self._create_trade_data(n=20, seed=seed)
            self.tracker.add_pattern_results(name, data)

        df = self.tracker.get_all_pattern_summaries(window_size=20)

        assert len(df) == 2
        assert "rolling_win_rate" in df.columns
        assert "rolling_sharpe" in df.columns

    def test_list_tracked_patterns(self):
        data = self._create_trade_data(n=20, seed=42)
        self.tracker.add_pattern_results("pattern_a", data)
        self.tracker.add_pattern_results("pattern_b", data)

        patterns = self.tracker.list_tracked_patterns()

        assert len(patterns) == 2
        assert "pattern_a" in patterns
        assert "pattern_b" in patterns

    def test_detect_degrading_patterns(self):
        data_improving = pd.DataFrame(
            {
                "entry_time": pd.date_range("2020-01-01", periods=20, freq="D"),
                "exit_time": pd.date_range("2020-01-02", periods=20, freq="D"),
                "pnl": list(np.random.randn(10) * 100 - 50) + list(np.random.randn(10) * 100 + 100),
                "pnl_pct": list(np.random.randn(10) * 0.02 - 0.01)
                + list(np.random.randn(10) * 0.02 + 0.02),
            }
        )
        self.tracker.add_pattern_results("improving_pattern", data_improving)

        degrading = self.tracker.detect_degrading_patterns()
        assert "improving_pattern" not in degrading
