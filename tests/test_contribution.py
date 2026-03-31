"""
Unit tests for Contribution Analysis System

Tests for all analysis classes:
- SignalEventLog (Layer 1)
- TradeAttributor (Layer 2)
- AblationEngine (Layer 3)
- SynergyAnalyzer (Layer 4)
- ContributionReport (Aggregation)
"""

import tempfile
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    AblationEngine,
    AblationResult,
    ContributionReport,
    SignalEvent,
    SignalEventLog,
    SynergyResult,
    TradeAttributor,
)


class TestSignalEventLog:
    """Tests for SignalEventLog (Layer 1)."""

    def test_signal_event_creation(self):
        """Test SignalEvent dataclass creation."""
        event = SignalEvent(
            bar_index=100,
            timestamp=pd.Timestamp("2020-01-01"),
            pattern_name="DoubleTop",
            pattern_category="classic",
            direction="Long",
            confidence=0.75,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit_1=101.0,
            pattern_type="Reversal",
        )

        assert event.bar_index == 100
        assert event.pattern_name == "DoubleTop"
        assert event.direction == "Long"
        assert event.confidence == 0.75
        assert event.led_to_trade is False

    def test_signal_event_log_recording(self):
        """Test that all detections are logged, not just threshold-passing."""
        log = SignalEventLog()

        # Record detections
        detections = [
            {
                "pattern_name": "DoubleTop",
                "pattern_category": "classic",
                "direction": "Long",
                "confidence": 0.75,
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit_1": 101.0,
                "pattern_type": "Reversal",
            },
            {
                "pattern_name": "MSL",
                "pattern_category": "basic",
                "direction": "Long",
                "confidence": 0.65,
                "entry_price": 100.0,
                "stop_loss": 99.5,
                "take_profit_1": 101.5,
                "pattern_type": "Reversal",
            },
        ]

        log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=detections,
            active_patterns=None,
            confluence_count=2,
            passed_threshold=False,
        )

        # Verify all detections are recorded
        assert len(log) == 2
        events = log.get_detections_for_bar(0)
        assert len(events) == 2
        assert events[0].pattern_name == "DoubleTop"
        assert events[1].pattern_name == "MSL"

    def test_signal_event_log_to_dataframe(self):
        """Test DataFrame conversion preserves all fields."""
        log = SignalEventLog()

        # Add events
        for i in range(5):
            log.record_bar_detections(
                bar_index=i,
                timestamp=pd.Timestamp("2020-01-01") + timedelta(days=i),
                all_detections=[
                    {
                        "pattern_name": f"Pattern{i}",
                        "pattern_category": "basic",
                        "direction": "Long",
                        "confidence": 0.7,
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit_1": 101.0,
                        "pattern_type": "Reversal",
                    }
                ],
            )

        df = log.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "bar_index" in df.columns
        assert "pattern_name" in df.columns
        assert "confidence" in df.columns

    def test_signal_event_log_detection_frequency(self):
        """Test detection frequency calculation."""
        log = SignalEventLog()

        # Add events for same pattern multiple times
        for i in range(10):
            log.record_bar_detections(
                bar_index=i,
                timestamp=pd.Timestamp("2020-01-01") + timedelta(days=i),
                all_detections=[
                    {
                        "pattern_name": "DoubleTop",
                        "pattern_category": "classic",
                        "direction": "Long",
                        "confidence": 0.7,
                        "entry_price": 100.0,
                        "stop_loss": 99.0,
                        "take_profit_1": 101.0,
                        "pattern_type": "Reversal",
                    }
                ],
            )

        freq = log.get_detection_frequency()

        assert not freq.empty
        assert freq.iloc[0]["pattern_name"] == "DoubleTop"
        assert freq.iloc[0]["detection_count"] == 10
        assert freq.iloc[0]["detection_frequency_pct"] == 100.0

    def test_signal_event_log_co_occurrence_matrix(self):
        """Test co-occurrence matrix calculation."""
        log = SignalEventLog()

        # Add events where patterns co-occur
        log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=[
                {
                    "pattern_name": "DoubleTop",
                    "pattern_category": "classic",
                    "direction": "Long",
                    "confidence": 0.7,
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit_1": 101.0,
                    "pattern_type": "Reversal",
                },
                {
                    "pattern_name": "MSL",
                    "pattern_category": "basic",
                    "direction": "Long",
                    "confidence": 0.65,
                    "entry_price": 100.0,
                    "stop_loss": 99.5,
                    "take_profit_1": 101.5,
                    "pattern_type": "Reversal",
                },
            ],
        )

        matrix = log.get_co_occurrence_matrix()

        assert not matrix.empty
        assert matrix.loc["DoubleTop", "MSL"] == 1
        assert matrix.loc["MSL", "DoubleTop"] == 1
        assert matrix.loc["DoubleTop", "DoubleTop"] == 1

    def test_signal_event_log_update_bar_passed_threshold(self):
        """Test updating bar metadata after trade execution."""
        log = SignalEventLog()

        log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=[
                {
                    "pattern_name": "DoubleTop",
                    "pattern_category": "classic",
                    "direction": "Long",
                    "confidence": 0.7,
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit_1": 101.0,
                    "pattern_type": "Reversal",
                }
            ],
        )

        # Update threshold
        log.update_bar_passed_threshold(
            bar_index=0,
            active_patterns=["DoubleTop"],
            confluence_count=1,
        )

        metadata = log.get_bar_metadata(0)
        assert metadata["passed_threshold"] is True
        assert metadata["active_patterns"] == ["DoubleTop"]

    def test_signal_event_log_clear(self):
        """Test clearing all events."""
        log = SignalEventLog()

        log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=[
                {
                    "pattern_name": "DoubleTop",
                    "pattern_category": "classic",
                    "direction": "Long",
                    "confidence": 0.7,
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit_1": 101.0,
                    "pattern_type": "Reversal",
                }
            ],
        )

        assert len(log) == 1

        log.clear()

        assert len(log) == 0


class TestTradeAttributor:
    """Tests for TradeAttributor (Layer 2)."""

    def _create_test_data(self):
        """Create test signal log and trades DataFrame."""
        # Create signal log
        signal_log = SignalEventLog()

        # Add detections for 3 bars
        for i in range(3):
            signal_log.record_bar_detections(
                bar_index=i,
                timestamp=pd.Timestamp("2020-01-01") + timedelta(days=i),
                all_detections=[
                    {
                        "pattern_name": "DoubleTop",
                        "pattern_category": "classic",
                        "direction": "Long",
                        "confidence": 0.7,
                        "entry_price": 100.0 + i,
                        "stop_loss": 99.0 + i,
                        "take_profit_1": 101.0 + i,
                        "pattern_type": "Reversal",
                    }
                ],
            )

            # Mark first bar as passed threshold
            if i == 0:
                signal_log.update_bar_passed_threshold(
                    bar_index=i,
                    active_patterns=["DoubleTop"],
                    confluence_count=1,
                )

        # Create trades DataFrame
        trades_df = pd.DataFrame(
            {
                "entry_time": [pd.Timestamp("2020-01-01")],
                "exit_time": [pd.Timestamp("2020-01-05")],
                "entry_price": [100.0],
                "exit_price": [102.0],
                "size": [100],
                "pl": [200.0],
                "pl_pct": [0.02],
                "is_long": [True],
            }
        )

        return signal_log, trades_df

    def test_trade_attributor_matching(self):
        """Test trades matched to correct signal events."""
        signal_log, trades_df = self._create_test_data()

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributed = attributor.attribute_trades()

        assert len(attributed) == 1
        assert attributed[0].contributing_patterns == ["DoubleTop"]
        assert attributed[0].confluence_count == 1

    def test_trade_attributor_unattributed(self):
        """Test unmatched trades handled gracefully."""
        signal_log = SignalEventLog()

        # Create trades with no matching signals
        trades_df = pd.DataFrame(
            {
                "entry_time": [pd.Timestamp("2020-01-10")],
                "exit_time": [pd.Timestamp("2020-01-15")],
                "entry_price": [100.0],
                "exit_price": [102.0],
                "size": [100],
                "pl": [200.0],
                "pl_pct": [0.02],
                "is_long": [True],
            }
        )

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributed = attributor.attribute_trades()

        assert len(attributed) == 1
        assert attributed[0].contributing_patterns == []

    def test_trade_attributor_pattern_stats(self):
        """Test pattern trade statistics calculation."""
        signal_log, trades_df = self._create_test_data()

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributor.attribute_trades()

        stats = attributor.get_pattern_trade_stats()

        assert not stats.empty
        assert stats.iloc[0]["pattern_name"] == "DoubleTop"
        assert stats.iloc[0]["trade_count"] == 1
        assert stats.iloc[0]["win_count"] == 1

    def test_trade_attributor_confluence_vs_performance(self):
        """Test confluence vs performance grouping."""
        signal_log, trades_df = self._create_test_data()

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributor.attribute_trades()

        confluence_perf = attributor.get_confluence_vs_performance()

        assert not confluence_perf.empty


class TestAblationEngine:
    """Tests for AblationEngine (Layer 3)."""

    def test_ablation_result_creation(self):
        """Test AblationResult dataclass creation."""
        result = AblationResult(
            excluded_pattern="DoubleTop",
            total_trades=10,
            win_rate=0.6,
            total_return_pct=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown_pct=-0.1,
            profit_factor=2.0,
            equity_final=115000,
            duration_seconds=10.0,
        )

        assert result.excluded_pattern == "DoubleTop"
        assert result.total_trades == 10
        assert result.win_rate == 0.6

    def test_ablation_exclude_patterns(self):
        """Test pattern exclusion actually removes detections."""
        # Create a simple DataFrame for testing
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "Open": np.random.randn(100).cumsum() + 100,
                "High": np.random.randn(100).cumsum() + 101,
                "Low": np.random.randn(100).cumsum() + 99,
                "Close": np.random.randn(100).cumsum() + 100,
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        # Create engine
        engine = AblationEngine(
            data=df,
            cash=100000,
            commission=0.001,
            output_dir=tempfile.mkdtemp(),
        )

        # Verify patterns are loaded
        assert len(engine.all_patterns) > 0

    def test_ablation_caching(self):
        """Test cache save/load round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AblationEngine(
                data=pd.DataFrame(),
                cash=100000,
                commission=0.001,
                output_dir=tmpdir,
            )

            # Create mock results
            engine.baseline_result = {
                "total_trades": 10,
                "win_rate": 0.6,
                "total_return_pct": 0.15,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "max_drawdown_pct": -0.1,
                "profit_factor": 2.0,
                "equity_final": 115000,
                "duration_seconds": 10.0,
            }

            engine.ablation_results = [
                AblationResult(
                    excluded_pattern="DoubleTop",
                    total_trades=9,
                    win_rate=0.55,
                    total_return_pct=0.12,
                    sharpe_ratio=1.3,
                    sortino_ratio=1.8,
                    max_drawdown_pct=-0.12,
                    profit_factor=1.8,
                    equity_final=112000,
                    duration_seconds=9.0,
                )
            ]

            # Save
            engine.save_results()

            # Load
            new_engine = AblationEngine(
                data=pd.DataFrame(),
                cash=100000,
                commission=0.001,
                output_dir=tmpdir,
            )

            success = new_engine.load_results()

            assert success
            assert new_engine.baseline_result is not None
            assert len(new_engine.ablation_results) == 1


class TestSynergyAnalyzer:
    """Tests for SynergyAnalyzer (Layer 4)."""

    def test_synergy_result_creation(self):
        """Test SynergyResult dataclass creation."""
        result = SynergyResult(
            pattern_a="DoubleTop",
            pattern_b="MSL",
            solo_a_return=0.10,
            solo_b_return=0.08,
            pair_return=0.25,
            expected_return=0.18,
            synergy_score=0.07,
            solo_a_sharpe=1.2,
            solo_b_sharpe=1.0,
            pair_sharpe=1.8,
            synergy_sharpe=0.6,
            co_occurrence_count=50,
            co_trade_count=10,
            co_trade_win_rate=0.7,
        )

        assert result.pattern_a == "DoubleTop"
        assert result.synergy_score == 0.07
        assert result.synergy_sharpe == 0.6

    def test_synergy_score_calculation(self):
        """Test synergy = pair - soloA - soloB."""
        # Synergy score should be pair_return - (solo_a_return + solo_b_return)
        solo_a = 0.10
        solo_b = 0.08
        pair = 0.25

        expected_synergy = pair - (solo_a + solo_b)
        assert expected_synergy == 0.07


class TestContributionReport:
    """Tests for ContributionReport (Aggregation)."""

    def test_contribution_report_roles(self):
        """Test role classification matches expected rules."""
        # Create mock data
        signal_log = SignalEventLog()
        signal_log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=[
                {
                    "pattern_name": "DoubleTop",
                    "pattern_category": "classic",
                    "direction": "Long",
                    "confidence": 0.7,
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit_1": 101.0,
                    "pattern_type": "Reversal",
                }
            ],
        )
        # Mark bar as passed threshold so attribution works
        signal_log.update_bar_passed_threshold(
            bar_index=0,
            active_patterns=["DoubleTop"],
            confluence_count=1,
        )

        trades_df = pd.DataFrame(
            {
                "entry_time": [pd.Timestamp("2020-01-01")],
                "exit_time": [pd.Timestamp("2020-01-05")],
                "entry_price": [100.0],
                "exit_price": [102.0],
                "size": [100],
                "pl": [200.0],
                "pl_pct": [0.02],
                "is_long": [True],
            }
        )

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributor.attribute_trades()

        # Create ablation results
        ablation_results = pd.DataFrame(
            {
                "pattern_name": ["DoubleTop", "MSL"],
                "delta_sharpe": [0.15, -0.05],
                "delta_return": [0.10, -0.03],
            }
        )

        report = ContributionReport(
            signal_log=signal_log,
            attributor=attributor,
            ablation_results=ablation_results,
        )

        roles = report.get_pattern_roles()

        # DoubleTop should be Primary Signal (high delta_sharpe)
        assert roles.get("DoubleTop") == "Primary Signal"

    def test_contribution_report_recommendations(self):
        """Test recommendations generation."""
        signal_log = SignalEventLog()
        signal_log.record_bar_detections(
            bar_index=0,
            timestamp=pd.Timestamp("2020-01-01"),
            all_detections=[
                {
                    "pattern_name": "DoubleTop",
                    "pattern_category": "classic",
                    "direction": "Long",
                    "confidence": 0.7,
                    "entry_price": 100.0,
                    "stop_loss": 99.0,
                    "take_profit_1": 101.0,
                    "pattern_type": "Reversal",
                }
            ],
        )

        trades_df = pd.DataFrame(
            {
                "entry_time": [pd.Timestamp("2020-01-01")],
                "exit_time": [pd.Timestamp("2020-01-05")],
                "entry_price": [100.0],
                "exit_price": [102.0],
                "size": [100],
                "pl": [200.0],
                "pl_pct": [0.02],
                "is_long": [True],
            }
        )

        attributor = TradeAttributor(
            signal_log=signal_log,
            trades_df=trades_df,
            tolerance_bars=1,
        )

        attributor.attribute_trades()

        ablation_results = pd.DataFrame(
            {
                "pattern_name": ["DoubleTop", "MSL"],
                "delta_sharpe": [0.15, -0.05],
                "delta_return": [0.10, -0.03],
            }
        )

        report = ContributionReport(
            signal_log=signal_log,
            attributor=attributor,
            ablation_results=ablation_results,
        )

        recommendations = report.get_recommendations()

        assert isinstance(recommendations, list)
        # Should have at least one recommendation
        assert len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
