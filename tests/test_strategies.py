"""
Unit Tests for Strategy Components

Tests the confluence scoring and configuration systems.
"""

import pytest

from src.config import (
    BacktestConfig,
    PatternConfig,
    RegimeConfig,
    RiskConfig,
    SignalConfig,
    TradingConfig,
)
from src.indicators.regime import MarketPhase, RegimeState, TrendDirection, VolatilityRegime
from src.patterns.base import PatternResult, PatternType, SignalDirection, TradeSignal
from src.strategies.confluence import ConfluenceScore, ConfluenceScorer, PatternCompatibility


class TestConfluenceScore:
    """Tests for ConfluenceScore dataclass."""

    def test_create_confluence_score(self):
        """Test creating a confluence score."""
        score = ConfluenceScore(
            score=0.75,
            direction=SignalDirection.LONG,
            confidence_level=3,
            pattern_count=3,
            patterns=["Double Bottom", "MSL", "Matching Lows"],
            regime_alignment=True,
            risk_reward_ratio=2.0,
            quality_score=0.8,
        )

        assert score.score == 0.75
        assert score.direction == SignalDirection.LONG
        assert score.confidence_level == 3
        assert len(score.patterns) == 3

    def test_confluence_score_to_dict(self):
        """Test converting confluence score to dict."""
        score = ConfluenceScore(
            score=0.8,
            direction=SignalDirection.SHORT,
            confidence_level=2,
            pattern_count=2,
            patterns=["Double Top", "Head and Shoulders"],
            regime_alignment=False,
            risk_reward_ratio=1.5,
            quality_score=0.7,
        )

        result = score.to_dict()

        assert isinstance(result, dict)
        assert result["score"] == 0.8
        assert result["direction"] == "Short"
        assert result["confidence_level"] == 2


class TestPatternCompatibility:
    """Tests for PatternCompatibility dataclass."""

    def test_compatible_patterns(self):
        """Test compatible pattern pair."""
        compat = PatternCompatibility(
            pattern_a="Double Bottom", pattern_b="MSL", compatible=True, reason="Same direction"
        )

        assert compat.compatible is True

    def test_incompatible_patterns(self):
        """Test incompatible pattern pair."""
        compat = PatternCompatibility(
            pattern_a="Double Top",
            pattern_b="Double Bottom",
            compatible=False,
            reason="Conflicting direction",
        )

        assert compat.compatible is False


class TestConfluenceScorer:
    """Tests for ConfluenceScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a confluence scorer."""
        return ConfluenceScorer()

    @pytest.fixture
    def sample_results(self):
        """Create sample pattern results."""
        signal1 = TradeSignal(
            pattern_name="Double Bottom",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.65,
        )
        signal2 = TradeSignal(
            pattern_name="Market Structure Low",
            direction=SignalDirection.LONG,
            entry_price=100.5,
            stop_loss=95.5,
            take_profit_1=111.0,
            confidence=0.55,
        )

        return [
            PatternResult(
                detected=True,
                pattern_name="Double Bottom",
                pattern_type=PatternType.REVERSAL,
                signal=signal1,
                pivot_points={"low": 95.0},
            ),
            PatternResult(
                detected=True,
                pattern_name="Market Structure Low",
                pattern_type=PatternType.REVERSAL,
                signal=signal2,
                pivot_points={"low": 96.0},
            ),
        ]

    @pytest.fixture
    def sample_regime(self):
        """Create sample regime state."""
        return RegimeState(
            trend_direction=TrendDirection.UPTREND,
            trend_strength=30.0,
            volatility_regime=VolatilityRegime.NORMAL,
            volatility_value=2.5,
            market_phase=MarketPhase.WEAK_BULL,
            sma_position="above",
            adx=30.0,
            atr=2.5,
            atr_ratio=1.0,
        )

    def test_scorer_initialization(self, scorer):
        """Test scorer initialization."""
        assert scorer.min_confidence == 0.50
        assert scorer.trend_alignment_bonus == 0.05
        assert scorer.regime_adaptation is True

    def test_calculate_confluence_basic(self, scorer, sample_results):
        """Test basic confluence calculation."""
        result = scorer.calculate_confluence(sample_results)

        assert result is not None
        assert isinstance(result, ConfluenceScore)
        assert result.direction == SignalDirection.LONG
        assert result.pattern_count == 2
        assert result.confidence_level >= 2

    def test_calculate_confluence_with_regime(self, scorer, sample_results, sample_regime):
        """Test confluence calculation with regime."""
        result = scorer.calculate_confluence(sample_results, sample_regime)

        assert result is not None
        assert result.regime_alignment is True  # LONG in UPTREND

    def test_calculate_confluence_empty_results(self, scorer):
        """Test confluence with empty results."""
        result = scorer.calculate_confluence([])

        assert result is None

    def test_calculate_confluence_no_signals(self, scorer):
        """Test confluence with results without signals."""
        results = [
            PatternResult(detected=False, pattern_name="Test", pattern_type=PatternType.REVERSAL)
        ]

        result = scorer.calculate_confluence(results)

        assert result is None

    def test_conflicting_directions(self, scorer):
        """Test handling of conflicting directions."""
        signal1 = TradeSignal(
            pattern_name="Double Bottom",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.65,
        )
        signal2 = TradeSignal(
            pattern_name="Double Top",
            direction=SignalDirection.SHORT,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit_1=90.0,
            confidence=0.60,
        )

        results = [
            PatternResult(
                detected=True,
                pattern_name="Double Bottom",
                pattern_type=PatternType.REVERSAL,
                signal=signal1,
            ),
            PatternResult(
                detected=True,
                pattern_name="Double Top",
                pattern_type=PatternType.REVERSAL,
                signal=signal2,
            ),
        ]

        result = scorer.calculate_confluence(results)

        # Should pick the direction with more patterns or higher confidence
        assert result is not None
        assert result.direction in [SignalDirection.LONG, SignalDirection.SHORT]

    def test_confluence_bonus_applied(self, scorer):
        """Test that confluence bonus is applied."""
        # Single pattern
        single_signal = TradeSignal(
            pattern_name="Double Bottom",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.60,
        )
        single_result = PatternResult(
            detected=True,
            pattern_name="Double Bottom",
            pattern_type=PatternType.REVERSAL,
            signal=single_signal,
        )

        single_score = scorer.calculate_confluence([single_result])

        # Multiple patterns
        multi_signal1 = TradeSignal(
            pattern_name="Double Bottom",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.60,
        )
        multi_signal2 = TradeSignal(
            pattern_name="MSL",
            direction=SignalDirection.LONG,
            entry_price=100.5,
            stop_loss=95.5,
            take_profit_1=111.0,
            confidence=0.60,
        )
        multi_results = [
            PatternResult(
                detected=True,
                pattern_name="Double Bottom",
                pattern_type=PatternType.REVERSAL,
                signal=multi_signal1,
            ),
            PatternResult(
                detected=True,
                pattern_name="MSL",
                pattern_type=PatternType.REVERSAL,
                signal=multi_signal2,
            ),
        ]

        multi_score = scorer.calculate_confluence(multi_results)

        # Multi-pattern should have higher score due to confluence bonus
        assert multi_score.score > single_score.score

    def test_rank_signals(self, scorer):
        """Test signal ranking."""
        signal1 = TradeSignal(
            pattern_name="Double Bottom",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit_1=110.0,
            confidence=0.75,
        )
        signal2 = TradeSignal(
            pattern_name="MSL",
            direction=SignalDirection.LONG,
            entry_price=100.5,
            stop_loss=95.5,
            take_profit_1=111.0,
            confidence=0.55,
        )
        signal3 = TradeSignal(
            pattern_name="Matching Lows",
            direction=SignalDirection.LONG,
            entry_price=101.0,
            stop_loss=96.0,
            take_profit_1=112.0,
            confidence=0.65,
        )

        results = [
            PatternResult(
                detected=True,
                pattern_name="Double Bottom",
                pattern_type=PatternType.REVERSAL,
                signal=signal1,
            ),
            PatternResult(
                detected=True, pattern_name="MSL", pattern_type=PatternType.REVERSAL, signal=signal2
            ),
            PatternResult(
                detected=True,
                pattern_name="Matching Lows",
                pattern_type=PatternType.REVERSAL,
                signal=signal3,
            ),
        ]

        ranked = scorer.rank_signals(results, top_n=2)

        assert len(ranked) == 2
        # Should be sorted by score (descending)
        assert ranked[0][1].score >= ranked[1][1].score


class TestPatternConfig:
    """Tests for PatternConfig dataclass."""

    def test_default_values(self):
        """Test default pattern config values."""
        config = PatternConfig()

        assert config.enabled is True
        assert config.lookback == 21
        assert config.tolerance == 0.05
        assert config.confidence_base == 0.50

    def test_custom_values(self):
        """Test custom pattern config values."""
        config = PatternConfig(enabled=False, lookback=50, tolerance=0.10, confidence_base=0.65)

        assert config.enabled is False
        assert config.lookback == 50
        assert config.tolerance == 0.10
        assert config.confidence_base == 0.65


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self):
        """Test default risk config values."""
        config = RiskConfig()

        assert config.initial_equity == 100000.0
        assert config.risk_per_trade == 0.02
        assert config.max_open_positions == 5

    def test_custom_values(self):
        """Test custom risk config values."""
        config = RiskConfig(initial_equity=50000.0, risk_per_trade=0.01, max_open_positions=3)

        assert config.initial_equity == 50000.0
        assert config.risk_per_trade == 0.01
        assert config.max_open_positions == 3


class TestTradingConfig:
    """Tests for TradingConfig class."""

    def test_default_initialization(self):
        """Test default trading config initialization."""
        config = TradingConfig()

        assert config.risk is not None
        assert config.signal is not None
        assert config.backtest is not None
        assert config.regime is not None

    def test_pattern_configs_exist(self):
        """Test that pattern configs are initialized."""
        config = TradingConfig()

        assert len(config.basic_patterns) > 0
        assert len(config.harmonic_patterns) > 0
        assert len(config.complex_patterns) > 0
        assert len(config.classic_patterns) > 0

    def test_get_pattern_config(self):
        """Test getting pattern config."""
        config = TradingConfig()

        # Should find double_top
        pattern_config = config.get_pattern_config("double_top")
        assert pattern_config is not None

        # Should find msl (Market Structure Low is keyed as 'msl')
        pattern_config = config.get_pattern_config("msl")
        assert pattern_config is not None

        # Should also find with partial match
        pattern_config = config.get_pattern_config("double_bottom")
        assert pattern_config is not None

    def test_to_dict(self):
        """Test converting config to dict."""
        config = TradingConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "risk" in config_dict
        assert "signal" in config_dict
        assert "basic_patterns" in config_dict

    def test_from_dict(self):
        """Test creating config from dict."""
        config_dict = {
            "risk": {"initial_equity": 50000.0, "risk_per_trade": 0.01},
            "signal": {"min_confidence": 0.60},
        }

        config = TradingConfig.from_dict(config_dict)

        assert config.risk.initial_equity == 50000.0
        assert config.risk.risk_per_trade == 0.01
        assert config.signal.min_confidence == 0.60

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading config as JSON."""
        config = TradingConfig()
        config.risk.initial_equity = 75000.0

        filepath = str(tmp_path / "config.json")
        config.save(filepath)

        loaded_config = TradingConfig.load(filepath)

        assert loaded_config.risk.initial_equity == 75000.0


class TestSignalConfig:
    """Tests for SignalConfig dataclass."""

    def test_default_values(self):
        """Test default signal config values."""
        config = SignalConfig()

        assert config.min_confidence == 0.50
        assert config.min_confluence_patterns == 2
        assert config.signal_decay_rate == 0.05

    def test_signal_validity_bars(self):
        """Test signal validity bars by category."""
        config = SignalConfig()

        assert "basic" in config.signal_validity_bars
        assert "harmonic" in config.signal_validity_bars
        assert config.signal_validity_bars["basic"] == 5
        assert config.signal_validity_bars["complex"] == 20


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default backtest config values."""
        config = BacktestConfig()

        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005
        assert config.use_take_profit_1 is True


class TestRegimeConfig:
    """Tests for RegimeConfig dataclass."""

    def test_default_values(self):
        """Test default regime config values."""
        config = RegimeConfig()

        assert config.adx_period == 14
        assert config.adx_trend_threshold == 25.0
        assert config.atr_period == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
