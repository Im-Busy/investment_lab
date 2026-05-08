"""Unit tests for Triple Barrier Labeling."""

import numpy as np
import pandas as pd
import pytest

from src.ml.triple_barrier import TripleBarrierLabeler


@pytest.fixture
def sample_ohlcv():
    """Generate OHLCV data with known barrier crossings."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)

    close = pd.Series(base + np.random.randn(n) * 0.1, index=dates)
    high = pd.Series(
        np.maximum(close, close.shift(1)) + np.abs(np.random.randn(n)) * 0.1, index=dates
    )
    low = pd.Series(
        np.minimum(close, close.shift(1)) - np.abs(np.random.randn(n)) * 0.1, index=dates
    )

    # Ensure high >= close >= low
    high = pd.Series(np.maximum(high, close), index=dates)
    low = pd.Series(np.minimum(low, close), index=dates)

    return {"close": close, "high": high, "low": low}


@pytest.fixture
def trending_up():
    """Strong uptrend where TP should always hit."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 130, n), index=dates)
    high = pd.Series(close.values + 1.0, index=dates)
    low = pd.Series(close.values - 0.5, index=dates)
    return {"close": close, "high": high, "low": low}


@pytest.fixture
def trending_down():
    """Strong downtrend where SL should always hit."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 70, n), index=dates)
    high = pd.Series(close.values + 0.5, index=dates)
    low = pd.Series(close.values - 1.0, index=dates)
    return {"close": close, "high": high, "low": low}


class TestTripleBarrierLabeler:
    """Tests for TripleBarrierLabeler class."""

    def test_fit_tp_hits(self, trending_up):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=trending_up["close"],
            high=trending_up["high"],
            low=trending_up["low"],
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert labels.iloc[0] == 1

    def test_fit_sl_hits(self, trending_down):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=trending_down["close"],
            high=trending_down["high"],
            low=trending_down["low"],
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert labels.iloc[0] == -1

    def test_fit_timeout(self, sample_ohlcv):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            take_profit=200.0,
            stop_loss=10.0,
            time_limit=5,
        )
        valid = labels.dropna()
        assert set(valid.unique()).issubset({1, -1, 0})

    def test_fit_returns_series(self, sample_ohlcv):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            take_profit=110.0,
            stop_loss=90.0,
            time_limit=10,
        )
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_ohlcv["close"])

    def test_fit_labels_valid_values(self, sample_ohlcv):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=sample_ohlcv["close"],
            take_profit=120.0,
            stop_loss=80.0,
            time_limit=10,
        )
        valid = labels.dropna()
        assert all(v in (1, -1, 0) for v in valid.values)

    def test_fit_attrs_set(self, sample_ohlcv):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=sample_ohlcv["close"],
            take_profit=120.0,
            stop_loss=80.0,
            time_limit=10,
        )
        assert "return_pct" in labels.attrs
        assert "barrier" in labels.attrs
        assert "bars_to_exit" in labels.attrs
        assert len(labels.attrs["return_pct"]) == len(labels)
        assert len(labels.attrs["barrier"]) == len(labels)
        assert len(labels.attrs["bars_to_exit"]) == len(labels)

    def test_fit_dynamic_barriers(self, trending_up):
        atr = pd.Series(np.full(50, 2.0), index=trending_up["close"].index)
        labeler = TripleBarrierLabeler(atr_mult_tp=1.5, atr_mult_sl=1.0)
        labels = labeler.fit(
            close=trending_up["close"],
            high=trending_up["high"],
            low=trending_up["low"],
            take_profit=None,
            stop_loss=None,
            time_limit=20,
            atr_series=atr,
        )
        assert labels.iloc[0] == 1

    def test_fit_dynamic_no_atr_raises(self, sample_ohlcv):
        labeler = TripleBarrierLabeler()
        with pytest.raises(ValueError, match="atr_series"):
            labeler.fit(
                close=sample_ohlcv["close"],
                take_profit=None,
                stop_loss=95.0,
                time_limit=10,
            )

    def test_fit_with_numpy_arrays(self):
        n = 50
        close = np.linspace(100, 110, n)
        high = close + 1.0
        low = close - 0.5
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=close,
            high=high,
            low=low,
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert labels.iloc[0] == 1

    def test_fit_no_high_low_uses_close(self, trending_up):
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=trending_up["close"],
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert labels.iloc[0] == 1

    def test_fit_per_bar_barriers(self, trending_up):
        n = len(trending_up["close"])
        tp_series = pd.Series(np.full(n, 105.0), index=trending_up["close"].index)
        sl_series = pd.Series(np.full(n, 95.0), index=trending_up["close"].index)
        labeler = TripleBarrierLabeler()
        labels = labeler.fit(
            close=trending_up["close"],
            high=trending_up["high"],
            low=trending_up["low"],
            take_profit=tp_series,
            stop_loss=sl_series,
            time_limit=20,
        )
        assert labels.iloc[0] == 1


class TestFitSingle:
    """Tests for the fit_single method."""

    def test_tp_hit(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([101, 102, 105, 104, 103])
        high_fwd = np.array([101.5, 103, 106, 105, 104])
        low_fwd = np.array([99, 98, 100, 101, 102])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            high_arr=high_fwd,
            low_arr=low_fwd,
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert result.label == 1
        assert result.barrier == "tp"
        assert result.bars_to_exit == 3
        assert abs(result.return_pct - 0.05) < 1e-6

    def test_sl_hit(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([99, 98, 97, 96, 95])
        high_fwd = np.array([100, 99, 98, 97, 96])
        low_fwd = np.array([98, 97, 96, 95, 94.5])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            high_arr=high_fwd,
            low_arr=low_fwd,
            take_profit=110.0,
            stop_loss=95.0,
            time_limit=20,
        )
        assert result.label == -1
        assert result.barrier == "sl"
        assert result.bars_to_exit == 4

    def test_timeout(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([100.1, 100.2, 100.3, 100.4, 100.5])
        high_fwd = np.array([100.5, 100.6, 100.7, 100.8, 100.9])
        low_fwd = np.array([99.5, 99.6, 99.7, 99.8, 99.9])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            high_arr=high_fwd,
            low_arr=low_fwd,
            take_profit=110.0,
            stop_loss=90.0,
            time_limit=5,
        )
        assert result.label == 0
        assert result.barrier == "time"

    def test_tp_before_sl(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([101, 105, 94, 93])
        high_fwd = np.array([102, 106, 95, 94])
        low_fwd = np.array([99, 98, 93, 92])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            high_arr=high_fwd,
            low_arr=low_fwd,
            take_profit=105.0,
            stop_loss=93.0,
            time_limit=20,
        )
        assert result.label == 1

    def test_sl_before_tp(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([99, 98, 93, 106])
        high_fwd = np.array([100, 99, 94, 107])
        low_fwd = np.array([98, 97, 92, 104])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            high_arr=high_fwd,
            low_arr=low_fwd,
            take_profit=105.0,
            stop_loss=93.0,
            time_limit=20,
        )
        assert result.label == -1

    def test_no_both_barriers(self):
        labeler = TripleBarrierLabeler()
        close_fwd = np.array([100.1, 100.2])
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=close_fwd,
            take_profit=None,
            stop_loss=None,
            time_limit=2,
        )
        assert result.label == 0
        assert result.barrier == "time"

    def test_empty_forward(self):
        labeler = TripleBarrierLabeler()
        result = labeler.fit_single(
            entry_price=100.0,
            close_arr=np.array([]),
            take_profit=105.0,
            stop_loss=95.0,
            time_limit=5,
        )
        assert result.label == 0


class TestLabelSummary:
    """Tests for label summary statistics."""

    def test_all_positive(self):
        labels = pd.Series([1, 1, 1, 1, 1])
        summary = TripleBarrierLabeler.label_summary(labels)
        assert summary["n_positive"] == 5
        assert summary["n_negative"] == 0
        assert summary["n_timeout"] == 0
        assert summary["pct_positive"] == 1.0
        assert summary["event_rate"] == 1.0

    def test_mixed(self):
        labels = pd.Series([1, 1, -1, 0, 0, 0, 1, -1, 0, 1])
        summary = TripleBarrierLabeler.label_summary(labels)
        assert summary["n_total"] == 10
        assert summary["n_positive"] == 4
        assert summary["n_negative"] == 2
        assert summary["n_timeout"] == 4

    def test_empty(self):
        labels = pd.Series([], dtype=float)
        summary = TripleBarrierLabeler.label_summary(labels)
        assert summary["n_total"] == 0


class TestFromPatternSignal:
    """Tests for from_pattern_signal convenience method."""

    def test_basic(self, trending_up):
        n = len(trending_up["close"])
        tp_series = pd.Series(np.full(n, 105.0), index=trending_up["close"].index)
        sl_series = pd.Series(np.full(n, 95.0), index=trending_up["close"].index)

        labels = TripleBarrierLabeler.from_pattern_signal(
            close=trending_up["close"],
            high=trending_up["high"],
            low=trending_up["low"],
            take_profit_series=tp_series,
            stop_loss_series=sl_series,
            time_limit=20,
        )
        assert labels.iloc[0] == 1
        assert pd.isna(labels.iloc[-1])

    def test_nan_where_no_signal(self, trending_up):
        n = len(trending_up["close"])
        tp_series = pd.Series(np.nan, index=trending_up["close"].index)
        sl_series = pd.Series(np.nan, index=trending_up["close"].index)
        tp_series.iloc[0] = 105.0
        sl_series.iloc[0] = 95.0

        labels = TripleBarrierLabeler.from_pattern_signal(
            close=trending_up["close"],
            high=trending_up["high"],
            low=trending_up["low"],
            take_profit_series=tp_series,
            stop_loss_series=sl_series,
            time_limit=20,
        )
        assert not pd.isna(labels.iloc[0])
        assert pd.isna(labels.iloc[1])

    def test_invalid_barriers_skipped(self, sample_ohlcv):
        n = len(sample_ohlcv["close"])
        tp_series = pd.Series(np.nan, index=sample_ohlcv["close"].index)
        sl_series = pd.Series(np.nan, index=sample_ohlcv["close"].index)
        tp_series.iloc[10] = 90.0  # TP below entry = invalid
        sl_series.iloc[10] = 95.0

        labels = TripleBarrierLabeler.from_pattern_signal(
            close=sample_ohlcv["close"],
            high=sample_ohlcv["high"],
            low=sample_ohlcv["low"],
            take_profit_series=tp_series,
            stop_loss_series=sl_series,
            time_limit=10,
        )
        assert pd.isna(labels.iloc[10])
