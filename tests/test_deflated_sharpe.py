"""Unit tests for DSR, PSR, and FDR computation."""

import numpy as np

from src.analysis.deflated_sharpe import (
    PSRResult,
    FDRResult,
    _expected_max_sharpe,
    _sharpe_ratio,
    compute_psr,
    compute_psr_from_returns,
    compute_dsr,
    compute_dsr_from_returns,
    benjamini_hochberg,
    deflated_sharpe_batch,
    dsr_significance,
)


class TestSharpeRatio:
    """Tests for internal Sharpe ratio computation."""

    def test_positive_returns(self):
        returns = np.full(252, 0.001)
        sr = _sharpe_ratio(returns)
        assert sr > 50

    def test_zero_returns(self):
        returns = np.zeros(100)
        sr = _sharpe_ratio(returns)
        assert sr == 0.0

    def test_single_observation(self):
        sr = _sharpe_ratio(np.array([0.01]))
        assert sr == 0.0

    def test_mixed_returns(self):
        np.random.seed(42)
        returns = np.random.randn(252) * 0.02 + 0.001
        sr = _sharpe_ratio(returns)
        assert -5 < sr < 5


class TestPSR:
    """Tests for Probabilistic Sharpe Ratio."""

    def test_psr_basic(self):
        result = compute_psr(sharpe=1.0, benchmark=0.0, n_obs=252)
        assert 0.99 <= result.psr <= 1.0
        assert not result.deflated
        assert result.sharpe == 1.0
        assert result.benchmark == 0.0

    def test_psr_high_sharpe(self):
        result = compute_psr(sharpe=3.0, benchmark=0.0, n_obs=252)
        assert result.psr > 0.99

    def test_psr_at_benchmark(self):
        result = compute_psr(sharpe=0.0, benchmark=0.0, n_obs=252)
        assert abs(result.psr - 0.5) < 0.1

    def test_psr_negative_sharpe(self):
        result = compute_psr(sharpe=-0.5, benchmark=0.0, n_obs=252)
        assert result.psr < 0.5

    def test_psr_with_skew_kurt(self):
        result = compute_psr(sharpe=1.0, benchmark=0.0, n_obs=252, skew=-0.5, kurt=5.0)
        assert 0.0 <= result.psr <= 1.0

    def test_psr_few_obs(self):
        result = compute_psr(sharpe=1.0, n_obs=1)
        assert result.psr == 0.0

    def test_psr_zero_denom(self):
        result = compute_psr(sharpe=5.0, benchmark=0.0, n_obs=100, skew=3.0, kurt=1.0)
        assert result.psr == 0.0

    def test_psr_from_returns(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02 + 0.001
        result = compute_psr_from_returns(returns)
        assert 0.0 <= result.psr <= 1.0
        assert not result.deflated

    def test_psr_from_returns_few_obs(self):
        result = compute_psr_from_returns(np.array([0.01, 0.02]))
        assert result.psr == 0.0

    def test_psr_standard_error(self):
        result = compute_psr(sharpe=2.0, benchmark=0.0, n_obs=252)
        assert result.standard_error > 0

    def test_psr_result_fields(self):
        result = compute_psr(sharpe=1.5, benchmark=0.5, n_obs=500, skew=-0.3, kurt=4.0)
        assert isinstance(result, PSRResult)
        assert isinstance(result.psr, float)
        assert not result.deflated


class TestDSR:
    """Tests for Deflated Sharpe Ratio."""

    def test_dsr_basic(self):
        result = compute_dsr(sharpe=2.0, n_obs=252, n_trials=10, n_simulations=2000)
        assert 0.0 <= result.psr <= 1.0
        assert result.deflated

    def test_dsr_low_sharpe_many_trials(self):
        result = compute_dsr(sharpe=0.3, n_obs=252, n_trials=500, n_simulations=2000)
        assert result.psr < 0.95

    def test_dsr_high_sharpe(self):
        result = compute_dsr(sharpe=4.0, n_obs=252, n_trials=5, n_simulations=2000)
        assert result.psr > 0.9

    def test_dsr_benchmark_grows_with_trials(self):
        r_few = compute_dsr(sharpe=2.0, n_obs=252, n_trials=3, n_simulations=2000)
        r_many = compute_dsr(sharpe=2.0, n_obs=252, n_trials=100, n_simulations=2000)
        assert r_few.benchmark < r_many.benchmark

    def test_dsr_from_returns(self):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02 + 0.002
        result = compute_dsr_from_returns(returns, n_trials=10, n_simulations=2000)
        assert 0.0 <= result.psr <= 1.0
        assert result.deflated

    def test_dsr_from_returns_few_obs(self):
        result = compute_dsr_from_returns(np.array([0.01]), n_trials=5)
        assert result.psr == 0.0

    def test_dsr_reproducible(self):
        r1 = compute_dsr(sharpe=2.0, n_obs=252, n_trials=10, n_simulations=2000, seed=42)
        r2 = compute_dsr(sharpe=2.0, n_obs=252, n_trials=10, n_simulations=2000, seed=42)
        assert abs(r1.psr - r2.psr) < 1e-10

    def test_dsr_result_fields(self):
        result = compute_dsr(sharpe=1.5, n_obs=252, n_trials=20, n_simulations=2000)
        assert isinstance(result, PSRResult)
        assert result.deflated


class TestExpectedMaxSharpe:
    """Tests for expected maximum Sharpe under the null."""

    def test_grows_with_trials(self):
        e_few = _expected_max_sharpe(n_trials=5, n_obs=252, n_simulations=2000)
        e_many = _expected_max_sharpe(n_trials=100, n_obs=252, n_simulations=2000)
        assert e_many > e_few

    def test_decreases_with_obs(self):
        e_short = _expected_max_sharpe(n_trials=10, n_obs=63, n_simulations=2000)
        e_long = _expected_max_sharpe(n_trials=10, n_obs=1008, n_simulations=2000)
        assert e_short > e_long

    def test_reproducible(self):
        e1 = _expected_max_sharpe(n_trials=10, n_obs=252, n_simulations=2000, seed=42)
        e2 = _expected_max_sharpe(n_trials=10, n_obs=252, n_simulations=2000, seed=42)
        assert abs(e1 - e2) < 1e-10

    def test_positive(self):
        e = _expected_max_sharpe(n_trials=50, n_obs=252, n_simulations=2000)
        assert e > 0


class TestFDR:
    """Tests for Benjamini-Hochberg FDR."""

    def test_no_significant(self):
        p_values = np.array([0.5, 0.6, 0.7, 0.8])
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected == 0
        assert len(result.rejected) == 0
        assert result.threshold == 0.0

    def test_all_significant(self):
        p_values = np.array([0.001, 0.002, 0.003, 0.004])
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected == 4

    def test_mixed(self):
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 20)
        p_values[0:3] = [1e-6, 1e-5, 0.001]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected >= 1

    def test_empty(self):
        result = benjamini_hochberg([], alpha=0.05)
        assert result.n_rejected == 0

    def test_nan_handling(self):
        p_values = np.array([0.01, np.nan, 0.02, 0.03])
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected > 0

    def test_strict_alpha(self):
        p_values = np.array([0.01, 0.02, 0.03, 0.5])
        result_loose = benjamini_hochberg(p_values, alpha=0.05)
        result_strict = benjamini_hochberg(p_values, alpha=0.001)
        assert result_strict.n_rejected <= result_loose.n_rejected

    def test_rejected_indices(self):
        p_values = np.array([0.6, 0.001, 0.7, 0.002])
        result = benjamini_hochberg(p_values, alpha=0.05)
        rejected_set = set(result.rejected)
        assert 1 in rejected_set
        assert 3 in rejected_set
        assert 0 not in rejected_set
        assert 2 not in rejected_set

    def test_result_type(self):
        result = benjamini_hochberg([0.01, 0.02, 0.5], alpha=0.05)
        assert isinstance(result, FDRResult)
        assert result.alpha == 0.05

    def test_threshold_value(self):
        p_values = np.array([0.01, 0.02, 0.03])
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.threshold > 0


class TestDeflatedSharpeBatch:
    """Tests for batch DSR computation."""

    def test_basic(self):
        sharpes = np.array([0.5, 1.0, 1.5, 2.0])
        results = deflated_sharpe_batch(sharpes, n_obs=252, n_trials=10, n_simulations=2000)
        assert len(results) == 4
        for r in results:
            assert r.deflated
            assert 0.0 <= r.psr <= 1.0

    def test_all_same_benchmark(self):
        sharpes = np.array([1.0, 2.0, 3.0])
        results = deflated_sharpe_batch(sharpes, n_obs=252, n_trials=10, n_simulations=2000)
        for r in results:
            assert r.benchmark == results[0].benchmark

    def test_with_skew_kurt(self):
        sharpes = np.array([1.0, 1.5])
        skews = np.array([-0.3, -0.1])
        kurts = np.array([4.0, 3.5])
        results = deflated_sharpe_batch(
            sharpes, n_obs=252, n_trials=10, skews=skews, kurts=kurts, n_simulations=2000
        )
        assert len(results) == 2


class TestDSRSignificance:
    """Tests for DSR significance interpretation."""

    def test_significant(self):
        flags = dsr_significance(0.97)
        assert flags["significant"]
        assert flags["borderline"]

    def test_borderline(self):
        flags = dsr_significance(0.85)
        assert not flags["significant"]
        assert flags["borderline"]

    def test_not_significant(self):
        flags = dsr_significance(0.50)
        assert not flags["significant"]
        assert not flags["borderline"]

    def test_custom_thresholds(self):
        thresholds = {"strong": 0.99, "weak": 0.70}
        flags = dsr_significance(0.95, thresholds=thresholds)
        assert not flags["strong"]
        assert flags["weak"]
