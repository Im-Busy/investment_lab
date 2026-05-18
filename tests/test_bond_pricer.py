"""Tests for B2: Bond Duration & Convexity (BondPricer)."""

import numpy as np
import pytest

from src.ml.fixed_income_models import BondPricer, BondResult


@pytest.fixture
def pricer():
    return BondPricer(face_value=100.0, frequency=2)


def test_price_par_bond(pricer):
    """At par: price = face value when YTM = coupon."""
    result = pricer.price(ytm=0.05, coupon=0.05, maturity=10.0)
    assert abs(result - 100.0) < 0.5


def test_price_discount_bond(pricer):
    """Discount bond: price < face value when YTM > coupon."""
    result = pricer.price(ytm=0.08, coupon=0.05, maturity=10.0)
    assert result < 95.0


def test_price_premium_bond(pricer):
    """Premium bond: price > face value when YTM < coupon."""
    result = pricer.price(ytm=0.03, coupon=0.05, maturity=10.0)
    assert result > 105.0


def test_price_zero_coupon(pricer):
    """Zero coupon bond converges to face value."""
    price = pricer.price(ytm=0.0, coupon=0.0, maturity=10.0)
    assert abs(price - 100.0) < 0.01


def test_compute_analytics(pricer):
    """Compute returns all analytics including duration and convexity."""
    result = pricer.compute(ytm=0.05, coupon=0.04, maturity=7.0)
    assert isinstance(result, BondResult)
    assert result.macaulay_duration > 0
    assert result.modified_duration > 0
    assert result.convexity > 0
    assert result.dv01 > 0


def test_duration_at_par(pricer):
    """Duration increases with maturity."""
    short = pricer.compute(ytm=0.04, coupon=0.04, maturity=2.0)
    long = pricer.compute(ytm=0.04, coupon=0.04, maturity=10.0)
    assert long.macaulay_duration > short.macaulay_duration


def test_convexity_positive(pricer):
    """Convexity is always positive for vanilla bonds."""
    result = pricer.compute(ytm=0.06, coupon=0.03, maturity=5.0)
    assert result.convexity > 0


def test_dv01_linear_approx(pricer):
    """DV01 approximates price change for 1bp yield move."""
    ytm = 0.05
    result = pricer.compute(ytm=ytm, coupon=0.04, maturity=5.0)
    estimated_delta = -result.dv01
    actual_p0 = pricer.price(ytm, 0.04, 5.0)
    actual_p1 = pricer.price(ytm + 0.0001, 0.04, 5.0)
    actual_delta = actual_p1 - actual_p0
    assert abs(estimated_delta - actual_delta) < 0.001


def test_zero_maturity(pricer):
    """Zero maturity returns face value."""
    result = pricer.compute(ytm=0.05, coupon=0.04, maturity=0.0)
    assert abs(result.price - 100.0) < 0.01
    assert result.macaulay_duration == 0.0
    assert result.dv01 == 0.0


def test_ytm_from_price(pricer):
    """YTM inversion recovers original yield."""
    price = pricer.price(ytm=0.06, coupon=0.04, maturity=5.0)
    recovered = pricer.ytm_from_price(price, coupon=0.04, maturity=5.0)
    assert abs(recovered - 0.06) < 0.001


def test_price_impact(pricer):
    """Price impact approximation via duration + convexity."""
    impact = pricer.price_impact(ytm=0.05, coupon=0.04, maturity=5.0, yield_change_bp=50)
    assert "pct_change" in impact
    assert abs(impact["pct_change"]) > 0.1


def test_bond_yield_series(pricer):
    """Convert price series to YTM series."""
    prices = np.array([105.0, 104.5, 105.2, 103.8, 106.0])
    ytms = pricer.bond_yield_series(prices, coupon=0.05, maturity=5.0)
    assert len(ytms) == len(prices)
    assert np.all(np.isfinite(ytms))


def test_etf_proxy(pricer):
    """ETF proxy pricing for TLT/IEF."""
    result = pricer.price_etf_proxy(ytm=0.045, avg_maturity=16.0, avg_coupon=0.03)
    assert result.price > 0
    assert result.macaulay_duration > 5
    assert result.convexity > 0


def test_to_dict(pricer):
    """BondResult serialization."""
    result = pricer.compute(ytm=0.05, coupon=0.04, maturity=5.0)
    d = result.to_dict()
    assert "modified_duration" in d
    assert "convexity" in d
    assert "dv01" in d
