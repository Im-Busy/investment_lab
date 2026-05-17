"""
D3: Structural Break Detection & Unit-Root Testing.

Unit-root tests determine whether a time series is stationary (mean-reverting)
or contains a stochastic trend (non-stationary). Structural break tests detect
abrupt changes in the mean, trend, or variance of a time series.

Tests:
  ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary).
    Rejection → series is stationary. Critical for mean-reversion strategies.
  KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = stationary.
    Complementary to ADF. Both ADF-reject + KPSS-accept → strong stationarity.
  Chow: Tests for a known structural break at a specific date.
    H0 = no break. F-statistic compares restricted vs unrestricted OLS.
  Bai-Perron: Discovers multiple unknown breakpoints simultaneously.
    Sequential L breaks vs L+1 breaks testing. Global minimizer of RSS.

Usage:
    >>> detector = StructuralBreakDetector()
    >>> result = detector.run(price_series)
    >>> result.is_stationary
    >>> break_dates = result.breakpoints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


@dataclass
class UnitRootResult:
    test_name: str
    statistic: float
    pvalue: float
    critical_values: dict
    is_stationary: bool
    nobs: int
    used_lag: Optional[int] = None


@dataclass
class Breakpoint:
    index: int
    date: Optional[str]
    f_statistic: float
    pvalue: float
    is_significant: bool


@dataclass
class StructuralBreakResult:
    series_name: str
    nobs: int
    adf: UnitRootResult
    kpss_result: UnitRootResult
    is_stationary: bool
    has_structural_break: bool
    breakpoints: List[Breakpoint] = field(default_factory=list)
    optimal_break_count: int = 0
    notes: List[str] = field(default_factory=list)


def adf_test(
    series: np.ndarray,
    maxlag: Optional[int] = None,
    autolag: str = "AIC",
    regression: str = "c",
    alpha: float = 0.05,
) -> UnitRootResult:
    """Augmented Dickey-Fuller unit root test.

    H0: series has a unit root (non-stationary).
    Rejection (p < alpha): series is stationary.

    Args:
        series: 1-D array of values.
        maxlag: Maximum lags to use. None = automatic.
        autolag: Lag selection criterion ("AIC", "BIC", "t-stat").
        regression: "c" = constant only, "ct" = constant + trend, "n" = none.
        alpha: Significance level for stationarity decision.

    Returns:
        UnitRootResult with test statistic, p-value, and stationarity verdict.
    """
    result = adfuller(
        series,
        maxlag=maxlag,
        autolag=autolag,
        regression=regression,
    )
    statistic, pvalue, usedlag, nobs, crit_values, _icbest = result
    is_stationary = pvalue < alpha

    return UnitRootResult(
        test_name="ADF",
        statistic=statistic,
        pvalue=pvalue,
        critical_values=crit_values,
        is_stationary=is_stationary,
        nobs=nobs,
        used_lag=usedlag,
    )


def kpss_test(
    series: np.ndarray,
    regression: str = "c",
    nlags: str = "auto",
    alpha: float = 0.05,
) -> UnitRootResult:
    """KPSS stationarity test.

    H0: series is stationary (trend-stationary).
    Rejection (p < alpha): series is non-stationary.

    Complementary to ADF. Both ADF-reject + KPSS-accept = strong stationarity.

    Args:
        series: 1-D array of values.
        regression: "c" = level stationary, "ct" = trend stationary.
        nlags: Lag selection ("auto", "legacy", or integer).
        alpha: Significance level for stationarity decision.

    Returns:
        UnitRootResult with test statistic, p-value, and stationarity verdict.
    """
    statistic, pvalue, lags, crit_values = kpss(
        series,
        regression=regression,
        nlags=nlags,
    )
    is_stationary = pvalue >= alpha

    return UnitRootResult(
        test_name="KPSS",
        statistic=statistic,
        pvalue=pvalue,
        critical_values=crit_values,
        is_stationary=is_stationary,
        nobs=len(series),
        used_lag=lags,
    )


def chow_test(
    series: np.ndarray,
    break_index: int,
    alpha: float = 0.05,
) -> Breakpoint:
    """Chow test for a known structural break at a specific index.

    H0: no structural break at break_index.
    Fits a linear trend model with constant on the full series and on
    each sub-period. Compares RSS via F-test.

    Args:
        series: 1-D array of values.
        break_index: Index where the break is suspected.
        alpha: Significance level.

    Returns:
        Breakpoint with F-statistic, p-value, and significance verdict.
    """
    n = len(series)
    if break_index < 3 or break_index > n - 3:
        return Breakpoint(
            index=break_index,
            date=None,
            f_statistic=0.0,
            pvalue=1.0,
            is_significant=False,
        )

    t = np.arange(n).reshape(-1, 1)
    X = add_constant(t)

    # Restricted model: all data
    model_full = OLS(series, X).fit()
    rss_full = np.sum(model_full.resid**2)

    # Unrestricted: fit separately on [0:break] and [break:]
    X1 = add_constant(t[:break_index])
    X2 = add_constant(t[break_index:])
    model1 = OLS(series[:break_index], X1).fit()
    model2 = OLS(series[break_index:], X2).fit()
    rss1 = np.sum(model1.resid**2)
    rss2 = np.sum(model2.resid**2)
    rss_unrestricted = rss1 + rss2

    k = X.shape[1]  # number of parameters (intercept + trend)
    df1 = k
    df2 = n - 2 * k

    if df2 <= 0 or rss_unrestricted < 1e-12:
        return Breakpoint(
            index=break_index,
            date=None,
            f_statistic=0.0,
            pvalue=1.0,
            is_significant=False,
        )

    f_stat = ((rss_full - rss_unrestricted) / df1) / (rss_unrestricted / df2)

    from scipy.stats import f as f_dist

    pvalue = 1.0 - f_dist.cdf(f_stat, df1, df2)

    return Breakpoint(
        index=break_index,
        date=None,
        f_statistic=f_stat,
        pvalue=pvalue,
        is_significant=pvalue < alpha,
    )


def bai_perron_test(
    series: np.ndarray,
    max_breaks: int = 5,
    trim: float = 0.15,
    alpha: float = 0.05,
) -> List[Breakpoint]:
    """Bai-Perron sequential breakpoint test.

    Discovers multiple unknown structural breaks by minimizing the global
    residual sum of squares. Uses sequential L vs L+1 testing.

    Args:
        series: 1-D array of values.
        max_breaks: Maximum number of breakpoints to search for.
        trim: Minimum fraction of observations between breaks (default 15%).
        alpha: Significance level for the sequential test.

    Returns:
        List of significant Breakpoint objects sorted by index.
    """
    n = len(series)
    min_segment = max(3, int(n * trim))
    breakpoints: List[Breakpoint] = []
    best_rss = float("inf")
    best_breaks: List[int] = []
    total_best_rss = float("inf")
    total_best_breaks: List[int] = []

    for num_breaks in range(1, max_breaks + 1):
        best_rss = float("inf")
        best_breaks = []

        candidates = _generate_break_candidates(n, num_breaks, min_segment)
        for breaks in candidates:
            rss = _compute_segmented_rss(series, breaks)
            if rss < best_rss:
                best_rss = rss
                best_breaks = breaks

        if not best_breaks:
            continue

        if best_rss < total_best_rss:
            total_best_rss = best_rss
            total_best_breaks = best_breaks

        # Sequential test: L breaks vs L+1 breaks
        if num_breaks > 1 and breakpoints:
            f_stat = _sequential_f_test(
                series,
                breakpoints_indices=[b.index for b in breakpoints],
                new_breaks=best_breaks,
                n=n,
            )
            if f_stat < 1.0:
                break

        breakpoints = []
        for bp_idx in sorted(best_breaks):
            chow = chow_test(series, bp_idx, alpha=alpha)
            breakpoints.append(chow)

    if not breakpoints and total_best_breaks:
        for bp_idx in sorted(total_best_breaks):
            chow = chow_test(series, bp_idx, alpha=alpha)
            breakpoints.append(chow)

    return breakpoints


def _generate_break_candidates(
    n: int,
    num_breaks: int,
    min_segment: int,
    max_candidates: int = 500,
) -> List[List[int]]:
    """Generate plausible breakpoint candidate combinations."""
    candidates: List[List[int]] = []
    step = max(1, (n - 2 * min_segment) // min(20, max_candidates // num_breaks))
    search_start = min_segment
    search_end = n - min_segment

    for i in range(search_start, search_end, step):
        if num_breaks == 1:
            candidates.append([i])
        elif num_breaks == 2:
            for j in range(i + min_segment, search_end, step):
                candidates.append([i, j])
        elif num_breaks <= 3:
            for j in range(i + min_segment, search_end, step):
                for k in range(j + min_segment, search_end, step):
                    candidates.append([i, j, k])
        else:
            breaks = [i]
            pos = i
            for _ in range(num_breaks - 1):
                pos += min_segment + (n - pos) // (num_breaks + 1)
                if pos >= search_end:
                    break
                breaks.append(min(pos, n - 1))
            if len(breaks) == num_breaks:
                candidates.append(breaks)

            if len(candidates) > max_candidates:
                break

        if len(candidates) > max_candidates:
            break

    return candidates[:max_candidates]


def _compute_segmented_rss(series: np.ndarray, breakpoints: List[int]) -> float:
    """Compute total RSS across all segments defined by breakpoints."""
    n = len(series)
    segments = [0] + sorted(breakpoints) + [n]
    total_rss = 0.0

    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        if end - start < 2:
            continue
        seg = series[start:end]
        seg_mean = np.mean(seg)
        total_rss += np.sum((seg - seg_mean) ** 2)

    return total_rss


def _sequential_f_test(
    series: np.ndarray,
    breakpoints_indices: List[int],
    new_breaks: List[int],
    n: int,
) -> float:
    """F-test comparing RSS with L vs L+1 breakpoints."""
    rss_l = _compute_segmented_rss(series, breakpoints_indices)
    rss_l1 = _compute_segmented_rss(series, new_breaks)
    if rss_l1 < 1e-12:
        return 0.0
    return rss_l / rss_l1


class StructuralBreakDetector:
    """Combined unit-root and structural break test suite.

    Runs ADF + KPSS for stationarity assessment and Bai-Perron
    for unknown structural break discovery. Produces a unified result
    indicating whether the series is stationary and whether regime
    shifts have occurred.

    Args:
        max_breaks: Maximum breakpoints for Bai-Perron search.
        alpha: Significance level for all tests.
        trim: Minimum segment fraction for Bai-Perron.
    """

    def __init__(
        self,
        max_breaks: int = 3,
        alpha: float = 0.05,
        trim: float = 0.15,
    ):
        self.max_breaks = max_breaks
        self.alpha = alpha
        self.trim = trim

    def run(
        self,
        series: np.ndarray,
        dates: Optional[List[str]] = None,
        series_name: str = "series",
        adf_regression: str = "c",
        kpss_regression: str = "c",
    ) -> StructuralBreakResult:
        """Run full structural break analysis on a time series.

        Args:
            series: 1-D array of values (prices, returns, spreads, etc.).
            dates: Optional list of date strings for breakpoint labeling.
            series_name: Identifier for the result.
            adf_regression: ADF regression type ("c", "ct", "n").
            kpss_regression: KPSS regression type ("c", "ct").

        Returns:
            StructuralBreakResult with stationarity, breakpoints, and notes.
        """
        series = np.asarray(series, dtype=float)
        valid = ~np.isnan(series)
        series = series[valid]
        if dates is not None:
            dates = [d for d, v in zip(dates, valid) if v]

        notes: List[str] = []
        n = len(series)

        if n < 20:
            return StructuralBreakResult(
                series_name=series_name,
                nobs=n,
                adf=UnitRootResult("ADF", 0.0, 1.0, {}, False, n),
                kpss_result=UnitRootResult("KPSS", 0.0, 1.0, {}, False, n),
                is_stationary=False,
                has_structural_break=False,
                notes=["Insufficient observations (< 20)"],
            )

        adf = adf_test(series, regression=adf_regression, alpha=self.alpha)
        kpss_r = kpss_test(series, regression=kpss_regression, alpha=self.alpha)

        # Strong stationarity: ADF rejects unit root + KPSS accepts stationarity
        is_stationary = adf.is_stationary and kpss_r.is_stationary

        if adf.is_stationary and not kpss_r.is_stationary:
            notes.append("ADF stationary but KPSS non-stationary: possible trend-stationarity")
        elif not adf.is_stationary and kpss_r.is_stationary:
            notes.append("ADF non-stationary but KPSS stationary: borderline, test on returns")

        breakpoints = bai_perron_test(
            series,
            max_breaks=self.max_breaks,
            trim=self.trim,
            alpha=self.alpha,
        )

        significant_breaks = [b for b in breakpoints if b.is_significant]
        if dates:
            for bp in significant_breaks:
                if bp.index < len(dates):
                    bp.date = dates[bp.index]

        has_break = len(significant_breaks) > 0

        if has_break:
            notes.append(f"Detected {len(significant_breaks)} significant structural break(s)")

        return StructuralBreakResult(
            series_name=series_name,
            nobs=n,
            adf=adf,
            kpss_result=kpss_r,
            is_stationary=is_stationary,
            has_structural_break=has_break,
            breakpoints=significant_breaks,
            optimal_break_count=len(significant_breaks),
            notes=notes,
        )

    def run_on_prices(
        self,
        prices: pd.Series,
        log_transform: bool = True,
    ) -> StructuralBreakResult:
        """Run analysis on price data, optionally on log prices.

        Args:
            prices: Price series with datetime index.
            log_transform: Use log prices (recommended for financial data).

        Returns:
            StructuralBreakResult with breakpoints labeled by date.
        """
        name = getattr(prices, "name", "prices")
        dates = [str(d.date()) for d in prices.index]

        if log_transform:
            series = np.log(prices.values)
            result = self.run(series, dates=dates, series_name=f"log_{name}")
        else:
            result = self.run(prices.values, dates=dates, series_name=name)

        return result

    def run_on_returns(
        self,
        prices: pd.Series,
    ) -> StructuralBreakResult:
        """Run analysis on log returns (first-differenced log prices).

        Returns are expected to be stationary. Structural breaks in
        returns indicate volatility regime shifts.

        Args:
            prices: Price series with datetime index.

        Returns:
            StructuralBreakResult for the return series.
        """
        name = getattr(prices, "name", "returns")
        log_prices = np.log(prices.values)
        returns = np.diff(log_prices)
        dates = [str(d.date()) for d in prices.index[1:]]

        return self.run(returns, dates=dates, series_name=f"{name}_returns")
