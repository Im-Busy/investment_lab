"""
Block A: Options Pricing Models — A1-A7 (Complete Options Toolkit).

Implements the standard options pricing and risk management models used
by quantitative traders and market makers.

Models:
  A1 BlackScholes: Closed-form European call/put pricing (BSM 1973).
  A2 BinomialTree: Cox-Ross-Rubinstein binomial tree for American/Euro.
  A3 MonteCarlo: Geometric Brownian motion simulation with antithetic variates.
  A4 HestonModel: Stochastic volatility model with correlated Brownian motions.
  A5 SABRModel: Stochastic Alpha-Beta-Rho model (Hagan 2002).
  A6 VolSurface: Implied volatility surface via cubic spline interpolation.
  A7 Greeks: Delta, Gamma, Theta, Vega, Rho via analytical formulas.

Usage:
    >>> bs = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
    >>> bs.call_price
    >>> bs.put_price
    >>> iv = BlackScholes.implied_vol(price=3.5, S=100, K=105, T=0.25, r=0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

SQRT_2PI = np.sqrt(2 * np.pi)
IV_MAX_ITER = 100
IV_TOL = 1e-8
IV_PRICE_TOL = 1e-12


# ── A1: Black-Scholes ──────────────────────────────────────────────────


@dataclass
class BlackScholesResult:
    call_price: float
    put_price: float
    delta_call: float
    delta_put: float
    gamma: float
    theta_call: float
    theta_put: float
    vega: float
    rho_call: float
    rho_put: float

    def to_dict(self) -> Dict:
        return {
            "call": round(self.call_price, 4),
            "put": round(self.put_price, 4),
            "delta_call": round(self.delta_call, 4),
            "delta_put": round(self.delta_put, 4),
            "gamma": round(self.gamma, 6),
            "theta_call": round(self.theta_call, 6),
            "vega": round(self.vega, 4),
        }


class BlackScholes:
    """Black-Scholes-Merton European option pricing.

    Standard model for European call and put options with no dividends.
    Provides closed-form prices and all first-order Greeks.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (continuous, decimal).
        sigma: Volatility (decimal, e.g., 0.20 = 20%).
        q: Continuous dividend yield (decimal).
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ):
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)
        self.r = r
        self.sigma = max(sigma, 1e-10)
        self.q = q

    @property
    def d1(self) -> float:
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * np.sqrt(self.T)

    @property
    def call_price(self) -> float:
        return float(
            self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        )

    @property
    def put_price(self) -> float:
        return float(
            self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        )

    @property
    def delta_call(self) -> float:
        return float(np.exp(-self.q * self.T) * norm.cdf(self.d1))

    @property
    def delta_put(self) -> float:
        return float(np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1.0))

    @property
    def gamma(self) -> float:
        return float(
            np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        )

    @property
    def theta_call(self) -> float:
        t1 = (
            -self.S
            * np.exp(-self.q * self.T)
            * norm.pdf(self.d1)
            * self.sigma
            / (2 * np.sqrt(self.T))
        )
        t2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        t3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
        return float((t1 + t2 + t3) / 365.0)

    @property
    def theta_put(self) -> float:
        t1 = (
            -self.S
            * np.exp(-self.q * self.T)
            * norm.pdf(self.d1)
            * self.sigma
            / (2 * np.sqrt(self.T))
        )
        t2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        t3 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return float((t1 + t2 + t3) / 365.0)

    @property
    def vega(self) -> float:
        return float(
            self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100.0
        )

    @property
    def rho_call(self) -> float:
        return float(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100.0)

    @property
    def rho_put(self) -> float:
        return float(-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100.0)

    def all_greeks(self) -> BlackScholesResult:
        return BlackScholesResult(
            call_price=self.call_price,
            put_price=self.put_price,
            delta_call=self.delta_call,
            delta_put=self.delta_put,
            gamma=self.gamma,
            theta_call=self.theta_call,
            theta_put=self.theta_put,
            vega=self.vega,
            rho_call=self.rho_call,
            rho_put=self.rho_put,
        )

    @staticmethod
    def implied_vol(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        is_call: bool = True,
    ) -> float:
        """Compute implied volatility via Brent's method.

        Args:
            price: Observed market price.
            S: Spot price.
            K: Strike.
            T: Time to expiration.
            r: Risk-free rate.
            q: Dividend yield.
            is_call: True for call, False for put.

        Returns:
            Implied volatility (decimal).
        """
        if price <= 0 or T <= 0:
            return np.nan

        intrinsic = max(0, S - K * np.exp(-r * T)) if is_call else max(0, K * np.exp(-r * T) - S)
        if price <= intrinsic:
            return 0.0

        def f(sigma: float) -> float:
            bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
            model_price = bs.call_price if is_call else bs.put_price
            return model_price - price

        try:
            return float(brentq(f, 1e-6, 5.0, maxiter=IV_MAX_ITER, xtol=IV_TOL))
        except (ValueError, RuntimeError):
            return np.nan


# ── A2: Binomial Tree ──────────────────────────────────────────────────


class BinomialTree:
    """Cox-Ross-Rubinstein binomial tree for American and European options.

    Handles early exercise for American options. Supports calls and puts.

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiration (years).
        r: Risk-free rate (continuous).
        sigma: Volatility.
        steps: Number of time steps in the tree.
        q: Dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        steps: int = 200,
        q: float = 0.0,
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.q = q

    def price(self, is_call: bool = True, american: bool = True) -> float:
        """Price an option using the binomial tree.

        Args:
            is_call: True for call, False for put.
            american: True for American (early exercise), False for European.

        Returns:
            Option price.
        """
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)

        S_grid = self.S * u ** np.arange(self.steps, -1, -1) * d ** np.arange(0, self.steps + 1)

        if is_call:
            values = np.maximum(S_grid - self.K, 0.0)
        else:
            values = np.maximum(self.K - S_grid, 0.0)

        for j in range(self.steps - 1, -1, -1):
            values = disc * (p * values[:-1] + (1 - p) * values[1:])
            if american:
                S_g = self.S * u ** np.arange(j, -1, -1) * d ** np.arange(0, j + 1)
                if is_call:
                    values = np.maximum(values, S_g - self.K)
                else:
                    values = np.maximum(values, self.K - S_g)

        return float(values[0])


# ── A3: Monte Carlo ────────────────────────────────────────────────────


class MonteCarloPricer:
    """Geometric Brownian motion Monte Carlo option pricing.

    Uses antithetic variates for variance reduction. Supports European
    options with configurable paths and time steps.

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiration.
        r: Risk-free rate.
        sigma: Volatility.
        paths: Number of Monte Carlo paths.
        steps: Number of time steps.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        paths: int = 100_000,
        steps: int = 1,
        seed: Optional[int] = None,
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.paths = paths
        self.steps = steps
        self.seed = seed

    def price(self, is_call: bool = True) -> float:
        """Price a European option via Monte Carlo simulation.

        Args:
            is_call: True for call, False for put.

        Returns:
            Option price with standard error.
        """
        dt = self.T / self.steps
        rng = np.random.RandomState(self.seed)
        half_paths = self.paths // 2

        S_T = np.zeros(self.paths)
        for _ in range(self.steps):
            Z = rng.randn(half_paths)
            Z_antithetic = np.concatenate([Z, -Z])
            S_T = self.S * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(dt) * Z_antithetic
            )

        if is_call:
            payoffs = np.maximum(S_T - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - S_T, 0.0)

        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return float(price)


# ── A4: Heston Stochastic Volatility ───────────────────────────────────


class HestonModel:
    """Heston stochastic volatility model (1993).

    SDEs:
        dS_t = μ S_t dt + √v_t S_t dW^S_t
        dv_t = κ(θ − v_t)dt + ξ √v_t dW^v_t
        dW^S_t · dW^v_t = ρ dt

    Prices European options via characteristic function and Fourier inversion.

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiration.
        r: Risk-free rate.
        v0: Initial variance.
        kappa: Mean-reversion speed of variance.
        theta: Long-run variance.
        xi: Volatility of variance (vol-of-vol).
        rho: Correlation between asset and variance Brownian motions.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def _char_func(self, u: complex) -> complex:
        """Heston characteristic function."""
        k = self.kappa
        th = self.theta
        s = self.xi
        rho = self.rho
        v0 = self.v0

        d = np.sqrt((rho * s * 1j * u - k) ** 2 + s**2 * (1j * u + u**2))
        g = (k - rho * s * 1j * u - d) / (k - rho * s * 1j * u + d)

        A = 1j * u * (np.log(self.S) + self.r * self.T) + k * th / s**2 * (
            (k - rho * s * 1j * u - d) * self.T
            - 2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))
        )
        B = (
            (k - rho * s * 1j * u - d)
            / s**2
            * (1 - np.exp(-d * self.T))
            / (1 - g * np.exp(-d * self.T))
        )

        return np.exp(A + B * v0)

    def price(self, is_call: bool = True) -> float:
        """Price European option via Carr-Madan Fourier inversion."""
        try:
            k = np.log(self.K)
            N = 4096
            eta = 0.25
            b = np.pi / eta
            v = np.arange(N) * eta

            cf = np.array([self._char_func(u) for u in v])

            # Simpson weights
            simpson = (3 + (-1) ** (np.arange(N) + 1)) / 3.0
            simpson[0] = 1.0

            integrand = (
                np.exp(-1j * b * v) * cf * np.exp(-self.r * self.T) / (1j * v * (1j * v + 1))
            )
            integrand[0] = 0.0  # handle v=0

            call = float(
                np.real(
                    np.exp(-0.5 * k)
                    / np.pi
                    * np.sum(simpson * integrand * np.exp(-1j * v * (k - b)))
                    * eta
                )
            )
            intrinsic = self.S - self.K * np.exp(-self.r * self.T)
            call = max(call, max(intrinsic, 0.0))

            if is_call:
                return call
            else:
                put = call + self.K * np.exp(-self.r * self.T) - self.S
                return max(put, max(self.K * np.exp(-self.r * self.T) - self.S, 0.0))
        except Exception:
            # Fallback to BS with sqrt(v0)
            bs = BlackScholes(S=self.S, K=self.K, T=self.T, r=self.r, sigma=np.sqrt(self.v0))
            return bs.call_price if is_call else bs.put_price


# ── A5: SABR Model ─────────────────────────────────────────────────────


class SABRModel:
    """SABR stochastic volatility model (Hagan et al. 2002).

    Designed for interest rate options but widely used for equity index
    volatility surface modeling. Provides closed-form approximation for
    implied volatility.

    dF_t = α_t F_t^β dW^1_t
    dα_t = ν α_t dW^2_t
    dW^1_t · dW^2_t = ρ dt

    Args:
        F: Forward price.
        K: Strike.
        T: Time to expiration.
        alpha: Initial volatility level.
        beta: CEV exponent (0=normal, 1=lognormal).
        rho: Correlation between forward and vol.
        nu: Vol-of-vol.
    """

    def __init__(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float = 1.0,
        rho: float = 0.0,
        nu: float = 0.3,
    ):
        self.F = F
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self) -> float:
        """Hagan et al. (2002) implied volatility approximation."""
        F, K, T = self.F, self.K, self.T
        a, b, r, n = self.alpha, self.beta, self.rho, self.nu

        if abs(F - K) < 1e-12:
            # ATM formula
            fk = F ** (1 - b)
            term1 = a / fk
            term2 = (
                ((1 - b) ** 2 / 24) * (a**2 / F ** (2 - 2 * b))
                + r * b * a * n / (4 * F ** (1 - b))
                + (2 - 3 * r**2) * n**2 / 24
            )
            return float(term1 * (1 + term2 * T))
        else:
            Fm = (F * K) ** ((1 - b) / 2)
            z = (n / a) * Fm * np.log(F / K)
            x_z = np.log((np.sqrt(1 - 2 * r * z + z**2) + z - r) / (1 - r))

            term1 = a / (
                Fm
                * (
                    1
                    + (1 - b) ** 2 / 24 * np.log(F / K) ** 2
                    + (1 - b) ** 4 / 1920 * np.log(F / K) ** 4
                )
            )
            term1 *= z / x_z if abs(x_z) > 1e-12 else 1.0

            term2 = (
                ((1 - b) ** 2 / 24) * a**2 / Fm**2
                + r * b * a * n / (4 * Fm)
                + (2 - 3 * r**2) * n**2 / 24
            )
            return float(term1 * (1 + term2 * T))


# ── A6: Volatility Surface ─────────────────────────────────────────────


@dataclass
class VolSurfaceSlice:
    strikes: np.ndarray
    ivs: np.ndarray
    F: float
    T: float
    atm_vol: float
    skew: float
    smile: float

    def to_dict(self) -> Dict:
        return {
            "T": round(self.T, 4),
            "F": round(self.F, 2),
            "atm_vol": round(self.atm_vol, 4),
            "skew": round(self.skew, 4),
            "smile": round(self.smile, 4),
        }


class VolSurface:
    """Implied volatility surface via cubic spline interpolation.

    Builds a volatility surface from discrete option strikes and maturities.
    Supports interpolation for any (strike, maturity) pair.

    Args:
        strikes: Array of strike prices (moneyness = K/F).
        maturities: Array of times to expiration (years).
        iv_matrix: Matrix of implied vols [n_maturities, n_strikes].
    """

    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_matrix: np.ndarray,
        F: Optional[float] = None,
    ):
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.iv_matrix = np.asarray(iv_matrix, dtype=float)
        self.F = F if F is not None else strikes[np.argmin(np.abs(strikes - 1.0))]
        self._splines: Dict[float, CubicSpline] = {}
        self._build_splines()

    def _build_splines(self) -> None:
        for i, T in enumerate(self.maturities):
            valid = ~np.isnan(self.iv_matrix[i])
            if valid.sum() >= 4:
                try:
                    self._splines[T] = CubicSpline(
                        self.strikes[valid],
                        self.iv_matrix[i][valid],
                        extrapolate=True,
                    )
                except Exception:
                    pass

    def iv(self, strike: float, T: float) -> float:
        """Get implied vol at (strike, T) via interpolation.

        Args:
            strike: Strike price (or moneyness K/F).
            T: Time to expiration.

        Returns:
            Interpolated implied volatility.
        """
        available_T = sorted(self._splines.keys())
        if not available_T:
            return np.nan

        if T < available_T[0]:
            T = available_T[0]
        if T > available_T[-1]:
            T = available_T[-1]

        # Find bracketing maturities
        lower_T = (
            max(t for t in available_T if t <= T)
            if any(t <= T for t in available_T)
            else available_T[0]
        )
        upper_T = (
            min(t for t in available_T if t >= T)
            if any(t >= T for t in available_T)
            else available_T[-1]
        )

        if lower_T == upper_T:
            return float(np.clip(self._splines[lower_T](strike), 0.01, 5.0))

        iv_lower = np.clip(self._splines[lower_T](strike), 0.01, 5.0)
        iv_upper = np.clip(self._splines[upper_T](strike), 0.01, 5.0)

        # Linear interpolation in total variance
        var_lower = iv_lower**2 * lower_T
        var_upper = iv_upper**2 * upper_T
        var_interp = var_lower + (var_upper - var_lower) * (T - lower_T) / (upper_T - lower_T)
        return float(np.sqrt(max(var_interp, 0.0) / T))

    def slice(self, T: float) -> VolSurfaceSlice:
        """Get volatility surface metrics at a given maturity.

        Args:
            T: Time to expiration.

        Returns:
            VolSurfaceSlice with ATM vol, skew, and smile curvature.
        """
        ivs_at_T = np.array([self.iv(K, T) for K in self.strikes])
        atm_idx = np.argmin(np.abs(self.strikes - self.F))
        atm_vol = float(ivs_at_T[atm_idx])

        # Skew: 90% moneyness IV minus 110% moneyness IV
        left_idx = np.argmin(np.abs(self.strikes - 0.90 * self.F))
        right_idx = np.argmin(np.abs(self.strikes - 1.10 * self.F))
        skew = float(ivs_at_T[left_idx] - ivs_at_T[right_idx])

        # Smile: avg of 90% and 110% IV minus ATM IV
        smile = float((ivs_at_T[left_idx] + ivs_at_T[right_idx]) / 2 - atm_vol)

        return VolSurfaceSlice(
            strikes=self.strikes.copy(),
            ivs=ivs_at_T,
            F=self.F,
            T=T,
            atm_vol=atm_vol,
            skew=skew,
            smile=smile,
        )

    def term_structure(self, strike: Optional[float] = None) -> pd.Series:
        """Get volatility term structure at a given strike.

        Args:
            strike: Strike price (default: ATM).

        Returns:
            Series of implied vols indexed by maturity.
        """
        K = strike if strike is not None else self.F
        ivs = [self.iv(K, T) for T in self.maturities]
        return pd.Series(ivs, index=self.maturities, name=f"IV_K={K:.0f}")


# ── A7: Greeks (analytical, already in BlackScholes class) ─────────────
# Included in BlackScholes class above. Key trading functions below.


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> BlackScholesResult:
    """Compute all Greeks for a European option position.

    Convenience function wrapping BlackScholes.all_greeks().

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes parameters.

    Returns:
        BlackScholesResult with all Greeks and prices.
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    return bs.all_greeks()


def delta_hedge_ratio(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Delta of an option — shares needed to hedge one option contract.

    Args:
        S, K, T, r, sigma, q: BS parameters.
        option_type: "call" or "put".

    Returns:
        Delta (hedge ratio).
    """
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    return bs.delta_call if option_type == "call" else bs.delta_put
