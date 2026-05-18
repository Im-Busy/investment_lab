"""
B7: Interest Rate Derivative Pricing — swaps, caps, floors, swaptions, and signals.

Rate derivatives are the deepest source of market-implied forward-looking
information for fixed income and macro regime signals.

Models:
    DiscountCurve: Bootstrap zero-coupon discount factors from swap/futures rates.
    InterestRateSwap: IRS pricing (NPV, par rate, DV01, carry/rolldown).
    SwaptionPricer: Payer/receiver swaption via Black (normal) model.
    CapFloorPricer: Cap/floor as portfolio of caplets/floorlets via Black.
    RateDerivativeSignal: Forward rates, swap spread, swaption vol regime for ML.

Usage:
    >>> curve = DiscountCurve.from_flat(0.05)
    >>> irs = InterestRateSwap(curve)
    >>> result = irs.price(notional=1_000_000, fixed_rate=0.05, maturity=5.0)
    >>> print(f"Swap NPV: ${result.npv:,.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

FREQ_SEMIANNUAL = 2
FREQ_QUARTERLY = 4


# ── Discount Curve ──────────────────────────────────────────────────────


@dataclass
class DiscountCurve:
    """Zero-coupon discount factor curve.

    Maps time-to-maturity (years) to present value of $1 at that time.
    Built from swap rates, bond yields, or fitted models (Nelson-Siegel, Vasicek).

    Args:
        maturities: Tenor points in years.
        discount_factors: PV of $1 at each maturity.
        zero_rates: Continuously compounded zero rates (optional, computed from DFs).
    """

    maturities: np.ndarray
    discount_factors: np.ndarray
    zero_rates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.zero_rates = -np.log(np.maximum(self.discount_factors, 1e-12)) / np.maximum(
            self.maturities, 1e-12
        )

    @classmethod
    def from_flat(cls, rate: float, maturities: Optional[np.ndarray] = None) -> DiscountCurve:
        """Create flat yield curve from a single continuously compounded rate."""
        if maturities is None:
            maturities = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])
        dfs = np.exp(-rate * maturities)
        return cls(maturities=maturities, discount_factors=dfs)

    @classmethod
    def from_zero_rates(
        cls, maturities: np.ndarray, zero_rates: np.ndarray, freq: int = FREQ_SEMIANNUAL
    ) -> DiscountCurve:
        """Build curve from zero-coupon rates using given compounding frequency."""
        dfs = (1.0 + zero_rates / freq) ** (-freq * maturities)
        return cls(maturities=maturities, discount_factors=dfs)

    @classmethod
    def from_swap_rates(
        cls,
        maturities: np.ndarray,
        swap_rates: np.ndarray,
        freq: int = FREQ_SEMIANNUAL,
    ) -> DiscountCurve:
        """Bootstrap discount factors from par swap rates.

        Uses iterative bootstrap: for each maturity, solve for DF such
        that the fixed leg PV equals the floating leg PV (both = 1 at par).
        """
        n = len(maturities)
        dfs = np.ones(n)
        for i in range(n):
            T = maturities[i]
            dt = 1.0 / freq
            n_periods = int(T / dt + 0.5)
            payment_times = np.linspace(dt, T, n_periods)

            if i == 0:
                dfs[i] = 1.0 / (1.0 + swap_rates[i] * T)
            else:
                known_dfs = np.array(
                    [
                        DiscountCurve._interp_dfs(maturities[:i], dfs[:i], t)
                        for t in payment_times[:-1]
                    ]
                )
                annuity = dt * np.sum(known_dfs)
                dfs[i] = (1.0 - swap_rates[i] * annuity) / (1.0 + swap_rates[i] * dt)
                dfs[i] = max(dfs[i], 1e-12)
        return cls(maturities=maturities, discount_factors=dfs)

    @staticmethod
    def _interp_dfs(maturities: np.ndarray, dfs: np.ndarray, target: float) -> float:
        """Linear interpolation on log discount factors."""
        if target <= maturities[0]:
            return float(np.exp(np.log(dfs[0]) * target / maturities[0]))
        if target >= maturities[-1]:
            return float(dfs[-1])
        i = np.searchsorted(maturities, target) - 1
        i = max(0, min(i, len(maturities) - 2))
        t0, t1 = maturities[i], maturities[i + 1]
        w = (target - t0) / (t1 - t0)
        return float(np.exp(np.log(dfs[i]) * (1 - w) + np.log(dfs[i + 1]) * w))

    def df(self, t: float) -> float:
        """Get discount factor at time t (interpolated)."""
        return self._interp_dfs(self.maturities, self.discount_factors, t)

    def zero_rate(self, t: float, freq: int = FREQ_SEMIANNUAL) -> float:
        """Get continuously compounded zero rate at maturity t."""
        d = self.df(t)
        return float(-np.log(max(d, 1e-12)) / max(t, 1e-12))

    def forward_rate(self, t1: float, t2: float, freq: int = FREQ_SEMIANNUAL) -> float:
        """Forward rate between t1 and t2."""
        df1 = self.df(t1)
        df2 = self.df(t2)
        tau = t2 - t1
        return float(freq * ((df1 / df2) ** (1.0 / (freq * tau)) - 1.0))


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class IRSResult:
    """Interest rate swap pricing result."""

    notional: float
    maturity: float
    fixed_rate: float
    swap_rate: float
    npv: float
    npv_receiver: float
    npv_payer: float
    dv01: float
    fixed_leg_pv: float
    floating_leg_pv: float

    def to_dict(self) -> Dict:
        return {
            "notional": self.notional,
            "maturity": self.maturity,
            "fixed_rate": round(self.fixed_rate, 6),
            "swap_rate": round(self.swap_rate, 6),
            "npv": round(self.npv, 2),
            "npv_receiver": round(self.npv_receiver, 2),
            "npv_payer": round(self.npv_payer, 2),
            "dv01": round(self.dv01, 2),
            "fixed_leg_pv": round(self.fixed_leg_pv, 2),
            "floating_leg_pv": round(self.floating_leg_pv, 2),
        }


@dataclass
class SwaptionResult:
    """Swaption pricing result via Black (normal) model."""

    option_type: str
    expiry: float
    underlying_swap_maturity: float
    strike: float
    forward_swap_rate: float
    normal_vol: float
    npv: float

    def to_dict(self) -> Dict:
        return {
            "option_type": self.option_type,
            "expiry": self.expiry,
            "underlying_swap_maturity": self.underlying_swap_maturity,
            "strike": round(self.strike, 6),
            "forward_swap_rate": round(self.forward_swap_rate, 6),
            "normal_vol": round(self.normal_vol, 6),
            "npv": round(self.npv, 2),
        }


@dataclass
class CapFloorResult:
    """Cap/floor pricing result as portfolio of caplets/floorlets."""

    option_type: str
    maturity: float
    strike: float
    npv: float
    n_caplets: int
    caplet_details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "option_type": self.option_type,
            "maturity": self.maturity,
            "strike": round(self.strike, 6),
            "npv": round(self.npv, 2),
            "n_caplets": self.n_caplets,
        }


# ── B7a: Interest Rate Swap ─────────────────────────────────────────────


class InterestRateSwap:
    """Plain vanilla interest rate swap pricing.

    Fixed leg: pay/receive fixed rate K on notional N at frequency F.
    Floating leg: receive/pay reference rate (SOFR/LIBOR) at same frequency.
    Net PV = (∑ DF_i * τ_i * (Fwd_i − K)) * N for payer.
    """

    def __init__(self, curve: DiscountCurve, freq: int = FREQ_SEMIANNUAL):
        self.curve = curve
        self.freq = freq

    def price(
        self,
        notional: float,
        fixed_rate: float,
        maturity: float,
        payer: bool = True,
    ) -> IRSResult:
        dt = 1.0 / self.freq
        n_periods = int(maturity / dt + 0.5)
        payment_times = np.linspace(dt, maturity, n_periods)

        dfs = np.array([self.curve.df(t) for t in payment_times])
        dfs_prev = np.array([self.curve.df(max(t - dt, 0)) for t in payment_times])

        forward_rates = (dfs_prev / dfs - 1.0) / dt

        fixed_pv = fixed_rate * dt * np.sum(dfs) * notional
        float_pv = np.sum(forward_rates * dt * dfs) * notional

        if payer:
            npv_val = float_pv - fixed_pv
        else:
            npv_val = fixed_pv - float_pv

        swap_rate = self._par_rate(maturity)
        dv01_val = self._dv01(notional, maturity)

        return IRSResult(
            notional=notional,
            maturity=maturity,
            fixed_rate=fixed_rate,
            swap_rate=swap_rate,
            npv=npv_val,
            npv_receiver=fixed_pv - float_pv,
            npv_payer=float_pv - fixed_pv,
            dv01=dv01_val,
            fixed_leg_pv=fixed_pv,
            floating_leg_pv=float_pv,
        )

    def _par_rate(self, maturity: float) -> float:
        dt = 1.0 / self.freq
        n_periods = int(maturity / dt + 0.5)
        payment_times = np.linspace(dt, maturity, n_periods)
        dfs = np.array([self.curve.df(t) for t in payment_times])
        annuity = dt * np.sum(dfs)
        return float((1.0 - self.curve.df(maturity)) / annuity)

    def _dv01(self, notional: float, maturity: float, bp: float = 0.0001) -> float:
        par = self._par_rate(maturity)
        npv_up = self.price(notional, par + bp, maturity).npv
        npv_down = self.price(notional, par - bp, maturity).npv
        return (npv_up - npv_down) / 2.0

    def par_swap_rates(self, maturities: List[float]) -> List[Tuple[float, float]]:
        """Compute par swap rates for multiple maturities."""
        return [(m, self._par_rate(m)) for m in maturities]


# ── B7b: Swaption Pricer ────────────────────────────────────────────────


class SwaptionPricer:
    """European swaption pricing via Black (normal) model.

    A payer swaption gives the right to enter a payer swap at strike K.
    A receiver swaption gives the right to enter a receiver swap at strike K.

    Black (normal) price:
        Payer:  A * [F * N(d1) − K * N(d2)]
        Receiver: A * [K * N(−d2) − F * N(−d1)]
    where
        d1 = (F − K) / (σ * √T)
        d2 = d1 − σ * √T
        A = annuity = τ * Σ DF(T_i) (forward annuity)
    """

    def __init__(self, curve: DiscountCurve, swap_freq: int = FREQ_SEMIANNUAL):
        self.curve = curve
        self.swap_freq = swap_freq

    def price(
        self,
        expiry: float,
        swap_maturity: float,
        strike: float,
        normal_vol: float,
        option_type: str = "payer",
        notional: float = 1_000_000.0,
    ) -> SwaptionResult:
        forward_swap_rate = self._forward_swap_rate(expiry, swap_maturity)
        annuity = self._compute_annuity(expiry, swap_maturity)

        T = expiry
        sigma_T = normal_vol * np.sqrt(max(T, 1e-10))

        if sigma_T < 1e-12:
            if option_type == "payer":
                pv = max(forward_swap_rate - strike, 0.0) * annuity * notional
            else:
                pv = max(strike - forward_swap_rate, 0.0) * annuity * notional
        else:
            d1 = (forward_swap_rate - strike) / sigma_T
            d2 = d1 - sigma_T

            if option_type == "payer":
                intrinsic = (forward_swap_rate - strike) * norm.cdf(d1) + sigma_T * norm.pdf(d1)
            else:
                intrinsic = (strike - forward_swap_rate) * norm.cdf(-d1) + sigma_T * norm.pdf(d1)

            pv = intrinsic * annuity * notional

        return SwaptionResult(
            option_type=option_type,
            expiry=expiry,
            underlying_swap_maturity=swap_maturity,
            strike=strike,
            forward_swap_rate=forward_swap_rate,
            normal_vol=normal_vol,
            npv=pv,
        )

    def _forward_swap_rate(self, expiry: float, swap_maturity: float) -> float:
        """Forward-starting swap rate from expiry to expiry+swap_maturity."""
        df_start = self.curve.df(expiry)
        df_end = self.curve.df(expiry + swap_maturity)
        dt = 1.0 / self.swap_freq
        n_periods = int(swap_maturity / dt + 0.5)
        payment_times = expiry + np.linspace(dt, swap_maturity, n_periods)
        dfs = np.array([self.curve.df(t) for t in payment_times])
        annuity = dt * np.sum(dfs)
        return float((df_start - df_end) / annuity)

    def _compute_annuity(self, expiry: float, swap_maturity: float) -> float:
        """PV of $1 paid at each swap coupon date, discounted to today."""
        dt = 1.0 / self.swap_freq
        n_periods = int(swap_maturity / dt + 0.5)
        payment_times = expiry + np.linspace(dt, swap_maturity, n_periods)
        dfs = np.array([self.curve.df(t) for t in payment_times])
        return float(dt * np.sum(dfs))

    def implied_vol(
        self,
        expiry: float,
        swap_maturity: float,
        strike: float,
        option_price: float,
        option_type: str = "payer",
        notional: float = 1_000_000.0,
    ) -> float:
        """Invert Black formula to find implied normal vol."""

        def objective(vol: float) -> float:
            return (
                self.price(expiry, swap_maturity, strike, max(vol, 1e-8), option_type, notional).npv
                - option_price
            )

        try:
            return float(brentq(objective, 1e-8, 2.0, xtol=1e-8))
        except (ValueError, RuntimeError):
            return np.nan


# ── B7c: Cap/Floor Pricer ───────────────────────────────────────────────


class CapFloorPricer:
    """Cap and floor pricing as portfolio of caplets/floorlets.

    Each caplet/floorlet is an option on a single forward rate period.
    Priced using Black (normal) model on each forward rate.

    Cap = Σ caplet_i — gives the right to receive max(F_i − K, 0) * τ * N
    Floor = Σ floorlet_i — gives the right to receive max(K − F_i, 0) * τ * N
    """

    def __init__(self, curve: DiscountCurve, freq: int = FREQ_QUARTERLY):
        self.curve = curve
        self.freq = freq

    def price(
        self,
        maturity: float,
        strike: float,
        normal_vol: float,
        option_type: str = "cap",
        notional: float = 1_000_000.0,
    ) -> CapFloorResult:
        dt = 1.0 / self.freq
        n_caplets = int(maturity / dt + 0.5)
        if n_caplets < 1:
            n_caplets = 1

        total_npv = 0.0
        details: List[Dict] = []

        for i in range(n_caplets):
            t_start = i * dt
            t_end = (i + 1) * dt

            if i == 0:
                continue  # Skip the stub (already past reset)

            df_pay = self.curve.df(t_end)
            forward = self.curve.forward_rate(t_start, t_end, self.freq)
            tau = dt

            if normal_vol < 1e-12:
                if option_type == "cap":
                    payoff = max(forward - strike, 0.0)
                else:
                    payoff = max(strike - forward, 0.0)
            else:
                sigma_T = normal_vol * np.sqrt(max(t_start, 1e-10))
                d1 = (
                    (forward - strike) / sigma_T
                    if sigma_T > 1e-12
                    else (1.0 if forward > strike else -1.0)
                )
                d2 = d1 - sigma_T

                if option_type == "cap":
                    payoff = (forward - strike) * norm.cdf(d1) + sigma_T * norm.pdf(d1)
                else:
                    payoff = (strike - forward) * norm.cdf(-d1) + sigma_T * norm.pdf(d1)

            caplet_npv = payoff * tau * df_pay * notional
            total_npv += caplet_npv
            details.append(
                {
                    "start": round(t_start, 4),
                    "end": round(t_end, 4),
                    "forward": round(forward, 6),
                    "df_pay": round(df_pay, 4),
                    "npv": round(caplet_npv, 2),
                }
            )

        return CapFloorResult(
            option_type=option_type,
            maturity=maturity,
            strike=strike,
            npv=total_npv,
            n_caplets=n_caplets,
            caplet_details=details,
        )


# ── B7d: Rate Derivative Signals ───────────────────────────────────────


class RateDerivativeSignal:
    """Extract ML features and macro signals from rate derivative data.

    Converts swap/sawaption/cap data into structured features for the
    CatBoost ML pipeline.

    Key signals:
        - Forward rate curves (expectations about future rates)
        - Swap spread vs Treasury (credit/macro stress)
        - Steepness of forward rate curve (economic expectations)
        - Swaption-implied volatility regime
        - Cap/floor skew (market-implied asymmetry)
    """

    def __init__(self, curve: DiscountCurve):
        self.curve = curve

    def forward_rate_curve(
        self, start_tenors: Optional[List[float]] = None, forward_period: float = 0.25
    ) -> Dict[str, float]:
        """Compute forward rate curve for multiple start tenors.

        Returns:
            Dict mapping tenor label to forward rate.
        """
        if start_tenors is None:
            start_tenors = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        return {
            f"fwd_{int(t * 12)}m": self.curve.forward_rate(t, t + forward_period)
            for t in start_tenors
        }

    def swap_spread_vs_treasury(
        self,
        swap_maturities: Optional[List[float]] = None,
        treasury_rates: Optional[Dict[float, float]] = None,
    ) -> Dict[str, float]:
        """Compute swap rate minus Treasury rate (swap spread).

        Positive swap spread = banks paying premium for fixed-rate exposure.
        Widening = financial stress / credit concerns.

        Args:
            swap_maturities: List of maturities to compute spread at.
            treasury_rates: Dict of {maturity: rate} for Treasury yields.
                            If None, estimates from curve itself.
        """
        if swap_maturities is None:
            swap_maturities = [2.0, 5.0, 10.0, 30.0]

        irs = InterestRateSwap(self.curve)
        result = {}
        for m in swap_maturities:
            swap_rate = irs._par_rate(m)
            tres_rate = treasury_rates.get(m, swap_rate) if treasury_rates else swap_rate * 0.85
            result[f"swap_spread_{int(m)}y"] = round(swap_rate - tres_rate, 6)
        return result

    def curve_steepness_signals(self) -> Dict[str, float]:
        """Yield curve steepness features using forward rate differences.

        Returns:
            Dict with steepness metrics (2s10s, 5s30s, forward vs spot).
        """
        irs = InterestRateSwap(self.curve)
        r2 = irs._par_rate(2.0)
        r5 = irs._par_rate(5.0)
        r10 = irs._par_rate(10.0)
        r30 = irs._par_rate(30.0)

        fwd_2y_1y = self.curve.forward_rate(1.0, 3.0)
        fwd_5y_5y = self.curve.forward_rate(5.0, 10.0)

        return {
            "swap_2s10s_spread": round(r10 - r2, 6),
            "swap_5s30s_spread": round(r30 - r5, 6),
            "fwd_2y1y_vs_spot_2y": round(fwd_2y_1y - r2, 6),
            "fwd_5y5y_vs_spot_10y": round(fwd_5y_5y - r10, 6),
            "curve_level": round(r10, 6),
        }

    def swaption_regime(
        self, expiry: float = 0.25, swap_maturity: float = 10.0, atm_vol: float = 0.008
    ) -> Dict[str, float]:
        """Extract swaption vol regime features.

        Args:
            expiry: Option expiry in years.
            swap_maturity: Underlying swap maturity.
            atm_vol: ATM normal vol (as decimal, e.g. 0.008 = 80bp).

        Returns:
            Dict with vol level, percentile classification, and signal.
        """
        pricer = SwaptionPricer(self.curve)
        forward = pricer._forward_swap_rate(expiry, swap_maturity)

        atm_price = pricer.price(expiry, swap_maturity, forward, atm_vol, "payer").npv
        otm_strike_up = forward * 1.02
        otm_strike_dn = forward * 0.98
        otm_price_up = pricer.price(expiry, swap_maturity, otm_strike_up, atm_vol, "payer").npv
        otm_price_dn = pricer.price(expiry, swap_maturity, otm_strike_dn, atm_vol, "receiver").npv

        volatility_regime = "LOW" if atm_vol < 0.005 else "NORMAL" if atm_vol < 0.010 else "HIGH"

        return {
            "swaption_atm_vol_bp": round(atm_vol * 10_000, 1),
            "swaption_forward_rate": round(forward, 6),
            "swaption_atm_npv": round(atm_price, 2),
            "swaption_skew": round(np.log(otm_price_up / max(otm_price_dn, 1e-12)), 4),
            "swaption_vol_regime": volatility_regime,
            "swaption_vol_regime_code": {"LOW": 0, "NORMAL": 1, "HIGH": 2}.get(
                volatility_regime, 1
            ),
        }

    def macro_rate_signals(self, history_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Generate a full set of rate derivative features for ML pipeline.

        Args:
            history_df: Optional historical DataFrame with 'rate' column for
                       computing percentile ranks and trend features.

        Returns:
            Dict of feature_name → value for CatBoost input.
        """
        features: Dict[str, float] = {}

        # Forward rate curve
        features.update(self.forward_rate_curve())

        # Curve steepness
        features.update(self.curve_steepness_signals())

        # Swaption regime
        try:
            features.update(
                {k: v for k, v in self.swaption_regime().items() if isinstance(v, (int, float))}
            )
        except Exception:
            pass

        return features


# ── Utility Functions ────────────────────────────────────────────────────


def compute_forward_curve_from_df(
    df: pd.DataFrame,
    price_col: str = "close",
    maturities: Optional[List[float]] = None,
) -> DiscountCurve:
    """Build a rough discount curve from a bond ETF price series.

    Uses the YTM approach: fits a simple yield from bond price,
    assumes flat curve for forwards.

    Args:
        df: DataFrame with bond ETF prices (e.g. TLT, IEF, SHY).
        price_col: Column name for price data.
        maturities: Maturities to build curve at.

    Returns:
        DiscountCurve built from YTM estimates.
    """
    if maturities is None:
        maturities = [3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    maturities = np.array(maturities)

    prices = df[price_col].values
    if len(prices) < 20:
        return DiscountCurve.from_flat(0.04)

    # Rough YTM from price: assume price ≈ 100 * exp(-ytm * maturity)
    # For TLT, use ~17yr effective maturity
    current_price = float(prices[-1])
    recent_avg = float(np.mean(prices[-20:]))
    pct_change = current_price / recent_avg - 1.0

    # Calibrate a curve with slope from price change
    base_rate = max(0.01, 0.03 - pct_change * 2.0)
    slope = -pct_change * 0.5

    zero_rates = base_rate + slope * (maturities - 1.0) / 29.0
    zero_rates = np.clip(zero_rates, 0.001, 0.20)

    return DiscountCurve.from_zero_rates(maturities, zero_rates)


def swap_spread_signal(
    curve: DiscountCurve,
    treasury_rates: Optional[Dict[float, float]] = None,
) -> float:
    """Compute 10-year swap spread as a macro stress signal.

    Args:
        curve: Current discount curve.
        treasury_rates: Dict of {maturity: rate} or None for rough estimate.

    Returns:
        Swap spread in basis points.
    """
    irs = InterestRateSwap(curve)
    swap_10y = irs._par_rate(10.0)
    if treasury_rates:
        tres_10y = treasury_rates.get(10.0, swap_10y * 0.9)
    else:
        tres_10y = swap_10y * 0.9
    return (swap_10y - tres_10y) * 10_000
