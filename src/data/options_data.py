"""P28-20: Options data coverage — Greeks, chain analysis, and ML features.

Provides options chain analysis via yfinance for index/ETF/equity
options. Computes Greeks approximations, put/call ratios, and unusual
options activity detection.

Source: stock-sdk Features — CFFEX index options, SSE ETF options,
commodity options with T-quotes and Greeks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

logger = logging.getLogger(__name__)

_N = norm.cdf
_N_PDF = norm.pdf


@dataclass
class OptionGreeks:
    """Approximate option Greeks computed from market data."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float | None = None

    def summary(self) -> str:
        return (
            f"delta={self.delta:+.3f} gamma={self.gamma:.4f} "
            f"theta={self.theta:.4f} vega={self.vega:.4f}"
        )


@dataclass
class OptionsChain:
    """Full options chain for a single expiration date."""

    ticker: str
    expiration: str
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    fetch_time: str = ""

    @property
    def pc_ratio_volume(self) -> float:
        """Put/call volume ratio."""
        call_vol = self.calls.get("volume", pd.Series(dtype=float)).sum()
        put_vol = self.puts.get("volume", pd.Series(dtype=float)).sum()
        if call_vol and call_vol > 0:
            return float(put_vol / call_vol)
        return 0.0

    @property
    def pc_ratio_oi(self) -> float:
        """Put/call open interest ratio."""
        call_oi = self.calls.get("openInterest", pd.Series(dtype=float)).sum()
        put_oi = self.puts.get("openInterest", pd.Series(dtype=float)).sum()
        if call_oi and call_oi > 0:
            return float(put_oi / call_oi)
        return 0.0

    @property
    def max_pain(self) -> float:
        """Max pain strike — price where option buyers lose the most."""
        strikes = self.calls.get("strike", pd.Series(dtype=float)).unique()
        if len(strikes) == 0:
            return 0.0
        pain = {}
        for k in strikes:
            call_loss = self._itm_value(self.calls, k, "call")
            put_loss = self._itm_value(self.puts, k, "put")
            pain[k] = call_loss + put_loss
        return float(min(pain, key=pain.get)) if pain else 0.0

    @property
    def atm_iv(self) -> float:
        """At-the-money implied volatility (average of call and put)."""
        spot = self.spot
        if not spot:
            return 0.0
        call_strikes = self.calls.get("strike", pd.Series(dtype=float))
        put_strikes = self.puts.get("strike", pd.Series(dtype=float))
        if len(call_strikes) == 0 and len(put_strikes) == 0:
            return 0.0
        all_strikes = pd.concat([call_strikes, put_strikes])
        atm_strike = all_strikes.iloc[(all_strikes - spot).abs().argmin()]
        call_iv = self._iv_at_strike(self.calls, atm_strike)
        put_iv = self._iv_at_strike(self.puts, atm_strike)
        return (call_iv + put_iv) / 2 if call_iv and put_iv else (call_iv or put_iv or 0.0)

    @staticmethod
    def _itm_value(df: pd.DataFrame, strike: float, option_type: str) -> float:
        oi_col = "openInterest" if "openInterest" in df.columns else "volume"
        if oi_col not in df.columns:
            return 0.0
        if option_type == "call":
            mask = df.get("strike", pd.Series(dtype=float)) < strike
        else:
            mask = df.get("strike", pd.Series(dtype=float)) > strike
        itm_oi = df.loc[mask, oi_col].sum()
        diff = abs(df.loc[mask, "strike"] - strike).fillna(0)
        return float((itm_oi * diff).sum()) / float(itm_oi) if itm_oi and itm_oi > 0 else 0.0

    @staticmethod
    def _iv_at_strike(df: pd.DataFrame, strike: float) -> float:
        iv_col = next(
            (c for c in ["impliedVolatility", "implied_volatility"] if c in df.columns), None
        )
        if iv_col is None:
            return 0.0
        match = df[df.get("strike", pd.Series(dtype=float)) == strike]
        if len(match) == 0:
            return 0.0
        return float(match[iv_col].iloc[0]) if match[iv_col].iloc[0] else 0.0


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> OptionGreeks:
    """Compute Black-Scholes Greeks for a single option.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (decimal).
        sigma: Implied volatility (decimal).
        option_type: 'call' or 'put'.

    Returns:
        OptionGreeks with delta, gamma, theta, vega.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return OptionGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = _N(d1)
        theta = (
            -(S * _N_PDF(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * _N(d2)
        ) / 365
    else:
        delta = _N(d1) - 1
        theta = (
            -(S * _N_PDF(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * _N(-d2)
        ) / 365

    gamma = _N_PDF(d1) / (S * sigma * np.sqrt(T))
    vega = S * _N_PDF(d1) * np.sqrt(T) / 100

    return OptionGreeks(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
    )


def fetch_options_chain(
    ticker: str,
    expiration: Optional[str] = None,
) -> Optional[OptionsChain]:
    """Fetch the options chain for a ticker from yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol.
        expiration: Expiration date string YYYY-MM-DD. Default: nearest.

    Returns:
        OptionsChain if data available, None otherwise.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        expirations = yf_ticker.options
        if not expirations:
            return None

        expiration = expiration or expirations[0]
        if expiration not in expirations:
            expiration = expirations[0]

        opt = yf_ticker.option_chain(expiration)
        spot = yf_ticker.history(period="5d")
        spot_price = float(spot["Close"].iloc[-1]) if not spot.empty else 0.0

        return OptionsChain(
            ticker=ticker,
            expiration=expiration,
            spot=spot_price,
            calls=opt.calls,
            puts=opt.puts,
            fetch_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    except Exception as e:
        logger.debug(f"Options chain failed for {ticker}: {e}")
        return None


def fetch_multiple_chains(
    tickers: list[str],
) -> dict[str, Optional[OptionsChain]]:
    """Fetch options chains for multiple tickers."""
    results = {}
    for ticker in tickers:
        results[ticker] = fetch_options_chain(ticker)
    return results


def options_chain_to_features(
    chain: OptionsChain,
) -> dict[str, float]:
    """Convert an options chain to ML-ready features.

    Args:
        chain: OptionsChain from fetch_options_chain().

    Returns:
        Dict of option-derived features for ML models.
    """
    features: dict[str, float] = {
        "options_pc_volume": chain.pc_ratio_volume,
        "options_pc_oi": chain.pc_ratio_oi,
        "options_atm_iv": chain.atm_iv,
        "options_max_pain": chain.max_pain,
        "options_spot": chain.spot,
    }

    if chain.spot > 0 and chain.max_pain > 0:
        features["options_max_pain_pct"] = (chain.spot - chain.max_pain) / chain.spot

    features["options_call_volume"] = (
        float(chain.calls["volume"].sum()) if "volume" in chain.calls.columns else 0.0
    )
    features["options_put_volume"] = (
        float(chain.puts["volume"].sum()) if "volume" in chain.puts.columns else 0.0
    )

    oi_c = "openInterest" if "openInterest" in chain.calls.columns else "volume"
    oi_p = "openInterest" if "openInterest" in chain.puts.columns else "volume"
    features["options_call_oi"] = (
        float(chain.calls[oi_c].sum()) if oi_c in chain.calls.columns else 0.0
    )
    features["options_put_oi"] = (
        float(chain.puts[oi_p].sum()) if oi_p in chain.puts.columns else 0.0
    )

    return features


def detect_unusual_options_activity(
    chain: OptionsChain,
    volume_threshold: float = 3.0,
) -> list[dict]:
    """Detect unusual options activity (high volume relative to OI).

    Args:
        chain: OptionsChain to scan.
        volume_threshold: Volume/OI ratio to flag as unusual.

    Returns:
        List of dicts describing unusual activity per strike/type.
    """
    unusual = []
    if "volume" not in chain.calls.columns or "openInterest" not in chain.calls.columns:
        return unusual

    oi_col = "openInterest"

    for side, df in [("call", chain.calls), ("put", chain.puts)]:
        for _, row in df.iterrows():
            oi = row.get(oi_col, 0) or 0
            vol = row.get("volume", 0) or 0
            if oi == 0:
                continue
            ratio = vol / oi
            if ratio >= volume_threshold:
                unusual.append(
                    {
                        "type": side,
                        "strike": float(row["strike"]),
                        "expiration": chain.expiration,
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "vol_oi_ratio": round(ratio, 1),
                        "last_price": float(row.get("lastPrice", 0) or 0),
                    }
                )

    return sorted(unusual, key=lambda x: x["vol_oi_ratio"], reverse=True)


def get_options_sentiment(
    ticker: str,
) -> dict[str, float]:
    """Quick options sentiment snapshot for a ticker.

    Returns put/call ratios, ATM IV, and max pain as a simple dict.
    """
    chain = fetch_options_chain(ticker)
    if chain is None:
        return {"pc_volume": 0.0, "pc_oi": 0.0, "atm_iv": 0.0, "max_pain_pct": 0.0}
    return {
        "pc_volume": round(chain.pc_ratio_volume, 3),
        "pc_oi": round(chain.pc_ratio_oi, 3),
        "atm_iv": round(chain.atm_iv, 3),
        "max_pain_pct": round((chain.spot - chain.max_pain) / chain.spot, 3)
        if chain.spot > 0 and chain.max_pain > 0
        else 0.0,
    }
