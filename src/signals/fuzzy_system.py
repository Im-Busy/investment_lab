"""P25: Fuzzy rule system for trading signals with 5-state membership.

Implements the fuzzy logic expert system from Dymova (2016) "Forex Trading
Expert System" and Sadeghi (2021) "Combined Ensemble SVM + Fuzzy NSGA-II".

Architecture:
- 4 indicators: dEMA + HBar + iV + modified BB
- 5-state trapezoidal membership: Buy, BuyHold, Hold, SellHold, Sell
- Mamdani Min inference with weighted averaging combination

Reference: Dymova et al. (2016), Sadeghi et al. (2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FuzzyState(Enum):
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2

    @property
    def label(self) -> str:
        labels = {
            FuzzyState.STRONG_SELL: "SELL",
            FuzzyState.SELL: "sell",
            FuzzyState.HOLD: "HOLD",
            FuzzyState.BUY: "buy",
            FuzzyState.STRONG_BUY: "BUY",
        }
        return labels[self]


# ── Membership Functions ──


@dataclass
class TrapezoidMF:
    """Trapezoidal membership function defined by 4 points (a, b, c, d)."""

    a: float  # left outer
    b: float  # left inner
    c: float  # right inner
    d: float  # right outer

    def membership(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        if self.b <= x <= self.c:
            return 1.0
        if self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        return (self.d - x) / (self.d - self.c)


@dataclass
class FuzzyVariable:
    """A fuzzy variable with membership functions for each linguistic term."""

    name: str
    membership_functions: dict[FuzzyState, TrapezoidMF]
    value_range: tuple[float, float]

    def fuzzify(self, value: float) -> dict[FuzzyState, float]:
        """Convert crisp value to fuzzy membership values."""
        return {state: mf.membership(value) for state, mf in self.membership_functions.items()}


# ── Default Membership Functions ──


def _make_fuzzy_variables() -> dict[str, FuzzyVariable]:
    """Create default fuzzy variables for the 4-indicator system."""

    return {
        "dEMA": FuzzyVariable(
            name="dEMA",
            membership_functions={
                FuzzyState.STRONG_SELL: TrapezoidMF(-999, -999, -0.01, -0.005),
                FuzzyState.SELL: TrapezoidMF(-0.01, -0.005, -0.002, 0.0),
                FuzzyState.HOLD: TrapezoidMF(-0.003, 0.0, 0.0, 0.003),
                FuzzyState.BUY: TrapezoidMF(0.0, 0.002, 0.005, 0.01),
                FuzzyState.STRONG_BUY: TrapezoidMF(0.005, 0.01, 999, 999),
            },
            value_range=(-0.05, 0.05),
        ),
        "HBar": FuzzyVariable(
            name="HBar",
            membership_functions={
                FuzzyState.STRONG_SELL: TrapezoidMF(-999, -999, -1.0, -0.5),
                FuzzyState.SELL: TrapezoidMF(-1.0, -0.5, -0.2, 0.0),
                FuzzyState.HOLD: TrapezoidMF(-0.3, 0.0, 0.0, 0.3),
                FuzzyState.BUY: TrapezoidMF(0.0, 0.2, 0.5, 1.0),
                FuzzyState.STRONG_BUY: TrapezoidMF(0.5, 1.0, 999, 999),
            },
            value_range=(-3.0, 3.0),
        ),
        "iV": FuzzyVariable(
            name="iV",
            membership_functions={
                FuzzyState.STRONG_SELL: TrapezoidMF(-999, -999, 0.3, 0.5),
                FuzzyState.SELL: TrapezoidMF(0.3, 0.5, 0.7, 0.85),
                FuzzyState.HOLD: TrapezoidMF(0.7, 0.85, 1.15, 1.3),
                FuzzyState.BUY: TrapezoidMF(1.15, 1.3, 1.5, 2.0),
                FuzzyState.STRONG_BUY: TrapezoidMF(1.5, 2.0, 999, 999),
            },
            value_range=(0.0, 5.0),
        ),
        "BB_position": FuzzyVariable(
            name="BB_position",
            membership_functions={
                FuzzyState.STRONG_SELL: TrapezoidMF(-999, -999, 0.0, 0.1),
                FuzzyState.SELL: TrapezoidMF(0.0, 0.1, 0.3, 0.5),
                FuzzyState.HOLD: TrapezoidMF(0.3, 0.5, 0.5, 0.7),
                FuzzyState.BUY: TrapezoidMF(0.5, 0.7, 0.9, 1.0),
                FuzzyState.STRONG_BUY: TrapezoidMF(0.9, 1.0, 999, 999),
            },
            value_range=(0.0, 1.0),
        ),
    }


# ── Fuzzy Rule ──


@dataclass
class FuzzyRule:
    """A single fuzzy AND rule: IF var1=state1 AND var2=state2 AND ... THEN output=state."""

    antecedents: dict[str, FuzzyState]  # variable → required state
    consequent: FuzzyState
    weight: float = 1.0

    def fire_strength(self, fuzzified: dict[str, dict[FuzzyState, float]]) -> float:
        """Compute rule firing strength (Mamdani Min: AND = minimum)."""
        strengths = [
            fuzzified[var].get(state, 0.0)
            for var, state in self.antecedents.items()
            if var in fuzzified
        ]
        if not strengths:
            return 0.0
        return min(strengths) * self.weight


# ── Fuzzy Inference System ──


@dataclass
class FuzzyResult:
    """Output of fuzzy inference system."""

    crisp_output: float  # continuous signal in [-2, 2]
    state: FuzzyState
    rule_firing_strengths: dict[int, float]
    variable_memberships: dict[str, dict[FuzzyState, float]]

    def to_signal(self) -> float:
        """Convert to signal strength in [-1, 1]."""
        return self.crisp_output / 2.0


DEFAULT_RULES = [
    FuzzyRule(
        {"dEMA": FuzzyState.STRONG_BUY, "HBar": FuzzyState.BUY, "BB_position": FuzzyState.BUY},
        FuzzyState.STRONG_BUY,
        weight=1.5,
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.BUY, "HBar": FuzzyState.BUY, "iV": FuzzyState.BUY},
        FuzzyState.BUY,
        weight=1.2,
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.BUY, "BB_position": FuzzyState.STRONG_BUY}, FuzzyState.BUY, weight=1.0
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.STRONG_SELL, "HBar": FuzzyState.SELL, "BB_position": FuzzyState.SELL},
        FuzzyState.STRONG_SELL,
        weight=1.5,
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.SELL, "HBar": FuzzyState.SELL, "iV": FuzzyState.SELL},
        FuzzyState.SELL,
        weight=1.2,
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.SELL, "BB_position": FuzzyState.STRONG_SELL},
        FuzzyState.SELL,
        weight=1.0,
    ),
    FuzzyRule(
        {"HBar": FuzzyState.STRONG_BUY, "iV": FuzzyState.STRONG_BUY}, FuzzyState.BUY, weight=0.8
    ),
    FuzzyRule(
        {"HBar": FuzzyState.STRONG_SELL, "iV": FuzzyState.STRONG_SELL}, FuzzyState.SELL, weight=0.8
    ),
    FuzzyRule(
        {"dEMA": FuzzyState.HOLD, "HBar": FuzzyState.HOLD, "iV": FuzzyState.HOLD},
        FuzzyState.HOLD,
        weight=0.5,
    ),
]


class FuzzyInferenceSystem:
    """Fuzzy logic inference system for trading signal generation.

    Mamdani Min inference + weighted averaging defuzzification.

    Usage:
        fis = FuzzyInferenceSystem()
        signal = fis.evaluate(price_data, ema_short, ema_long, volume_short, volume_long)
        print(signal.to_signal())  # -1.0 to 1.0
    """

    def __init__(self, rules: list[FuzzyRule] | None = None) -> None:
        self.variables = _make_fuzzy_variables()
        self.rules = rules or DEFAULT_RULES

    def add_rule(self, rule: FuzzyRule) -> None:
        self.rules.append(rule)

    def set_rules(self, rules: list[FuzzyRule]) -> None:
        self.rules = rules

    def fuzzify_inputs(self, values: dict[str, float]) -> dict[str, dict[FuzzyState, float]]:
        """Convert crisp input values to fuzzy membership values."""
        result = {}
        for var_name, value in values.items():
            if var_name in self.variables:
                var = self.variables[var_name]
                result[var_name] = var.fuzzify(value)
        return result

    def evaluate(self, values: dict[str, float]) -> FuzzyResult:
        """Run fuzzy inference: fuzzify → infer → defuzzify.

        Args:
            values: Dict of variable_name → crisp value.
                    Required: dEMA, HBar, iV, BB_position

        Returns:
            FuzzyResult with crisp_output, state, and firing details.
        """
        fuzzified = self.fuzzify_inputs(values)

        state_strengths: dict[FuzzyState, float] = {s: 0.0 for s in FuzzyState}

        firing = {}
        for i, rule in enumerate(self.rules):
            strength = rule.fire_strength(fuzzified)
            if strength > 0:
                firing[i] = strength
                current = state_strengths.get(rule.consequent, 0.0)
                state_strengths[rule.consequent] = max(current, strength)

        weights = np.array(
            [
                state_strengths.get(FuzzyState.STRONG_SELL),
                state_strengths.get(FuzzyState.SELL),
                state_strengths.get(FuzzyState.HOLD),
                state_strengths.get(FuzzyState.BUY),
                state_strengths.get(FuzzyState.STRONG_BUY),
            ]
        )
        centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        total_weight = float(np.sum(weights))
        if total_weight > 1e-12:
            crisp = float(np.sum(weights * centers) / total_weight)
        else:
            crisp = 0.0

        distances = np.abs(centers - crisp)
        nearest_state = FuzzyState(int(centers[np.argmin(distances)]))

        return FuzzyResult(
            crisp_output=crisp,
            state=nearest_state,
            rule_firing_strengths=firing,
            variable_memberships=fuzzified,
        )

    def evaluate_df(
        self,
        df: pd.DataFrame,
        ema_short_col: str = "ema_short",
        ema_long_col: str = "ema_long",
        hbar_col: str = "hbar",
        iv_col: str = "ivol",
        bb_lower_col: str = "bb_lower",
        bb_upper_col: str = "bb_upper",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """Evaluate fuzzy system on a DataFrame of indicators.

        Returns DataFrame with crisp_output, state, and signal columns.
        """
        results = []
        for i in range(len(df)):
            close = float(df[close_col].iloc[i])
            ema_s = float(df[ema_short_col].iloc[i])
            ema_l = float(df[ema_long_col].iloc[i])
            dEMA = (ema_s - ema_l) / ema_l if ema_l != 0 else 0.0
            hbar_val = float(df[hbar_col].iloc[i])
            iv_val = float(df[iv_col].iloc[i])
            bb_l = float(df[bb_lower_col].iloc[i])
            bb_u = float(df[bb_upper_col].iloc[i])
            bb_pos = (close - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) != 0 else 0.5

            result = self.evaluate(
                {
                    "dEMA": dEMA,
                    "HBar": hbar_val,
                    "iV": iv_val,
                    "BB_position": np.clip(bb_pos, 0.0, 1.0),
                }
            )
            results.append(
                {
                    "crisp_output": result.crisp_output,
                    "state": result.state.value,
                    "signal": result.to_signal(),
                }
            )

        return pd.DataFrame(results, index=df.index)
