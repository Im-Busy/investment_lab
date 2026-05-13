"""
ML-Powered Strategy for backtesting.py using CatBoost Pattern Classifier V3.

Precomputes feature-based probability scores at init time, then uses them
for entry/exit decisions with configurable volatility gate, consecutive
confirmation, trailing stop, conviction-based position scaling, and
C8 pattern boost filter.

Usage:
    from backtesting import Backtest
    from src.strategies.ml_strategy import MLStrategy
    import pandas as pd

    bt = Backtest(df, MLStrategy, cash=10_000, commission=0.001)
    stats = bt.run()
    bt.plot()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from src.ml.feature_engineering import FeatureExtractor
from src.signals.pattern_boost import PatternBoostFilter


class MLStrategy(Strategy):
    """Strategy driven by CatBoost pattern classifier probability scores.

    Parameters:
        model_path: Path to trained PatternClassifier pickle file.
        entry_threshold: Minimum probability to enter a long position (default 0.50).
        exit_threshold: Probability below which to exit a long position (default 0.35).
        tp_atr_mult: Take-profit multiplier on ATR(14) (default 3.0).
        sl_atr_mult: Stop-loss multiplier on ATR(14) (default 1.5).
        risk_pct: Fraction of equity to risk per trade (default 0.02).

        ── C7 Refinement Parameters ──
        vol_gate_threshold: Max vol_regime allowed for entry (None = disabled).
            vol_regime = ATR(20)/ATR(100). Default None (off).
            Recommended: 1.5 (skip entries during extreme vol expansion).
        confirm_bars: Require prob >= entry_threshold for N consecutive bars
            before entering. Default 1 (off). Recommended: 2-3.
        use_trail_stop: Replace fixed TP with ATR-trailing stop. Default False.
            Trail stop starts sl_atr_mult * ATR below highest high since entry.
        trail_atr_mult: ATR multiplier for trailing stop distance (default 3.0).
        conviction_scale: Boost position size for higher-confidence signals.
            If True, size scales linearly from prob=entry_threshold (1x) to
            prob=max observed (~0.60) (2x). Default False.

        ── C8 Pattern Boost Parameters ──
        use_pattern_boost: Enable chart pattern signal boost. When True, runs
            34 pattern detectors each bar and adds a boost (0.0–0.10) to the
            ML probability when patterns confirm the ML direction. Default False.
        pattern_boost_scale: Base scaling factor (default 0.05). Multiplied by
            average pattern reliability weight + confluence bonus.
        pattern_boost_max: Maximum boost cap (default 0.10).
        pattern_confluence_bonus: Additional boost per extra pattern beyond
            the first one firing (default 0.02).
    """

    # ── Base parameters ──
    model_path: str = "models/pattern_classifier_v3_SPY_20260511_224704.pkl"
    ticker: str = ""
    entry_threshold: float = 0.50
    exit_threshold: float = 0.35
    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 1.5
    risk_pct: float = 0.02

    # ── C7 Refinement toggles ──
    vol_gate_threshold: float | None = None
    confirm_bars: int = 1
    use_trail_stop: bool = False
    trail_atr_mult: float = 3.0
    conviction_scale: bool = False

    # ── C12 Multi-TP exit toggle ──
    use_multi_tp: bool = False

    # ── C8 Pattern Boost toggles ──
    use_pattern_boost: bool = False
    pattern_boost_scale: float = 0.05
    pattern_boost_max: float = 0.10
    pattern_confluence_bonus: float = 0.02

    # ── Internal state ──
    _model: PatternClassifier = None
    _feature_names: list[str] = []
    _probs: pd.Series = None
    _atr: pd.Series = None
    _vol_regime: pd.Series = None
    _confirm_count: int = 0
    _trail_high: float = 0.0
    _trail_sl: float = 0.0
    _pattern_booster: PatternBoostFilter | None = None
    _ohlcv_df: pd.DataFrame | None = None
    # C12: Multi-TP state
    _partial_exit_done: bool = False
    _entry_price: float = 0.0
    _entry_atr: float = 0.0
    _sl_price: float = 0.0

    def init(self) -> None:
        """Load model, extract features, precompute predictions, ATR, vol_regime."""
        # ── B10: Per-sector model resolution ──
        model_file = self.model_path
        if self.ticker:
            from src.ml.sector_map import SECTOR_MAP

            sector = SECTOR_MAP.get(self.ticker, "general")
            if sector != "general":
                candidates = sorted(
                    Path("models").glob(f"pattern_classifier_v3_{sector}_*.pkl"),
                    reverse=True,
                )
                if candidates:
                    model_file = str(candidates[0])
                    print(f"[MLStrategy] {self.ticker} → sector '{sector}' → {model_file}")
                else:
                    print(
                        f"[MLStrategy] {self.ticker} → sector '{sector}'"
                        f" (no model found, using default: {model_file})"
                    )

        # ── Load model ──
        self._model = PatternClassifier()
        self._model.load(model_file)
        self._feature_names = list(self._model.feature_names_)

        # ── Build OHLCV DataFrame from backtesting.py's internal data ──
        ohlcv = pd.DataFrame(
            {
                "Open": self.data.Open.s,
                "High": self.data.High.s,
                "Low": self.data.Low.s,
                "Close": self.data.Close.s,
                "Volume": self.data.Volume.s,
            },
            index=self.data.index,
        )

        # ── Extract features ──
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(ohlcv, include_forward_returns=False)

        # ── Extract vol_regime for volatility gate ──
        if self.vol_gate_threshold is not None and "vol_regime" in features.columns:
            self._vol_regime = features["vol_regime"].ffill().bfill().fillna(1.0)
        else:
            self._vol_regime = pd.Series(1.0, index=features.index)

        # ── Align features with model's expected columns ──
        present = [c for c in self._feature_names if c in features.columns]
        missing = [c for c in self._feature_names if c not in features.columns]
        fx = features[present].ffill().bfill().fillna(0)
        for c in missing:
            fx[c] = 0.0
        fx = fx[self._feature_names]

        # ── Precompute probabilities ──
        preds_df = self._model.predict(fx)
        self._probs = preds_df["probability_profitable"]

        # ── Precompute ATR(14) ──
        high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        self._atr = tr.rolling(14).mean()

        # ── Register indicators for backtesting.py plotting ──
        self.prob_line = self.I(
            lambda: self._probs.values,
            name="ML Prob",
            color="purple",
            overlay=False,
        )
        self.atr_line = self.I(
            lambda: self._atr.values,
            name="ATR(14)",
            color="gray",
            overlay=False,
        )

        # ── C8: Precompute pattern boost signals ──
        if self.use_pattern_boost:
            self._ohlcv_df = ohlcv.copy()
            self._pattern_booster = PatternBoostFilter(
                max_boost=self.pattern_boost_max,
                boost_scale=self.pattern_boost_scale,
                confluence_bonus=self.pattern_confluence_bonus,
            )
            self._pattern_booster.precompute(self._ohlcv_df)

    def next(self) -> None:
        """Evaluate ML signal and manage position."""
        i = len(self.data) - 1  # current bar index

        prob = self._probs.iloc[i]
        if pd.isna(prob):
            return

        atr_val = self._atr.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            return

        price = self.data.Close[-1]

        # ── C8: Pattern boost computation ──
        if self.use_pattern_boost and self._pattern_booster is not None:
            boost = self._pattern_booster.get_boost(i)
            # Apply directional boost: bull boost for long-biased, bear boost suppresses
            if prob >= self.entry_threshold and boost.bull_boost > 0:
                prob = min(1.0, prob + boost.bull_boost)
            elif prob <= self.exit_threshold and boost.bear_boost > 0:
                # Bearish pattern confluence = confirm exit / suppress long entry
                prob = max(0.0, prob - boost.bear_boost * 0.5)

        # ── Trailing stop management (when in position) ──
        if self.position and self.position.is_long:
            if self.use_trail_stop:
                if price > self._trail_high:
                    self._trail_high = price
                    self._trail_sl = self._trail_high - self.trail_atr_mult * atr_val

                if price <= self._trail_sl:
                    self.position.close()
                    self._confirm_count = 0
                    return

            # ── C12: Multi-TP exit (partial TP at 50% of target, breakeven SL after) ──
            elif self.use_multi_tp:
                tp_half = self._entry_price + (self.tp_atr_mult * self._entry_atr / 2.0)
                tp_full = self._entry_price + (self.tp_atr_mult * self._entry_atr)

                if not self._partial_exit_done and price >= tp_half:
                    self.position.close(portion=0.5)
                    self._sl_price = self._entry_price  # move SL to breakeven
                    self._partial_exit_done = True

                if price >= tp_full:
                    self.position.close()
                    self._confirm_count = 0
                    return

                if price <= self._sl_price:
                    self.position.close()
                    self._confirm_count = 0
                    return

            # ── Exit on signal reversal ──
            if prob <= self.exit_threshold:
                self.position.close()
                self._confirm_count = 0
                return

            # ── Fixed TP exit (if trail stop disabled and not multi-TP) ──
            if not self.use_trail_stop and not self.use_multi_tp:
                return

        # ── Entry logic ──
        if self.position:
            return

        # ── C7.1: Volatility gate ──
        if self.vol_gate_threshold is not None:
            vr = self._vol_regime.iloc[i]
            if pd.notna(vr) and vr > self.vol_gate_threshold:
                self._confirm_count = 0
                return

        # ── C7.2: Consecutive confirmation ──
        if prob >= self.entry_threshold:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        if self._confirm_count < self.confirm_bars:
            return

        # ── C7.3: Determine exit levels ──
        sl_price = price - self.sl_atr_mult * atr_val

        if self.use_trail_stop:
            tp_price = None
            self._trail_high = price
            self._trail_sl = sl_price
        elif self.use_multi_tp:
            tp_price = None
            sl_price = None  # managed manually in next()
            self._entry_price = price
            self._entry_atr = atr_val
            self._sl_price = price - self.sl_atr_mult * atr_val
            self._partial_exit_done = False
        else:
            tp_price = price + self.tp_atr_mult * atr_val

        # ── C7.4: Position scaling by conviction ──
        risk_amount = self.equity * self.risk_pct
        size = max(
            1, int(risk_amount / (price - (self._sl_price if self.use_multi_tp else sl_price)))
        )

        if self.conviction_scale and prob > self.entry_threshold:
            max_prob = 0.60
            scale = 1.0 + (prob - self.entry_threshold) / (max_prob - self.entry_threshold)
            scale = min(scale, 2.0)
            size = max(1, int(size * scale))

        self.buy(size=size, sl=sl_price, tp=tp_price)
        self._confirm_count = 0
