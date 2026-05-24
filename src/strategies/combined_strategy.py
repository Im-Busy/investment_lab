"""
AB1: Combined strategy — dynamic ML + Rules-First pattern weights.

Signal = w_rules * rules_score + (1 - w_rules) * ml_prob

Weighting modes:
  - static: Fixed weight (e.g., 0.5 = equal blend)
  - regime-adaptive: Bull → more ML, Bear → more Rules (via SimpleTrendRegimeDetector)
  - reciprocal-sharpe: Rolling 60d Sharpe per signal source, invert to weight
  - signal-conflict: When ML and Rules disagree, trust Rules (override to 0.8)

Entry: combined_score >= entry_threshold
Exit: ATR trailing stop OR combined_score < exit_threshold

Usage:
    from backtesting import Backtest
    from src.strategies.combined_strategy import CombinedStrategy

    bt = Backtest(df, CombinedStrategy, cash=10_000, commission=0.001)
    stats = bt.run(
        ml_model_path="models/pattern_classifier_v3_SPY.pkl",
        weight_mode="regime-adaptive",
        entry_threshold=0.55,
        trail_stop_atr=3.0,
    )
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# ── Pattern reliability weights (synced with rules_first_strategy.py) ──
PATTERN_RELIABILITY: dict[str, float] = {
    "Gartley Pattern": 0.85,
    "ABC Pattern": 0.75,
    "Symmetric Triangle": 0.77,
    "Bollinger Bands": 0.65,
    "Head and Shoulders": 0.70,
    "Inverse Head and Shoulders": 0.70,
    "Double Top": 0.70,
    "Double Bottom": 0.70,
    "Triple Top": 0.75,
    "Triple Bottom": 0.75,
    "Trader Vic 2B": 0.72,
    "Ascending Triangle": 0.78,
    "Descending Triangle": 0.78,
    "Rectangle": 0.68,
    "Wedge": 0.65,
    "Dead Cat Bounce": 0.50,
    "Cup and Handle": 0.65,
    "Spike and Ledge": 0.72,
    "Three Hills and a Mountain": 0.78,
    "Parabolic Arc": 0.70,
    "Donchian Channel Breakout": 0.62,
    "Gap Pattern": 0.60,
    "Bull Flag": 0.72,
    "Bear Flag": 0.72,
    "Pennant": 0.68,
    "Flag": 0.68,
    "Market Structure Low": 0.65,
    "Market Structure High": 0.65,
    "Matching Lows": 0.60,
    "NR7ID": 0.55,
    "N-Bar Decline": 0.58,
    "Floor Pivot Breakout": 0.55,
    "Two Bar Reversal": 0.60,
    "Doji": 0.40,
    "Harami": 0.45,
    "Hammer": 0.45,
    "Engulfing": 0.55,
    "Dark Cloud Cover": 0.50,
    "Piercing Line": 0.50,
}


class CombinedStrategy(Strategy):
    """Regime-adaptive ML + rules-first pattern combination strategy.

    Parameters:
        ml_model_path: Path to ML model (.pkl) or regime router config (.json).
        use_regime_router: If True, ml_model_path is a JSON config for RegimeRouter.
            If False, ml_model_path is a single CatBoost model .pkl.
        weight_mode: How to blend ML and Rules signals.
            Options: 'static', 'regime-adaptive', 'reciprocal-sharpe', 'signal-conflict'.
        rules_weight: Static weight for rules-first signal (0.5 = equal blend).
            Used when weight_mode='static'.
        rules_weight_min: Minimum rules weight in adaptive modes (default 0.3).
        rules_weight_max: Maximum rules weight in adaptive modes (default 0.8).
        entry_threshold: Combined score required to enter long.
        exit_threshold: Combined score below which to exit.
        trail_stop_atr: ATR multiplier for trailing stop.
        min_reliability: Minimum pattern reliability to include.
        confluence_bonus: Boost for multiple patterns firing.
        volume_confirm: Enable volume confirmation on rules signal.
    """

    # ── Model config ──
    ml_model_path: str = "models/pattern_classifier_v3_SPY_20260514_195235.pkl"
    use_regime_router: bool = True
    regime_router_config: str = "models/regime_router_SPY.json"

    # ── Weighting ──
    weight_mode: str = "regime-adaptive"
    rules_weight: float = 0.5
    rules_weight_min: float = 0.3
    rules_weight_max: float = 0.8

    # ── Entry/exit ──
    entry_threshold: float = 0.55
    exit_threshold: float = 0.30
    trail_stop_atr: float = 3.0

    # ── Rules-first params ──
    min_reliability: float = 0.70
    confluence_bonus: float = 0.10
    volume_confirm: bool = True

    # ── Multi-TP exit (H1 Phase 20) ──
    use_multi_tp: bool = True
    tp1_atr: float = 1.5
    tp1_size: float = 0.5
    move_sl_to_be: bool = True

    # ── Phase 21 new-tech gates (opt-in only — gates OFF by default) ──
    use_vix_gate: bool = False
    vix_gate_stress_mult: float = 0.30
    vix_gate_elevated_mult: float = 0.75
    use_yield_curve_gate: bool = False
    yield_curve_inversion_mult: float = 0.50
    yield_curve_near_inversion_mult: float = 0.75

    # ── Internal state ──
    _ml_probs: np.ndarray | None = None
    _rules_scores: np.ndarray | None = None
    _combined_scores: np.ndarray | None = None
    _atr: np.ndarray | None = None
    _regime: np.ndarray | None = None
    _df: pd.DataFrame | None = None
    _signals_cache: dict[str, np.ndarray] | None = None
    _reliability: dict[str, float] | None = None
    _patterns: list = None
    _trail_high: float = 0.0
    _tp1_hit: bool = False
    _entry_price: float = 0.0
    _vix_mults: np.ndarray | None = None
    _yield_curve_mults: np.ndarray | None = None

    def init(self) -> None:
        """Precompute all signals: ML probabilities, rules scores, ATR, regime."""
        self._df = self._build_df()
        self._init_patterns()
        self._precompute_rules_signals()
        self._precompute_ml_probs()
        self._precompute_atr()
        self._detect_regime()
        self._precompute_combined_scores()
        self._init_new_tech_gates()

    # ── DataFrame construction ──

    def _build_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": self.data.Open.s,
                "High": self.data.High.s,
                "Low": self.data.Low.s,
                "Close": self.data.Close.s,
                "Volume": self.data.Volume.s,
            },
            index=self.data.index,
        )

    # ── Pattern detection (shared with RulesFirstStrategy) ──

    def _init_patterns(self) -> None:
        from src.patterns.basic.floor_pivot import FloorPivotBreakout
        from src.patterns.basic.matching_lows import MatchingLows
        from src.patterns.basic.msl import MarketStructureLow
        from src.patterns.basic.n_bar_decline import NBarDecline
        from src.patterns.basic.nr7id import NR7ID
        from src.patterns.basic.two_bar_reversal import TwoBarReversal
        from src.patterns.breakout.donchian import DonchianChannelBreakout
        from src.patterns.breakout.gap import GapPattern
        from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine
        from src.patterns.candlestick.doji import Doji
        from src.patterns.candlestick.engulfing import Engulfing
        from src.patterns.candlestick.hammer import Hammer
        from src.patterns.candlestick.harami import Harami
        from src.patterns.classic.ascending_triangle import AscendingTriangle
        from src.patterns.classic.dead_cat_bounce import DeadCatBounce
        from src.patterns.classic.descending_triangle import DescendingTriangle
        from src.patterns.classic.double_bottom import DoubleBottom
        from src.patterns.classic.double_top import DoubleTop
        from src.patterns.classic.rectangle import Rectangle
        from src.patterns.classic.trader_vic_2b import TraderVic2B
        from src.patterns.classic.triple_bottom import TripleBottom
        from src.patterns.classic.triple_top import TripleTop
        from src.patterns.classic.wedge import Wedge
        from src.patterns.complex.cup_handle import CupAndHandle
        from src.patterns.complex.head_shoulders import HeadAndShoulders
        from src.patterns.complex.parabolic_arc import ParabolicArc
        from src.patterns.complex.spike_ledge import SpikeAndLedge
        from src.patterns.complex.three_hills import ThreeHillsMountain
        from src.patterns.continuation.flag import Flag
        from src.patterns.continuation.pennant import Pennant
        from src.patterns.harmonic.abc import ABCPattern
        from src.patterns.harmonic.bollinger import BollingerBands
        from src.patterns.harmonic.gartley import GartleyPattern
        from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle

        self._patterns = [
            GartleyPattern(),
            ABCPattern(),
            SymmetricTriangle(),
            BollingerBands(),
            DoubleTop(),
            DoubleBottom(),
            TraderVic2B(),
            TripleTop(),
            TripleBottom(),
            AscendingTriangle(),
            DescendingTriangle(),
            Rectangle(),
            Wedge(),
            DeadCatBounce(),
            CupAndHandle(),
            HeadAndShoulders(),
            SpikeAndLedge(),
            ThreeHillsMountain(),
            ParabolicArc(),
            DonchianChannelBreakout(),
            GapPattern(),
            Flag(),
            Pennant(),
            MarketStructureLow(),
            MatchingLows(),
            NR7ID(),
            NBarDecline(),
            FloorPivotBreakout(),
            TwoBarReversal(),
            Doji(),
            Harami(),
            Hammer(),
            Engulfing(),
            DarkCloudCover(),
            PiercingLine(),
        ]

        self._reliability = {p.name: PATTERN_RELIABILITY.get(p.name, 0.55) for p in self._patterns}
        self._reliability = {
            k: v for k, v in self._reliability.items() if v >= self.min_reliability
        }

    def _precompute_rules_signals(self) -> None:
        """Precompute all pattern signals and rules-based aggregate scores."""
        self._signals_cache = {}
        for pattern in self._patterns:
            if pattern.name not in self._reliability:
                continue
            try:
                signals = pattern.detect_vectorized(self._df)
                self._signals_cache[pattern.name] = signals.astype(np.int8)
            except Exception:
                logger.debug(
                    "CombinedStrategy: signal precompute failed for %s",
                    pattern.name,
                    exc_info=True,
                )

        n = len(self._df)
        self._rules_scores = np.zeros(n)
        for i in range(n):
            self._rules_scores[i] = self._compute_rules_score(i)

        logger.info(
            "CombinedStrategy: %d/%d patterns active, %d bars",
            len(self._signals_cache),
            len(self._patterns),
            n,
        )

    def _compute_rules_score(self, idx: int) -> float:
        total_weight = 0.0
        total_count = 0

        for name, signals in self._signals_cache.items():
            if idx >= len(signals):
                continue
            sig = signals[idx]
            if sig == 0:
                continue
            weight = self._reliability.get(name, 0.0)
            total_weight += weight * sig
            total_count += 1

        score = total_weight

        if total_count >= 2:
            score += self.confluence_bonus * np.sign(total_weight)

        if self.volume_confirm and idx > 0:
            vol = self._df["Volume"].iloc[: idx + 1]
            if len(vol) > 20:
                rel_vol = vol.iloc[-1] / max(vol.iloc[-20:].mean(), 1e-10)
                vol_mult = np.clip(rel_vol, 0.5, 2.0)
                score *= vol_mult

        return np.tanh(score)

    # ── ML probability precomputation ──

    def _precompute_ml_probs(self) -> None:
        """Precompute ML probabilities via RegimeRouter or single model."""
        from src.ml.feature_engineering import FeatureExtractor

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(self._df, include_forward_returns=False)

        if self.use_regime_router:
            self._ml_probs = self._regime_router_predict(features)
        else:
            self._ml_probs = self._single_model_predict(features)

    def _regime_router_predict(self, features: pd.DataFrame) -> np.ndarray:
        from src.ml.regime_router import RegimeRouter
        from src.ml.simple_regime import SimpleTrendRegimeDetector

        regime_config = json.loads(Path(self.regime_router_config).read_text())
        detector = SimpleTrendRegimeDetector(ma_period=regime_config.get("ma_period", 200))
        detector.fit(self._df)

        model_paths = {
            regime: Path(path) for regime, path in regime_config["regime_models"].items()
        }
        fallback_path = Path(regime_config["fallback_model"])
        router = RegimeRouter(detector, model_paths, fallback_path)

        probs = router.predict(features, price_data=self._df)
        logger.info(
            "CombinedStrategy: RegimeRouter loaded %d models, detector=%s",
            router.n_models,
            regime_config["detector"],
        )
        return probs.values

    def _single_model_predict(self, features: pd.DataFrame) -> np.ndarray:
        from src.ml.pattern_classifier import PatternClassifier

        model = PatternClassifier()
        model.load(self.ml_model_path)

        feature_names = model.feature_names_
        present = [c for c in feature_names if c in features.columns]
        missing = [c for c in feature_names if c not in features.columns]
        fx = features[present].ffill().bfill().fillna(0)
        for c in missing:
            fx[c] = 0.0
        fx = fx[feature_names]

        preds = model.predict(fx)
        if isinstance(preds, pd.DataFrame):
            prob_col = (
                "probability_profitable"
                if "probability_profitable" in preds.columns
                else preds.columns[-1]
            )
            return preds[prob_col].values
        return preds.values

    # ── Regime detection ──

    def _detect_regime(self) -> None:
        """Detect regime for adaptive weighting (Bull=1, Bear=-1, Unknown=0)."""
        if self.weight_mode == "regime-adaptive":
            from src.ml.simple_regime import SimpleTrendRegimeDetector

            detector = SimpleTrendRegimeDetector(ma_period=200)
            detector.fit(self._df)
            regimes = detector.predict(self._df)
            self._regime = np.where(regimes.values == "Bull", 1, -1)
        else:
            self._regime = np.zeros(len(self._df), dtype=int)

    # ── Combined score computation ──

    def _precompute_combined_scores(self) -> None:
        """Compute combined scores for all bars using the selected weighting mode."""
        n = len(self._df)
        self._combined_scores = np.zeros(n)

        if self.weight_mode == "static":
            self._combined_scores = (
                self.rules_weight * self._rules_scores + (1 - self.rules_weight) * self._ml_probs
            )
        elif self.weight_mode == "regime-adaptive":
            for i in range(n):
                w_rules = self._regime_weight(i)
                self._combined_scores[i] = (
                    w_rules * self._rules_scores[i] + (1 - w_rules) * self._ml_probs[i]
                )
        elif self.weight_mode == "reciprocal-sharpe":
            self._combined_scores = self._reciprocal_sharpe_blend()
        elif self.weight_mode == "signal-conflict":
            for i in range(n):
                w_rules = self._conflict_weight(i)
                self._combined_scores[i] = (
                    w_rules * self._rules_scores[i] + (1 - w_rules) * self._ml_probs[i]
                )

    def _regime_weight(self, idx: int) -> float:
        """Regime-adaptive: Bull → weight ML more (lower w_rules), Bear → weight Rules more."""
        if self._regime is None or idx >= len(self._regime):
            return 0.5
        regime = self._regime[idx]
        if regime == 1:  # Bull
            return self.rules_weight_min
        elif regime == -1:  # Bear
            return self.rules_weight_max
        else:
            return 0.5

    def _reciprocal_sharpe_blend(self) -> np.ndarray:
        """Reciprocal Sharpe: rolling 60d Sharpe per source, invert to weight."""
        n = len(self._df)
        scores = np.zeros(n)
        lookback = 60
        daily_rf = 0.0

        for i in range(n):
            if i < lookback:
                w = 0.5
            else:
                rules_slice = self._rules_scores[i - lookback : i]
                ml_slice = self._ml_probs[i - lookback : i]

                rules_ret = np.diff(rules_slice)
                ml_ret = np.diff(ml_slice)

                rules_mean = np.mean(rules_ret) if len(rules_ret) > 0 else 0
                rules_std = np.std(rules_ret) if len(rules_ret) > 0 else 1
                ml_mean = np.mean(ml_ret) if len(ml_ret) > 0 else 0
                ml_std = np.std(ml_ret) if len(ml_ret) > 0 else 1

                sharpe_rules = (rules_mean - daily_rf) / max(rules_std, 1e-10)
                sharpe_ml = (ml_mean - daily_rf) / max(ml_std, 1e-10)

                w_raw = sharpe_rules / max(sharpe_rules + sharpe_ml, 1e-10)
                w = np.clip(w_raw, self.rules_weight_min, self.rules_weight_max)

            scores[i] = w * self._rules_scores[i] + (1 - w) * self._ml_probs[i]

        return scores

    def _conflict_weight(self, idx: int) -> float:
        """Signal-conflict: when ML and Rules disagree, trust Rules (w=0.8)."""
        rules_dir = np.sign(self._rules_scores[idx])
        ml_dir = 1 if self._ml_probs[idx] >= 0.5 else -1

        if rules_dir == 0:
            return 0.3
        if rules_dir == ml_dir:
            return 0.5
        return 0.8  # conflict → trust rules

    # ── ATR precomputation ──

    def _precompute_atr(self) -> None:
        high, low, close = self._df["High"], self._df["Low"], self._df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        self._atr = tr.rolling(14).mean().bfill().fillna(close * 0.02).to_numpy()

    # ── Phase 21: New-tech gates ──

    def _init_new_tech_gates(self) -> None:
        """Initialize VIX and yield curve macro regime gates."""
        n = len(self._df)

        if self.use_vix_gate:
            try:
                from src.signals.vix_regime_gate import VixRegimeGate

                gate = VixRegimeGate(
                    stress_mult=self.vix_gate_stress_mult,
                    elevated_mult=self.vix_gate_elevated_mult,
                )
                gate.fit(start=str(self._df.index[0].date()))
                self._vix_mults = np.ones(n, dtype=np.float64)
                for i in range(n):
                    d = self._df.index[i]
                    if hasattr(d, "date"):
                        d = d.date()
                    self._vix_mults[i] = gate.multiplier(date=d)
            except Exception:
                logger.debug("VIX gate init failed", exc_info=True)
                self._vix_mults = np.ones(n, dtype=np.float64)
        else:
            self._vix_mults = np.ones(n, dtype=np.float64)

        if self.use_yield_curve_gate:
            try:
                from src.signals.yield_curve_gate import YieldCurveGate

                gate = YieldCurveGate(
                    inversion_mult=self.yield_curve_inversion_mult,
                    near_inversion_mult=self.yield_curve_near_inversion_mult,
                )
                gate.fit(start=str(self._df.index[0].date()))
                self._yield_curve_mults = np.ones(n, dtype=np.float64)
                for i in range(n):
                    d = self._df.index[i]
                    if hasattr(d, "date"):
                        d = d.date()
                    self._yield_curve_mults[i] = gate.multiplier(date=d)
            except Exception:
                logger.debug("Yield curve gate init failed", exc_info=True)
                self._yield_curve_mults = np.ones(n, dtype=np.float64)
        else:
            self._yield_curve_mults = np.ones(n, dtype=np.float64)

    # ── Per-bar execution ──

    def next(self) -> None:
        idx = len(self.data) - 1
        n = len(self._df)
        min_bars = max(
            (getattr(p, "min_bars_required", 20) for p in self._patterns),
            default=20,
        )
        if idx < min_bars or idx >= n:
            return

        score = float(self._combined_scores[idx])
        # ── Phase 21: Apply macro regime gates ──
        if self._vix_mults is not None and idx < len(self._vix_mults):
            score *= self._vix_mults[idx]
        if self._yield_curve_mults is not None and idx < len(self._yield_curve_mults):
            score *= self._yield_curve_mults[idx]
        current_close = float(self.data.Close[-1])
        atr = float(self._atr[idx]) if idx < len(self._atr) else 0.0
        if atr <= 0:
            atr = current_close * 0.02

        if self.position:
            self._trail_high = max(self._trail_high, current_close)
            trail_sl = self._trail_high - self.trail_stop_atr * atr

            if self.use_multi_tp and not self._tp1_hit:
                tp1_price = self._entry_price + self.tp1_atr * atr
                if current_close >= tp1_price:
                    self.position.close(portion=self.tp1_size)
                    self._tp1_hit = True
                    if self.move_sl_to_be:
                        self._trail_high = self._entry_price
                        trail_sl = self._entry_price
                elif current_close <= trail_sl:
                    self.position.close()
                    self._trail_high = 0.0
                elif score < self.exit_threshold:
                    self.position.close()
                    self._trail_high = 0.0
            elif current_close <= trail_sl:
                self.position.close()
                self._trail_high = 0.0
            elif score < self.exit_threshold:
                self.position.close()
                self._trail_high = 0.0
        else:
            if score >= self.entry_threshold:
                self.buy()
                self._trail_high = current_close
                self._entry_price = current_close
                self._tp1_hit = False
