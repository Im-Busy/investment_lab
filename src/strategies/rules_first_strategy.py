"""
B1: Rules-first multi-pattern trading strategy for backtesting.py.

No ML model. Pure rule-based signals from:
1. 34+ pattern detectors (7 categories)
2. Reliability weights from NCFE/Duddella research
3. Volume/volatility confirmation
4. ATR trailing stop (proven +31% Sharpe boost)
5. Confluence bonus (multi-pattern agreement)

Signal Formula:
    raw_score = Σ(pattern_detected × reliability_weight × direction)
    signal_score = sigmoid(raw_score + confluence_bonus + volume_confirm)

Entry: signal_score > entry_threshold (default 0.55)
Exit: ATR trailing stop or signal drops below exit_threshold

Direction: long-only (backtesting.py handles position sizing).

Usage:
    from backtesting import Backtest
    from src.strategies.rules_first_strategy import RulesFirstStrategy

    bt = Backtest(df, RulesFirstStrategy, cash=10_000, commission=0.001)
    stats = bt.run(entry_threshold=0.55, trail_stop_atr=3.0)
    bt.plot()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.signals.ir_weighting import IRWeighting  # noqa: E402
from src.signals.pattern_quality_registry import PatternQualityRegistry  # noqa: E402

logger = logging.getLogger(__name__)

# ── Pattern reliability weights from NCFE/Duddella research ──
PATTERN_RELIABILITY: dict[str, float] = {
    "Gartley Pattern": 0.85,
    "ABC Pattern": 0.75,
    "Symmetric Triangle": 0.77,
    "Bollinger Bands": 0.65,
    "Head and Shoulders": 0.87,
    "Inverse Head and Shoulders": 0.87,
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
    "Cup and Handle": 0.80,
    "Spike and Ledge": 0.72,
    "Three Hills and a Mountain": 0.78,
    "Pipe Pattern": 0.55,
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
    # FMZ strategy conversions (PineScript/JS → Python)
    "Alpha Beast": 0.55,
    "Multi-Factor Trend": 0.60,
    "Momentum ZigZag": 0.50,
    "EMA-MACD HF": 0.55,
    "Adaptive Bollinger": 0.50,
    "AI Volatility Breakout": 0.55,
    # C14: Extended harmonic patterns (conservative initial weights)
    "Butterfly Pattern": 0.55,
    "Bat Pattern": 0.55,
    "Crab Pattern": 0.50,
    "Cypher Pattern": 0.55,
    "Shark Pattern": 0.50,
    # D7a-d: Technical indicator patterns
    "Keltner Channel": 0.60,
    "Williams %R": 0.52,
    "CCI": 0.50,
    "Ichimoku Cloud": 0.58,
}


class RulesFirstStrategy(Strategy):
    """Rule-based multi-pattern trading strategy with trailing stop.

    Parameters:
        entry_threshold: Minimum signal score to enter long (default 0.55).
        exit_threshold: Signal score below which to exit (default 0.30).
        trail_stop_atr: ATR multiplier for trailing stop (default 3.0).
        min_reliability: Minimum pattern reliability to include (default 0.40).
        confluence_bonus: Boost for multi-pattern agreement (default 0.10).
        volume_confirm: Enable volume confirmation (default True).
    """

    entry_threshold: float = 0.55
    exit_threshold: float = 0.30
    trail_stop_atr: float = 3.0
    min_reliability: float = 0.40
    confluence_bonus: float = 0.10
    volume_confirm: bool = True
    use_short: bool = True
    use_ir_weights: bool = False
    ir_weighting_window: int = 252
    ir_weighting_mode: str = "scalar"
    use_multi_tp: bool = True
    tp1_atr: float = 1.5
    tp1_size: float = 0.5
    move_sl_to_be: bool = True

    use_quality_registry: bool = True
    quality_registry_path: str = "reports/pattern_gate/all_patterns.json"

    use_multi_factor: bool = False
    multi_factor_weight: float = 0.15
    multi_factor_file: str = ""

    # Q1: VIX regime gate
    use_vix_gate: bool = False
    vix_gate_stress_mult: float = 0.30
    vix_gate_elevated_mult: float = 0.75

    # Q2: Yield curve macro gate
    use_yield_curve_gate: bool = False
    yield_curve_inversion_mult: float = 0.50
    yield_curve_near_inversion_mult: float = 0.75

    # RF3.1: GARCH dynamic ATR trail
    use_garch_atr: bool = False
    garch_model: str = "egarch"
    garch_window: int = 252
    garch_trail_mult_min: float = 2.0
    garch_trail_mult_max: float = 4.0

    # RF3.2: Options sentiment signal modifier
    use_options_sentiment: bool = False
    options_sentiment_weight: float = 0.10

    # RF3.3: Kelly dynamic position sizing
    use_kelly_sizing: bool = False
    kelly_fraction: float = 0.5
    kelly_max_allocation: float = 0.25

    # P1.5: VIX regime-adaptive position sizing
    use_vix_regime_sizing: bool = False
    vix_regime_size_penalty: float = 0.50
    vix_regime_high_vol_cap: float = 0.50
    vix_regime_crisis_cap: float = 0.25

    # RF3.4: Order book microstructure signals
    use_order_book: bool = False
    order_book_weight: float = 0.05

    # P24-17: Signal-strength dynamic position sizing
    use_signal_strength_sizing: bool = False
    use_voting_signal: bool = False

    # Phase 27C: TadGAN regime anomaly gate
    use_tadgan_gate: bool = False
    tadgan_model_path: str = ""
    tadgan_gate_threshold_pct: float = 95.0
    voting_signal_weight: float = 0.15
    use_rules_catalog: bool = False
    rules_catalog_weight: float = 0.10
    use_divergence: bool = False
    divergence_weight: float = 0.20
    use_wm_bollinger: bool = False
    wm_bollinger_weight: float = 0.15
    signal_strength_min_size: float = 0.5
    signal_strength_max_size: float = 1.0

    _strategy_ref: Optional[list] = None

    def init(self) -> None:
        """Precompute pattern signals and indicators for all bars."""
        if self._strategy_ref is not None:
            self._strategy_ref.append(self)

        self._patterns = self._init_patterns()
        self._reliability = {p.name: PATTERN_RELIABILITY.get(p.name, 0.55) for p in self._patterns}
        self._reliability = {
            k: v for k, v in self._reliability.items() if v >= self.min_reliability
        }

        self._df = self._build_df()
        self._n_bars = len(self._df)
        self._signals_cache: dict[str, np.ndarray] = {}

        self._precompute_signals()
        self._precompute_atr()

        if self.use_voting_signal:
            self._init_voting_signal()
        if self.use_rules_catalog:
            self._init_rules_catalog()
        if self.use_divergence:
            self._init_divergence()
        if self.use_wm_bollinger:
            self._init_wm_bollinger()

        self._quality_registry: PatternQualityRegistry | None = None
        self._quality_mults: dict[str, float] = {}
        if self.use_quality_registry:
            self._init_quality_registry()

        self._ir_weighting = None
        if self.use_ir_weights:
            self._init_ir_weights()

        self._multi_factor_scalar: float = 1.0
        if self.use_multi_factor and self.multi_factor_file:
            self._init_multi_factor()

        self._vix_mults: np.ndarray | None = None
        if self.use_vix_gate:
            self._init_vix_gate()

        self._vix_regime_sizes: np.ndarray | None = None
        if self.use_vix_regime_sizing:
            self._init_vix_regime_sizing()

        self._yield_curve_mults: np.ndarray | None = None
        if self.use_yield_curve_gate:
            self._init_yield_curve_gate()

        self._garch_trail_mults: np.ndarray | None = None
        if self.use_garch_atr:
            self._init_garch_trail()

        self._options_sentiment_mults: np.ndarray | None = None
        if self.use_options_sentiment:
            self._init_options_sentiment()

        self._kelly_alloc = None
        if self.use_kelly_sizing:
            self._init_kelly_sizing()

        self._order_book_mult: float | np.ndarray = 1.0
        if self.use_order_book:
            self._init_order_book()

        self._tadgan_mults: np.ndarray | None = None
        if self.use_tadgan_gate:
            self._init_tadgan_gate()

        self._trail_high: float = 0.0
        self._trail_low: float = float("inf")
        self._trail_sl: float = 0.0
        self._entry_price: float = 0.0
        self._tp1_hit: bool = False
        self._tp2_hit: bool = False

    def _init_patterns(self):
        """Instantiate all pattern detectors."""
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
        from src.patterns.complex.pipe import PipePattern
        from src.patterns.continuation.flag import Flag
        from src.patterns.continuation.pennant import Pennant
        from src.patterns.harmonic.abc import ABCPattern
        from src.patterns.harmonic.bollinger import BollingerBands
        from src.patterns.harmonic.extended import (
            ButterflyPattern,
            BatPattern,
            CrabPattern,
            CypherPattern,
            SharkPattern,
        )
        from src.patterns.harmonic.gartley import GartleyPattern
        from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle
        from src.patterns.technical.keltner_channel import KeltnerChannelPattern
        from src.patterns.technical.williams_r import WilliamsRPattern
        from src.patterns.technical.cci import CCIPattern
        from src.patterns.technical.ichimoku import IchimokuPattern

        return [
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
            PipePattern(),
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
            ButterflyPattern(),
            BatPattern(),
            CrabPattern(),
            CypherPattern(),
            SharkPattern(),
            KeltnerChannelPattern(),
            WilliamsRPattern(),
            CCIPattern(),
            IchimokuPattern(),
        ]

    def _build_df(self) -> pd.DataFrame:
        """Build OHLCV DataFrame from backtesting.py internal data."""
        df = pd.DataFrame(
            {
                "Open": self.data.Open.s,
                "High": self.data.High.s,
                "Low": self.data.Low.s,
                "Close": self.data.Close.s,
                "Volume": self.data.Volume.s,
            },
            index=self.data.index,
        )
        return df

    def _precompute_signals(self) -> None:
        """Precompute all pattern signals vectorized and compute aggregate score."""
        active_reliability = {k: v for k, v in self._reliability.items()}

        for pattern in self._patterns:
            if pattern.name not in active_reliability:
                continue
            try:
                signals = pattern.detect_vectorized(self._df)
                self._signals_cache[pattern.name] = signals.astype(np.int8)
            except Exception:
                logger.debug("Signal precompute failed for %s", pattern.name, exc_info=True)

        logger.info(
            "RulesFirstStrategy: %d/%d patterns active, %d bars",
            len(self._signals_cache),
            len(self._patterns),
            self._n_bars,
        )

    def _compute_score(self, idx: int) -> float:
        """Compute aggregate signal score for a single bar."""
        total_weight = 0.0
        total_count = 0

        ir_weights: Optional[dict[str, float]] = None
        ir_scalars: Optional[dict[str, float]] = None
        if self._ir_weighting is not None:
            if self._ir_weighting.mode == "scalar":
                ir_scalars = self._ir_weighting.get_scalars(idx)
            else:
                ir_weights = self._ir_weighting.get_weights(idx)

        for name, signals in self._signals_cache.items():
            if idx >= len(signals):
                continue
            sig = signals[idx]
            if sig == 0:
                continue
            base_weight = self._reliability.get(name, 0.0)
            if self._quality_mults:
                base_weight *= self._quality_mults.get(
                    name, PatternQualityRegistry.DEFAULT_MULTIPLIER
                )
            if ir_scalars is not None:
                scalar = ir_scalars.get(name, 0.5)
                weight = base_weight * scalar
            elif ir_weights is not None:
                weight = ir_weights.get(name, 0.0)
            else:
                weight = base_weight
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

        if self.use_multi_factor and self._multi_factor_scalar != 1.0:
            score *= self._multi_factor_scalar

        if self._vix_mults is not None and idx < len(self._vix_mults):
            score *= self._vix_mults[idx]

        if self._yield_curve_mults is not None and idx < len(self._yield_curve_mults):
            score *= self._yield_curve_mults[idx]

        if self._options_sentiment_mults is not None and idx < len(self._options_sentiment_mults):
            score *= self._options_sentiment_mults[idx]

        if self.use_order_book and idx < len(self._order_book_mult):
            score *= float(self._order_book_mult[idx])

        return np.tanh(score)

    def _precompute_atr(self) -> None:
        """Precompute ATR(14) for all bars."""
        high = self._df["High"]
        low = self._df["Low"]
        close = self._df["Close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        self._atr = tr.rolling(14).mean().bfill().fillna(close * 0.02).to_numpy()

    def _init_quality_registry(self) -> None:
        """Load pattern quality registry and map multipliers to pattern names."""
        try:
            registry_path = project_root / self.quality_registry_path
            if not registry_path.exists():
                logger.warning("Quality registry not found: %s", self.quality_registry_path)
                return
            self._quality_registry = PatternQualityRegistry.load(registry_path)
            self._quality_mults = {
                name: self._quality_registry.get_multiplier(name) for name in self._signals_cache
            }
            pass_count = sum(1 for v in self._quality_mults.values() if v >= 1.0)
            fail_strong = sum(1 for v in self._quality_mults.values() if 0.4 <= v < 1.0)
            fail_weak = sum(1 for v in self._quality_mults.values() if 0 < v < 0.4)
            excluded = sum(1 for v in self._quality_mults.values() if v == 0.0)
            logger.info(
                "Quality registry loaded: %d pass, %d fail_strong, %d fail_weak, %d excluded",
                pass_count,
                fail_strong,
                fail_weak,
                excluded,
            )
        except Exception:
            logger.debug("Failed to init quality registry", exc_info=True)

    def _init_ir_weights(self) -> None:
        """Precompute rolling IR weights for all pattern signals."""
        close_arr = self._df["Close"].to_numpy(dtype=np.float64)
        self._ir_weighting = IRWeighting(
            window=self.ir_weighting_window, mode=self.ir_weighting_mode
        )
        self._ir_weighting.fit(close_arr, self._signals_cache)
        logger.info(
            "IR weights initialized (%s, %d-bar window)",
            self.ir_weighting_mode,
            self.ir_weighting_window,
        )

    def _init_multi_factor(self) -> None:
        """Load fundamental sector score from CSV file."""
        try:
            df = pd.read_csv(self.multi_factor_file, index_col=0)
            if "score" in df.columns:
                score = float(df["score"].iloc[-1])
            elif len(df.columns) > 0:
                score = float(df.iloc[:, 0].iloc[-1])
            else:
                score = 0.0
            self._multi_factor_scalar = 1.0 + self.multi_factor_weight * np.tanh(score)
            logger.info("Multi-factor scalar: %.3f (raw=%.3f)", self._multi_factor_scalar, score)
        except Exception:
            logger.debug(
                "Failed to load multi-factor file: %s", self.multi_factor_file, exc_info=True
            )
            self._multi_factor_scalar = 1.0

    def _init_vix_gate(self) -> None:
        """Q1: Precompute VIX regime multipliers for all bars."""
        try:
            from src.signals.vix_regime_gate import VixRegimeGate

            gate = VixRegimeGate(
                stress_mult=self.vix_gate_stress_mult,
                elevated_mult=self.vix_gate_elevated_mult,
            )
            dates = self._df.index
            start_str = str(dates[0].date())
            gate.fit(start=start_str)
            self._vix_mults = np.ones(len(dates), dtype=np.float64)
            for i, d in enumerate(dates):
                self._vix_mults[i] = gate.multiplier(date=d)
            num_stress = np.sum(self._vix_mults < 1.0)
            logger.info(
                "VIX gate initialized: %d/%d bars gated (%.1f%%)",
                num_stress,
                len(dates),
                100 * num_stress / max(len(dates), 1),
            )
        except Exception:
            logger.warning("VIX gate init failed, disabling", exc_info=True)
            self._vix_mults = np.ones(self._n_bars, dtype=np.float64)

    def _init_yield_curve_gate(self) -> None:
        """Q2: Precompute yield curve inversion multipliers for all bars."""
        try:
            from src.signals.yield_curve_gate import YieldCurveGate

            gate = YieldCurveGate(
                inversion_mult=self.yield_curve_inversion_mult,
                near_inversion_mult=self.yield_curve_near_inversion_mult,
            )
            dates = self._df.index
            start_str = str(dates[0].date())
            gate.fit(start=start_str)
            self._yield_curve_mults = np.ones(len(dates), dtype=np.float64)
            for i, d in enumerate(dates):
                self._yield_curve_mults[i] = gate.multiplier(date=d)
            num_gated = np.sum(self._yield_curve_mults < 1.0)
            logger.info(
                "Yield curve gate initialized: %d/%d bars gated (%.1f%%)",
                num_gated,
                len(dates),
                100 * num_gated / max(len(dates), 1),
            )
        except Exception:
            logger.warning("Yield curve gate init failed, disabling", exc_info=True)
            self._yield_curve_mults = np.ones(self._n_bars, dtype=np.float64)

    def _init_garch_trail(self) -> None:
        """RF3.1: Precompute GARCH-volatility-scaled trail multipliers."""
        try:
            from src.ml.garch_forecaster import GARCHForecaster

            close = self._df["Close"]
            returns = close.pct_change().dropna()
            forecaster = GARCHForecaster(
                model=self.garch_model, window=self.garch_window, retrain_every=63
            )
            forecaster.fit(returns)
            forecasts = forecaster.forecast(horizon=1)
            aligned = forecasts.reindex(self._df.index).fillna(method="ffill").fillna(0.02)
            median_vol = np.median(aligned[aligned > 0]) if (aligned > 0).any() else 0.02
            vol_ratio = np.clip(aligned / max(median_vol, 1e-6), 0.5, 2.0)
            self._garch_trail_mults = self.garch_trail_mult_min + (
                self.garch_trail_mult_max - self.garch_trail_mult_min
            ) * (1 - vol_ratio)
            low_garch = np.sum(aligned < median_vol * 0.8)
            logger.info(
                "GARCH trail: %d bars vol<80%% median → wider trail",
                low_garch,
            )
        except Exception:
            logger.warning("GARCH trail init failed, disabling", exc_info=True)
            self._garch_trail_mults = None

    def _init_options_sentiment(self) -> None:
        """RF3.2: Precompute options sentiment signal modifiers."""
        try:
            from src.signals.options_sentiment import OptionsSentimentProvider

            provider = OptionsSentimentProvider()
            dates = self._df.index
            start_str = str(dates[0].date())
            provider.fit(start=start_str)
            sentiment = provider.get_sentiment(dates)
            self._options_sentiment_mults = (
                (1.0 + self.options_sentiment_weight * sentiment.fillna(0.0))
                .clip(0.8, 1.2)
                .to_numpy(dtype=np.float64)
            )
            logger.info(
                "Options sentiment: mean=%.3f, range=[%.3f, %.3f]",
                float(sentiment.mean()),
                float(sentiment.min()),
                float(sentiment.max()),
            )
        except Exception:
            logger.warning("Options sentiment init failed, disabling", exc_info=True)
            self._options_sentiment_mults = None

    def _init_kelly_sizing(self) -> None:
        """RF3.3: Initialize Kelly allocator for dynamic position sizing."""
        try:
            from src.risk.kelly_allocator import KellyAllocator

            self._kelly_alloc = KellyAllocator(
                kelly_fraction=self.kelly_fraction,
                max_allocation=self.kelly_max_allocation,
                method="classic",
            )
            self._kelly_signal_history: list[float] = []
            logger.info(
                "Kelly allocator: fraction=%.2f, max=%.2f",
                self.kelly_fraction,
                self.kelly_max_allocation,
            )
        except Exception:
            logger.warning("Kelly allocator init failed, disabling", exc_info=True)
            self._kelly_alloc = None

    def _init_order_book(self) -> None:
        """RF3.4: Precompute order book microstructure features."""
        try:
            from src.signals.order_book_features import OrderBookFeatures

            ob = OrderBookFeatures()
            features = ob.compute(self._df)
            bai = features.get("bai", pd.Series(0.0, index=self._df.index))
            self._order_book_mult = np.clip(
                1.0 + self.order_book_weight * bai.fillna(0.0).to_numpy(), 0.85, 1.15
            )
            logger.info("Order book features: BAI mean=%.4f", float(bai.mean()))
        except Exception:
            logger.warning("Order book features init failed, disabling", exc_info=True)
            self._order_book_mult = 1.0

    def _init_voting_signal(self) -> None:
        """B6: Precompute 6-indicator voting signals and cache as a signal source."""
        try:
            from src.signals.indicator_voting import compute_voting_signals

            close_arr = self._df["Close"].to_numpy(dtype=np.float64)
            vote = compute_voting_signals(close_arr)
            self._signals_cache["voting_6indicator"] = vote.astype(np.int8)
            self._reliability["voting_6indicator"] = self.voting_signal_weight
            n_signals = int(np.sum(np.abs(vote)))
            logger.info(
                "Voting signal: %d non-zero bars, weight=%.2f", n_signals, self.voting_signal_weight
            )
        except Exception:
            logger.warning("Voting signal init failed, disabling", exc_info=True)

    def _init_rules_catalog(self) -> None:
        """B1: Precompute 35-rule catalog signals and cache as signal sources."""
        try:
            from src.signals.rules_catalog import generate_rules_catalog

            close = self._df["Close"].to_numpy(dtype=np.float64)
            rules = generate_rules_catalog(close)
            n_added = 0
            for name, sig in rules.items():
                if int(np.sum(np.abs(sig))) > 0:
                    self._signals_cache[name] = sig.astype(np.int8)
                    self._reliability[name] = self.rules_catalog_weight
                    n_added += 1
            logger.info(
                "Rules catalog: %d active rules, weight=%.2f", n_added, self.rules_catalog_weight
            )
        except Exception:
            logger.warning("Rules catalog init failed, disabling", exc_info=True)

    def _init_divergence(self) -> None:
        """B10: Precompute divergence signals and cache as signal sources."""
        try:
            from src.signals.divergence_detector import detect_all_divergences

            divs = detect_all_divergences(self._df)
            for name, sig in divs.items():
                if int(np.sum(np.abs(sig))) > 0:
                    self._signals_cache[name] = sig.astype(np.int8)
                    self._reliability[name] = self.divergence_weight
            logger.info(
                "Divergence signals: %d types, weight=%.2f", len(divs), self.divergence_weight
            )
        except Exception:
            logger.warning("Divergence init failed, disabling", exc_info=True)

    def _init_wm_bollinger(self) -> None:
        """B2: Precompute W/M-Bollinger signals and cache as signal sources."""
        try:
            from src.patterns.bollinger.wm_patterns import detect_wm_bollinger

            wm = detect_wm_bollinger(self._df)
            if "w_bottom" in wm.columns and int(np.sum(np.abs(wm["w_bottom"].to_numpy()))) > 0:
                self._signals_cache["w_bottom"] = wm["w_bottom"].to_numpy(dtype=np.int8)
                self._reliability["w_bottom"] = self.wm_bollinger_weight
            if "m_top" in wm.columns and int(np.sum(np.abs(wm["m_top"].to_numpy()))) > 0:
                self._signals_cache["m_top"] = wm["m_top"].to_numpy(dtype=np.int8)
                self._reliability["m_top"] = self.wm_bollinger_weight
            logger.info("W/M Bollinger: weight=%.2f", self.wm_bollinger_weight)
        except Exception:
            logger.warning("W/M Bollinger init failed, disabling", exc_info=True)

    def next(self) -> None:
        """Execute trading logic for current bar."""
        idx = len(self.data) - 1
        min_bars = max(
            (getattr(p, "min_bars_required", 20) for p in self._patterns),
            default=20,
        )
        if idx < min_bars or idx >= self._n_bars:
            return

        score = self._compute_score(idx)
        current_close = float(self.data.Close[-1])
        atr = float(self._atr[idx]) if idx < len(self._atr) else 0.0
        if atr <= 0:
            atr = current_close * 0.02

        trail_atr_mult = self.trail_stop_atr
        if self._garch_trail_mults is not None and idx < len(self._garch_trail_mults):
            trail_atr_mult = float(self._garch_trail_mults[idx])

        if self.position:
            if self.position.is_long:
                self._trail_high = max(self._trail_high, current_close)
                trail_sl = self._trail_high - trail_atr_mult * atr

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
                else:
                    if current_close <= trail_sl:
                        self.position.close()
                        self._trail_high = 0.0
                    elif score < self.exit_threshold:
                        self.position.close()
                        self._trail_high = 0.0
            else:
                self._trail_low = min(self._trail_low, current_close)
                trail_sl = self._trail_low + trail_atr_mult * atr

                if self.use_multi_tp and not self._tp1_hit:
                    tp1_price = self._entry_price - self.tp1_atr * atr
                    if current_close <= tp1_price:
                        self.position.close(portion=self.tp1_size)
                        self._tp1_hit = True
                        if self.move_sl_to_be:
                            self._trail_low = self._entry_price
                            trail_sl = self._entry_price
                    elif current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                    elif score > -self.exit_threshold:
                        self.position.close()
                        self._trail_low = float("inf")
                else:
                    if current_close >= trail_sl:
                        self.position.close()
                        self._trail_low = float("inf")
                    elif score > -self.exit_threshold:
                        self.position.close()
                        self._trail_low = float("inf")
        else:
            kelly_size = self._compute_kelly_size(score) if self._kelly_alloc else 1.0
            kelly_size = self._compute_signal_strength_size(score, kelly_size)
            kelly_size = self._apply_regime_size_penalty(idx, kelly_size)
            if self._tadgan_mults is not None and idx < len(self._tadgan_mults):
                if self._tadgan_mults[idx] < 0.5:
                    return
            if score >= self.entry_threshold:
                self.buy(size=kelly_size)
                self._trail_high = current_close
                self._trail_low = float("inf")
                self._entry_price = current_close
                self._tp1_hit = False
                self._tp2_hit = False
            elif self.use_short and score <= -self.entry_threshold:
                self.sell(size=kelly_size)
                self._trail_low = current_close
                self._trail_high = 0.0
                self._entry_price = current_close
                self._tp1_hit = False
                self._tp2_hit = False
                self._trail_high = current_close

    def _compute_kelly_size(self, score: float) -> float:
        """RF3.3: Compute Kelly-derived position size fraction from signal strength."""
        if self._kelly_alloc is None:
            return 1.0
        prob = max(0.30, min(0.80, 0.50 + float(score) * 0.25))
        edge = self._kelly_alloc.estimate_edge_from_probability(prob)
        alloc = self._kelly_alloc.compute(edge)
        return max(0.25, alloc.adjusted_fraction)

    def _compute_signal_strength_size(self, score: float, base_size: float) -> float:
        """P24-17: Signal-strength position sizing — magnitude → lot scaling.

        |abs(score)| ≥ 0.45 → 3 lots, 0.05-0.15 → 2 lots, < 0.05 → 1 lot.
        Maps into a size multiplier on base position size.
        """
        if not self.use_signal_strength_sizing:
            return base_size
        abs_score = abs(float(score))
        if abs_score >= 0.45:
            lots = 3
        elif abs_score >= 0.15:
            lots = 2
        else:
            lots = 1
        multiplier = float(lots) / 3.0
        return float(
            np.clip(
                base_size * multiplier, self.signal_strength_min_size, self.signal_strength_max_size
            )
        )

    def _init_vix_regime_sizing(self) -> None:
        """P1.5: Precompute VIX-regime-based position size caps for all bars.

        Uses the existing VIX regime gate to determine regime per bar:
          - COMPLACENT (VIX < 15): full size (1.0)
          - NORMAL (15 <= VIX < 25): full size (1.0)
          - ELEVATED (25 <= VIX < 35): cap at high_vol_cap (default 0.50)
          - STRESS (VIX >= 35): cap at crisis_cap (default 0.25)

        Source: arXiv:2601.19504 — ATR-based sizing + regime filter
                arXiv:2509.01393 — volatility targeting + regime penalty overlay
        """
        self._vix_regime_sizes = np.ones(self._n_bars, dtype=np.float64)
        try:
            from src.signals.vix_regime_gate import VixRegimeGate

            gate = VixRegimeGate(
                stress_mult=self.vix_regime_crisis_cap,
                elevated_mult=self.vix_regime_high_vol_cap,
            )
            dates = self._df.index
            start_str = str(dates[0].date())
            gate.fit(start=start_str)
            for i, d in enumerate(dates):
                regime = gate.regime(date=d)
                multiplier = gate.multiplier(date=d)
                if regime == "STRESS":
                    self._vix_regime_sizes[i] = multiplier
                elif regime == "ELEVATED":
                    self._vix_regime_sizes[i] = multiplier
                else:
                    self._vix_regime_sizes[i] = 1.0
            num_capped = np.sum(self._vix_regime_sizes < 1.0)
            logger.info(
                "VIX regime sizing: %d/%d bars capped (%.1f%%)",
                num_capped,
                len(dates),
                100 * num_capped / max(len(dates), 1),
            )
        except Exception:
            logger.warning("VIX regime sizing init failed, disabling", exc_info=True)
            self._vix_regime_sizes = np.ones(self._n_bars, dtype=np.float64)

    def _init_tadgan_gate(self) -> None:
        """Phase 27C: Load TadGAN model and precompute anomaly mask for all bars.

        TadGAN detects regime anomalies (crisis events). When active,
        entries are blocked during anomalous bars to avoid trading
        during market dislocations.
        """
        try:
            from src.ml.anomaly_detection import TadGAN, prepare_tadgan_samples

            dates = self._df.index
            tadgan = TadGAN.load(self.tadgan_model_path, seq_len=100)
            samples = prepare_tadgan_samples(self._df, seq_len=100, step=1)
            if len(samples) == 0:
                logger.warning("TadGAN: no samples, disabling")
                self._tadgan_mults = np.ones(self._n_bars, dtype=np.float64)
                return

            scores = tadgan.compute_anomaly_scores(samples)
            threshold = np.percentile(scores, self.tadgan_gate_threshold_pct)
            anomaly_mask = scores > threshold

            self._tadgan_mults = np.ones(self._n_bars, dtype=np.float64)
            for i in range(len(samples)):
                bar_idx = min(99 + i, self._n_bars - 1)
                if anomaly_mask[i]:
                    self._tadgan_mults[bar_idx] = 0.0

            n_blocked = np.sum(self._tadgan_mults < 0.5)
            logger.info(
                "TadGAN gate initialized: %d/%d bars blocked (%.1f%%)",
                n_blocked,
                self._n_bars,
                100 * n_blocked / max(self._n_bars, 1),
            )
        except Exception:
            logger.warning("TadGAN gate init failed, disabling", exc_info=True)
            self._tadgan_mults = np.ones(self._n_bars, dtype=np.float64)

    def _apply_regime_size_penalty(self, idx: int, base_size: float) -> float:
        """P1.5: Apply VIX regime-based position size cap.

        Reduces position size when:
          - VIX is elevated (25-35): max 50% of base size
          - VIX is stressed (>35): max 25% of base size

        In all other regimes, passes through unchanged.
        """
        if self._vix_regime_sizes is None or idx >= len(self._vix_regime_sizes):
            return base_size
        regime_cap = float(self._vix_regime_sizes[idx])
        return min(base_size, regime_cap)
