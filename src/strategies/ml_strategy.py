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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Strategy

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from src.ml.dynamic_ensemble import DynamicEnsemble, DynamicEnsembleConfig
from src.ml.feature_engineering import FeatureExtractor
from src.signals.pattern_boost import PatternBoostFilter
from src.signals.sentiment_scorer import (
    SentimentSignalModifier,
    SyntheticSentimentProvider,
)


class MLStrategy(Strategy):
    """Strategy driven by CatBoost pattern classifier probability scores.

    Parameters:
        model_path: Path to trained PatternClassifier pickle file.
        entry_threshold: Minimum probability to enter a long position (default 0.50).
        exit_threshold: Probability below which to exit a long position (default 0.35).
        tp_atr_mult: Take-profit multiplier on ATR(14) (default 3.0).
        sl_atr_mult: Stop-loss multiplier on ATR(14) (default 1.5).
        risk_pct: Fraction of equity to risk per trade (default 0.02).

        ── C2 Sentiment Modifier ──
        use_sentiment: Enable sentiment score adjustment to ML probabilities.
            Uses SyntheticSentimentProvider for testing (AR(1) process).
            Positive sentiment boosts bullish signals, negative suppresses.
            Real data sources (CSV, FinBERT) deferred to post-C6. Default False.
        sentiment_weight: How much sentiment influences signals (default 0.15).
            Formula: adjusted_prob = prob * (1 + weight * sentiment_score)
            Clamped to [0, 1] after adjustment.

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
    model_path: str = "models/pattern_classifier_v3_SPY_20260514_195235.pkl"
    model_paths: list[str] | None = None
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
    use_multi_tp: bool = True

    # ── C8 Pattern Boost toggles ──
    use_pattern_boost: bool = False
    pattern_boost_scale: float = 0.05
    pattern_boost_max: float = 0.10
    pattern_confluence_bonus: float = 0.02

    # ── B12 Dynamic Ensemble toggle ──
    use_dynamic_ensemble: bool = False
    dynamic_ensemble_path: str = "models/dynamic_ensemble_SPY"

    # ── B13 Meta-Labeler toggle ──
    use_meta_label: bool = False
    meta_label_path: str = "models/meta_labeler_v2_SPY_20260514_125515.pkl"

    # ── C2 Sentiment toggle ──
    use_sentiment: bool = False
    sentiment_weight: float = 0.15

    # ── C3 Event filter toggle ──
    use_event_filter: bool = False

    # ── N1 LM Dictionary Sentiment ──
    use_lm_sentiment: bool = False
    lm_sentiment_weight: float = 0.15

    # ── C4 RL execution toggle ──
    use_rl_execution: bool = False
    rl_model_path: str = "models/rl_executor_dqn.pt"

    # ── C5 Kelly Allocator ──
    use_kelly: bool = False
    kelly_fraction: float = 0.5  # half-Kelly default
    kelly_max_allocation: float = 0.25

    # ── C5 Crash Filter ──
    use_crash_filter: bool = False
    crash_threshold: float = 0.50  # crash_risk_score threshold to skip entry

    # ── E2 Chronos Foundation Model ──
    use_chronos: bool = False
    chronos_model_size: str = "chronos-2"
    chronos_weight: float = 0.30

    # ── E5 Cross-Asset Inference ──
    use_cross_asset: bool = False

    # ── Q1 Multi-Factor Fundamentals ──
    use_fundamentals: bool = False

    # ── A+B Regime Router ──
    use_regime_router: bool = False
    regime_router_config: str = "models/regime_router_SPY.json"

    # ── Internal state ──
    _model: PatternClassifier = None
    _models: list[PatternClassifier] = []
    _dynamic_ensemble: DynamicEnsemble | None = None
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
    # B13: Meta-Labeler state
    _meta_labeler = None
    _meta_context: pd.DataFrame | None = None
    # C2: Sentiment state
    _sentiment_modifier: SentimentSignalModifier | None = None
    _sentiment_scores: pd.Series | None = None
    # N1: LM Sentiment state
    _lm_sentiment_modifier: object | None = None
    _lm_headlines: pd.Series | None = None
    # C3: Event filter state
    _event_suppressor: object | None = None
    _event_days: pd.Series | None = None
    # C4: RL execution state
    _rl_executor: object | None = None
    _rl_confirm_streak: int = 0
    _rl_entry_price: float = 0.0

    # C5: Kelly allocator state
    _kelly_allocator: object | None = None

    # C5: Crash filter state
    _crash_detector: object | None = None
    _crash_risk: pd.Series | None = None

    # E2: Chronos signal generator state
    _chronos_gen: object | None = None

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
                    print(f"[MLStrategy] {self.ticker} -> sector '{sector}' -> {model_file}")
                else:
                    print(
                        f"[MLStrategy] {self.ticker} -> sector '{sector}'"
                        f" (no model found, using default: {model_file})"
                    )

        # ── Load model(s) ──
        if self.use_dynamic_ensemble:
            self._dynamic_ensemble = DynamicEnsemble()
            self._dynamic_ensemble.load(self.dynamic_ensemble_path)
            self._model = self._dynamic_ensemble.models[0]
            print(
                f"[MLStrategy] Loaded DynamicEnsemble: {len(self._dynamic_ensemble.models)} models"
            )
        elif self.model_paths:
            self._models = []
            for mp in self.model_paths:
                m = PatternClassifier()
                m.load(mp)
                self._models.append(m)
            self._model = self._models[0]
            print(f"[MLStrategy] Loaded {len(self._models)} bagged CPCV models")
        else:
            self._model = PatternClassifier()
            self._model.load(model_file)
            self._models = []

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

        # ── E5: Cross-asset feature extraction ──
        if self.use_cross_asset:
            from src.ml.cross_asset_features import (
                CrossAssetFeatureExtractor,
                load_market_data,
            )

            market_data = load_market_data(ohlcv)
            ca_extractor = CrossAssetFeatureExtractor(market_data=market_data)
            ca_features = ca_extractor.extract(ohlcv)
            features = features.join(ca_features, how="inner")
            present_ca = list(ca_features.columns)
            print(
                f"[MLStrategy] Cross-asset features enabled -- "
                f"{len(present_ca)} features, market tickers: {list(market_data.keys())}"
            )

        # ── Q1: Multi-factor fundamental features ──
        if self.use_fundamentals:
            from src.ml.fundamental_features import (
                QuarterlyFundamentalProvider,
            )

            fund_provider = QuarterlyFundamentalProvider()
            fund_features = fund_provider.build_time_series(
                self.ticker or "SPY",
                features.index,
            )
            features = features.join(fund_features, how="left")
            present_fund = [c for c in fund_features.columns if c in features.columns]
            print(
                f"[MLStrategy] Fundamental features enabled -- "
                f"{len(present_fund)} features joined (forward-filled quarterly)"
            )

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
        self._features_df = fx

        # ── Precompute probabilities ──
        if self._dynamic_ensemble is not None:
            self._probs = pd.Series(
                self._dynamic_ensemble.predict(fx),
                index=fx.index,
            )
        elif self._models:
            all_probs = []
            for m in self._models:
                preds_df = m.predict(fx)
                all_probs.append(preds_df["probability_profitable"])
            self._probs = pd.concat(all_probs, axis=1).mean(axis=1)
        else:
            preds_df = self._model.predict(fx)
            self._probs = preds_df["probability_profitable"]

        # ── A+B: RegimeRouter (per-regime model dispatch) ──
        if self.use_regime_router:
            from src.ml.regime_router import RegimeRouter
            from src.ml.simple_regime import SimpleTrendRegimeDetector

            regime_config = json.loads(Path(self.regime_router_config).read_text())
            detector = SimpleTrendRegimeDetector(ma_period=regime_config.get("ma_period", 200))
            detector.fit(ohlcv)

            model_paths = {
                regime: Path(path) for regime, path in regime_config["regime_models"].items()
            }
            fallback_path = Path(regime_config["fallback_model"])
            router = RegimeRouter(detector, model_paths, fallback_path)
            self._probs = router.predict(fx, price_data=ohlcv)
            print(
                f"[MLStrategy] RegimeRouter active: {router.n_models} regime models, "
                f"detector={regime_config['detector']}"
            )

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

        # ── B13: Precompute meta-labeler context features ──
        if self.use_meta_label:
            from src.ml.meta_labeler_v2 import MetaLabelerV2
            from src.ml.simple_meta_labeler import MetaLabelContextFeatures

            self._meta_labeler = MetaLabelerV2()
            self._meta_labeler.load(self.meta_label_path)
            print(f"[MLStrategy] Loaded MetaLabelerV2: {self.meta_label_path}")

            ctx_gen = MetaLabelContextFeatures(
                adx_window=14,
                vol_window=20,
                vol_lookback=60,
                cluster_window=20,
            )
            self._meta_context = ctx_gen.generate(
                ohlcv,
                signal_dates=features.index,
                primary_probs=self._probs,
            )
            # Add primary_prob to match MetaLabelerV2 feature expectations
            aligned = self._probs.reindex(self._meta_context.index)
            self._meta_context["primary_prob"] = aligned.fillna(0.5)

        # ── C2: Precompute sentiment scores ──
        if self.use_sentiment:
            provider = SyntheticSentimentProvider(seed=42)
            self._sentiment_modifier = SentimentSignalModifier(
                sentiment_provider=provider,
                sentiment_weight=self.sentiment_weight,
                min_sentiment_abs=0.1,
            )
            sentiment_raw = provider.get_sentiment(
                dates=pd.DatetimeIndex(self.data.index),
                symbol=self.ticker,
            )
            self._sentiment_scores = sentiment_raw.where(sentiment_raw.abs() >= 0.1, 0.0)
            print(
                f"[MLStrategy] Sentiment enabled (weight={self.sentiment_weight}, "
                f"non-neutral={int((self._sentiment_scores != 0).sum())} bars)"
            )

        # ── N1: Precompute LM sentiment scores via synthetic headlines ──
        if self.use_lm_sentiment:
            from src.signals.sentiment.dictionary import (
                LMTradingSignalModifier,
            )

            self._lm_sentiment_modifier = LMTradingSignalModifier(
                sentiment_weight=self.lm_sentiment_weight,
            )
            n_bars = len(self.data.Close)
            close_series = (
                self.data.Close.s if hasattr(self.data.Close, "s") else pd.Series(self.data.Close)
            )
            returns = close_series.pct_change().fillna(0.0)
            headlines = []
            for r in returns:
                if r > 0.015:
                    headlines.append(
                        "strong performance profit growth exceeds expectations "
                        "revenue up margins expanding"
                    )
                elif r > 0.005:
                    headlines.append("modest gains improving conditions steady growth")
                elif r < -0.015:
                    headlines.append(
                        "faces headwinds revenue decline concerns losses downturn risk uncertainty"
                    )
                elif r < -0.005:
                    headlines.append("modest decline weakness pressure soft demand")
                else:
                    headlines.append("")
            self._lm_headlines = pd.Series(headlines, index=pd.DatetimeIndex(self.data.index))
            active = int((self._lm_headlines != "").sum())
            print(
                f"[MLStrategy] LM Sentiment enabled "
                f"(weight={self.lm_sentiment_weight}, "
                f"active_bars={active}/{n_bars})"
            )

        # ── C3: Precompute event days for signal suppression ──
        if self.use_event_filter:
            from src.signals.event_detector import EventSignalSuppressor

            self._event_suppressor = EventSignalSuppressor(min_impact="HIGH")
            self._event_days = self._event_suppressor.preload_event_dates(
                pd.DatetimeIndex(self.data.index),
                symbol=self.ticker,
            )
            event_count = int(self._event_days.sum())
            print(
                f"[MLStrategy] Event filter enabled — "
                f"{event_count} high-impact event days suppressed"
            )

        # ── C4: Load RL execution model ──
        if self.use_rl_execution:
            from src.rl.rl_trade_executor import RLTradeExecutor

            self._rl_executor = RLTradeExecutor(model_path=self.rl_model_path)
            self._rl_confirm_streak = 0
            self._rl_entry_price = 0.0
            if not self._rl_executor.is_trained:
                print(
                    f"[MLStrategy] RL execution model not found at "
                    f"{self.rl_model_path} — train with scripts/train_rl_executor.py"
                )
            else:
                print("[MLStrategy] RL execution agent loaded")

        # ── C5: Initialize Kelly allocator ──
        if self.use_kelly:
            from src.risk.kelly_allocator import KellyAllocator

            self._kelly_allocator = KellyAllocator(
                kelly_fraction=self.kelly_fraction,
                max_allocation=self.kelly_max_allocation,
                method="classic",
            )
            print(
                f"[MLStrategy] Kelly allocator enabled "
                f"(fraction={self.kelly_fraction}, max={self.kelly_max_allocation})"
            )

        # ── C5: Precompute crash risk scores ──
        if self.use_crash_filter:
            from src.risk.crash_factor import BehavioralCrashDetector

            self._crash_detector = BehavioralCrashDetector()
            crash_df = self._crash_detector.aggregate_crash_risk(ohlcv)
            self._crash_risk = crash_df["crash_risk_score"]
            crash_bars = int((self._crash_risk >= self.crash_threshold).sum())
            print(
                f"[MLStrategy] Crash filter enabled — "
                f"{crash_bars} bars flagged as crash regime "
                f"(threshold={self.crash_threshold})"
            )

        # ── E2: Lazy-init Chronos signal generator ──
        if self.use_chronos:
            from src.signals.chronos_signal import ChronosSignalGenerator

            self._chronos_gen = ChronosSignalGenerator(
                model_size=self.chronos_model_size,
                device="cpu",
                forecast_horizon=5,
                context_length=126,
            )
            print(
                f"[MLStrategy] Chronos enabled — "
                f"model={self.chronos_model_size} weight={self.chronos_weight}"
            )

    def next(self) -> None:
        """Evaluate ML signal and manage position."""
        i = len(self.data) - 1  # current bar index

        prob = self._probs.iloc[i]
        if pd.isna(prob):
            return

        # ── E2: Chronos signal blend ──
        if self.use_chronos and self._chronos_gen is not None:
            chronos_signal = self._chronos_gen.generate_signal(self.data.Close.s)
            if chronos_signal != 0.0:
                prob = self.chronos_weight * chronos_signal + (1 - self.chronos_weight) * prob

        atr_val = self._atr.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            return

        price = self.data.Close[-1]

        # ── B12: Dynamic Ensemble weight update ──
        if self._dynamic_ensemble is not None and i > 0:
            prev_close = self.data.Close[-2]
            if prev_close > 0:
                bar_return = (self.data.Close[-1] / prev_close) - 1.0
                self._dynamic_ensemble.update_weights(bar_return)
            # Recompute probability for this bar with updated weights
            prob = self._dynamic_ensemble.predict(self._features_df.iloc[[i]])[0]
            self._probs.iloc[i] = prob

        # ── C8: Pattern boost computation ──
        if self.use_pattern_boost and self._pattern_booster is not None:
            boost = self._pattern_booster.get_boost(i)
            # Apply directional boost: bull boost for long-biased, bear boost suppresses
            if prob >= self.entry_threshold and boost.bull_boost > 0:
                prob = min(1.0, prob + boost.bull_boost)
            elif prob <= self.exit_threshold and boost.bear_boost > 0:
                # Bearish pattern confluence = confirm exit / suppress long entry
                prob = max(0.0, prob - boost.bear_boost * 0.5)

        # ── C2: Sentiment adjustment ──
        if self.use_sentiment and self._sentiment_scores is not None:
            idx = self.data.index[i]
            sentiment = self._sentiment_scores.loc[idx]
            if abs(sentiment) >= 0.1:
                adjustment = 1.0 + self.sentiment_weight * sentiment
                self._probs.iloc[i] = float(np.clip(prob * adjustment, 0, 1))
                prob = self._probs.iloc[i]

        # ── N1: LM Dictionary sentiment adjustment ──
        if self.use_lm_sentiment and self._lm_sentiment_modifier is not None:
            idx = self.data.index[i]
            headline = self._lm_headlines.loc[idx]
            if headline:
                prob = self._lm_sentiment_modifier.adjust_signal(prob, headline)
                self._probs.iloc[i] = prob

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

        # ── C3: Event-day entry suppression ──
        if self.use_event_filter and self._event_days is not None:
            idx = self.data.index[i]
            if idx in self._event_days.index and self._event_days.loc[idx]:
                self._confirm_count = 0
                return

        # ── C5: Crash filter entry suppression ──
        if self.use_crash_filter and self._crash_risk is not None:
            crash_score = float(self._crash_risk.iloc[i])
            if pd.notna(crash_score) and crash_score >= self.crash_threshold:
                self._confirm_count = 0
                return

        # ── C4: RL-based execution (replaces threshold entry/exit) ──
        if self.use_rl_execution and self._rl_executor is not None and self._rl_executor.is_trained:
            # Check crash filter before RL execution
            if self.use_crash_filter and self._crash_risk is not None:
                crash_score = float(self._crash_risk.iloc[i])
                if pd.notna(crash_score) and crash_score >= self.crash_threshold:
                    self._rl_confirm_streak = 0
                    return

            # Update RL confirm streak
            if prob >= self.entry_threshold:
                self._rl_confirm_streak += 1
            else:
                self._rl_confirm_streak = 0

            # Build observation vector
            vr_i = float(self._vol_regime.iloc[i]) if pd.notna(self._vol_regime.iloc[i]) else 1.0
            atr_pct = float(atr_val / price) if price > 0 else 0.01
            sent = (
                float(self._sentiment_scores.iloc[i]) if self._sentiment_scores is not None else 0.0
            )
            is_event = (
                float(self._event_days.iloc[i])
                if self._event_days is not None and self.data.index[i] in self._event_days.index
                else 0.0
            )
            position_pnl = 0.0
            if self.position and self.position.is_long and self._rl_entry_price > 0:
                position_pnl = float(price / self._rl_entry_price - 1.0)
            mkt_5d = (
                float(price / self.data.Close[-min(6, i + 1)]) - 1.0
                if i >= 5 and self.data.Close[-min(6, i + 1)] > 0
                else 0.0
            )

            state = np.array(
                [
                    float(np.clip(prob, 0, 1)),
                    float(np.clip(vr_i, 0, 10)),
                    float(np.clip(atr_pct, 0, 1)),
                    float(np.clip(sent, -1, 1)),
                    float(min(self._rl_confirm_streak, 100)),
                    float(position_pnl),
                    float(mkt_5d),
                    float(is_event),
                ],
                dtype=np.float32,
            )

            action = self._rl_executor.predict(state)

            # ── RL Exit decision ──
            if self.position and self.position.is_long and action == 2:
                self.position.close()
                self._rl_confirm_streak = 0
                self._rl_entry_price = 0.0
                return

            # ── RL Entry decision ──
            if not self.position and action == 1:
                sl_price = price - self.sl_atr_mult * atr_val
                if self.use_trail_stop:
                    tp_price = None
                    self._trail_high = price
                    self._trail_sl = sl_price
                else:
                    tp_price = price + self.tp_atr_mult * atr_val
                risk_amount = self._get_risk_amount(prob, price, sl_price)
                size = max(1, int(risk_amount / (price - sl_price)))
                self.buy(size=size, sl=sl_price, tp=tp_price)
                self._rl_entry_price = price
                self._rl_confirm_streak = 0
            return

        # ── C7.2: Consecutive confirmation ──
        if prob >= self.entry_threshold:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        if self._confirm_count < self.confirm_bars:
            return

        # ── B13: Meta-Labeler filter ──
        if (
            self.use_meta_label
            and self._meta_labeler is not None
            and self._meta_context is not None
        ):
            meta_row = self._meta_context.loc[self._meta_context.index == self.data.index[i]]
            if len(meta_row) == 0:
                self._confirm_count = 0
                return
            meta_pred = self._meta_labeler.predict(meta_row.iloc[0].to_dict())
            if not meta_pred.take_trade:
                self._confirm_count = 0
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
        risk_amount = self._get_risk_amount(prob, price, sl_price)
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

    def _get_risk_amount(self, prob: float, entry_price: float, stop_price: float) -> float:
        """Calculate risk amount for position sizing.

        Uses Kelly allocator if enabled (C5), otherwise fixed fractional sizing.
        """
        if self.use_kelly and self._kelly_allocator is not None:
            edge = self._kelly_allocator.estimate_edge_from_probability(
                model_probability=prob,
            )
            allocation = self._kelly_allocator.compute(edge, self.equity)
            if not self._kelly_allocator.should_trade(allocation):
                return self.equity * self.risk_pct
            return self.equity * allocation.adjusted_fraction
        return self.equity * self.risk_pct
