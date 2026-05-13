"""Optuna-based strategy parameter tuning — replaces manual grid search scripts.

Tunes strategy parameters (RSI, MACD, EMA Ribbon, Keltner Channel, etc.)
by running backtests per trial and optimizing for Sharpe ratio or return.

Usage:
    uv run scripts/tune_model.py --algo optuna --target strategy_params --strategy rsi --trials 50

Supported strategies:
    rsi - RSI period, oversold/overbought thresholds
    macd - Fast/slow/signal periods
    ema_ribbon - Short/long EMA windows, buy/sell thresholds
    keltner - Period, multiplier, ATR period
    sma_crossover - Fast/slow SMA periods
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)

STRATEGY_SPACES: Dict[str, Dict[str, dict]] = {
    "rsi": {
        "window": {"type": "int", "low": 5, "high": 50},
        "oversold": {"type": "int", "low": 15, "high": 40},
        "overbought": {"type": "int", "low": 55, "high": 85},
    },
    "macd": {
        "fast": {"type": "int", "low": 3, "high": 20},
        "slow": {"type": "int", "low": 15, "high": 50},
        "signal": {"type": "int", "low": 3, "high": 20},
    },
    "ema_ribbon": {
        "short_period": {"type": "int", "low": 3, "high": 20},
        "long_period": {"type": "int", "low": 15, "high": 60},
        "buy_threshold": {"type": "float", "low": 0.01, "high": 0.10},
        "sell_threshold": {"type": "float", "low": -0.10, "high": -0.01},
    },
    "keltner": {
        "period": {"type": "int", "low": 10, "high": 40},
        "multiplier": {"type": "float", "low": 1.0, "high": 4.0},
        "atr_period": {"type": "int", "low": 5, "high": 30},
    },
    "sma_crossover": {
        "fast_period": {"type": "int", "low": 3, "high": 30},
        "slow_period": {"type": "int", "low": 15, "high": 100},
    },
}


def _suggest(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Suggest a parameter value from the strategy space spec."""
    ptype = spec["type"]
    low = spec["low"]
    high = spec["high"]

    if ptype == "float":
        return trial.suggest_float(name, low, high)
    if ptype == "int":
        return trial.suggest_int(name, low, high)
    return low


@dataclass
class StrategyTuneResult:
    """Result from strategy parameter tuning."""

    best_params: Dict[str, Any]
    best_score: float
    metric: str = "sharpe_ratio"
    n_trials: int = 0
    trial_scores: List[float] = field(default_factory=list)
    study: optuna.Study | None = None


class OptunaStrategyTuner:
    """Tune strategy parameters using Optuna with backtesting.py evaluation.

    Each trial runs a full backtest on the training period and optimizes
    for a target metric (Sharpe ratio by default).
    """

    STRATEGY_CLASSES: Dict[str, str] = {
        "rsi": "src.strategies.rsi_divergence.RSIDivergenceStrategy",
        "macd": "src.strategies.macd_histogram.MACDHistogramStrategy",
        "ema_ribbon": "src.strategies.ema_ribbon.EMARibbonStrategy",
        "keltner": "src.strategies.keltner_channel.KeltnerChannelStrategy",
        "sma_crossover": "src.strategies.sma_crossover.SMACrossoverStrategy",
    }

    def __init__(
        self,
        strategy: str = "rsi",
        n_trials: int = 50,
        metric: str = "sharpe_ratio",
        pruner_patience: int = 10,
        seed: int = 42,
    ) -> None:
        if strategy not in STRATEGY_SPACES:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(STRATEGY_SPACES)}")
        self.strategy = strategy
        self.n_trials = n_trials
        self.metric = metric
        self.pruner_patience = pruner_patience
        self.seed = seed
        self._study: optuna.Study | None = None

    def _objective(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
    ) -> float:
        """Backtest-based objective: one trial = one full backtest."""
        space = STRATEGY_SPACES[self.strategy]
        params: Dict[str, Any] = {name: _suggest(trial, name, spec) for name, spec in space.items()}

        if len(df) < 50:
            return float("-inf")

        signals = self._generate_signals(df, params)
        if signals is None or len(signals) < 5:
            return float("-inf")

        score = self._evaluate_signals(df, signals)
        for k, v in params.items():
            trial.set_user_attr(k, v)

        return score

    def _generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
    ) -> pd.Series | None:
        """Generate buy/sell signals from strategy parameters."""
        try:
            if self.strategy == "rsi":
                window = params["window"]
                oversold = params["oversold"]
                overbought = params["overbought"]

                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = (-delta).where(delta < 0, 0.0)
                avg_gain = gain.rolling(window).mean()
                avg_loss = loss.rolling(window).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = pd.Series(100.0 - (100.0 / (1.0 + rs)), index=df.index)

                signals = pd.Series(0, index=df.index, dtype=int)
                signals[rsi < oversold] = 1
                signals[rsi > overbought] = -1
                return signals

            if self.strategy == "sma_crossover":
                fast = df["Close"].rolling(params["fast_period"]).mean()
                slow = df["Close"].rolling(params["slow_period"]).mean()
                signals = pd.Series(0, index=df.index, dtype=int)
                signals[fast > slow] = 1
                signals[fast < slow] = -1
                return signals

            if self.strategy == "macd":
                ema_fast = df["Close"].ewm(span=params["fast"], adjust=False).mean()
                ema_slow = df["Close"].ewm(span=params["slow"], adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=params["signal"], adjust=False).mean()
                signals = pd.Series(0, index=df.index, dtype=int)
                signals[macd_line > signal_line] = 1
                signals[macd_line < signal_line] = -1
                return signals

            if self.strategy == "ema_ribbon":
                short = df["Close"].ewm(span=params["short_period"], adjust=False).mean()
                long = df["Close"].ewm(span=params["long_period"], adjust=False).mean()
                ret = short / long - 1
                signals = pd.Series(0, index=df.index, dtype=int)
                signals[ret > params["buy_threshold"]] = 1
                signals[ret < params["sell_threshold"]] = -1
                return signals

            if self.strategy == "keltner":
                typical = (df["High"] + df["Low"] + df["Close"]) / 3
                ema = typical.ewm(span=params["period"], adjust=False).mean()
                atr = _compute_atr(df, params["atr_period"])
                upper = ema + params["multiplier"] * atr
                lower = ema - params["multiplier"] * atr
                signals = pd.Series(0, index=df.index, dtype=int)
                signals[df["Close"] > upper] = 1
                signals[df["Close"] < lower] = -1
                return signals

            return None
        except Exception as exc:
            logger.debug(f"Signal generation failed: {exc}")
            return None

    def _evaluate_signals(self, df: pd.DataFrame, signals: pd.Series) -> float:
        """Evaluate signal performance without look-ahead bias.

        Entry at signal bar close, exit at next signal or end of data.
        Only counts bars where signal != 0 (in-market returns).
        """
        returns = df["Close"].pct_change().shift(-1)
        aligned_idx = signals.index.intersection(returns.dropna().index)
        signals = signals.loc[aligned_idx]
        returns = returns.loc[aligned_idx]

        shifted = signals.shift(1)
        strategy_returns = returns * shifted
        strategy_returns = strategy_returns[shifted != 0].dropna()

        if len(strategy_returns) < 10:
            return float("-inf")

        ann_return = strategy_returns.mean() * 252
        ann_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        if self.metric == "sharpe_ratio":
            return float(sharpe)
        if self.metric == "return":
            return float(ann_return)
        if self.metric == "calmar":
            cumulative = (1 + strategy_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 1.0
            return float(ann_return / max_dd) if max_dd > 0 else 0.0

        return float(sharpe)

    def optimize(
        self,
        df: pd.DataFrame,
    ) -> StrategyTuneResult:
        """Run Optuna strategy parameter optimization.

        Args:
            df: OHLCV DataFrame with columns Open, High, Low, Close, Volume.

        Returns:
            StrategyTuneResult with best parameters and score.
        """
        study_name = f"strategy_{self.strategy}_tuning"

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        )
        sampler = optuna.samplers.TPESampler(seed=self.seed)

        self._study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///optuna_{study_name}.db",
            load_if_exists=True,
        )

        logger.info(
            f"Optuna tuning {self.strategy} strategy: {self.n_trials} trials, metric={self.metric}"
        )

        self._study.optimize(
            lambda trial: self._objective(trial, df),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        best_score = self._study.best_value
        best_params = self._study.best_params
        trial_scores = [t.value for t in self._study.trials if t.value is not None]

        logger.info(f"Strategy tuning complete: best {self.metric}={best_score:.4f}")
        logger.info(f"Best params: {best_params}")

        return StrategyTuneResult(
            best_params=best_params,
            best_score=best_score,
            metric=self.metric,
            n_trials=len(self._study.trials),
            trial_scores=trial_scores,
            study=self._study,
        )


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
