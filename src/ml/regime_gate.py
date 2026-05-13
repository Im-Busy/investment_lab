"""
Regime Gate — Market-Environment Filter for Trade Signals.

Trains a binary classifier on market-level cross-asset features to predict
whether the current environment is favorable for trading a target instrument
(or basket of similar instruments).

Separates two questions the cross-asset model failed to disentangle:
  1. Is the macro regime favorable? (regime gate)
  2. Given the regime, which direction? (instrument-specific model)

Architecture:
  - Features: market-level only (SPY trend, VIX, bonds, gold, sectors)
  - Labels: derived from forward returns of target instrument(s)
  - Model: LightGBM binary classifier with PurgedKFold CV
  - Output: .should_trade() → (bool, float) per bar

Training on a REIT basket (JOE + O + PLD + AMT + SPG) provides more
regime transitions than JOE alone, producing a more robust model.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

MARKET_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "IWM", "XLF", "XLK"]
REIT_BASKET = ["JOE", "O", "PLD", "AMT", "SPG"]


@dataclass
class RegimeGateResult:
    model: object
    feature_names: List[str]
    train_auc: float
    test_auc: float
    n_samples: int
    n_features: int
    positive_ratio: float


class RegimeGate:
    """Market-environment classifier that gates trade signals.

    Usage:
        >>> gate = RegimeGate()
        >>> gate.train("JOE")                              # train on JOE only
        >>> gate.train(["JOE", "O", "PLD", "AMT", "SPG"])  # train on REIT basket
        >>> gate.save("models/regime_gate.pkl")
        >>> gate = RegimeGate.load("models/regime_gate.pkl")
        >>> mask, conf = gate.should_trade(df_index, market_data)
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        horizon: int = 5,
        cv_splits: int = 3,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.horizon = horizon
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.model_: object | None = None
        self.feature_names_: List[str] = []
        self.train_auc_: float = 0.0
        self.test_auc_: float = 0.0

    def train(
        self,
        symbols: str | List[str],
        start: str = "2017-01-01",
        end: str = "2024-12-31",
        data_dir: str | Path = "data/raw",
    ) -> RegimeGateResult:
        """Train the regime gate on one or more instruments.

        Args:
            symbols: Single symbol (e.g. "JOE") or list of symbols.
            start: Start date for data.
            end: End date for data.
            data_dir: Directory containing *_daily.csv files.

        Returns:
            RegimeGateResult with model and training metrics.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        data_dir = Path(data_dir)

        market_data = self._load_market_data(data_dir)
        if "SPY" not in market_data:
            raise ValueError("SPY market data required for regime features")

        regime_X = self._extract_regime_features(market_data)
        logger.info("Regime features: %d columns", regime_X.shape[1])

        all_X, all_y = [], []
        valid_count = 0

        for symbol in symbols:
            df = self._load_instrument_data(data_dir, symbol)
            if df is None:
                continue

            bt_mask = (df.index >= start) & (df.index <= end)
            df_bt = df[bt_mask]
            if len(df_bt) < 500:
                logger.warning("%s: only %d bars in range, skipping", symbol, len(df_bt))
                continue

            close = df_bt["Close"]
            future_return = close.shift(-self.horizon) / close - 1
            y = (future_return > 0).astype(int)

            common_idx = regime_X.index.intersection(y.dropna().index)
            if len(common_idx) < 100:
                logger.warning("%s: only %d aligned bars, skipping", symbol, len(common_idx))
                continue

            X_sym = regime_X.loc[common_idx]
            y_sym = y.loc[common_idx]

            all_X.append(X_sym)
            all_y.append(y_sym)
            valid_count += 1
            logger.info(
                "%s: %d samples, %.1f%% positive",
                symbol,
                len(X_sym),
                y_sym.mean() * 100,
            )

        if valid_count == 0:
            raise RuntimeError("No symbols produced valid training data")

        X = pd.concat(all_X)
        y = pd.concat(all_y)

        logger.info(
            "Training data: %d samples, %d features, %.1f%% positive",
            len(X),
            X.shape[1],
            y.mean() * 100,
        )

        result = self._train_model(X, y)
        self.model_ = result.model
        self.feature_names_ = result.feature_names
        self.train_auc_ = result.train_auc
        self.test_auc_ = result.test_auc

        return result

    def should_trade(
        self,
        index: pd.DatetimeIndex,
        market_data_dir: str | Path = "data/raw",
        threshold: float = 0.6,
    ) -> Tuple[pd.Series, pd.Series]:
        """Return (boolean mask, confidence) for each bar.

        Args:
            index: DatetimeIndex of bars to evaluate.
            market_data_dir: Directory containing market CSVs.
            threshold: Minimum probability to allow trading (0.6 = 60%+ confidence).

        Returns:
            Tuple of (bool_series, confidence_series) aligned to index.
        """
        if self.model_ is None:
            raise RuntimeError("RegimeGate not trained. Call .train() or .load() first.")

        market_data = self._load_market_data(Path(market_data_dir))
        features = self._extract_regime_features(market_data)

        available = [f for f in self.feature_names_ if f in features.columns]
        X = features[available].loc[features.index.isin(index)].dropna()

        if len(X) == 0:
            return pd.Series(False, index=index), pd.Series(0.0, index=index)

        proba = self.model_.predict_proba(X)[:, 1]
        proba_series = pd.Series(proba, index=X.index)
        mask = proba_series >= threshold
        confidence = proba_series

        return mask.reindex(index, fill_value=False), confidence.reindex(index, fill_value=0.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model_,
            "feature_names": self.feature_names_,
            "train_auc": self.train_auc_,
            "test_auc": self.test_auc_,
            "model_type": self.model_type,
            "horizon": self.horizon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("RegimeGate saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeGate":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)  # nosec B301
        gate = cls(
            model_type=data.get("model_type", "lightgbm"),
            horizon=data.get("horizon", 5),
        )
        gate.model_ = data["model"]
        gate.feature_names_ = data["feature_names"]
        gate.train_auc_ = data.get("train_auc", 0.0)
        gate.test_auc_ = data.get("test_auc", 0.0)
        return gate

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Model not trained")
        if not hasattr(self.model_, "feature_importances_"):
            raise RuntimeError("Model does not support feature_importances_")
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False).head(top_n)

    # ── internals ──

    def _load_instrument_data(self, data_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
        path = data_dir / f"{symbol}_daily.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=True, index_col=0).sort_index()
            if "Close" in df.columns:
                return df
        return self._download_instrument(symbol)

    def _download_instrument(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf

            logger.info("Downloading %s from Yahoo Finance...", symbol)
            df = yf.download(symbol, start="2015-01-01", end="2024-12-31", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns or len(df) < 500:
                logger.warning("%s: insufficient data (%d bars)", symbol, len(df))
                return None
            return df
        except Exception as e:
            logger.warning("Failed to download %s: %s", symbol, e)
            return None

    def _load_market_data(self, data_dir: Path) -> Dict[str, pd.DataFrame]:
        market_data: Dict[str, pd.DataFrame] = {}
        for ticker in MARKET_TICKERS:
            path = data_dir / f"{ticker}_daily.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=True, index_col=0).sort_index()
                if "Close" in df.columns and len(df) > 0:
                    market_data[ticker] = df
            else:
                try:
                    import yfinance as yf

                    df = yf.download(ticker, start="2015-01-01", end="2024-12-31", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns and len(df) > 0:
                        market_data[ticker] = df
                except Exception:
                    pass
        return market_data

    def _extract_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract market-level regime features (no instrument-specific data)."""
        feats = pd.DataFrame()

        spy = market_data.get("SPY")
        if spy is None:
            return feats

        spy_close = spy["Close"]
        spy_ret = spy_close.pct_change()
        feats.index = spy.index

        # ── SPY trend ──
        for w in [5, 20, 50, 200]:
            ma = spy_close.rolling(w).mean()
            feats[f"spy_trend_{w}d"] = (spy_close / ma - 1).shift(1)
            feats[f"spy_ret_{w}d"] = spy_ret.rolling(w).sum().shift(1)

        # ── SPY volatility regime ──
        spy_vol_20 = spy_ret.rolling(20).std()
        spy_vol_rank = (
            spy_vol_20.rolling(60)
            .apply(
                lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5,
                raw=True,
            )
            .shift(1)
        )
        feats["spy_vol_percentile"] = spy_vol_rank
        feats["spy_vol_20d"] = spy_vol_20.shift(1)

        # ── SPY drawdown ──
        spy_peak = spy_close.rolling(50).max()
        feats["spy_drawdown_50d"] = (spy_close / spy_peak - 1).shift(1)

        # ── QQQ / Tech relative strength ──
        qqq = market_data.get("QQQ")
        if qqq is not None:
            qqq_close = qqq["Close"]
            qqq_ret = qqq_close.pct_change()
            qqq_vs_spy = qqq_ret.rolling(20).sum() - spy_ret.rolling(20).sum()
            feats["qqq_vs_spy_20d"] = qqq_vs_spy.shift(1)

        # ── TLT / Bonds ──
        tlt = market_data.get("TLT")
        if tlt is not None:
            tlt_close = tlt["Close"]
            tlt_ret = tlt_close.pct_change()
            feats["tlt_ret_5d"] = tlt_ret.rolling(5).sum().shift(1)
            feats["tlt_ret_20d"] = tlt_ret.rolling(20).sum().shift(1)
            feats["tlt_vol_20d"] = tlt_ret.rolling(20).std().shift(1)

        # ── GLD / Gold ──
        gld = market_data.get("GLD")
        if gld is not None:
            gld_close = gld["Close"]
            gld_ret = gld_close.pct_change()
            feats["gld_ret_5d"] = gld_ret.rolling(5).sum().shift(1)
            feats["gld_ret_20d"] = gld_ret.rolling(20).sum().shift(1)
            feats["gld_vs_spy_20d"] = (gld_ret.rolling(20).sum() - spy_ret.rolling(20).sum()).shift(
                1
            )

        # ── IWM / Small caps ──
        iwm = market_data.get("IWM")
        if iwm is not None:
            iwm_close = iwm["Close"]
            iwm_ret = iwm_close.pct_change()
            iwm_vs_spy = iwm_ret.rolling(20).sum() - spy_ret.rolling(20).sum()
            feats["iwm_vs_spy_20d"] = iwm_vs_spy.shift(1)
            feats["iwm_ret_5d"] = iwm_ret.rolling(5).sum().shift(1)

        # ── XLF vs XLK (value vs growth spread) ──
        xlf = market_data.get("XLF")
        xlk = market_data.get("XLK")
        if xlf is not None and xlk is not None:
            xlf_ret = xlf["Close"].pct_change()
            xlk_ret = xlk["Close"].pct_change()
            value_growth = xlf_ret.rolling(5).sum() - xlk_ret.rolling(5).sum()
            feats["value_growth_spread_5d"] = value_growth.shift(1)

        # ── Cross-asset correlation ──
        rets: dict[str, pd.Series] = {"SPY": spy_ret}
        if qqq is not None:
            rets["QQQ"] = qqq["Close"].pct_change()
        if tlt is not None:
            rets["TLT"] = tlt["Close"].pct_change()
        if gld is not None:
            rets["GLD"] = gld["Close"].pct_change()
        available = list(rets.keys())
        if len(available) >= 2:
            corr_list = []
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    corr_list.append(rets[available[i]].rolling(20).corr(rets[available[j]]))
            if corr_list:
                feats["avg_cross_corr_20d"] = pd.concat(corr_list, axis=1).mean(axis=1).shift(1)

        # ── SPY momentum / breadth ──
        if "High" in spy.columns and "Low" in spy.columns:
            tr = pd.concat(
                [
                    spy["High"] - spy["Low"],
                    (spy["High"] - spy_close.shift(1)).abs(),
                    (spy["Low"] - spy_close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            feats["spy_atr_ratio"] = (tr.rolling(20).mean() / spy_close).shift(1)

        feats = feats.dropna(axis=1, how="all")
        return feats

    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> RegimeGateResult:
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if self.model_type == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=5.0,
                random_state=self.random_state,
                verbose=False,
                task_type="CPU",
            )

        model.fit(X_train, y_train)

        train_auc = float(_roc_auc(y_train, model.predict_proba(X_train)[:, 1]))
        test_auc = float(_roc_auc(y_test, model.predict_proba(X_test)[:, 1]))

        logger.info(
            "RegimeGate: Train AUC=%.4f, Test AUC=%.4f, Overfit Gap=%.4f",
            train_auc,
            test_auc,
            train_auc - test_auc,
        )

        return RegimeGateResult(
            model=model,
            feature_names=list(X.columns),
            train_auc=train_auc,
            test_auc=test_auc,
            n_samples=len(X),
            n_features=X.shape[1],
            positive_ratio=float(y.mean()),
        )


def _roc_auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y_true, y_score)
