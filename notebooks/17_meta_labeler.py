# %% [markdown]
# # Meta-Labeling — Signal Filtering for Trading Models
#
# Meta-labeling trains a secondary classifier on historical trade outcomes
# to filter out low-quality signals from the primary model.
#
# **Workflow:**
# 1. Select ticker + model below
# 2. Run primary model backtest → generates trades with outcomes (TP/SL/Time)
# 3. Extract features at each trade entry bar
# 4. Train meta-labeler (LightGBM vs CatBoost, auto-selects best)
# 5. Filter trades: only execute where meta-labeler says "take"
# 6. Compare filtered vs unfiltered backtest
#
# **Configurable:** ticker, model path, top-pct, TP/SL multipliers.

# %% [markdown]
# ## Configuration — Change These

# %%
import sys
from pathlib import Path

project_root = Path().absolute().parent if Path().absolute().name == "notebooks" else Path().absolute()
sys.path.insert(0, str(project_root))

# ── CONFIG (change these) ──────────────────────────────────────────────
TICKER = "JOE"
TOP_PCT = 12.5                    # top N% of signals to trade
HORIZON = 5                       # max bars to hold
ATR_MULT_TP = 1.5                 # take-profit ATR multiplier
ATR_MULT_SL = 1.0                 # stop-loss ATR multiplier

# Which model to meta-label
MODEL_PATH = str(
    project_root
    / "models"
    / "pattern_classifier_20260509_051553_v3_baseline_JOE_catboost.pkl"
)

# Date range
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"

# Capital
INITIAL_CAPITAL = 100_000.0
COMMISSION = 0.001
RISK_PER_TRADE = 0.02

# Meta-labeler params
META_PROB_THRESHOLD = 0.5         # 0.5 = balanced, >0.5 = more selective
META_CV_SPLITS = 5                # PurgedKFold splits

# %% [markdown]
# ## 1. Load Data

# %%
import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.feature_engineering import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CSV_PATH = project_root / "data" / "raw" / f"{TICKER}_daily.csv"
df = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0).sort_index()

bt_mask = (df.index >= START_DATE) & (df.index <= END_DATE)
df_bt = df[bt_mask].copy()
df_full = df  # full dataset for rolling feature calculation

logger.info(f"Ticker: {TICKER}")
logger.info(f"Full data: {len(df_full)} bars ({df_full.index[0].date()} to {df_full.index[-1].date()})")
logger.info(f"Backtest:  {len(df_bt)} bars ({df_bt.index[0].date()} to {df_bt.index[-1].date()})")

# %% [markdown]
# ## 2. Feature Extraction (Primary Model)

# %%
extractor = FeatureExtractor()
features = extractor.extract_all_features(df_full)
logger.info(f"Instrument features: {features.shape[1]} columns")

# %% [markdown]
# ## 3. Load Primary Model & Generate Signals

# %%
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

catboost_model = model_data["model"]
model_feature_names = model_data["feature_names"]

available = [f for f in model_feature_names if f in features.columns]
missing = set(model_feature_names) - set(available)
if missing:
    logger.info(f"Missing features: {sorted(missing)}")

X_all = features[available].loc[features.index.isin(df_bt.index)].dropna()
logger.info(f"Valid prediction bars: {len(X_all)} / {len(df_bt)}")

# Get raw probabilities
raw_probs = catboost_model.predict_proba(X_all)[:, 1]
probs_series = pd.Series(raw_probs, index=X_all.index)
logger.info(f"Mean prob: {probs_series.mean():.4f}, std: {probs_series.std():.4f}")

# Select top N% as signals
n_signals = max(1, int(len(df_bt) * TOP_PCT / 100))
signal_dates = probs_series.nlargest(n_signals).index
logger.info(f"Top {TOP_PCT:.1f}% = {n_signals} signals (of {len(df_bt)} bars)")

# %% [markdown]
# ## 4. Run Backtest — Generate Trades with Outcomes

# %%
# ── ATR calculation ──
high, low, close = df_full["High"], df_full["Low"], df_full["Close"]
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# ── Backtest engine ──
@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    pnl: float
    exit_reason: str         # TP, SL, Time, EOD
    bars_held: int
    position_size: int

trades: List[Trade] = []
in_trade = False
entry_price = 0.0
tp_price = 0.0
sl_price = 0.0
position_size = 0
bars_in_trade = 0
entry_date: Optional[pd.Timestamp] = None
signal_set = set(signal_dates)
bt_indices = list(df_bt.index)

for i in range(len(bt_indices)):
    idx = bt_indices[i]
    bar = df_bt.iloc[i]

    if in_trade:
        bars_in_trade += 1
        exit_price = None
        exit_reason = "held"

        if bar["Low"] <= sl_price:
            exit_price = sl_price
            exit_reason = "SL"
        elif bar["High"] >= tp_price:
            exit_price = tp_price
            exit_reason = "TP"
        elif bars_in_trade >= HORIZON:
            exit_price = bar["Close"]
            exit_reason = "Time"

        if exit_price is not None:
            trade_ret = (exit_price - entry_price) / entry_price
            trade_pnl = position_size * (exit_price - entry_price)
            trade_pnl -= position_size * entry_price * COMMISSION * 2
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=idx,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=trade_ret * 100,
                pnl=trade_pnl,
                exit_reason=exit_reason,
                bars_held=bars_in_trade,
                position_size=position_size,
            ))
            in_trade = False
        continue

    if idx not in signal_set:
        continue

    c = bar["Close"]
    if pd.isna(c):
        continue

    _atr = atr.get(idx)
    if _atr is None:
        _atr_idx = atr.index.get_indexer([idx], method="ffill")[0]
        _atr = atr.iloc[_atr_idx] if _atr_idx >= 0 else None
    if _atr is None or pd.isna(_atr) or _atr <= 0:
        continue

    risk_amount = INITIAL_CAPITAL * RISK_PER_TRADE
    sl_dist = ATR_MULT_SL * _atr
    pos = int(risk_amount / sl_dist) if sl_dist > 0 else 0
    if pos == 0:
        continue

    entry_price = c
    tp_price = c + ATR_MULT_TP * _atr
    sl_price = c - ATR_MULT_SL * _atr
    entry_date = idx
    position_size = pos
    bars_in_trade = 0
    in_trade = True

# Close any open trade at end
if in_trade:
    last_bar = df_bt.iloc[-1]
    last_close = last_bar["Close"]
    trade_ret = (last_close - entry_price) / entry_price
    trade_pnl = position_size * (last_close - entry_price)
    trade_pnl -= position_size * entry_price * COMMISSION * 2
    trades.append(Trade(
        entry_date=entry_date,
        exit_date=df_bt.index[-1],
        entry_price=entry_price,
        exit_price=last_close,
        return_pct=trade_ret * 100,
        pnl=trade_pnl,
        exit_reason="EOD",
        bars_held=bars_in_trade,
        position_size=position_size,
    ))

logger.info(f"Primary model generated {len(trades)} trades")

trades_df = pd.DataFrame([
    {"entry_date": t.entry_date, "exit_date": t.exit_date, "return": t.return_pct,
     "pnl": t.pnl, "exit_reason": t.exit_reason, "bars_held": t.bars_held}
    for t in trades
])

n_tp = (trades_df["exit_reason"] == "TP").sum()
n_sl = (trades_df["exit_reason"] == "SL").sum()
n_time = (trades_df["exit_reason"].isin(["Time", "EOD"])).sum()
logger.info(f"Exit breakdown: TP={n_tp}, SL={n_sl}, Time/EOD={n_time}")

# %% [markdown]
# ## 5. Build Meta-Labeler Features at Each Trade Entry

# %%
# Use ALL features from the primary model (37 for baseline) as meta-labeler context
# Goal: the meta-labeler sees the same market state the primary model saw

entry_dates = [t.entry_date for t in trades]
meta_features = features[available].loc[features.index.isin(entry_dates)]

# Label each trade: 1 = profitable (TP hit), 0 = unprofitable (SL or Time)
meta_labels = pd.Series(
    [(t.exit_reason == "TP") for t in trades],
    index=entry_dates,
    dtype=int,
)

# Filter out rows where features are NaN
valid_mask = meta_features.dropna().index.intersection(meta_labels.dropna().index)
X_meta = meta_features.loc[valid_mask]
y_meta = meta_labels.loc[valid_mask]

logger.info(f"Meta-labeler training data: {len(X_meta)} trades, {X_meta.shape[1]} features")
logger.info(f"Profitable trades: {y_meta.sum()} ({y_meta.mean()*100:.1f}%)")

if len(X_meta) < 20:
    logger.warning("Very few trades — meta-labeler will be unreliable. Increase TOP_PCT.")

# %% [markdown]
# ## 6. Train Meta-Labeler with PurgedKFold CV

# %%
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

cv_results = []

# ── PurgedKFold ──
from src.ml.purged_cv import PurgedKFold

n_splits = min(META_CV_SPLITS, len(X_meta) // 3)
if n_splits < 2:
    # Not enough data — train on first 70%, test on last 30%
    split_idx = int(len(X_meta) * 0.7)
    train_idx = list(range(split_idx))
    test_idx = list(range(split_idx, len(X_meta)))
    folds = [(train_idx, test_idx)]
else:
    splitter = PurgedKFold(n_splits=n_splits, pct_embargo=0.05, label_span=HORIZON)
    folds = list(splitter.split(X_meta))

# ── Try both LightGBM and CatBoost ──
lgbm_scores = []
cb_scores = []

for fold_i, (train_idx, test_idx) in enumerate(folds, 1):
    if len(test_idx) < 5:
        continue
    X_tr, X_te = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
    y_tr, y_te = y_meta.iloc[train_idx], y_meta.iloc[test_idx]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        continue

    # LightGBM
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  min_child_samples=10, random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr)
        lgbm_proba = lgbm.predict_proba(X_te)[:, 1]
        lgbm_scores.append(float(roc_auc_score(y_te, lgbm_proba)))
    except Exception as e:
        logger.debug(f"LightGBM fold {fold_i} failed: {e}")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                min_data_in_leaf=10, random_state=42, verbose=False, task_type="CPU")
        cb.fit(X_tr, y_tr)
        cb_proba = cb.predict_proba(X_te)[:, 1]
        cb_scores.append(float(roc_auc_score(y_te, cb_proba)))
    except Exception as e:
        logger.debug(f"CatBoost fold {i+1} failed: {e}")

lgbm_mean = float(np.mean(lgbm_scores)) if lgbm_scores else 0.5
cb_mean = float(np.mean(cb_scores)) if cb_scores else None

logger.info(f"LightGBM mean AUC: {lgbm_mean:.4f}")
logger.info(f"CatBoost mean AUC: {cb_mean:.4f}" if cb_mean is not None else "CatBoost not available")

# ── Select best model ──
USE_MODEL = "lgbm"
if cb_mean is not None and cb_mean >= 1.10 * lgbm_mean and lgbm_mean > 0:
    USE_MODEL = "catboost"
    logger.info(f"Using CatBoost (AUC {cb_mean:.4f} >= 1.10x LightGBM {lgbm_mean:.4f})")
else:
    logger.info(f"Using LightGBM (AUC {lgbm_mean:.4f})")

# Train final model on all data
if USE_MODEL == "catboost":
    from catboost import CatBoostClassifier
    meta_model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05,
                                    min_data_in_leaf=10, random_state=42, verbose=False, task_type="CPU")
else:
    import lightgbm as lgb
    meta_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                    min_child_samples=10, random_state=42, verbose=-1)

meta_model.fit(X_meta, y_meta)

# %% [markdown]
# ## 7. Find Optimal Meta-Label Threshold

# %%
from sklearn.metrics import roc_curve

all_proba = meta_model.predict_proba(X_meta)[:, 1]
fpr, tpr, thresholds = roc_curve(y_meta, all_proba)
j_scores = tpr - fpr
best_threshold = float(thresholds[np.argmax(j_scores)])

logger.info(f"Optimal threshold (Youden's J): {best_threshold:.4f}")

# Use user-configured threshold if different from 0.5
if META_PROB_THRESHOLD != 0.5:
    best_threshold = META_PROB_THRESHOLD
    logger.info(f"Overriding with user threshold: {best_threshold}")

# %% [markdown]
# ## 8. Meta-Label Filtering — Filtered Backtest

# %%
# For each trade entry, get features and meta-label prediction
# Only execute trades where meta_model says P(win) > threshold

meta_entry_probs = meta_model.predict_proba(X_meta)[:, 1]
meta_accept = meta_entry_probs >= best_threshold

logger.info(f"Meta-labeler accepts {meta_accept.sum()} / {len(meta_accept)} trades "
            f"({meta_accept.mean()*100:.1f}%)")

# Which trades survive?
accepted_dates = set(X_meta.index[meta_accept])

# Re-run backtest with only accepted signals
filtered_trades = [t for t in trades if t.entry_date in accepted_dates]

# Compute metrics for filtered trades
def compute_metrics(trades_list, label):
    if not trades_list:
        return {label: {"trades": 0, "return": 0, "win_rate": 0, "profit_factor": 0}}

    td = pd.DataFrame([
        {"return": t.return_pct, "pnl": t.pnl, "exit_reason": t.exit_reason}
        for t in trades_list
    ])

    total_pnl = td["pnl"].sum()
    total_return = total_pnl / INITIAL_CAPITAL * 100

    win_trades = td[td["return"] > 0]
    loss_trades = td[td["return"] <= 0]
    win_rate = len(win_trades) / len(td) * 100

    gross_profit = win_trades["pnl"].sum() if len(win_trades) > 0 else 0
    gross_loss = abs(loss_trades["pnl"].sum()) if len(loss_trades) > 0 else 1.0
    profit_factor = gross_profit / gross_loss

    cumulative = td["pnl"].cumsum()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / (peak + INITIAL_CAPITAL + 1e-10) * 100
    max_dd = abs(drawdown.min())

    n_tp = (td["exit_reason"] == "TP").sum()
    n_sl = (td["exit_reason"] == "SL").sum()

    return {
        "trades": len(td),
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "n_tp": n_tp,
        "n_sl": n_sl,
    }

unfiltered_metrics = compute_metrics(trades, "unfiltered")
filtered_metrics = compute_metrics(filtered_trades, "filtered")

# %% [markdown]
# ## 9. Results Comparison

# %%
print(f"\n{'='*70}")
print(f"  Meta-Labeling Results — {TICKER} | {Path(MODEL_PATH).name[:50]}")
print(f"{'='*70}")
print(f"{'':<25} {'Unfiltered':>20} {'Filtered':>20}")
print(f"{'-'*70}")
print(f"  Number of Trades        {unfiltered_metrics['trades']:>20d}   {filtered_metrics['trades']:>20d}")
print(f"  Total Return %          {unfiltered_metrics['total_return']:>20.2f}   {filtered_metrics['total_return']:>20.2f}")
print(f"  Win Rate %              {unfiltered_metrics['win_rate']:>20.1f}   {filtered_metrics['win_rate']:>20.1f}")
print(f"  Profit Factor           {unfiltered_metrics['profit_factor']:>20.2f}   {filtered_metrics['profit_factor']:>20.2f}")
print(f"  Max Drawdown %          {unfiltered_metrics['max_drawdown']:>20.2f}   {filtered_metrics['max_drawdown']:>20.2f}")
print(f"  TP / SL                 {str(unfiltered_metrics['n_tp'])+'/'+str(unfiltered_metrics['n_sl']):>20}   {str(filtered_metrics['n_tp'])+'/'+str(filtered_metrics['n_sl']):>20}")
print(f"{'='*70}")

# Calculate uplift
return_uplift = filtered_metrics["total_return"] - unfiltered_metrics["total_return"]
wr_uplift = filtered_metrics["win_rate"] - unfiltered_metrics["win_rate"]
pf_uplift = filtered_metrics["profit_factor"] - unfiltered_metrics["profit_factor"]

print(f"\n  Return Uplift:    {return_uplift:+.2f}%")
print(f"  Win Rate Uplift:  {wr_uplift:+.1f}%")
print(f"  Profit Factor:    {pf_uplift:+.2f}")
print(f"  Meta-label AUC:   {lgbm_mean if USE_MODEL == 'lgbm' else cb_mean:.4f}")
print(f"  Model used:       {USE_MODEL}")
print(f"  Threshold:        {best_threshold:.4f}")

# Feature importance
print(f"\n--- Top 10 Meta-Label Features ---")
if hasattr(meta_model, "feature_importances_"):
    importance = meta_model.feature_importances_
    top_features = sorted(zip(model_feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in top_features:
        print(f"  {name:<30} {imp:.4f}")

print(f"\n{'='*70}")

# %% [markdown]
# ## 10. Equity Curve Comparison

# %%
import matplotlib.pyplot as plt

unfiltered_cum = pd.DataFrame([
    {"date": t.entry_date, "pnl": t.pnl} for t in trades
]).set_index("date")["pnl"].cumsum() + INITIAL_CAPITAL

if filtered_trades:
    filtered_cum = pd.DataFrame([
        {"date": t.entry_date, "pnl": t.pnl} for t in filtered_trades
    ]).set_index("date")["pnl"].cumsum() + INITIAL_CAPITAL
else:
    filtered_cum = unfiltered_cum.copy()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(unfiltered_cum.index, unfiltered_cum.values, label="Unfiltered", alpha=0.7, linewidth=1.5)
ax.plot(filtered_cum.index, filtered_cum.values, label="Meta-Label Filtered", alpha=0.9, linewidth=2)
ax.set_title(f"{TICKER} Meta-Labeling — Equity Curve", fontsize=14)
ax.set_ylabel("Equity ($)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Try Different Thresholds (Sensitivity)

# %%
thresholds_to_try = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
sensitivity = []

for thresh in thresholds_to_try:
    accept = meta_entry_probs >= thresh
    accepted_set = set(X_meta.index[accept])
    ft = [t for t in trades if t.entry_date in accepted_set]

    if ft:
        td = pd.DataFrame([{"return": t.return_pct, "pnl": t.pnl} for t in ft])
        ret = td["pnl"].sum() / INITIAL_CAPITAL * 100
        win = (td["return"] > 0).mean() * 100
        pf = td[td["return"] > 0]["pnl"].sum() / max(1, abs(td[td["return"] <= 0]["pnl"].sum()))
    else:
        ret, win, pf = 0, 0, 0

    sensitivity.append({
        "threshold": thresh,
        "trades": len(ft),
        "return": round(ret, 2),
        "win_rate": round(win, 1),
        "profit_factor": round(pf, 2),
    })

sens_df = pd.DataFrame(sensitivity)
print(f"\n{'='*70}")
print("  Threshold Sensitivity")
print(f"{'='*70}")
print(f"  {'Thresh':<10} {'Trades':>8} {'Return %':>10} {'Win Rate %':>12} {'Profit Factor':>15}")
print(f"  {'-'*55}")
for _, r in sens_df.iterrows():
    print(f"  {r['threshold']:<10.2f} {r['trades']:>8.0f} {r['return']:>10.2f} {r['win_rate']:>12.1f} {r['profit_factor']:>15.2f}")
print(f"{'='*70}")
