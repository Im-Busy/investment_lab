"""Autonomous training loop with independence filtering, consecutive confirmation,
and cross-group generalization testing.

Wraps existing tools (train_ml_pipeline_v3, run_ml_backtest, tune_model) into an
orchestration loop with guardrails against overfitting and look-ahead bias.

Usage:
    uv run scripts/autonomous_train_loop.py \\
        --tickers "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA" \\
        --max-iterations 20 --confirmations 3 --timeout-hours 8

    uv run scripts/autonomous_train_loop.py \\
        --tickers "SPY,QQQ,XLK,XLF,XLE,XLV,XLI" \\
        --horizon 10 --fast --trail-stop

    uv run scripts/autonomous_train_loop.py \\
        --tickers "JOE,SPY,QQQ,TLT,GLD,IWM" \\
        --phase 4 --model models/pattern_classifier_v3_SPY_20260511.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

project_root = Path(__file__).parent.parent
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autonomous_loop")

# ── Constants ──────────────────────────────────────────────────────────
BESTS_PATH = Path("BESTS.md")
LOOP_LOG_PATH = Path("reports/autonomous_loop/loop_log.jsonl")
LOOP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path("reports/autonomous_loop/checkpoint.json")
PHASE_STATE_PATH = Path("reports/autonomous_loop/phase_state.json")

CONDITION_FEATURES = {
    "trail": {"trail_stop": True, "conviction": False, "vol_gate": None},
    "trail+conv": {"trail_stop": True, "conviction": True, "vol_gate": None},
    "baseline": {"trail_stop": False, "conviction": False, "vol_gate": None},
    "vol_gate": {"trail_stop": True, "conviction": True, "vol_gate": 1.3},
}

SWEEP_HORIZONS = [3, 5, 10, 20]
SWEEP_ENTRY_THRESHOLDS = [0.40, 0.45, 0.50, 0.55]
SWEEP_TRAIL_ATRS = [2.0, 2.5, 3.0, 3.5, 4.0]

# ── 1. Ticker Independence Clustering ───────────────────────────────────


def load_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Load daily close returns for all tickers into a single DataFrame."""
    returns = {}
    for t in tickers:
        path = Path(f"data/raw/{t}_daily.csv")
        if not path.exists():
            logger.warning(f"No data for {t}, downloading...")
            import yfinance as yf

            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
        else:
            df = pd.read_csv(path, parse_dates=True, index_col=0)

        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        if "Close" in df.columns:
            returns[t] = df["Close"].pct_change().dropna()
        else:
            logger.warning(f"No Close column for {t}, skipping")
            return pd.DataFrame()

    if not returns:
        return pd.DataFrame()

    result = pd.DataFrame(returns)
    return result.dropna()


# ── ETF-Component Leakage Detection ────────────────────────────────────

# Known ETFs and their major component stocks (by weight, approximate).
# If both the ETF and any component are in the same ticker list, cross-asset
# features create look-ahead leakage.
KNOWN_ETF_COMPONENTS: dict[str, set[str]] = {
    "SPY": {
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "BRK/B",
        "JPM",
        "JNJ",
        "V",
        "UNH",
        "XOM",
        "WMT",
        "MA",
        "PG",
        "HD",
        "CVX",
        "BAC",
        "ABBV",
        "PEP",
        "KO",
        "MRK",
        "AVGO",
        "COST",
        "ORCL",
        "WFC",
        "CSCO",
        "ACN",
    },
    "QQQ": {
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "META",
        "GOOGL",
        "GOOG",
        "TSLA",
        "AVGO",
        "COST",
        "PEP",
        "ADBE",
        "CSCO",
        "CMCSA",
        "TXN",
        "NFLX",
        "AMD",
        "INTC",
        "QCOM",
        "INTU",
        "HON",
        "AMGN",
        "ISRG",
        "BKNG",
        "SBUX",
        "GILD",
        "ADP",
    },
    "IWM": set(),  # Too many components to enumerate; flag as ETF regardless
    "DIA": {
        "AAPL",
        "MSFT",
        "AMZN",
        "JPM",
        "GS",
        "HD",
        "MCD",
        "CAT",
        "UNH",
        "CRM",
        "V",
        "PG",
        "JNJ",
        "WMT",
        "KO",
        "CVX",
        "BA",
        "HON",
        "IBM",
        "TRV",
    },
    "XLK": {
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "CRM",
        "ORCL",
        "CSCO",
        "ACN",
        "ADBE",
        "QCOM",
        "TXN",
        "INTU",
        "AMD",
        "INTC",
        "IBM",
        "NOW",
        "AMAT",
        "LRCX",
    },
    "XLF": {
        "BRK-B",
        "BRK/B",
        "JPM",
        "V",
        "MA",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "BLK",
        "SCHW",
        "C",
        "CB",
        "MMC",
        "PGR",
        "ICE",
        "CME",
        "COF",
        "USB",
    },
    "XLE": {
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "MPC",
        "SLB",
        "PSX",
        "VLO",
        "OXY",
        "WMB",
        "KMI",
        "HES",
        "FANG",
        "DVN",
        "HAL",
        "BKR",
    },
    "XLV": {
        "UNH",
        "JNJ",
        "ABBV",
        "LLY",
        "MRK",
        "PFE",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "AMGN",
        "ISRG",
        "SYK",
        "GILD",
        "VRTX",
        "CI",
        "ZTS",
        "REGN",
        "CVS",
    },
    "XLI": {
        "CAT",
        "RTX",
        "UNP",
        "HON",
        "UBER",
        "BA",
        "GE",
        "LMT",
        "DE",
        "ADP",
        "ETN",
        "WM",
        "FDX",
        "ITW",
        "CSX",
        "PH",
        "NOC",
        "EMR",
        "CARR",
        "RSG",
    },
}

LEAKAGE_ETFS = {"IWM", "MDY"}  # ETFs with too many components to list


def check_etf_leakage(tickers: list[str]) -> list[str]:
    """Check for ETF-vs-component leakage in the ticker list.

    Returns a list of warning messages (empty if no issues found).
    """
    ticker_set = {t.upper() for t in tickers}
    warnings = []

    for etf, components in KNOWN_ETF_COMPONENTS.items():
        if etf not in ticker_set:
            continue
        overlap = components & ticker_set
        if overlap:
            warnings.append(
                f"ETF LEAKAGE: {etf} contains {sorted(overlap)} as components. "
                f"Cross-asset features will embed look-ahead bias. "
                f"Recommend moving these to separate independence groups."
            )

    for etf in LEAKAGE_ETFS:
        if etf in ticker_set:
            warnings.append(
                f"ETF WARNING: {etf} is a broad-market ETF with many components. "
                f"Cross-asset features with any single stock in the list risk leakage."
            )

    return warnings


def build_exclusion_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """Return list of (ticker_a, ticker_b) pairs that should not share cross-asset features."""
    ticker_set = {t.upper() for t in tickers}
    excluded: list[tuple[str, str]] = []
    for etf, components in KNOWN_ETF_COMPONENTS.items():
        if etf not in ticker_set:
            continue
        for c in components & ticker_set:
            excluded.append((etf, c))
    return excluded


def cluster_tickers(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    max_corr: float = 0.70,
    min_groups: int = 3,
) -> tuple[list[list[str]], pd.DataFrame]:
    """Cluster tickers into independent groups using hierarchical clustering
    on 1 - |correlation| distance.

    Args:
        tickers: Ticker symbols.
        start, end: Date range for correlation computation.
        max_corr: Maximum allowed absolute correlation within a group.
        min_groups: Minimum number of groups (loosens threshold if needed).

    Returns:
        (groups, corr_matrix) where groups is a list of ticker lists.
    """
    ret = load_returns(tickers, start, end)
    if ret.empty:
        logger.error("No return data loaded")
        return [[t] for t in tickers], pd.DataFrame()

    corr = ret.corr()
    distance = 1 - corr.abs()
    distance.values[np.diag_indices_from(distance)] = 0
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    threshold = 1 - max_corr
    clusters = fcluster(linkage_matrix, threshold, criterion="distance")
    n_clusters = len(set(clusters))

    if n_clusters < min_groups:
        threshold = linkage_matrix[:, 2][-(min_groups - 1)]
        clusters = fcluster(linkage_matrix, threshold, criterion="distance")

    groups: list[list[str]] = []
    for cid in sorted(set(clusters)):
        group = [tickers[i] for i in range(len(tickers)) if clusters[i] == cid]
        groups.append(group)

    logger.info(
        f"Clustered {len(tickers)} tickers into {len(groups)} groups at max_corr={max_corr}"
    )
    for i, g in enumerate(groups):
        logger.info(f"  Group {i + 1}: {g}")

    return groups, corr


def select_leaders(
    groups: list[list[str]],
    model_path: str,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    horizon: int = 5,
) -> list[str]:
    """Select the best ticker from each group based on walk-forward IC.

    Trains a quick CatBoost model on the first group's leader, then computes
    walk-forward IC for every ticker in every group. Returns the ticker with
    the highest mean rank IC per group.

    If no model exists yet, falls back to highest average daily volume.
    """
    from src.ml.pattern_classifier import PatternClassifier
    from src.ml.walk_forward import walk_forward_per_ticker

    all_tickers = [t for g in groups for t in g]

    # Load or create a model for IC computation
    m = PatternClassifier()
    model_loaded = False
    if model_path and Path(model_path).exists():
        try:
            m.load(model_path)
            model_loaded = True
        except Exception as e:
            logger.warning(f"Could not load model {model_path}: {e}")

    if not model_loaded:
        # Quick train on the first ticker for IC evaluation
        first_ticker = all_tickers[0]
        logger.info(f"Training quick IC model on {first_ticker} for leader selection...")
        try:
            from src.ml.features import FeatureEngineer
            from sklearn.model_selection import TimeSeriesSplit
            import catboost as cb

            eng = FeatureEngineer()
            df = load_returns([first_ticker], start, end)
            df = eng.add_features(df[[first_ticker]] if isinstance(df, pd.DataFrame) else df)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(c).strip() for c in df.columns.values]

            # Quick TSCV
            tscv = TimeSeriesSplit(n_splits=3)
            X = df.dropna()
            y = (X.iloc[:, 0].shift(-horizon) > X.iloc[:, 0]).astype(int)
            X = X.iloc[: len(y), :]
            y = y.dropna()
            X = X.loc[y.index]

            model = cb.CatBoostClassifier(
                iterations=100,
                depth=5,
                verbose=0,
                random_seed=42,
            )
            splits = list(tscv.split(X))
            for train_idx, test_idx in splits[:1]:
                model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=0)
            m.model = model
        except Exception as e:
            logger.warning(f"Quick IC model training failed: {e}, falling back to volume proxy")
            return _select_by_volume(groups, start, end)

    try:
        wf = walk_forward_per_ticker(
            m,
            all_tickers,
            initial_train_days=3 * 252,
            step_days=6 * 21,
            horizon=horizon,
        )
    except Exception as e:
        logger.warning(f"Walk-forward IC failed: {e}, falling back to volume proxy")
        return _select_by_volume(groups, start, end)

    leaders = []
    for group in groups:
        if len(group) == 1:
            leaders.append(group[0])
            continue

        best_ticker = max(group, key=lambda t: wf[t].mean_rank_ic if t in wf else -999)
        best_ic = wf[best_ticker].mean_rank_ic if best_ticker in wf else float("nan")
        logger.info(f"  Group leader: {best_ticker} (IC={best_ic:.4f}) from {group}")
        leaders.append(best_ticker)

    return leaders


def _select_by_volume(groups: list[list[str]], start: str, end: str) -> list[str]:
    """Fallback: select leader by average daily volume."""
    leaders = []
    for group in groups:
        if len(group) == 1:
            leaders.append(group[0])
            continue
        best_vol = 0
        best_ticker = group[0]
        for t in group:
            path = Path(f"data/raw/{t}_daily.csv")
            if path.exists():
                df = pd.read_csv(path, index_col=0)
                if "Volume" in df.columns:
                    avg_vol = df["Volume"].mean()
                    if avg_vol > best_vol:
                        best_vol = avg_vol
                        best_ticker = t
        logger.info(f"  Group leader (by volume): {best_ticker} from {group}")
        leaders.append(best_ticker)
    return leaders


# ── 2. BESTS.md Parser ──────────────────────────────────────────────────


@dataclass
class BestEntry:
    rank: int
    config: str
    return_pct: float
    sharpe: float
    trades: int
    win_pct: float
    profit_factor: float
    max_dd: float
    exposure: float
    ann_return: float = 0.0
    condition: str = ""

    @property
    def score(self) -> float:
        return self.sharpe


def parse_bests() -> dict[str, dict[str, BestEntry]]:
    """Parse BESTS.md into a structured dict of condition -> config -> BestEntry.

    Returns empty dict if BESTS.md doesn't exist.
    """
    if not BESTS_PATH.exists():
        return {}

    text = BESTS_PATH.read_text(encoding="utf-8")
    conditions: dict[str, dict[str, BestEntry]] = {}

    current_condition = "overall_sharpe"
    in_table = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("## Overall Best by Sharpe"):
            current_condition = "overall_sharpe"
            in_table = False
            continue
        if stripped.startswith("## Overall Best by Return"):
            current_condition = "overall_return"
            in_table = False
            continue
        if stripped.startswith("### With Trail Stop + Conviction"):
            current_condition = "trail+conv"
            in_table = False
            continue
        if stripped.startswith("### With Trail Stop"):
            current_condition = "trail"
            in_table = False
            continue
        if stripped.startswith("### Baseline"):
            current_condition = "baseline"
            in_table = False
            continue
        if stripped.startswith("### With Volatility Gate"):
            current_condition = "vol_gate"
            in_table = False
            continue
        if stripped.startswith("---") or stripped.startswith("## Model Probability"):
            in_table = False
            continue

        if stripped.startswith("| Rank |"):
            in_table = True
            continue
        if stripped.startswith("|------"):
            continue

        if in_table and stripped.startswith("|"):
            parts = [p.strip() for p in stripped.split("|")[1:-1]]
            try:
                if len(parts) >= 8 and parts[0].replace("*", "").strip().isdigit():
                    entry = BestEntry(
                        rank=int(parts[0].replace("*", "").strip()),
                        config=parts[1].replace("`", "").strip(),
                        return_pct=float(parts[2].replace("%", "").replace("**", "").strip()),
                        sharpe=float(parts[3].replace("**", "").strip()),
                        trades=int(parts[4].strip()),
                        win_pct=float(parts[5].replace("%", "").strip()),
                        profit_factor=float(parts[6].replace("**", "").strip()),
                        max_dd=float(parts[7].replace("%", "").strip()),
                        exposure=float(parts[8].replace("%", "").strip()) if len(parts) > 8 else 0,
                        ann_return=float(parts[9].replace("%", "").strip())
                        if len(parts) > 9
                        else 0,
                        condition=current_condition,
                    )
                    conditions.setdefault(current_condition, {})[entry.config] = entry
            except (ValueError, IndexError):
                continue

    return conditions


def get_best_sharpe(condition: str = "overall_sharpe") -> BestEntry | None:
    """Get the best-by-Sharpe entry for a condition."""
    bests = parse_bests()
    entries = bests.get(condition, {})
    if not entries:
        return None
    return max(entries.values(), key=lambda e: e.sharpe)


def is_new_best(
    result: dict[str, Any],
    condition: str = "trail",
    min_trades: int = 20,
) -> tuple[bool, BestEntry | None]:
    """Check if a backtest result is a new best for its condition.

    Returns (is_new_best, previous_best).
    """
    prev = get_best_sharpe(condition)
    if result.get("num_trades", 0) < min_trades:
        return False, prev
    if prev is None:
        return True, None
    new_sharpe = result.get("sharpe", 0)
    return new_sharpe > prev.sharpe, prev


# ── 3. Backtest Runner Wrappers ─────────────────────────────────────────


def run_backtest(
    model_path: str,
    tickers: list[str],
    start: str | None = None,
    cash: float = 100_000,
    entry_threshold: float = 0.50,
    trail_stop: bool = False,
    trail_atr: float = 3.0,
    conviction: bool = False,
    vol_gate: float | None = None,
    confirm: int = 1,
    multi_tp: bool = True,
) -> list[dict[str, Any]]:
    """Run backtest on multiple tickers and return aggregated results."""
    from run_ml_backtest import run_single

    results = []
    for ticker in tickers:
        try:
            r = run_single(
                symbol=ticker,
                model_path=model_path,
                cash=cash,
                start=start,
                entry_threshold=entry_threshold,
                vol_gate=vol_gate,
                confirm=confirm,
                trail_stop=trail_stop,
                trail_atr=trail_atr,
                conviction=conviction,
                multi_tp=multi_tp,
            )
            results.append(r)
        except Exception as e:
            logger.warning(f"Backtest failed for {ticker}: {e}")

    return results


def aggregate_backtest_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-ticker backtest results into a single summary."""
    if not results:
        return {}

    df = pd.DataFrame(results)
    total_equity = df["equity_final"].sum()
    total_trades = df["num_trades"].sum()

    return {
        "n_tickers": len(results),
        "total_trades": int(total_trades),
        "mean_sharpe": round(float(df["sharpe"].mean()), 3),
        "mean_return_pct": round(float(df["return_pct"].mean()), 2),
        "mean_win_rate": round(float(df["win_rate"].mean()), 2),
        "mean_profit_factor": round(float(df["profit_factor"].mean()), 2),
        "mean_max_dd": round(float(df["max_drawdown"].mean()), 2),
        "mean_exposure": round(float(df["exposure"].mean()), 2),
        "beat_bh": int((df["return_pct"] > df["buy_hold_return"]).sum()),
        "best_sharpe": round(float(df["sharpe"].max()), 3),
        "best_return": round(float(df["return_pct"].max()), 2),
        "tickers": [r["symbol"] for r in results],
        "per_ticker": results,
    }


# ── 4. Configuration Generation ─────────────────────────────────────────


def config_to_str(config: dict[str, Any]) -> str:
    """Convert a config dict to a BESTS.md config string."""
    parts = []
    parts.append(f"et={config.get('entry_threshold', 0.50)}")
    if config.get("trail_stop"):
        parts.append("trail")
    if config.get("conviction"):
        parts.append("conv")
    if config.get("vol_gate"):
        parts.append(f"vg={config['vol_gate']}")
    return " ".join(parts)


def generate_config_variants(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate config variants from a base configuration.

    Produces: baseline, trail, trail+conv, vol-gate variants.
    """
    variants = []

    baseline = {**base, "trail_stop": False, "conviction": False, "vol_gate": None}
    variants.append(baseline)

    trail = {**base, "trail_stop": True, "conviction": False, "vol_gate": None}
    variants.append(trail)

    trail_conv = {**base, "trail_stop": True, "conviction": True, "vol_gate": None}
    variants.append(trail_conv)

    for vg in [1.3, 1.5, 1.8]:
        variants.append({**base, "trail_stop": True, "conviction": True, "vol_gate": vg})

    return variants


# ── 5. Consecutive Confirmation Logic ───────────────────────────────────


@dataclass
class LoopState:
    ticker_groups: list[list[str]] = field(default_factory=list)
    current_model_path: str = ""
    current_config: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    consecutive_improvements: int = 0
    best_sharpe: float = 0.0
    best_config: dict[str, Any] = field(default_factory=dict)
    best_model_path: str = ""
    locked: bool = False
    start_time: float = 0.0
    timeout_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def decide_next_action(
    state: LoopState,
    condition: str,
    aggregate: dict[str, Any],
    confirmations_required: int,
) -> str:
    """Decide whether to continue, adjust, or stop.

    Returns: 'continue', 'adjust', 'locked', or 'timeout'.
    """
    if state.timeout_seconds > 0 and (time.time() - state.start_time) > state.timeout_seconds:
        return "timeout"

    current_sharpe = aggregate.get("mean_sharpe", 0)

    if current_sharpe > state.best_sharpe:
        state.consecutive_improvements += 1
        state.best_sharpe = current_sharpe
        state.best_config = dict(state.current_config)
        state.best_model_path = state.current_model_path
        logger.info(
            f"  IMPROVEMENT (#{state.consecutive_improvements}/{confirmations_required}): "
            f"Sharpe {state.best_sharpe:.3f}"
        )
    else:
        state.consecutive_improvements = 0
        logger.info(f"  No improvement (Sharpe {current_sharpe:.3f} <= {state.best_sharpe:.3f})")

    if state.consecutive_improvements >= confirmations_required:
        state.locked = True
        return "locked"

    if state.iteration >= 3 and state.consecutive_improvements == 0:
        return "adjust"

    return "continue"


def adjust_params(config: dict[str, Any], iteration: int) -> dict[str, Any]:
    """Suggest parameter adjustments when stuck.

    Alternates between lowering entry threshold (more trades), adjusting
    trail ATR, and trying different horizons.
    """
    new = dict(config)
    phase = iteration % 4

    if phase == 0:
        new["entry_threshold"] = max(0.35, new.get("entry_threshold", 0.50) - 0.05)
        logger.info(f"  Adjust: lower entry threshold → {new['entry_threshold']}")
    elif phase == 1:
        new["trail_atr"] = new.get("trail_atr", 3.0) + 0.5
        logger.info(f"  Adjust: wider trail ATR → {new['trail_atr']}")
    elif phase == 2:
        new["entry_threshold"] = min(0.60, new.get("entry_threshold", 0.50) + 0.05)
        logger.info(f"  Adjust: raise entry threshold → {new['entry_threshold']}")
    else:
        new["trail_atr"] = max(1.5, new.get("trail_atr", 3.0) - 0.5)
        logger.info(f"  Adjust: tighter trail ATR → {new['trail_atr']}")

    return new


# ── 6. Main Loop ────────────────────────────────────────────────────────


def run_phase_1_ticker_selection(
    tickers: list[str],
    start: str,
    end: str,
    max_corr: float,
) -> list[list[str]]:
    """PHASE 1: Download data, cluster tickers into independent groups."""
    logger.info("=" * 70)
    logger.info("PHASE 1: TICKER SELECTION + INDEPENDENCE FILTER")
    logger.info("=" * 70)

    all_tickers = list(dict.fromkeys(tickers))
    logger.info(f"Input tickers: {len(all_tickers)}")

    # Check for ETF-vs-component leakage
    leakage_warnings = check_etf_leakage(all_tickers)
    for w in leakage_warnings:
        logger.warning(w)

    exclusion_pairs = build_exclusion_pairs(all_tickers)
    if exclusion_pairs:
        logger.info(f"Cross-asset exclusions: {exclusion_pairs}")
        ex_path = Path("reports/autonomous_loop/exclusion_pairs.json")
        ex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ex_path, "w") as f:
            json.dump([[a, b] for a, b in exclusion_pairs], f)

    for t in all_tickers:
        path = Path(f"data/raw/{t}_daily.csv")
        if not path.exists():
            logger.info(f"  Downloading {t}...")
            import yfinance as yf

            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                logger.warning(f"  No data for {t}, skipping")
                all_tickers.remove(t)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)

    if len(all_tickers) < 2:
        logger.info("Only 1 ticker, skipping clustering")
        return [all_tickers]

    groups, corr = cluster_tickers(all_tickers, start, end, max_corr)
    return groups


def run_phase_2_tuning(
    groups: list[list[str]],
    start: str,
    end: str,
    horizon: int,
    fast: bool,
) -> dict[str, Any]:
    """PHASE 2: Per-group leader tuning via GWO."""
    logger.info("=" * 70)
    logger.info("PHASE 2: PER-GROUP LEADER TUNING")
    logger.info("=" * 70)

    if fast:
        logger.info("--fast mode: skipping tuning, using defaults")
        return {}

    leaders = [g[0] for g in groups]
    tuning_results = {}

    for leader in leaders:
        logger.info(f"Tuning leader: {leader}")
        try:
            from tune_model import run_tuning

            result = run_tuning(
                symbol=leader,
                start=start,
                end=end,
                target="pattern_classifier",
                n_wolves=10,
                max_iter=20,
                horizon=horizon,
                output_dir="reports/ml_tuning",
            )
            tuning_results[leader] = result
        except Exception as e:
            logger.warning(f"Tuning failed for {leader}: {e}")

    return tuning_results


def run_phase_3_cross_group_test(
    groups: list[list[str]],
    horizon: int,
    start: str,
    end: str,
    fast: bool,
) -> dict[str, Any]:
    """PHASE 3: Cross-group generalization test.

    Train on groups A,B,C → test on groups D,E (completely non-overlapping).
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: CROSS-GROUP GENERALIZATION TEST")
    logger.info("=" * 70)

    if len(groups) < 2:
        logger.info("Only 1 group, skipping cross-group test")
        return {}

    n_train = max(2, len(groups) * 3 // 5)
    train_groups = groups[:n_train]
    test_groups = groups[n_train:]

    train_tickers = [t for g in train_groups for t in g]
    test_tickers_set = {t for g in test_groups for t in g}

    logger.info(f"Train groups: {[g[0] for g in train_groups]} ({len(train_tickers)} tickers)")
    logger.info(f"Test groups:  {[g[0] for g in test_groups]} ({len(test_tickers_set)} tickers)")

    from train_ml_pipeline_v3 import run_pipeline

    result = run_pipeline(
        tickers=train_tickers,
        start=start,
        end=end,
        horizon=horizon,
        fast=fast,
        use_cross_asset=len(train_tickers) > 1,
        skip_tuning=fast,
        run_walk_forward=False,
        label_type="triple_barrier",
    )

    model_path = result.get("model_path", "")
    if not model_path:
        model_files = sorted(
            Path("models").glob("pattern_classifier_v3_*.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        if model_files:
            model_path = str(model_files[-1])
            logger.info(f"Guessed model path: {model_path}")

    test_tickers = list(test_tickers_set)
    if model_path and test_tickers:
        logger.info("Testing cross-group generalization...")
        results = run_backtest(
            model_path=model_path,
            tickers=test_tickers,
            trail_stop=True,
            entry_threshold=0.50,
        )
        agg = aggregate_backtest_results(results)
        logger.info(f"Cross-group OOS: mean Sharpe={agg.get('mean_sharpe', 'N/A')}")
        return {"model_path": model_path, "oos_results": agg, "test_tickers": test_tickers}

    return {"model_path": model_path}


def run_phase_4_refinement_loop(
    groups: list[list[str]],
    horizon: int,
    start: str,
    end: str,
    fast: bool,
    max_iterations: int,
    confirmations_required: int,
    timeout_hours: float,
    trail_stop: bool,
    conviction: bool,
    entry_threshold: float,
    trail_atr: float,
    initial_model_path: str = "",
    resume: bool = False,
    label_type: str = "triple_barrier",
) -> LoopState:
    """PHASE 4: Autonomous refinement loop with consecutive confirmation."""
    logger.info("=" * 70)
    logger.info("PHASE 4: AUTONOMOUS REFINEMENT LOOP")
    logger.info("=" * 70)
    logger.info(f"Max iterations: {max_iterations}, Confirmations: {confirmations_required}")
    logger.info(f"Timeout: {timeout_hours}h | Fast: {fast}")

    start_iteration = 1
    if resume:
        ckpt = load_checkpoint()
        if ckpt:
            logger.info(f"Resuming from iteration {ckpt['iteration']}/{max_iterations}")
            state = LoopState(
                ticker_groups=ckpt["ticker_groups"],
                start_time=time.time() - ckpt.get("elapsed_seconds", 0),
                timeout_seconds=timeout_hours * 3600 if timeout_hours > 0 else 0,
                current_config=ckpt["current_config"],
                current_model_path=ckpt.get("current_model_path", ""),
                iteration=ckpt["iteration"],
                consecutive_improvements=ckpt.get("consecutive_improvements", 0),
                best_sharpe=ckpt.get("best_sharpe", 0),
                best_config=ckpt.get("best_config", {}),
                best_model_path=ckpt.get("best_model_path", ""),
                locked=ckpt.get("locked", False),
                history=ckpt.get("history", []),
            )
            if state.locked:
                logger.info("Resumed state already locked, returning immediately")
                return state
            start_iteration = state.iteration + 1
            groups = state.ticker_groups
        else:
            logger.info("No checkpoint found, starting fresh")
    if not resume or start_iteration == 1:
        state = LoopState(
            ticker_groups=groups,
            start_time=time.time(),
            timeout_seconds=timeout_hours * 3600 if timeout_hours > 0 else 0,
            current_config={
                "entry_threshold": entry_threshold,
                "trail_stop": trail_stop,
                "conviction": conviction,
                "trail_atr": trail_atr,
                "vol_gate": None,
            },
            current_model_path=initial_model_path,
        )

    all_tickers = [t for g in state.ticker_groups for t in g]

    for iteration in range(start_iteration, max_iterations + 1):
        state.iteration = iteration

        elapsed = time.time() - state.start_time
        if state.timeout_seconds > 0 and elapsed > state.timeout_seconds:
            logger.info(f"Timeout reached ({elapsed / 3600:.1f}h), stopping")
            break

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Iteration {iteration}/{max_iterations} | Elapsed: {elapsed / 3600:.1f}h")
        logger.info(f"Config: {config_to_str(state.current_config)}")
        logger.info(f"{'─' * 60}")

        # Step A: Train model
        logger.info("Training model...")
        from train_ml_pipeline_v3 import run_pipeline

        pipeline_result = run_pipeline(
            tickers=all_tickers,
            start=start,
            end=end,
            horizon=horizon,
            fast=fast,
            use_cross_asset=len(all_tickers) > 1,
            skip_tuning=fast,
            run_walk_forward=False,
            label_type=label_type,
        )

        model_path = pipeline_result.get("model_path", "")
        if not model_path:
            model_files = sorted(
                Path("models").glob("pattern_classifier_v3_*.pkl"),
                key=lambda p: p.stat().st_mtime,
            )
            if model_files:
                model_path = str(model_files[-1])
        state.current_model_path = model_path
        logger.info(f"Model: {model_path}")

        # Step B: Backtest
        logger.info("Backtesting...")
        backtest_results = run_backtest(
            model_path=model_path,
            tickers=all_tickers,
            cash=100_000,
            entry_threshold=state.current_config["entry_threshold"],
            trail_stop=state.current_config["trail_stop"],
            trail_atr=state.current_config["trail_atr"],
            conviction=state.current_config["conviction"],
            vol_gate=state.current_config.get("vol_gate"),
        )
        aggregate = aggregate_backtest_results(backtest_results)

        # Step C: Print summary
        if aggregate:
            logger.info(
                f"  Sharpe: {aggregate['mean_sharpe']:.3f} | "
                f"Return: {aggregate['mean_return_pct']:.1f}% | "
                f"Win: {aggregate['mean_win_rate']:.1f}% | "
                f"Trades: {aggregate['total_trades']} | "
                f"PF: {aggregate['mean_profit_factor']:.2f} | "
                f"DD: {aggregate['mean_max_dd']:.1f}% | "
                f"Beat B&H: {aggregate['beat_bh']}/{aggregate['n_tickers']}"
            )

        # Step D: Check if new best
        condition = "trail" if state.current_config["trail_stop"] else "baseline"
        is_best, prev_best = is_new_best(
            {
                "sharpe": aggregate.get("mean_sharpe", 0),
                "num_trades": aggregate.get("total_trades", 0),
            },
            condition=condition,
        )
        if is_best and prev_best:
            logger.info(
                f"  NEW BEST: Sharpe {aggregate['mean_sharpe']:.3f} > {prev_best.sharpe:.3f}"
            )

        # Step E: Decide next action
        action = decide_next_action(state, condition, aggregate, confirmations_required)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "config": config_to_str(state.current_config),
            "model_path": model_path,
            "aggregate": aggregate,
            "action": action,
            "best_sharpe": state.best_sharpe,
            "consecutive": state.consecutive_improvements,
        }
        state.history.append(log_entry)
        _write_loop_log(log_entry)
        save_checkpoint(state)
        _log_to_mlflow(
            run_id=state.current_model_path.replace("\\", "/").split("/")[-1].replace(".pkl", ""),
            iteration=iteration,
            config=state.current_config,
            aggregate=aggregate,
            model_path=model_path,
            tickers=all_tickers,
        )

        if action == "locked":
            logger.info("LOCKED: 3+ consecutive improvements confirmed!")
            logger.info(f"Best config: {config_to_str(state.best_config)}")
            logger.info(f"Best Sharpe: {state.best_sharpe:.3f}")
            logger.info(f"Best model: {state.best_model_path}")
            _update_bests_md(state, all_tickers, start, end)
            _finish_mlflow_run(
                state.current_model_path.replace("\\", "/").split("/")[-1].replace(".pkl", ""),
                {"status": "locked", "best_sharpe": state.best_sharpe, "iterations": iteration},
            )
            if CHECKPOINT_PATH.exists():
                CHECKPOINT_PATH.unlink()
            break
        elif action == "timeout":
            logger.info("Timeout reached, stopping with current best")
            _finish_mlflow_run(
                state.current_model_path.replace("\\", "/").split("/")[-1].replace(".pkl", ""),
                {"status": "timeout", "best_sharpe": state.best_sharpe, "iterations": iteration},
            )
            break
        elif action == "adjust":
            logger.info("Stuck — adjusting parameters")
            state.current_config = adjust_params(state.current_config, iteration)
        else:
            logger.info("Continuing with same config...")

    return state


def run_phase_5_param_sweep(
    groups: list[list[str]],
    horizon: int,
    start: str,
    end: str,
    fast: bool,
    model_path: str,
    use_optuna: bool = True,
    n_trials: int = 30,
    pareto: bool = False,
) -> list[dict[str, Any]]:
    """PHASE 5: Parameter space crawl with Bayesian optimization.

    Uses Optuna TPE to efficiently search horizon × entry_threshold × trail_atr
    space. Falls back to grid sweep if Optuna is unavailable or disabled.
    When --pareto is set, uses multi-objective optimization (Sharpe + MaxDD + WinRate).
    """
    logger.info("=" * 70)
    logger.info("PHASE 5: PARAMETER SPACE CRAWL")
    logger.info("=" * 70)

    all_tickers = [t for g in groups for t in g]
    sweep_results: list[dict[str, Any]] = []

    if use_optuna:
        try:
            sweep_results = _optuna_sweep(
                all_tickers,
                model_path,
                start,
                n_trials=n_trials,
                pareto=pareto,
            )
        except Exception as e:
            logger.warning(f"Optuna sweep failed ({e}), falling back to grid sweep")
            use_optuna = False

    if not use_optuna:
        sweep_results = _grid_sweep(
            all_tickers,
            horizon,
            model_path,
            start,
        )

    return sweep_results


def _optuna_sweep(
    tickers: list[str],
    model_path: str,
    start: str | None,
    n_trials: int = 30,
    pareto: bool = False,
) -> list[dict[str, Any]]:
    """Run Optuna TPE optimization over entry_threshold × trail_atr × conviction.

    When pareto=True, uses multi-objective optimization minimizing three objectives
    simultaneously: maximize Sharpe, minimize |MaxDD|, maximize WinRate.
    """
    import optuna

    logger.info(
        f"Running Optuna {'Pareto multi-objective' if pareto else 'Bayesian'} sweep: {n_trials} trials"
    )

    if pareto:

        def objective(trial: optuna.Trial) -> tuple[float, float, float]:
            et = trial.suggest_float("entry_threshold", 0.35, 0.60, step=0.05)
            trail_atr = trial.suggest_float("trail_atr", 1.5, 5.0, step=0.5)
            use_conviction = trial.suggest_categorical("conviction", [False, True])
            vol_gate = trial.suggest_categorical("vol_gate", [None, 1.3, 1.5, 1.8])

            try:
                results = run_backtest(
                    model_path=model_path,
                    tickers=tickers,
                    entry_threshold=et,
                    trail_stop=True,
                    trail_atr=trail_atr,
                    conviction=use_conviction,
                    vol_gate=vol_gate,
                )
                agg = aggregate_backtest_results(results)
                sharpe = agg.get("mean_sharpe", 0)
                trades = agg.get("total_trades", 0)

                if trades < 20:
                    return (-999.0, 999.0, 0.0)

                trial.set_user_attr("return_pct", agg.get("mean_return_pct", 0))
                trial.set_user_attr("trades", trades)
                trial.set_user_attr("win_rate", agg.get("mean_win_rate", 0))
                trial.set_user_attr("profit_factor", agg.get("mean_profit_factor", 0))
                trial.set_user_attr("max_dd", agg.get("mean_max_dd", 0))

                logger.info(
                    f"  Trial {trial.number}: et={et} atr={trail_atr} "
                    f"conv={use_conviction} vg={vol_gate} → "
                    f"Sharpe={sharpe:.3f} DD={abs(agg.get('mean_max_dd', 0)):.1f}% WR={agg.get('mean_win_rate', 0):.1f}%"
                )
                return (
                    float(sharpe),
                    float(abs(agg.get("mean_max_dd", 0))),
                    float(agg.get("mean_win_rate", 0) / 100.0),
                )
            except Exception as e:
                logger.warning(f"  Trial {trial.number} failed: {e}")
                return (-999.0, 999.0, 0.0)

        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        pareto_front = []
        for trial in study.best_trials:
            if trial.values[0] > -999:
                pareto_front.append(
                    {
                        "params": trial.params,
                        "sharpe": trial.values[0],
                        "max_dd": -trial.values[1],
                        "win_rate": trial.values[2] * 100,
                    }
                )

        logger.info(f"Pareto frontier: {len(pareto_front)} non-dominated solutions")
        for i, sol in enumerate(sorted(pareto_front, key=lambda x: x["sharpe"], reverse=True)[:5]):
            logger.info(
                f"  #{i + 1}: {sol['params']} → "
                f"Sharpe={sol['sharpe']:.3f}, DD={sol['max_dd']:.1f}%, WR={sol['win_rate']:.1f}%"
            )

        results = []
        for trial in study.trials:
            if trial.values[0] > -999:
                results.append(
                    {
                        "entry_threshold": trial.params.get("entry_threshold"),
                        "trail_atr": trial.params.get("trail_atr"),
                        "conviction": trial.params.get("conviction"),
                        "vol_gate": trial.params.get("vol_gate"),
                        "sharpe": trial.values[0],
                        "max_dd": -trial.values[1],
                        "win_rate": trial.values[2] * 100,
                        "return_pct": trial.user_attrs.get("return_pct"),
                        "trades": trial.user_attrs.get("trades"),
                        "profit_factor": trial.user_attrs.get("profit_factor"),
                    }
                )

        sweep_path = Path("reports/autonomous_loop/pareto_sweep.json")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(
                {
                    "pareto_front": pareto_front,
                    "trials": results,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Saved Pareto results to {sweep_path}")

        return results
    else:

        def objective(trial: optuna.Trial) -> float:
            et = trial.suggest_float("entry_threshold", 0.35, 0.60, step=0.05)
            trail_atr = trial.suggest_float("trail_atr", 1.5, 5.0, step=0.5)
            use_conviction = trial.suggest_categorical("conviction", [False, True])
            vol_gate = trial.suggest_categorical("vol_gate", [None, 1.3, 1.5, 1.8])

            try:
                results = run_backtest(
                    model_path=model_path,
                    tickers=tickers,
                    entry_threshold=et,
                    trail_stop=True,
                    trail_atr=trail_atr,
                    conviction=use_conviction,
                    vol_gate=vol_gate,
                )
                agg = aggregate_backtest_results(results)
                sharpe = agg.get("mean_sharpe", 0)
                trades = agg.get("total_trades", 0)

                if trades < 20:
                    return -999.0

                trial.set_user_attr("return_pct", agg.get("mean_return_pct", 0))
                trial.set_user_attr("trades", trades)
                trial.set_user_attr("win_rate", agg.get("mean_win_rate", 0))
                trial.set_user_attr("profit_factor", agg.get("mean_profit_factor", 0))
                trial.set_user_attr("max_dd", agg.get("mean_max_dd", 0))

                logger.info(
                    f"  Trial {trial.number}: et={et} atr={trail_atr} "
                    f"conv={use_conviction} vg={vol_gate} → "
                    f"Sharpe={sharpe:.3f} Trades={trades}"
                )
                return float(sharpe)
            except Exception as e:
                logger.warning(f"  Trial {trial.number} failed: {e}")
                return -999.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=3, interval_steps=1
            ),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        results = []
        for trial in study.trials:
            if trial.value and trial.value > -999:
                results.append(
                    {
                        "entry_threshold": trial.params.get("entry_threshold"),
                        "trail_atr": trial.params.get("trail_atr"),
                        "conviction": trial.params.get("conviction"),
                        "vol_gate": trial.params.get("vol_gate"),
                        "sharpe": trial.value,
                        "return_pct": trial.user_attrs.get("return_pct"),
                        "trades": trial.user_attrs.get("trades"),
                        "win_rate": trial.user_attrs.get("win_rate"),
                        "profit_factor": trial.user_attrs.get("profit_factor"),
                        "max_dd": trial.user_attrs.get("max_dd"),
                    }
                )

        best = study.best_params
        best_value = study.best_value
        logger.info(f"Optuna best: {best} → Sharpe={best_value:.3f}")

        sweep_path = Path("reports/autonomous_loop/optuna_sweep.json")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(
                {
                    "best_params": best,
                    "best_value": best_value,
                    "trials": results,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Saved Optuna results to {sweep_path}")

        return results


def _grid_sweep(
    tickers: list[str],
    horizon: int,
    model_path: str,
    start: str | None,
) -> list[dict[str, Any]]:
    """Fallback brute-force grid sweep."""
    logger.info("Running grid sweep as fallback")
    sweep_results = []

    for et in SWEEP_ENTRY_THRESHOLDS:
        for trail_atr in SWEEP_TRAIL_ATRS:
            try:
                results = run_backtest(
                    model_path=model_path,
                    tickers=tickers,
                    entry_threshold=et,
                    trail_stop=True,
                    trail_atr=trail_atr,
                    conviction=False,
                )
                agg = aggregate_backtest_results(results)
                sweep_results.append(
                    {
                        "entry_threshold": et,
                        "trail_atr": trail_atr,
                        "sharpe": agg.get("mean_sharpe", 0),
                        "return_pct": agg.get("mean_return_pct", 0),
                        "trades": agg.get("total_trades", 0),
                    }
                )
                logger.info(f"  et={et} atr={trail_atr} → Sharpe={agg.get('mean_sharpe', 0):.3f}")
            except Exception as e:
                logger.warning(f"  Grid sweep failed: {e}")

    if sweep_results:
        sweep_path = Path("reports/autonomous_loop/param_sweep.json")
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        logger.info(f"Saved grid sweep results to {sweep_path}")

        best = max(sweep_results, key=lambda x: x["sharpe"])
        logger.info(f"Best grid config: {best}")

    return sweep_results


# ── 7. BESTS.md Update ─────────────────────────────────────────────────


def _update_bests_md(state: LoopState, tickers: list[str], start: str, end: str) -> None:
    """Update BESTS.md with the locked best result."""
    if not state.best_config or not state.best_model_path:
        return

    config_str = config_to_str(state.best_config)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    new_section = f"""

## Autonomous Loop Best (Locked)

| Rank | Config | Return | Sharpe | Trades | Win% | PF | MaxDD | Exp% |
|------|--------|--------|--------|--------|------|-----|-------|------|
| **1** | `{config_str}` | {state.best_sharpe:.1f}% | {state.best_sharpe:.3f} | — | — | — | — | — |

- **Ticker group**: {tickers}
- **Model**: {state.best_model_path}
- **Iterations**: {state.iteration}
- **Locked at**: {now}

"""

    content = (
        BESTS_PATH.read_text(encoding="utf-8")
        if BESTS_PATH.exists()
        else "# Backtest Leaderboard\n"
    )
    if "## Autonomous Loop Best" not in content:
        content += new_section
    else:
        start_marker = "## Autonomous Loop Best"
        footer_marker = "*Last updated:"
        parts = content.split(start_marker, 1)
        if footer_marker in parts[1]:
            after = parts[1].split(footer_marker, 1)
            content = parts[0] + new_section + footer_marker + after[1]
        else:
            content = parts[0] + new_section

    content = content.rstrip() + f"\n\n*Last updated: {now}*\n"
    BESTS_PATH.write_text(content, encoding="utf-8")
    logger.info(f"Updated {BESTS_PATH}")


def _write_loop_log(entry: dict[str, Any]) -> None:
    try:
        with open(LOOP_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def _log_to_mlflow(
    run_id: str,
    iteration: int,
    config: dict[str, Any],
    aggregate: dict[str, Any],
    model_path: str = "",
    tickers: list[str] | None = None,
) -> None:
    """Log iteration results to MLflow. Non-blocking — failures are silent."""
    try:
        from src.ml.mlflow_logger import MlflowExperimentLogger

        ml = MlflowExperimentLogger(run_id=run_id)
        ml._start_run()
        if ml._run:
            ml._mlflow.set_tag("tickers", ",".join(tickers or []))
            ml._mlflow.set_tag("phase", "autonomous_refinement")
            ml._mlflow.log_param("iteration", iteration)
            ml._mlflow.log_params({f"cfg.{k}": v for k, v in config.items()})
            ml._mlflow.log_metric("sharpe", aggregate.get("mean_sharpe", 0), step=iteration)
            ml._mlflow.log_metric("return_pct", aggregate.get("mean_return_pct", 0), step=iteration)
            ml._mlflow.log_metric("win_rate", aggregate.get("mean_win_rate", 0), step=iteration)
            ml._mlflow.log_metric(
                "profit_factor", aggregate.get("mean_profit_factor", 0), step=iteration
            )
            ml._mlflow.log_metric("max_dd", aggregate.get("mean_max_dd", 0), step=iteration)
            ml._mlflow.log_metric("trades", aggregate.get("total_trades", 0), step=iteration)
            if model_path and Path(model_path).exists():
                ml._mlflow.log_artifact(str(model_path))
    except Exception:
        pass


def _finish_mlflow_run(run_id: str, results: dict[str, Any]) -> None:
    """Log final results and close the MLflow run."""
    try:
        from src.ml.mlflow_logger import MlflowExperimentLogger

        ml = MlflowExperimentLogger(run_id=run_id)
        if ml._run:
            ml._mlflow.log_dict(results, "final_results.json")
            ml._mlflow.end_run()
    except Exception:
        pass


def save_checkpoint(state: LoopState) -> None:
    """Save loop state to JSON for resume on crash/timeout."""
    data = {
        "ticker_groups": [[t for t in g] for g in state.ticker_groups],
        "current_model_path": state.current_model_path,
        "current_config": state.current_config,
        "iteration": state.iteration,
        "consecutive_improvements": state.consecutive_improvements,
        "best_sharpe": state.best_sharpe,
        "best_config": state.best_config,
        "best_model_path": state.best_model_path,
        "locked": state.locked,
        "elapsed_seconds": time.time() - state.start_time,
        "history": state.history,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint() -> dict | None:
    """Load saved loop state. Returns None if no checkpoint exists."""
    if not CHECKPOINT_PATH.exists():
        return None
    return json.loads(CHECKPOINT_PATH.read_text())


def save_phase_state(phase: int, **kwargs: Any) -> None:
    """Track which phases completed with which outputs."""
    PHASE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if PHASE_STATE_PATH.exists():
        state = json.loads(PHASE_STATE_PATH.read_text())
    state[f"phase_{phase}"] = {"timestamp": datetime.now().isoformat(), **kwargs}
    with open(PHASE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── 8. CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous training loop with independence filtering and consecutive confirmation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/autonomous_train_loop.py --tickers "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA" --max-iterations 20 --confirmations 3 --timeout-hours 8
  uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,XLK,XLF" --phase 4 --fast --model models/my_model.pkl
  uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ,IWM,TLT,GLD" --horizon 10 --trail-stop --entry-threshold 0.45
""",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward return horizon in days",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum refinement iterations",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        default=3,
        help="Consecutive improvements required to lock",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=0,
        help="Maximum runtime in hours (0 = no limit)",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.70,
        help="Maximum allowed correlation within a group",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5"],
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip tuning for faster iterations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to existing model (skips training in phase 4)",
    )
    parser.add_argument(
        "--trail-stop",
        action="store_true",
        help="Enable ATR trailing stop",
    )
    parser.add_argument(
        "--trail-atr",
        type=float,
        default=3.0,
        help="ATR multiplier for trailing stop",
    )
    parser.add_argument(
        "--conviction",
        action="store_true",
        help="Scale position by conviction",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.50,
        help="ML probability entry threshold",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="triple_barrier",
        choices=["triple_barrier", "next_bar"],
        help="Label type: triple_barrier (forward horizon) or next_bar (no look-ahead)",
    )
    parser.add_argument(
        "--no-trail-stop",
        action="store_true",
        help="Disable trailing stop (use fixed TP/SL)",
    )
    parser.add_argument(
        "--skip-phase-4",
        action="store_true",
        help="Skip the autonomous refinement loop",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume Phase 4 from last checkpoint",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for Phase 5 Bayesian sweep (0 = use grid sweep)",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Use multi-objective Pareto optimization (Sharpe + MaxDD + WinRate)",
    )

    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        logger.error("No tickers provided")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("AUTONOMOUS TRAINING LOOP")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizon: {args.horizon}d | Fast: {args.fast}")
    logger.info(f"Max iterations: {args.max_iterations} | Confirmations: {args.confirmations}")
    logger.info(f"Timeout: {args.timeout_hours}h")
    logger.info("=" * 60)

    use_trail = not args.no_trail_stop and args.trail_stop

    run_all = args.phase == "all"

    # Phase 1: Ticker Selection + Independence Filter
    if run_all or args.phase == "1":
        groups = run_phase_1_ticker_selection(tickers, args.start, args.end, args.max_corr)
    else:
        groups = [[t] for t in tickers]

    if not groups:
        logger.error("No valid ticker groups")
        sys.exit(1)

    if run_all or args.phase == "1":
        save_phase_state(1, groups=groups, tickers=list(dict.fromkeys(tickers)))

    # Phase 2: Per-Group Tuning
    if run_all or args.phase == "2":
        run_phase_2_tuning(groups, args.start, args.end, args.horizon, args.fast)

    # Phase 3: Cross-Group Generalization
    cross_group_result: dict[str, Any] = {}
    if (run_all or args.phase == "3") and len(groups) > 1:
        cross_group_result = run_phase_3_cross_group_test(
            groups,
            args.horizon,
            args.start,
            args.end,
            args.fast,
        )

    if run_all or args.phase == "3":
        save_phase_state(3, model_path=cross_group_result.get("model_path", ""))

    # Phase 4: Autonomous Refinement Loop
    initial_model = args.model or cross_group_result.get("model_path", "")
    if not args.model and not cross_group_result.get("model_path"):
        phase_state = json.loads(PHASE_STATE_PATH.read_text()) if PHASE_STATE_PATH.exists() else {}
        if "phase_3" in phase_state:
            initial_model = phase_state["phase_3"].get("model_path", "")
    if (run_all or args.phase == "4") and not args.skip_phase_4:
        state = run_phase_4_refinement_loop(
            groups=groups,
            horizon=args.horizon,
            start=args.start,
            end=args.end,
            fast=args.fast,
            max_iterations=args.max_iterations,
            confirmations_required=args.confirmations,
            timeout_hours=args.timeout_hours,
            trail_stop=use_trail,
            conviction=args.conviction,
            entry_threshold=args.entry_threshold,
            trail_atr=args.trail_atr,
            initial_model_path=initial_model,
            resume=args.resume,
            label_type=args.label_type,
        )
        best_model = state.best_model_path or initial_model
    else:
        best_model = initial_model

    if run_all or args.phase == "4":
        st = locals().get("state")
        save_phase_state(
            4,
            model_path=st.best_model_path if st else "",
            best_config=st.best_config if st else {},
            best_sharpe=st.best_sharpe if st else 0,
            iterations=st.iteration if st else 0,
        )

    # Phase 5: Parameter Space Crawl
    sweep_results = []
    if (run_all or args.phase == "5") and best_model:
        sweep_results = run_phase_5_param_sweep(
            groups,
            args.horizon,
            args.start,
            args.end,
            args.fast,
            best_model,
            use_optuna=args.optuna_trials > 0,
            n_trials=args.optuna_trials if args.optuna_trials > 0 else 30,
            pareto=args.pareto,
        )

    if run_all or args.phase == "5":
        save_phase_state(5, sweep_results=sweep_results[-1] if sweep_results else {})

    logger.info("\n" + "=" * 60)
    logger.info("LOOP COMPLETE")
    logger.info(f"Log: {LOOP_LOG_PATH}")
    if best_model:
        logger.info(f"Model: {best_model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
