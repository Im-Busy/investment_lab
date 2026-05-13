"""Autonomous ticker screener for basket training inclusion.

Usage:
    uv run scripts/screen_tickers.py TICKER1 TICKER2 TICKER3 ...
    uv run scripts/screen_tickers.py --batch utilities
    uv run scripts/screen_tickers.py --batch all

Tier 1: Fundamental screen (relaxed v2 criteria)
Tier 2: Download data + walk-forward IC test
Tier 3: Correlation check (manual review)
Logs results to docs/ticker-test-log.md
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from src.ml.walk_forward import walk_forward_per_ticker

# ── Config ──────────────────────────────────────────────────────────
MIN_MC = 200e6
MIN_VOL_NOTIONAL_M = 5  # $M/day
MIN_INST = 0.25
MIN_YEARS = 5
VOL_LOW = 0.12
VOL_HIGH = 0.55
IC_THRESHOLD = 0.03
MODEL_GLOB = "models/pattern_classifier_v3_SPY_*.pkl"
LOG_PATH = Path("docs/ticker-test-log.md")

# ── Batch presets (sectors under-represented in basket) ─────────────
BATCHES: dict[str, list[str]] = {
    "utilities": ["XLU", "DUK", "SO", "NEE", "AEP", "D", "EXC", "SRE", "PEG", "ED"],
    "real_estate": ["XLRE", "VNQ", "O", "PLD", "AMT", "SPG", "PSA", "WELL", "AVB", "EQR"],
    "transportation": ["XTN", "UNP", "UPS", "FDX", "CSX", "NSC", "DAL", "UAL", "LUV", "AAL"],
    "metals_mining": ["FCX", "NEM", "AEM", "GOLD", "AA", "NUE", "STLD", "CLF", "X", "SCCO"],
    "consumer_staples": ["XLP", "COST", "WMT", "PG", "KO", "PEP", "PM", "MO", "CL", "KMB"],
    "energy_singles": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "HAL"],
    "healthcare_singles": [
        "JNJ",
        "MRK",
        "ABBV",
        "PFE",
        "BMY",
        "GILD",
        "AMGN",
        "REGN",
        "VRTX",
        "BIIB",
    ],
    "midcap_niche": ["KODK", "JOE", "CAR", "REPL", "CRVL", "HIFS", "LMT", "CAT", "HD"],
}


def find_latest_model() -> str | None:
    models = sorted(Path().glob(MODEL_GLOB))
    return str(models[-1]) if models else None


def fundamental_screen(ticker: str) -> dict[str, Any]:
    """Tier 1: check fundamental criteria. Returns dict with results."""
    result: dict[str, Any] = {"ticker": ticker, "passed": False, "failures": []}
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        result["failures"].append(f"yfinance error: {e}")
        return result

    mc = info.get("marketCap", 0) or 0
    vol_shares = info.get("averageVolume", 0) or 0
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 50
    )
    inst = info.get("heldPercentInstitutions", 0) or 0
    sector = info.get("sector", "N/A")
    name = info.get("shortName", ticker)
    exchange = info.get("exchange", "")

    result["name"] = name
    result["sector"] = sector
    result["mc"] = mc
    result["inst"] = inst
    result["price"] = price

    # Fetch OHLCV for history + volatility
    try:
        df = yf.download(ticker, start="2010-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        result["failures"].append("Download failed")
        return result

    if len(df) < 50:
        result["failures"].append(f"Only {len(df)} bars")
        return result

    years = len(df) / 252
    ann_vol = float(df["Close"].pct_change().std() * (252**0.5))
    notional_m = vol_shares * price / 1e6
    result["years"] = years
    result["ann_vol"] = ann_vol
    result["notional_m"] = notional_m
    result["df"] = df

    # Checks
    if mc < MIN_MC:
        result["failures"].append(f"MC ${mc / 1e6:.0f}M < ${MIN_MC / 1e6:.0f}M")
    if notional_m < MIN_VOL_NOTIONAL_M:
        result["failures"].append(f"Vol ${notional_m:.0f}M/d")
    if inst < MIN_INST:
        result["failures"].append(f"Inst {inst:.0%}")
    if years < MIN_YEARS:
        result["failures"].append(f"Hist {years:.1f}y")
    if ann_vol < VOL_LOW:
        result["failures"].append(f"Vol {ann_vol:.1%}")
    if ann_vol > VOL_HIGH:
        result["failures"].append(f"Vol {ann_vol:.1%}")

    result["passed"] = len(result["failures"]) == 0
    return result


def append_to_log(results: list[dict[str, Any]]) -> None:
    """Append test results to ticker-test-log.md."""
    today = date.today()
    lines = []
    for r in results:
        mc_str = f"${r.get('mc', 0) / 1e9:.1f}B"
        vol_str = f"${r.get('notional_m', 0):.0f}M"
        inst_str = f"{r.get('inst', 0):.0%}"
        years_str = f"{r.get('years', 0):.0f}yr"
        ann_str = f"{r.get('ann_vol', 0):.1%}"
        ic_str = f"{r.get('ic', 0):+.3f}" if r.get("ic") is not None else "—"
        verdict = r.get("verdict", "—")
        notes = r.get("notes", "")
        lines.append(
            f"| {today} | {r['ticker']} | {r.get('name', r['ticker'])} "
            f"| {r.get('sector', 'N/A')} | {mc_str} | {vol_str} | {inst_str} "
            f"| {years_str} | {ann_str} | {ic_str} | {verdict} | {notes} |"
        )

    if not LOG_PATH.exists():
        with open(LOG_PATH, "w") as f:
            f.write("# Ticker Test Log\n\n")
            f.write(
                "| Date | Ticker | Name | Sector | MC | Vol $M/d | Inst% | History | AnnVol% | IC | Verdict | Notes |\n"
            )
            f.write(
                "|------|--------|------|--------|----|---------|-------|---------|---------|----|---------|-------|\n"
            )

    with open(LOG_PATH, "a") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/screen_tickers.py TICKER1 TICKER2 ...")
        print("       uv run scripts/screen_tickers.py --batch <batch_name>")
        print(f"       Batches: {list(BATCHES.keys())}")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        batch_name = sys.argv[2] if len(sys.argv) > 2 else "all"
        if batch_name == "all":
            tickers = list(dict.fromkeys([t for batch in BATCHES.values() for t in batch]))
        else:
            tickers = BATCHES.get(batch_name, [])
        if not tickers:
            print(f"Unknown batch: {batch_name}. Available: {list(BATCHES.keys())}")
            sys.exit(1)
        print(f"Batch '{batch_name}': {len(tickers)} tickers")
    else:
        tickers = sys.argv[1:]

    model_path = find_latest_model()
    if not model_path:
        print("No model found. Train one first.")
        sys.exit(1)
    print(f"Model: {model_path}")

    # ── Tier 1: Fundamental Screen ──
    print(f"\n{'=' * 70}")
    print("TIER 1: Fundamental Screen")
    print(f"{'=' * 70}")
    passed: list[dict] = []
    failed: list[dict] = []

    for t in tickers:
        r = fundamental_screen(t)
        if r["passed"]:
            passed.append(r)
        else:
            failed.append(r)

    print(f"\nPassed: {len(passed)}/{len(tickers)}")
    for r in failed:
        print(f"  FAIL {r['ticker']:>6}: {'; '.join(r['failures'])}")

    # Log fundamental-only failures
    if failed:
        for r in failed:
            r["ic"] = None
            r["verdict"] = "FAIL (fundamental)"
            r["notes"] = "; ".join(r["failures"])
        append_to_log(failed)

    if not passed:
        print("\nNo tickers passed fundamental screen.")
        return

    # ── Tier 2: Download + Walk-Forward IC ──
    print(f"\n{'=' * 70}")
    print("TIER 2: Walk-Forward IC Test")
    print(f"{'=' * 70}")

    for r in passed:
        path = Path(f"data/raw/{r['ticker']}_daily.csv")
        if not path.exists():
            r["df"].to_csv(path)

    m = PatternClassifier()
    m.load(model_path)

    wf_results = walk_forward_per_ticker(
        m,
        [r["ticker"] for r in passed],
        initial_train_days=3 * 252,
        step_days=6 * 21,
        horizon=5,
    )

    print(
        f"\n{'Ticker':>6} | {'Name':<30} | {'Sector':<22} | {'IC':>8} | {'Steps':>5} | {'Verdict'}"
    )
    print("-" * 100)
    for r in passed:
        t = r["ticker"]
        wf = wf_results[t]
        ic = wf.mean_rank_ic
        verdict = "PASS" if ic > IC_THRESHOLD else "FAIL"
        print(
            f"{t:>6} | {r['name']:<30} | {r['sector']:<22} | {ic:+8.4f} | {wf.n_steps:>5} | {verdict}"
        )
        r["ic"] = ic
        r["verdict"] = verdict

    # Sort by IC
    passed.sort(key=lambda x: x["ic"], reverse=True)

    # Log results
    append_to_log(passed)

    # Summary
    n_pass = sum(1 for r in passed if r["verdict"] == "PASS")
    print(f"\nSummary: {n_pass}/{len(passed)} passed IC > {IC_THRESHOLD}")
    if n_pass > 0:
        print("PASSED:")
        for r in passed:
            if r["verdict"] == "PASS":
                print(f"  {r['ticker']} ({r['name']}): IC={r['ic']:+.4f}")


if __name__ == "__main__":
    main()
