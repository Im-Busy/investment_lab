"""
Batch backtest all 20 tickers individually with IS + OOS, comprehensive analysis.
Production config: et=0.55, mr=0.70, multi-tp=ON, quality-registry=ON, gates=OFF
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from scripts.backtest_rules_first import run_single

TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLE",
    "GLD",
    "SLV",
    "XLF",
    "XLV",
    "NUE",
    "STLD",
    "HAL",
    "MPC",
    "EOG",
    "INTC",
    "AMD",
    "LMT",
    "JNJ",
    "MRK",
    "NEM",
    "CN_CATL",
]

IS_START = "2016-01-01"
IS_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2026-05-27"

CATEGORY_MAP = {
    "SPY": "Broad Index",
    "QQQ": "Broad Index",
    "XLK": "Sector/Tech",
    "XLE": "Sector/Energy",
    "XLF": "Sector/Financial",
    "XLV": "Sector/Health",
    "GLD": "Commodity",
    "SLV": "Commodity",
    "NUE": "Materials/Steel",
    "STLD": "Materials/Steel",
    "HAL": "Energy/OilSvc",
    "MPC": "Energy/Refining",
    "EOG": "Energy/E&P",
    "INTC": "Tech/Semis",
    "AMD": "Tech/Semis",
    "LMT": "Defense",
    "JNJ": "Pharma",
    "MRK": "Pharma",
    "NEM": "Gold Mining",
    "CN_CATL": "Battery/EV",
}

OUTPUT_DIR = Path("outputs/individual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_annual_pct(return_pct: float, start: str, end: str) -> float:
    """Compute annualized return from cumulative return and date range."""
    try:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        years = (end_dt - start_dt).days / 365.25
        if years <= 0 or return_pct <= -100:
            return 0.0
        return ((1 + return_pct / 100) ** (1 / max(years, 0.25)) - 1) * 100
    except Exception:
        return 0.0


def run_ticker(symbol: str) -> dict | None:
    """Run IS and OOS backtests for a single ticker."""
    print(f"\n{'=' * 60}")
    print(f"  {symbol}  |  IS: {IS_START} -> {IS_END}  |  OOS: {OOS_START} -> {OOS_END}")
    print(f"{'=' * 60}")

    try:
        is_result = run_single(
            symbol,
            start=IS_START,
            end=IS_END,
            entry_threshold=0.55,
            min_reliability=0.70,
            use_multi_tp=True,
            use_quality_registry=True,
            use_vix_gate=False,
            use_yield_curve_gate=False,
            use_garch_atr=False,
            use_options_sentiment=False,
            use_kelly_sizing=False,
            use_vix_regime_sizing=False,
            volume_confirm=True,
            use_short=False,
            use_order_book=False,
            use_signal_strength_sizing=False,
            use_voting_signal=False,
            use_rules_catalog=False,
            use_divergence=False,
            use_wm_bollinger=False,
            use_tadgan_gate=False,
        )
    except Exception as e:
        print(f"  IS FAILED: {e}")
        is_result = None

    time.sleep(0.5)

    try:
        oos_result = run_single(
            symbol,
            start=OOS_START,
            end=OOS_END,
            entry_threshold=0.55,
            min_reliability=0.70,
            use_multi_tp=True,
            use_quality_registry=True,
            use_vix_gate=False,
            use_yield_curve_gate=False,
            use_garch_atr=False,
            use_options_sentiment=False,
            use_kelly_sizing=False,
            use_vix_regime_sizing=False,
            volume_confirm=True,
            use_short=False,
            use_order_book=False,
            use_signal_strength_sizing=False,
            use_voting_signal=False,
            use_rules_catalog=False,
            use_divergence=False,
            use_wm_bollinger=False,
            use_tadgan_gate=False,
        )
    except Exception as e:
        print(f"  OOS FAILED: {e}")
        oos_result = None

    combined = {
        "symbol": symbol,
        "category": CATEGORY_MAP.get(symbol, "Other"),
        "config": {
            "et": 0.55,
            "mr": 0.70,
            "multi_tp": True,
            "quality_registry": True,
            "gates": False,
        },
        "is": None,
        "oos": None,
    }

    if is_result:
        is_annual = compute_annual_pct(is_result["return_pct"], IS_START, IS_END)
        combined["is"] = {
            "return_pct": round(is_result["return_pct"], 2),
            "sharpe": round(is_result["sharpe"], 3),
            "sortino": round(is_result["sortino"], 3),
            "calmar": round(is_result["calmar"], 3),
            "max_dd_pct": round(is_result["max_dd_pct"], 2),
            "trades": is_result["trades"],
            "win_rate_pct": round(is_result["win_rate_pct"], 1),
            "profit_factor": round(is_result["profit_factor"], 2),
            "exposure_pct": round(is_result["exposure_pct"], 1),
            "annual_pct": round(is_annual, 1),
        }
        print(
            f"  IS: Ret={is_result['return_pct']:.1f}%  Sharpe={is_result['sharpe']:.3f}  "
            f"MaxDD={is_result['max_dd_pct']:.1f}%  Trades={is_result['trades']}  "
            f"Win={is_result['win_rate_pct']:.1f}%  PF={is_result['profit_factor']:.2f}"
        )

    if oos_result:
        oos_annual = compute_annual_pct(oos_result["return_pct"], OOS_START, OOS_END)
        combined["oos"] = {
            "return_pct": round(oos_result["return_pct"], 2),
            "sharpe": round(oos_result["sharpe"], 3),
            "sortino": round(oos_result["sortino"], 3),
            "calmar": round(oos_result["calmar"], 3),
            "max_dd_pct": round(oos_result["max_dd_pct"], 2),
            "trades": oos_result["trades"],
            "win_rate_pct": round(oos_result["win_rate_pct"], 1),
            "profit_factor": round(oos_result["profit_factor"], 2),
            "exposure_pct": round(oos_result["exposure_pct"], 1),
            "annual_pct": round(oos_annual, 1),
        }
        print(
            f"  OOS: Ret={oos_result['return_pct']:.1f}%  Sharpe={oos_result['sharpe']:.3f}  "
            f"MaxDD={oos_result['max_dd_pct']:.1f}%  Trades={oos_result['trades']}  "
            f"Win={oos_result['win_rate_pct']:.1f}%  PF={oos_result['profit_factor']:.2f}"
        )

    if is_result and oos_result:
        delta_sharpe = round(oos_result["sharpe"] - is_result["sharpe"], 3)
        combined["delta_sharpe"] = delta_sharpe
        print(f"  Delta Sharpe: {delta_sharpe:+.3f}")

    if is_result:
        combined["bh"] = {
            "is_return_pct": is_result["bh_return_pct"],
            "is_sharpe": is_result["bh_sharpe"],
            "is_max_dd_pct": is_result["bh_max_dd_pct"],
            "oos_return_pct": oos_result["bh_return_pct"] if oos_result else None,
            "oos_sharpe": oos_result["bh_sharpe"] if oos_result else None,
            "oos_max_dd_pct": oos_result["bh_max_dd_pct"] if oos_result else None,
        }

    # Save individual result
    output_path = OUTPUT_DIR / f"{symbol}.json"
    output_path.write_text(json.dumps(combined, indent=2))
    print(f"  Saved: {output_path}")

    return combined


def classify_pattern(is_result: dict | None, oos_result: dict | None) -> str:
    """Classify ticker pattern based on IS/OOS returns."""
    if is_result is None or oos_result is None:
        return "N/A"
    is_ret = is_result["return_pct"]
    oos_ret = oos_result["return_pct"]
    if is_ret <= 0 and oos_ret > 0:
        return "Phoenix (IS- → OOS+)"
    elif is_ret > 0 and oos_ret <= 0:
        return "Death Cross (IS+ → OOS-)"
    elif is_ret > 0 and oos_ret > 0:
        return "Consistent Winner"
    elif is_ret <= 0 and oos_ret <= 0:
        return "Consistent Loser"
    return "Mixed"


def assign_tier(oos_sharpe: float, trades: int, win_rate: float) -> str:
    """Assign S/A/B/F tier based on OOS Sharpe + trade count + win rate."""
    if oos_sharpe >= 1.0 and trades >= 10 and win_rate >= 45:
        return "S"
    elif oos_sharpe >= 0.5 and trades >= 8 and win_rate >= 40:
        return "A"
    elif oos_sharpe >= 0.0 and trades >= 5 and win_rate >= 35:
        return "B"
    else:
        return "F"


def produce_analysis(results: list[dict], output_path: Path) -> None:
    """Generate comprehensive analysis markdown."""
    valid = [r for r in results if r.get("is") and r.get("oos")]
    if not valid:
        print("No valid results to analyze.")
        return

    # Sort by OOS Sharpe descending
    valid.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)

    lines = []
    lines.append("# Individual Backtest Analysis — Production Config")
    lines.append("")
    lines.append("**Config:** et=0.55, mr=0.70, multi-tp=ON, quality-registry=ON, gates=OFF")
    lines.append(f"**IS Period:** {IS_START} → {IS_END} | **OOS Period:** {OOS_START} → {OOS_END}")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: Individual Results ──
    lines.append("## 1. Individual Backtest Results")
    lines.append("")

    for r in valid:
        sym = r["symbol"]
        cat = r["category"]
        isd = r["is"]
        oosd = r["oos"]
        ds = r.get("delta_sharpe", 0)
        bh = r.get("bh", {})
        pattern = classify_pattern(isd, oosd)
        tier = assign_tier(oosd["sharpe"], oosd["trades"], oosd["win_rate_pct"])

        lines.append(f"### {sym} — {cat} | Tier: {tier} | Pattern: {pattern}")
        lines.append("")

        # IS table
        lines.append("**In-Sample (2016-2024)**")
        lines.append("")
        lines.append(
            "| Return% | Sharpe | Sortino | Calmar | MaxDD% | Trades | Win% | PF | Exposure% | Annual% |"
        )
        lines.append(
            "|---------|--------|---------|--------|--------|--------|------|----|-----------|---------|"
        )
        lines.append(
            f"| {isd['return_pct']:.1f} | {isd['sharpe']:.3f} | {isd['sortino']:.3f} | {isd['calmar']:.3f} "
            f"| {isd['max_dd_pct']:.1f} | {isd['trades']} | {isd['win_rate_pct']:.1f} "
            f"| {isd['profit_factor']:.2f} | {isd['exposure_pct']:.1f} | {isd['annual_pct']:.1f} |"
        )
        lines.append("")

        # OOS table
        lines.append("**Out-of-Sample (2025-2026)**")
        lines.append("")
        lines.append(
            "| Return% | Sharpe | Sortino | Calmar | MaxDD% | Trades | Win% | PF | Exposure% | Δ Sharpe | Annual% |"
        )
        lines.append(
            "|---------|--------|---------|--------|--------|--------|------|----|-----------|----------|---------|"
        )
        lines.append(
            f"| {oosd['return_pct']:.1f} | {oosd['sharpe']:.3f} | {oosd['sortino']:.3f} | {oosd['calmar']:.3f} "
            f"| {oosd['max_dd_pct']:.1f} | {oosd['trades']} | {oosd['win_rate_pct']:.1f} "
            f"| {oosd['profit_factor']:.2f} | {oosd['exposure_pct']:.1f} | {ds:+.3f} | {oosd['annual_pct']:.1f} |"
        )
        lines.append("")

        # B&H comparison
        if bh:
            lines.append("**Buy & Hold Comparison**")
            lines.append("")
            lines.append("| Period | B&H Return% | B&H Sharpe | B&H MaxDD% |")
            lines.append("|--------|-------------|------------|------------|")
            bh_is = (
                f"{bh.get('is_return_pct', 0):.1f}"
                if bh.get("is_return_pct") is not None
                else "N/A"
            )
            bh_is_sh = f"{bh.get('is_sharpe', 0):.3f}" if bh.get("is_sharpe") is not None else "N/A"
            bh_is_dd = (
                f"{bh.get('is_max_dd_pct', 0):.1f}"
                if bh.get("is_max_dd_pct") is not None
                else "N/A"
            )
            bh_oos = (
                f"{bh.get('oos_return_pct', 0):.1f}"
                if bh.get("oos_return_pct") is not None
                else "N/A"
            )
            bh_oos_sh = (
                f"{bh.get('oos_sharpe', 0):.3f}" if bh.get("oos_sharpe") is not None else "N/A"
            )
            bh_oos_dd = (
                f"{bh.get('oos_max_dd_pct', 0):.1f}"
                if bh.get("oos_max_dd_pct") is not None
                else "N/A"
            )
            lines.append(f"| IS | {bh_is} | {bh_is_sh} | {bh_is_dd} |")
            lines.append(f"| OOS | {bh_oos} | {bh_oos_sh} | {bh_oos_dd} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Section 2: Ranking Table ──
    lines.append("## 2. Ranking Table (Sorted by OOS Sharpe)")
    lines.append("")
    lines.append(
        "| Rank | Ticker | Category | OOS Sharpe | OOS Return% | OOS MaxDD% | Trades | Win% | PF | Δ Sharpe | Pattern | Tier |"
    )
    lines.append(
        "|------|--------|----------|------------|-------------|------------|--------|------|----|----------|---------|------|"
    )
    for i, r in enumerate(valid, 1):
        oosd = r["oos"]
        isd = r["is"]
        ds = r.get("delta_sharpe", 0)
        pattern = classify_pattern(isd, oosd)
        tier = assign_tier(oosd["sharpe"], oosd["trades"], oosd["win_rate_pct"])
        lines.append(
            f"| {i} | {r['symbol']} | {r['category']} | {oosd['sharpe']:.3f} | {oosd['return_pct']:.1f} "
            f"| {oosd['max_dd_pct']:.1f} | {oosd['trades']} | {oosd['win_rate_pct']:.1f} "
            f"| {oosd['profit_factor']:.2f} | {ds:+.3f} | {pattern} | {tier} |"
        )
    lines.append("")

    # ── Section 3: Category Breakdown ──
    lines.append("## 3. Category Breakdown (Mean OOS Sharpe)")
    lines.append("")
    cat_data: dict[str, list[dict]] = {}
    for r in valid:
        cat = r["category"]
        cat_data.setdefault(cat, []).append(r)

    lines.append(
        "| Category | Count | Mean OOS Sharpe | Mean OOS Return% | Mean Win% | Mean Trades |"
    )
    lines.append(
        "|----------|-------|-----------------|------------------|-----------|-------------|"
    )
    for cat in sorted(cat_data.keys()):
        tickers_in_cat = cat_data[cat]
        n = len(tickers_in_cat)
        mean_sh = sum(t["oos"]["sharpe"] for t in tickers_in_cat) / n
        mean_ret = sum(t["oos"]["return_pct"] for t in tickers_in_cat) / n
        mean_win = sum(t["oos"]["win_rate_pct"] for t in tickers_in_cat) / n
        mean_tr = sum(t["oos"]["trades"] for t in tickers_in_cat) / n
        lines.append(
            f"| {cat} | {n} | {mean_sh:.3f} | {mean_ret:.1f} | {mean_win:.1f} | {mean_tr:.1f} |"
        )
    lines.append("")

    # ── Section 4: IS→OOS Sharpe Correlation ──
    lines.append("## 4. IS → OOS Sharpe Correlation")
    lines.append("")
    is_sharpes = [r["is"]["sharpe"] for r in valid]
    oos_sharpes = [r["oos"]["sharpe"] for r in valid]
    try:
        corr_series = pd.Series(is_sharpes).corr(pd.Series(oos_sharpes))
    except Exception:
        corr_series = 0.0
    lines.append(f"**Pearson r (IS Sharpe vs OOS Sharpe):** {corr_series:.3f}")
    lines.append("")
    lines.append("- Positive correlation → IS Sharpe is predictive of OOS")
    lines.append("- Negative correlation → IS performance is misleading")
    lines.append("")

    # ── Section 5: Pattern Classification ──
    lines.append("## 5. Pattern Classification")
    lines.append("")
    phoenix = [r for r in valid if "Phoenix" in classify_pattern(r["is"], r["oos"])]
    death_cross = [r for r in valid if "Death Cross" in classify_pattern(r["is"], r["oos"])]
    winners = [r for r in valid if "Consistent Winner" in classify_pattern(r["is"], r["oos"])]
    losers = [r for r in valid if "Consistent Loser" in classify_pattern(r["is"], r["oos"])]

    lines.append("### Phoenix (IS negative → OOS positive)")
    lines.append("")
    if phoenix:
        for r in phoenix:
            lines.append(
                f"- **{r['symbol']}** ({r['category']}): IS {r['is']['return_pct']:.1f}% → OOS {r['oos']['return_pct']:.1f}%"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Death Cross (IS positive → OOS negative)")
    lines.append("")
    if death_cross:
        for r in death_cross:
            lines.append(
                f"- **{r['symbol']}** ({r['category']}): IS {r['is']['return_pct']:.1f}% → OOS {r['oos']['return_pct']:.1f}%"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Consistent Winners")
    lines.append("")
    if winners:
        for r in winners:
            lines.append(
                f"- **{r['symbol']}** ({r['category']}): IS {r['is']['return_pct']:.1f}% / OOS {r['oos']['return_pct']:.1f}%"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Consistent Losers")
    lines.append("")
    if losers:
        for r in losers:
            lines.append(
                f"- **{r['symbol']}** ({r['category']}): IS {r['is']['return_pct']:.1f}% / OOS {r['oos']['return_pct']:.1f}%"
            )
    else:
        lines.append("- None")
    lines.append("")

    # ── Section 6: Recommended Basket ──
    lines.append("## 6. Recommended Basket (Tier Assignments)")
    lines.append("")
    lines.append("Tiers based on OOS Sharpe + trade count + win rate:")
    lines.append("- **S**: Sharpe >= 1.0, Trades >= 10, Win% >= 45")
    lines.append("- **A**: Sharpe >= 0.5, Trades >= 8, Win% >= 40")
    lines.append("- **B**: Sharpe >= 0.0, Trades >= 5, Win% >= 35")
    lines.append("- **F**: Below thresholds")
    lines.append("")

    for tier in ["S", "A", "B", "F"]:
        tier_tickers = [
            r
            for r in valid
            if assign_tier(r["oos"]["sharpe"], r["oos"]["trades"], r["oos"]["win_rate_pct"]) == tier
        ]
        lines.append(f"### Tier {tier} ({len(tier_tickers)} tickers)")
        lines.append("")
        if tier_tickers:
            for r in tier_tickers:
                lines.append(
                    f"- **{r['symbol']}** ({r['category']}): Sharpe {r['oos']['sharpe']:.3f}, "
                    f"Return {r['oos']['return_pct']:.1f}%, MaxDD {r['oos']['max_dd_pct']:.1f}%, "
                    f"Trades {r['oos']['trades']}, Win {r['oos']['win_rate_pct']:.1f}%, PF {r['oos']['profit_factor']:.2f}"
                )
        else:
            lines.append("- None")
        lines.append("")

    # ── Section 7: Full Data Tables ──
    lines.append("## 7. Full Data — In-Sample (2016-2024)")
    lines.append("")
    lines.append(
        "| Ticker | Category | Return% | Sharpe | Sortino | Calmar | MaxDD% | Trades | Win% | PF | Exposure% | Annual% | B&H Ret% | B&H Sharpe | B&H MaxDD% |"
    )
    lines.append(
        "|--------|----------|---------|--------|---------|--------|--------|--------|------|----|-----------|---------|----------|------------|------------|"
    )
    for r in valid:
        isd = r["is"]
        bh = r.get("bh", {})
        lines.append(
            f"| {r['symbol']} | {r['category']} | {isd['return_pct']:.1f} | {isd['sharpe']:.3f} "
            f"| {isd['sortino']:.3f} | {isd['calmar']:.3f} | {isd['max_dd_pct']:.1f} "
            f"| {isd['trades']} | {isd['win_rate_pct']:.1f} | {isd['profit_factor']:.2f} "
            f"| {isd['exposure_pct']:.1f} | {isd['annual_pct']:.1f} "
            f"| {bh.get('is_return_pct', 0):.1f} | {bh.get('is_sharpe', 0):.3f} | {bh.get('is_max_dd_pct', 0):.1f} |"
        )
    lines.append("")

    lines.append("## 8. Full Data — Out-of-Sample (2025-2026)")
    lines.append("")
    lines.append(
        "| Ticker | Category | Return% | Sharpe | Sortino | Calmar | MaxDD% | Trades | Win% | PF | Exposure% | Δ Sharpe | Annual% | B&H Ret% | B&H Sharpe | B&H MaxDD% |"
    )
    lines.append(
        "|--------|----------|---------|--------|---------|--------|--------|--------|------|----|-----------|----------|---------|----------|------------|------------|"
    )
    for r in valid:
        oosd = r["oos"]
        ds = r.get("delta_sharpe", 0)
        bh = r.get("bh", {})
        lines.append(
            f"| {r['symbol']} | {r['category']} | {oosd['return_pct']:.1f} | {oosd['sharpe']:.3f} "
            f"| {oosd['sortino']:.3f} | {oosd['calmar']:.3f} | {oosd['max_dd_pct']:.1f} "
            f"| {oosd['trades']} | {oosd['win_rate_pct']:.1f} | {oosd['profit_factor']:.2f} "
            f"| {oosd['exposure_pct']:.1f} | {ds:+.3f} | {oosd['annual_pct']:.1f} "
            f"| {bh.get('oos_return_pct', 0):.1f} | {bh.get('oos_sharpe', 0):.3f} | {bh.get('oos_max_dd_pct', 0):.1f} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(f"*Analysis generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAnalysis saved to {output_path}")


def main():
    print("=" * 70)
    print("  BATCH INDIVIDUAL BACKTEST — 20 Tickers, IS + OOS")
    print("  Config: et=0.55, mr=0.70, multi-tp=ON, quality-registry=ON")
    print("=" * 70)

    all_results = []
    for i, ticker in enumerate(TICKERS, 1):
        print(f"\n[{i}/{len(TICKERS)}] Processing {ticker}...")
        result = run_ticker(ticker)
        if result:
            all_results.append(result)

    print(f"\n\nCompleted: {len(all_results)}/{len(TICKERS)} tickers")

    produce_analysis(all_results, Path("reports/individual_backtest_analysis.md"))
    print("\nDone.")


if __name__ == "__main__":
    main()
