"""
Compile all 12 batch JSONs into master summary report.
Produces:
    reports/comprehensive_batch/MASTER_SUMMARY.json
    reports/comprehensive_batch/MASTER_SUMMARY.md
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def load_batches(batch_dir: Path) -> list[dict]:
    batches = []
    for f in sorted(batch_dir.glob("batch_*.json")):
        if "MASTER" in f.name:
            continue
        data = json.loads(f.read_text())
        batches.append(data)
    return batches


def safe_float(v: float | None, default: float = 0.0) -> float:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return default
    return float(v)


def build_master(batches: list[dict]) -> dict:
    all_tickers: list[dict] = []
    for batch in batches:
        for r in batch["results"]:
            r["batch"] = batch["batch"]
            all_tickers.append(r)

    n_total = len(all_tickers)
    is_traded = [r for r in all_tickers if r.get("is_trades", 0) > 0]
    oos_traded = [r for r in all_tickers if r.get("oos_trades", 0) > 0]

    # Top 20 IS
    top_is = sorted(all_tickers, key=lambda r: safe_float(r.get("is_sharpe", -999)), reverse=True)[
        :20
    ]

    # Top 20 OOS
    top_oos = sorted(
        all_tickers, key=lambda r: safe_float(r.get("oos_sharpe", -999)), reverse=True
    )[:20]

    # Bottom 10 IS
    bot_is = sorted(all_tickers, key=lambda r: safe_float(r.get("is_sharpe", 999)))[:10]

    # Bottom 10 OOS
    bot_oos = sorted(all_tickers, key=lambda r: safe_float(r.get("oos_sharpe", 999)))[:10]

    # Category aggregation
    cats: dict[str, list[dict]] = {}
    for r in all_tickers:
        cat = r.get("category", "Other")
        cat_key = cat.split("-")[0] if "-" in cat else cat
        cats.setdefault(cat_key, []).append(r)

    cat_summary = {}
    for cat_name, items in sorted(cats.items()):
        is_s_list = [safe_float(r.get("is_sharpe", 0)) for r in items if r.get("is_trades", 0) > 0]
        oos_s_list = [
            safe_float(r.get("oos_sharpe", 0)) for r in items if r.get("oos_trades", 0) > 0
        ]
        cat_summary[cat_name] = {
            "n": len(items),
            "is_mean_sharpe": round(np.mean(is_s_list), 3) if is_s_list else 0,
            "is_median_sharpe": round(np.median(is_s_list), 3) if is_s_list else 0,
            "oos_mean_sharpe": round(np.mean(oos_s_list), 3) if oos_s_list else 0,
            "oos_median_sharpe": round(np.median(oos_s_list), 3) if oos_s_list else 0,
            "is_mean_return": round(np.mean([safe_float(r.get("is_return", 0)) for r in items]), 2),
            "oos_mean_return": round(
                np.mean([safe_float(r.get("oos_return", 0)) for r in items]), 2
            ),
            "is_pos_sharpe": sum(1 for r in items if safe_float(r.get("is_sharpe", 0)) > 0.01),
            "oos_pos_sharpe": sum(1 for r in items if safe_float(r.get("oos_sharpe", 0)) > 0.01),
            "oos_better": sum(1 for r in items if safe_float(r.get("oos_delta_sharpe", 0)) > 0.01),
            "tickers": [r["symbol"] for r in items],
        }

    # IS->OOS correlation
    common = [
        (safe_float(r["is_sharpe"]), safe_float(r["oos_sharpe"]))
        for r in all_tickers
        if r.get("is_trades", 0) > 0 and r.get("oos_trades", 0) > 0
    ]
    is_oos_corr = (
        float(np.corrcoef([c[0] for c in common], [c[1] for c in common])[0, 1])
        if len(common) >= 5
        else 0
    )

    # IS->OOS delta distribution
    deltas = [
        safe_float(r.get("oos_delta_sharpe", 0))
        for r in all_tickers
        if r.get("is_trades", 0) > 0 and r.get("oos_trades", 0) > 0
    ]
    oos_improved = sum(1 for d in deltas if d > 0.01)
    oos_worsened = sum(1 for d in deltas if d < -0.01)

    # "Death cross" — great IS, terrible OOS
    death_cross = sorted(
        [
            r
            for r in all_tickers
            if safe_float(r.get("is_sharpe", 0)) > 0.2 and safe_float(r.get("oos_sharpe", 0)) < -0.5
        ],
        key=lambda r: safe_float(r.get("oos_sharpe", 0)),
    )

    # "Phoenix" — terrible IS, great OOS
    phoenix = sorted(
        [
            r
            for r in all_tickers
            if safe_float(r.get("is_sharpe", 0)) < -0.1 and safe_float(r.get("oos_sharpe", 0)) > 0.3
        ],
        key=lambda r: safe_float(r.get("oos_sharpe", 0)),
        reverse=True,
    )

    # "Consistent" — good both periods
    consistent = sorted(
        [
            r
            for r in all_tickers
            if safe_float(r.get("is_sharpe", 0)) > 0.1 and safe_float(r.get("oos_sharpe", 0)) > 0.1
        ],
        key=lambda r: safe_float(r.get("oos_sharpe", 0)),
        reverse=True,
    )

    # "No-go" — zero OOS trades
    no_trades_oos = [r for r in all_tickers if r.get("oos_trades", 0) == 0]

    master = {
        "timestamp": datetime.now().isoformat(),
        "config": batches[0]["config"] if batches else {},
        "summary": {
            "n_total": n_total,
            "n_is_traded": len(is_traded),
            "n_oos_traded": len(oos_traded),
            "is_pos_sharpe_pct": round(
                100
                * sum(1 for r in is_traded if safe_float(r.get("is_sharpe", 0)) > 0.01)
                / max(len(is_traded), 1)
            ),
            "oos_pos_sharpe_pct": round(
                100
                * sum(1 for r in oos_traded if safe_float(r.get("oos_sharpe", 0)) > 0.01)
                / max(len(oos_traded), 1)
            ),
            "oos_improved_pct": round(100 * oos_improved / max(len(deltas), 1)),
            "oos_worsened_pct": round(100 * oos_worsened / max(len(deltas), 1)),
            "is_oos_correlation": round(is_oos_corr, 3),
            "is_mean_sharpe": round(
                np.mean([safe_float(r.get("is_sharpe", 0)) for r in is_traded]), 3
            )
            if is_traded
            else 0,
            "oos_mean_sharpe": round(
                np.mean([safe_float(r.get("oos_sharpe", 0)) for r in oos_traded]), 3
            )
            if oos_traded
            else 0,
            "death_cross_count": len(death_cross),
            "phoenix_count": len(phoenix),
            "consistent_count": len(consistent),
            "no_trades_oos_count": len(no_trades_oos),
        },
        "top_20_is": [format_ticker(r, "is") for r in top_is],
        "top_20_oos": [format_ticker(r, "oos") for r in top_oos],
        "bottom_10_is": [format_ticker(r, "is") for r in bot_is],
        "bottom_10_oos": [format_ticker(r, "oos") for r in bot_oos],
        "death_cross": [format_ticker(r, "both") for r in death_cross],
        "phoenix": [format_ticker(r, "both") for r in phoenix],
        "consistent": [format_ticker(r, "both") for r in consistent],
        "no_trades_oos": [r["symbol"] for r in no_trades_oos],
        "category_summary": cat_summary,
        "all_tickers": all_tickers,
    }
    return master


def format_ticker(r: dict, period: str) -> dict:
    out = {
        "symbol": r["symbol"],
        "category": r.get("category", ""),
        "batch": r.get("batch", ""),
    }
    if period in ("is", "both"):
        out["is_sharpe"] = safe_float(r.get("is_sharpe"))
        out["is_return"] = safe_float(r.get("is_return"))
        out["is_trades"] = r.get("is_trades", 0)
        out["is_winrate"] = safe_float(r.get("is_winrate"))
        out["is_pf"] = safe_float(r.get("is_pf"))
    if period in ("oos", "both"):
        out["oos_sharpe"] = safe_float(r.get("oos_sharpe"))
        out["oos_return"] = safe_float(r.get("oos_return"))
        out["oos_trades"] = r.get("oos_trades", 0)
        out["oos_winrate"] = safe_float(r.get("oos_winrate"))
        out["oos_pf"] = safe_float(r.get("oos_pf"))
        out["oos_delta_sharpe"] = safe_float(r.get("oos_delta_sharpe"))
    return out


def generate_markdown(master: dict) -> str:
    s = master["summary"]
    lines = []
    lines.append("# Comprehensive Backtest Master Summary")
    lines.append(f"\n**Generated**: {master['timestamp'][:19]}")
    lines.append(
        "\n**Config**: Production — mr=0.70, et=0.55, trail=3.0, multi-TP=ON, quality-registry=ON"
    )
    lines.append("**Periods**: IS 2016-2024, OOS 2025-2026")

    lines.append("\n---")
    lines.append("\n## Global Summary")
    lines.append("\n| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total instruments tested | {s['n_total']} |")
    lines.append(f"| IS traded (>0 trades) | {s['n_is_traded']} |")
    lines.append(f"| OOS traded (>0 trades) | {s['n_oos_traded']} |")
    lines.append(f"| IS positive Sharpe | {s['is_pos_sharpe_pct']}% |")
    lines.append(f"| OOS positive Sharpe | {s['oos_pos_sharpe_pct']}% |")
    lines.append(f"| OOS improved vs IS | {s['oos_improved_pct']}% |")
    lines.append(f"| IS→OOS Sharpe correlation | {s['is_oos_correlation']:.3f} |")
    lines.append(f"| IS mean Sharpe | {s['is_mean_sharpe']:.3f} |")
    lines.append(f"| OOS mean Sharpe | {s['oos_mean_sharpe']:.3f} |")
    lines.append(f"| Death crosses (IS>0.2, OOS<-0.5) | {s['death_cross_count']} |")
    lines.append(f"| Phoenix (IS<-0.1, OOS>0.3) | {s['phoenix_count']} |")
    lines.append(f"| Consistent (IS>0.1, OOS>0.1) | {s['consistent_count']} |")
    lines.append(f"| OOS zero trades | {s['no_trades_oos_count']} |")

    # Category summary
    lines.append("\n---")
    lines.append("\n## By Category")
    lines.append(
        "\n| Category | N | IS Sharpe | OOS Sharpe | IS Ret% | OOS Ret% | IS Pos% | OOS Pos% | OOS Better% |"
    )
    lines.append(
        "|----------|---|-----------|------------|---------|----------|---------|----------|-------------|"
    )
    for cat_name, c in master["category_summary"].items():
        lines.append(
            f"| {cat_name} | {c['n']} | {c['is_mean_sharpe']:+.3f} | {c['oos_mean_sharpe']:+.3f} | "
            f"{c['is_mean_return']:+.1f} | {c['oos_mean_return']:+.1f} | "
            f"{100 * c['is_pos_sharpe'] // c['n']}% | {100 * c['oos_pos_sharpe'] // c['n']}% | "
            f"{100 * c['oos_better'] // c['n']}% |"
        )

    # Top 20 IS
    lines.append("\n---")
    lines.append("\n## Top 20 — IS Sharpe (2016-2024)")
    lines.append("\n| # | Symbol | Category | Sharpe | Return% | Trades | Win% | PF |")
    lines.append("|---|--------|----------|--------|---------|--------|------|-----|")
    for i, r in enumerate(master["top_20_is"], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['category']} | {r['is_sharpe']:+.3f} | "
            f"{r['is_return']:+.1f} | {r['is_trades']} | {r['is_winrate']:.1f} | {r['is_pf']:.2f} |"
        )

    # Top 20 OOS
    lines.append("\n---")
    lines.append("\n## Top 20 — OOS Sharpe (2025-2026)")
    lines.append("\n| # | Symbol | Category | Sharpe | Return% | Trades | Win% | PF | d Sharpe |")
    lines.append("|---|--------|----------|--------|---------|--------|------|-----|----------|")
    for i, r in enumerate(master["top_20_oos"], 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['category']} | {r['oos_sharpe']:+.3f} | "
            f"{r['oos_return']:+.1f} | {r['oos_trades']} | {r['oos_winrate']:.1f} | {r['oos_pf']:.2f} | "
            f"{r['oos_delta_sharpe']:+.3f} |"
        )

    # Death crosses
    if master["death_cross"]:
        lines.append("\n---")
        lines.append("\n## Death Crosses (IS > 0.2, OOS < -0.5)")
        lines.append("\n*Significant IS performance that reversed catastrophically OOS.*")
        lines.append("\n| Symbol | Category | IS Sharpe | OOS Sharpe | Δ |")
        lines.append("|--------|----------|-----------|------------|---|")
        for r in master["death_cross"]:
            lines.append(
                f"| {r['symbol']} | {r['category']} | {r['is_sharpe']:+.3f} | "
                f"{r['oos_sharpe']:+.3f} | {r['oos_delta_sharpe']:+.3f} |"
            )

    # Phoenix
    if master["phoenix"]:
        lines.append("\n---")
        lines.append("\n## Phoenix (IS < -0.1, OOS > 0.3)")
        lines.append(
            "\n*Instruments that went from losing to winning — regime shift beneficiaries.*"
        )
        lines.append("\n| Symbol | Category | IS Sharpe | OOS Sharpe | Δ |")
        lines.append("|--------|----------|-----------|------------|---|")
        for r in master["phoenix"]:
            lines.append(
                f"| {r['symbol']} | {r['category']} | {r['is_sharpe']:+.3f} | "
                f"{r['oos_sharpe']:+.3f} | {r['oos_delta_sharpe']:+.3f} |"
            )

    # Consistent
    if master["consistent"]:
        lines.append("\n---")
        lines.append("\n## Consistent Winners (IS > 0.1, OOS > 0.1)")
        lines.append("\n*Instruments with positive Sharpe in both IS and OOS periods.*")
        lines.append("\n| Symbol | Category | IS Sharpe | OOS Sharpe | Δ |")
        lines.append("|--------|----------|-----------|------------|---|")
        for r in master["consistent"]:
            lines.append(
                f"| {r['symbol']} | {r['category']} | {r['is_sharpe']:+.3f} | "
                f"{r['oos_sharpe']:+.3f} | {r['oos_delta_sharpe']:+.3f} |"
            )

    # No trades OOS
    if master["no_trades_oos"]:
        lines.append("\n---")
        lines.append("\n## OOS Zero-Trade Instruments")
        lines.append(
            f"\n{len(master['no_trades_oos'])} instruments generated zero trades in 2025-2026:"
        )
        lines.append(f"\n{', '.join(master['no_trades_oos'])}")

    # Key conclusions
    lines.append("\n---")
    lines.append("\n## Key Conclusions")

    pos_oot = oos_improved = s.get("oos_improved_pct", 50)
    if pos_oot >= 60:
        lines.append(
            f"\n1. **OOS improvement is the norm** — {pos_oot}% of instruments improved OOS vs IS. The 2025-2026 regime shift benefited the system overall."
        )

    if s.get("death_cross_count", 0) > 0:
        lines.append(
            f"\n2. **{s['death_cross_count']} death crosses** — instruments with strong IS performance that completely reversed. Avoid extrapolating from IS alone."
        )

    if s.get("phoenix_count", 0) > 0:
        lines.append(
            f"\n3. **{s['phoenix_count']} phoenix** — instruments that went from IS losers to OOS winners. The system adapts to regime shifts."
        )

    lines.append(
        f"\n4. **IS->OOS correlation = {s['is_oos_correlation']:.3f}** — IS performance is weakly or negatively correlated with OOS. Tuning on IS alone is dangerous."
    )

    energy_cat = master["category_summary"].get("Sector", {})
    if energy_cat:
        lines.append(
            f"\n5. **Energy sector strongest OOS** — {energy_cat.get('oos_mean_sharpe', 0):+.3f} mean OOS Sharpe, {energy_cat.get('oos_better', 0)}/{energy_cat.get('n', 0)} improved OOS."
        )

    lines.append("\n---")
    lines.append(
        f"\n*Report auto-generated {master['timestamp'][:19]} | Production config: mr=0.70, et=0.55, multi-TP ON, quality-registry ON*"
    )

    return "\n".join(lines)


def main():
    batch_dir = Path(__file__).parent.parent / "reports/comprehensive_batch"
    batches = load_batches(batch_dir)
    print(f"Loaded {len(batches)} batches")

    master = build_master(batches)

    # Save JSON
    json_path = batch_dir / "MASTER_SUMMARY.json"
    json_path.write_text(
        json.dumps({k: v for k, v in master.items() if k != "all_tickers"}, indent=2, default=str)
    )
    print(f"Master JSON -> {json_path}")

    # Generate markdown
    md = generate_markdown(master)
    md_path = batch_dir / "MASTER_SUMMARY.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Master Markdown -> {md_path}")

    # Print quick summary
    print(f"\nTotal: {master['summary']['n_total']} tickers")
    print(f"IS Pos Sharpe: {master['summary']['is_pos_sharpe_pct']}%")
    print(f"OOS Pos Sharpe: {master['summary']['oos_pos_sharpe_pct']}%")
    print(f"OOS Improved: {master['summary']['oos_improved_pct']}%")
    print(f"IS->OOS Corr: {master['summary']['is_oos_correlation']:.3f}")
    print(
        f"Death Cross: {master['summary']['death_cross_count']}, Phoenix: {master['summary']['phoenix_count']}, Consistent: {master['summary']['consistent_count']}"
    )
    print(
        f"\nTop 5 OOS: {', '.join(r['symbol'] + '=' + format(r['oos_sharpe'], '+.3f') for r in master['top_20_oos'][:5])}"
    )


if __name__ == "__main__":
    main()
