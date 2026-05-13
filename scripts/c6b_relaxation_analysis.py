"""
C6b — Edge Relaxation Analysis: test what fundamentally drives the model's edge.

Sequentially relaxes thresholds on volume, market cap, and volatility to find
the binding constraint. Uses the 33-ticker model to test on ALL available tickers.

Usage:
    uv run scripts/c6b_relaxation_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.pattern_classifier import PatternClassifier
from src.ml.walk_forward import walk_forward_per_ticker

MODEL_PATH = "models/pattern_classifier_v3_SPY_20260511_164601.pkl"
DATA_DIR = "data/raw"
OUTPUT_DIR = Path("reports/c6b_relaxation")
LOG_PATH = Path("docs/ticker-test-log.md")

SKIP_TICKERS = {"BTC_USD", "EURUSD_X", "IAU"}


def parse_fundamentals(log_path: Path) -> pd.DataFrame:
    """Parse fundamental data from ticker-test-log.md table rows."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if (
            line.startswith("|")
            and not line.startswith("|---")
            and not line.startswith("| Date")
            and not line.startswith("| #")
        ):
            parts = [p.strip() for p in line.split("|")][1:-1]
            if len(parts) < 9:
                continue
            date_str = parts[0]
            try:
                pd.Timestamp(date_str)
                has_date = True
            except (ValueError, TypeError):
                has_date = False
            if has_date and len(parts) >= 10:
                rows.append(
                    {
                        "date": parts[0],
                        "ticker": parts[1],
                        "name": parts[2],
                        "sector": parts[3],
                        "market_cap_str": parts[4],
                        "vol_m_str": parts[5],
                        "inst_pct_str": parts[6],
                        "history_str": parts[7],
                        "ann_vol_str": parts[8],
                        "ic_str": parts[9] if len(parts) > 9 else "",
                        "verdict": parts[10] if len(parts) > 10 else "",
                    }
                )
    df = pd.DataFrame(rows)

    def parse_dollar(s):
        s = str(s).strip().replace("$", "").replace(",", "")
        if not s or s in ("—", "", "nan"):
            return None
        try:
            if "B" in s:
                return float(s.replace("B", "")) * 1000
            elif "M" in s:
                return float(s.replace("M", ""))
            else:
                return float(s)
        except (ValueError, TypeError):
            return None

    def parse_pct(s):
        s = str(s).strip().replace("%", "")
        if not s or s in ("—", "", "nan"):
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def parse_years(s):
        s = str(s).strip().replace("yr", "")
        if not s or s in ("—", "", "nan"):
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    df["market_cap_m"] = df["market_cap_str"].apply(parse_dollar)
    df["daily_vol_m"] = df["vol_m_str"].apply(parse_dollar)
    df["inst_pct"] = df["inst_pct_str"].apply(parse_pct)
    df["history_years"] = df["history_str"].apply(parse_years)
    df["ann_vol_pct"] = df["ann_vol_str"].apply(parse_pct)
    df["ic_old"] = df["ic_str"].apply(parse_pct)
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    return df


def get_available_tickers(data_dir: str) -> list[str]:
    tickers = []
    for p in Path(data_dir).glob("*_daily.csv"):
        name = p.stem.replace("_daily", "")
        if name not in SKIP_TICKERS:
            tickers.append(name)
    return sorted(tickers)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("C6b — Edge Relaxation Analysis")
    print("=" * 100)

    print(f"\nParsing fundamentals from {LOG_PATH}...")
    fund_df = parse_fundamentals(LOG_PATH)
    print(f"  Parsed {len(fund_df)} unique tickers with fundamental data")

    print(f"\nLoading model: {MODEL_PATH}")
    model = PatternClassifier()
    model.load(MODEL_PATH)
    print(f"  Features: {len(model.feature_names_)}")

    available = get_available_tickers(DATA_DIR)
    print(f"\nTickers with data files: {len(available)}")

    print(f"\nRunning walk-forward on {len(available)} tickers...")
    wf_results = walk_forward_per_ticker(
        model=model,
        tickers=available,
        data_dir=DATA_DIR,
        initial_train_days=3 * 252,
        step_days=6 * 21,
        horizon=5,
    )

    wf_rows = []
    for t in available:
        if t in wf_results:
            wf_rows.append(
                {
                    "ticker": t,
                    "wf_ic": round(wf_results[t].mean_rank_ic, 4),
                    "wf_hit_rate": round(wf_results[t].mean_rank_ic, 4),
                    "wf_ic_std": 0.0,
                    "wf_n_steps": wf_results[t].n_steps,
                }
            )
    wf_df = pd.DataFrame(wf_rows)
    print(f"  Walk-forward complete: {len(wf_df)} tickers evaluated")

    merged = wf_df.merge(fund_df, on="ticker", how="left")
    valid = merged.dropna(subset=["wf_ic"])
    print(f"  Valid results: {len(valid)} tickers")

    # ---- Volume stratification ----
    print("\n" + "=" * 100)
    print("ANALYSIS 1: Daily Notional Volume vs Walk-Forward IC")
    print("=" * 100)

    vol_bins = [
        ("$0-$5M", 0, 5),
        ("$5M-$20M", 5, 20),
        ("$20M-$100M", 20, 100),
        ("$100M-$500M", 100, 500),
        ("$500M-$1B", 500, 1000),
        ("$1B+", 1000, 999999),
    ]
    vol_data = valid.dropna(subset=["daily_vol_m"])
    print(f"\n  Tickers with volume data: {len(vol_data)}")
    print(
        f"\n  {'Volume Bin':<20} {'N':>5} {'Mean IC':>10} {'Std IC':>10} {'Pct Pass':>10} {'Med IC':>10}"
    )
    for label, lo, hi in vol_bins:
        sub = vol_data[(vol_data["daily_vol_m"] >= lo) & (vol_data["daily_vol_m"] < hi)]
        if len(sub) == 0:
            continue
        print(
            f"  {label:<20} {len(sub):>5} {sub['wf_ic'].mean():>10.4f} {sub['wf_ic'].std():>10.4f} {(sub['wf_ic'] > 0.03).mean() * 100:>9.1f}% {sub['wf_ic'].median():>10.4f}"
        )

    # ---- Market Cap stratification ----
    print("\n" + "=" * 100)
    print("ANALYSIS 2: Market Cap vs Walk-Forward IC")
    print("=" * 100)
    mcap_bins = [
        ("$0-$200M", 0, 200),
        ("$200M-$1B", 200, 1000),
        ("$1B-$10B", 1000, 10000),
        ("$10B-$100B", 10000, 100000),
        ("$100B+", 100000, 9999999),
    ]
    mcap_data = valid.dropna(subset=["market_cap_m"])
    print(f"\n  Tickers with MC data: {len(mcap_data)}")
    for label, lo, hi in mcap_bins:
        sub = mcap_data[(mcap_data["market_cap_m"] >= lo) & (mcap_data["market_cap_m"] < hi)]
        if len(sub) == 0:
            continue
        print(
            f"  {label:<20} {len(sub):>5} {sub['wf_ic'].mean():>10.4f} {sub['wf_ic'].std():>10.4f} {(sub['wf_ic'] > 0.03).mean() * 100:>9.1f}% {sub['wf_ic'].median():>10.4f}"
        )

    # ---- Volatility stratification ----
    print("\n" + "=" * 100)
    print("ANALYSIS 3: Annualized Volatility vs Walk-Forward IC")
    print("=" * 100)
    vol_bins_pct = [
        ("<15% (low)", 0, 15),
        ("15-25% (med)", 15, 25),
        ("25-35% (high)", 25, 35),
        ("35-50% (v.high)", 35, 50),
        ("50%+ (extreme)", 50, 999),
    ]
    volpct_data = valid.dropna(subset=["ann_vol_pct"])
    print(f"\n  Tickers with vol data: {len(volpct_data)}")
    for label, lo, hi in vol_bins_pct:
        sub = volpct_data[(volpct_data["ann_vol_pct"] >= lo) & (volpct_data["ann_vol_pct"] < hi)]
        if len(sub) == 0:
            continue
        print(
            f"  {label:<20} {len(sub):>5} {sub['wf_ic'].mean():>10.4f} {sub['wf_ic'].std():>10.4f} {(sub['wf_ic'] > 0.03).mean() * 100:>9.1f}% {sub['wf_ic'].median():>10.4f}"
        )

    # ---- Sequential Relaxation ----
    print("\n" + "=" * 100)
    print("SEQUENTIAL RELAXATION: Cumulative Mean IC as volume threshold loosens")
    print("=" * 100)
    vol_sorted = vol_data.sort_values("daily_vol_m", ascending=False).copy()
    vol_sorted["cum_mean_ic"] = vol_sorted["wf_ic"].expanding().mean()
    vol_sorted["cum_n"] = range(1, len(vol_sorted) + 1)
    thresholds = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 0]
    prev_ic = None
    for thresh in thresholds:
        sub = vol_sorted if thresh == 0 else vol_sorted[vol_sorted["daily_vol_m"] >= thresh]
        if len(sub) == 0:
            continue
        cum_ic = sub["wf_ic"].mean()
        ch = f"{cum_ic - prev_ic:+.4f}" if prev_ic is not None else "--"
        print(f"  >= ${thresh}M/d  n={len(sub):>4}  cum_IC={cum_ic:.4f}  delta={ch}")
        prev_ic = cum_ic

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    sd = vol_data.dropna(subset=["daily_vol_m", "wf_ic"])
    ax.scatter(sd["daily_vol_m"], sd["wf_ic"], alpha=0.6, s=40)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03")
    ax.set_xscale("log")
    ax.set_xlabel("Daily Notional Volume ($M)")
    ax.set_ylabel("Walk-Forward Rank IC")
    ax.set_title("IC vs Daily Volume")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ms = mcap_data.dropna(subset=["market_cap_m", "wf_ic"])
    ax.scatter(ms["market_cap_m"], ms["wf_ic"], alpha=0.6, s=40)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03")
    ax.set_xscale("log")
    ax.set_xlabel("Market Cap ($M)")
    ax.set_ylabel("Walk-Forward Rank IC")
    ax.set_title("IC vs Market Cap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    vs = volpct_data.dropna(subset=["ann_vol_pct", "wf_ic"])
    ax.scatter(vs["ann_vol_pct"], vs["wf_ic"], alpha=0.6, s=40)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Walk-Forward Rank IC")
    ax.set_title("IC vs Volatility")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cum_data = []
    for thresh in [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0]:
        sub = vol_sorted if thresh == 0 else vol_sorted[vol_sorted["daily_vol_m"] >= thresh]
        if len(sub) >= 3:
            cum_data.append({"n": len(sub), "mean_ic": sub["wf_ic"].mean(), "thresh": thresh})
    cum_df = pd.DataFrame(cum_data)
    ax.plot(cum_df["n"], cum_df["mean_ic"], "o-", color="steelblue", lw=2, ms=6)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03")
    ax.set_xlabel("Number of Tickers (most liquid first)")
    ax.set_ylabel("Cumulative Mean IC")
    ax.set_title("Sequential Relaxation: Cum. Mean IC\n(most -> least liquid)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "relaxation_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    csv_path = OUTPUT_DIR / "relaxation_data.csv"
    valid.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    hi_vol = vol_sorted[vol_sorted["daily_vol_m"] >= 100]
    lo_vol = vol_sorted[vol_sorted["daily_vol_m"] < 10]
    print("\n  Volume cliff:")
    print(f"    High vol (>= $100M/d, n={len(hi_vol)}): mean IC = {hi_vol['wf_ic'].mean():.4f}")
    print(f"    Low vol (< $10M/d, n={len(lo_vol)}):   mean IC = {lo_vol['wf_ic'].mean():.4f}")

    # Correlation coefficients
    from scipy import stats

    for label, data, col in [
        ("Volume", vol_data, "daily_vol_m"),
        ("Market Cap", mcap_data, "market_cap_m"),
        ("Volatility", volpct_data, "ann_vol_pct"),
    ]:
        d = data.dropna(subset=[col, "wf_ic"])
        if len(d) > 5:
            r, p = stats.spearmanr(d[col], d["wf_ic"])
            print(f"    {label} spearman r = {r:+.3f} (p={p:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
