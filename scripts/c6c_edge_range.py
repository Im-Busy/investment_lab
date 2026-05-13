"""C6c — Edge Sweet Spot: granular volatility range testing using existing walk-forward data."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("reports/c6b_relaxation/relaxation_data.csv")
OUT = Path("reports/c6c_edge_range")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["ann_vol_pct", "wf_ic", "daily_vol_m"]).copy()

    print(f"Tickers with full data: {len(df)}")

    # ── 1. Granular volatility bins (2% steps) ──
    vol_min, vol_max = 12, 55
    step = 2
    bins = list(range(vol_min, vol_max + step, step))
    bin_edges = [(lo, lo + step) for lo in bins[:-1]]

    print("\n" + "=" * 90)
    print("1. GRANULAR VOLATILITY RANGE ANALYSIS (2% steps)")
    print("=" * 90)
    print(
        f"{'Vol Range':<15} {'N':>5} {'Mean IC':>9} {'Median IC':>9} {'Pct>0.03':>9} {'CumMean':>9}"
    )
    print("-" * 65)

    rows = []
    cum_n, cum_wt = 0, 0.0
    for lo, hi in bin_edges:
        mask = (df["ann_vol_pct"] >= lo) & (df["ann_vol_pct"] < hi)
        sub = df[mask]
        n = len(sub)
        if n == 0:
            continue
        mean_ic = sub["wf_ic"].mean()
        median_ic = sub["wf_ic"].median()
        pass_pct = (sub["wf_ic"] > 0.03).mean() * 100
        cum_n += n
        cum_wt += sub["wf_ic"].sum()
        cum_mean = cum_wt / cum_n
        marker = " <<< PEAK" if mean_ic > 0.05 else ""
        above = " >0.03" if cum_mean > 0.03 else " <0.03"
        print(
            f"{lo}-{hi}%{'':>7} {n:>5} {mean_ic:>9.4f} {median_ic:>9.4f} {pass_pct:>8.0f}% {cum_mean:>9.4f}{above}{marker}"
        )
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "mean_ic": mean_ic,
                "median_ic": median_ic,
                "pass_pct": pass_pct,
                "cum_mean": cum_mean,
            }
        )

    vol_bin_df = pd.DataFrame(rows)
    vol_bin_df.to_csv(OUT / "vol_bins_2pct.csv", index=False)

    # Find the cliff: last bin where mean_IC > 0.03
    cliff = vol_bin_df[vol_bin_df["mean_ic"] > 0.03]
    if not cliff.empty:
        last_good = cliff.iloc[-1]
        print(f"\nEdge cliff: mean IC drops below 0.03 after vol > {last_good['hi']}%")

    # Find peak
    peak = vol_bin_df.loc[vol_bin_df["mean_ic"].idxmax()]
    print(f"Peak: {peak['lo']}-{peak['hi']}% vol, mean IC = {peak['mean_ic']:.4f}")

    # ── 2. Cumulative IC vs expanding volatility range ──
    print("\n" + "=" * 90)
    print("2. CUMULATIVE: Expanding volatility inclusion (start from center, expand outward)")
    print("=" * 90)
    df_sorted = df.sort_values("ann_vol_pct")
    # Start from 15-25% (the known sweet spot), expand outward
    center = df[(df["ann_vol_pct"] >= 15) & (df["ann_vol_pct"] <= 25)]
    print(f"  Center (15-25%): n={len(center)}, mean_IC={center['wf_ic'].mean():.4f}")

    expansions = [(15, 25), (14, 26), (13, 28), (12, 30), (12, 35), (12, 40), (12, 45), (12, 55)]
    for lo, hi in expansions:
        sub = df[(df["ann_vol_pct"] >= lo) & (df["ann_vol_pct"] <= hi)]
        direction = "expand" if hi > 25 else "center"
        print(
            f"  [{lo}-{hi}%]: n={len(sub):>3}, mean_IC={sub['wf_ic'].mean():.4f}, pass={((sub['wf_ic'] > 0.03).mean() * 100):.0f}%"
        )

    # ── 3. Sector × Volatility interaction ──
    print("\n" + "=" * 90)
    print("3. SECTOR × VOLATILITY INTERACTION")
    print("=" * 90)
    df["vol_group"] = pd.cut(
        df["ann_vol_pct"], bins=[0, 20, 30, 40, 99], labels=["<20%", "20-30%", "30-40%", ">40%"]
    )
    pivot = df.pivot_table(
        values="wf_ic", index="sector", columns="vol_group", aggfunc=["mean", "count"]
    )
    # Flatten
    if not pivot.empty:
        for sector in df["sector"].dropna().unique():
            sec_data = df[df["sector"] == sector]
            if len(sec_data) >= 3:
                n = len(sec_data)
                ic = sec_data["wf_ic"].mean()
                vol = sec_data["ann_vol_pct"].mean()
                print(f"  {sector:<25} n={n:>2}  mean_IC={ic:.4f}  mean_vol={vol:.0f}%")

    # ── 4. Hard thresholds ──
    print("\n" + "=" * 90)
    print("4. HARD THRESHOLD TEST: what IC at strict vol cuts?")
    print("=" * 90)
    for cutoff in [20, 22, 25, 28, 30, 35]:
        sub = df[df["ann_vol_pct"] <= cutoff]
        n = len(sub)
        mean_ic = sub["wf_ic"].mean()
        pass_pct = (sub["wf_ic"] > 0.03).mean() * 100
        neg_pct = (sub["wf_ic"] < 0).mean() * 100
        print(
            f"  <= {cutoff}% vol: n={n:>3}  mean_IC={mean_ic:.4f}  pass={pass_pct:.0f}%  neg={neg_pct:.0f}%"
        )

    # ── 5. Plot: granular vol bins ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    centers = [(r["lo"] + r["hi"]) / 2 for _, r in vol_bin_df.iterrows()]
    ax.bar(
        centers,
        vol_bin_df["mean_ic"],
        width=1.8,
        color=["#2ecc71" if ic > 0.03 else "#e74c3c" for ic in vol_bin_df["mean_ic"]],
        edgecolor="white",
        alpha=0.85,
    )
    ax.axhline(y=0, color="gray", ls="-", alpha=0.5)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03 (PASS)")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Mean Walk-Forward IC")
    ax.set_title("Mean IC by 2% Volatility Bins")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        df["ann_vol_pct"],
        df["wf_ic"],
        c=df["daily_vol_m"].clip(upper=2000),
        cmap="viridis",
        alpha=0.7,
        s=60,
        edgecolors="black",
        linewidth=0.3,
    )
    ax.axhline(y=0, color="gray", ls="-", alpha=0.5)
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5)
    ax.axvline(x=25, color="orange", ls=":", alpha=0.8, label="vol=25%")
    ax.axvline(x=35, color="red", ls=":", alpha=0.8, label="vol=35% (cliff)")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Walk-Forward IC")
    ax.set_title("IC vs Volatility (color=volume size)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    cum_data = []
    df_by_vol = df.sort_values("ann_vol_pct")
    df_by_vol["cum_mean"] = df_by_vol["wf_ic"].expanding().mean()
    ax.fill_between(
        df_by_vol["ann_vol_pct"], 0, df_by_vol["cum_mean"], alpha=0.3, color="steelblue"
    )
    ax.plot(
        df_by_vol["ann_vol_pct"],
        df_by_vol["cum_mean"],
        "o-",
        color="steelblue",
        lw=2,
        ms=3,
        alpha=0.8,
    )
    ax.axhline(y=0.03, color="green", ls="--", alpha=0.5, label="IC=0.03")
    ax.set_xlabel("Volatility ≤ X% (sorted ascending)")
    ax.set_ylabel("Cumulative Mean IC")
    ax.set_title("Cumulative: include all tickers ≤ vol X%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "edge_sweet_spot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {OUT / 'edge_sweet_spot.png'}")

    # ── Summary ──
    print("\n" + "=" * 90)
    print("SUMMARY: The Edge Lives At...")
    print("=" * 90)
    sweet = df[df["ann_vol_pct"] <= 25]
    cliff_zone = df[(df["ann_vol_pct"] > 25) & (df["ann_vol_pct"] <= 35)]
    dead = df[df["ann_vol_pct"] > 35]
    print(
        f"  Sweet spot (vol <= 25%):  n={len(sweet)},  mean_IC={sweet['wf_ic'].mean():.4f},  pass={((sweet['wf_ic'] > 0.03).mean() * 100):.0f}%"
    )
    print(
        f"  Cliff zone (25-35%):      n={len(cliff_zone)},  mean_IC={cliff_zone['wf_ic'].mean():.4f},  pass={((cliff_zone['wf_ic'] > 0.03).mean() * 100):.0f}%"
    )
    print(
        f"  Dead zone (>35%):          n={len(dead)},  mean_IC={dead['wf_ic'].mean():.4f},  pass={((dead['wf_ic'] > 0.03).mean() * 100):.0f}%"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
