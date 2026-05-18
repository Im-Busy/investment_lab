"""
Benchmark 8 regime detectors on SPY: HMM, GMM, PCA+KMeans, CNN, R2/RD,
Changepoint, Path Signature, Macro.

Compares: regime count, stability (avg duration, transitions/yr),
regime-conditional returns, inter-detector correlation.

Usage:
    uv run scripts/benchmark_regimes.py SPY --start 2015-01-01 --end 2024-12-31
    uv run scripts/benchmark_regimes.py SPY --detectors hmm,gmm,pca_kmeans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.hmm_regime import HMMRegimeDetector
from src.ml.gmm_regime import GMMRegimeDetector
from src.ml.pca_kmeans_regime import PCAKMeansRegimeDetector

DETECTOR_CLASSES: dict[str, type] = {
    "hmm": HMMRegimeDetector,
    "gmm": GMMRegimeDetector,
    "pca_kmeans": PCAKMeansRegimeDetector,
}


def _safe_import(name: str, class_name: str) -> type | None:
    try:
        mod = __import__(f"src.ml.{name}", fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception:
        return None


_OPTIONAL_CLASSES: dict[str, tuple[str, str]] = {
    "cnn": ("cnn_regime", "CNNRegimeDetector"),
    "r2_rd": ("r2_rd_regime", "R2RDRegimeDetector"),
    "change_point": ("change_point_regime", "ChangePointRegimeDetector"),
    "path_signature": ("path_signature_regime", "PathSignatureRegimeDetector"),
    "macro": ("macro_regime", "MacroRegimeDetector"),
}

for _key, (_mod, _cls) in _OPTIONAL_CLASSES.items():
    dt = _safe_import(_mod, _cls)
    if dt is not None:
        DETECTOR_CLASSES[_key] = dt


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.dropna()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0 if col == "Volume" else df.iloc[:, 0]
    df.columns = [c.capitalize() for c in df.columns]
    return df


def compute_regime_stability(labels: pd.Series) -> dict:
    """Compute regime stability metrics from a label series."""
    transitions = (labels != labels.shift(1)).sum() - 1
    n = len(labels)
    duration_by_regime = {}
    current = labels.iloc[0]
    run_start = 0
    runs: list[float] = []
    for i in range(1, n):
        if labels.iloc[i] != current:
            runs.append(i - run_start)
            duration_by_regime.setdefault(current, []).append(i - run_start)
            current = labels.iloc[i]
            run_start = i
    runs.append(n - run_start)
    duration_by_regime.setdefault(current, []).append(n - run_start)

    avg_duration = float(np.mean(runs)) if runs else 0.0
    transitions_per_year = transitions / (n / 252) if n >= 252 else 0.0
    freq = labels.value_counts().to_dict()
    freq_pct = {k: round(v / n * 100, 1) for k, v in freq.items()}
    regime_avg_duration = {k: float(np.mean(v)) for k, v in duration_by_regime.items()}

    return {
        "n_bars": n,
        "n_transitions": int(transitions),
        "transitions_per_year": round(transitions_per_year, 1),
        "avg_regime_duration_bars": round(avg_duration, 1),
        "regime_frequency_pct": freq_pct,
        "regime_avg_duration_bars": regime_avg_duration,
    }


def compute_regime_returns(labels: pd.Series, close: pd.Series) -> dict:
    """Compute forward 1d, 5d, 20d returns conditional on each regime."""
    index_union = labels.index.intersection(close.index)
    labels = labels.loc[index_union]
    close = close.loc[index_union]

    results: dict[str, dict[str, float]] = {}
    for regime in labels.unique():
        mask = labels == regime
        n_signals = mask.sum()
        if n_signals < 10:
            continue
        regime_close = close[mask]
        fwd = {
            "fwd_1d": close.shift(-1).loc[mask].mean(),
            "fwd_5d": close.shift(-5).loc[mask].mean(),
            "fwd_20d": close.shift(-20).loc[mask].mean(),
        }
        returns = {k: round((v / regime_close.mean() - 1) * 100, 2) for k, v in fwd.items()}
        returns["n_signals"] = int(n_signals)
        results[str(regime)] = returns
    return results


def compute_detector_correlation(labels_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise Cramer's V (chi-square) between detectors."""
    names = list(labels_dict.keys())
    n = len(names)
    corr = pd.DataFrame(np.eye(n), index=names, columns=names)
    for i in range(n):
        for j in range(i + 1, n):
            common = labels_dict[names[i]].index.intersection(labels_dict[names[j]].index)
            a = labels_dict[names[i]].loc[common]
            b = labels_dict[names[j]].loc[common]
            tbl = pd.crosstab(a, b)
            chi2 = float(
                np.sum((tbl.values - tbl.values.mean()) ** 2 / tbl.values.mean().clip(1e-10))
            )
            n_obs = len(common)
            min_dim = min(tbl.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n_obs * min_dim))) if min_dim > 0 else 0.0
            corr.iloc[i, j] = cramers_v
            corr.iloc[j, i] = cramers_v
    return corr


def _create_detector(name: str, detector_cls: type, n_states: int) -> object:
    """Create detector with appropriate args for each type."""
    if name in ("hmm",):
        return detector_cls(n_states=n_states, min_samples=50)
    elif name in ("gmm",):
        return detector_cls(n_range=(n_states, n_states + 2), min_samples=50)
    elif name in ("pca_kmeans",):
        return detector_cls(k_range=(n_states, n_states + 2), min_samples=50)
    else:
        return detector_cls(min_samples=50)


def benchmark_detector(
    name: str,
    detector_cls: type,
    features: pd.DataFrame,
    close: pd.Series,
    n_states: int = 4,
) -> dict | None:
    """Run a single detector benchmark."""
    try:
        detector = _create_detector(name, detector_cls, n_states)
        detector.fit(features)
    except Exception as e:
        return {"name": name, "error": str(e), "passed": False}

    try:
        labels = detector.predict(features)
        _probs = detector.predict_proba(features)
        summary = detector.get_regime_summary()
    except Exception as e:
        return {"name": name, "error": f"predict: {e}", "passed": False}

    stability = compute_regime_stability(labels)
    regime_returns = compute_regime_returns(labels, close)

    return {
        "name": name,
        "passed": True,
        "n_regimes": summary.n_regimes,
        "regime_labels": summary.regime_labels,
        "label_distribution": summary.label_distribution,
        "stability": stability,
        "regime_returns": regime_returns,
        "metadata": summary.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark regime detectors")
    parser.add_argument("symbol", default="SPY", nargs="?", help="Ticker symbol")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated list of detectors to test (default: all available)",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=4,
        help="Number of states for detectors (default: 4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: reports/regimes/benchmark.json)",
    )
    args = parser.parse_args()

    df = load_data(args.symbol)
    if args.start:
        df = df.loc[args.start :]
    if args.end:
        df = df.loc[: args.end]

    print(f"Loaded {args.symbol}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    extractor = FeatureExtractor()
    features = extractor.extract_all_features(df, include_forward_returns=False)
    features = features.dropna()
    close = df.loc[features.index, "Close"]
    print(f"Features: {features.shape[1]} columns, {len(features)} rows")

    if args.detectors:
        selected = [d.strip() for d in args.detectors.split(",") if d.strip() in DETECTOR_CLASSES]
    else:
        selected = list(DETECTOR_CLASSES.keys())

    print(f"Benchmarking {len(selected)} detectors: {selected}\n")

    results = {}
    labels_dict: dict[str, pd.Series] = {}
    for name in selected:
        cls = DETECTOR_CLASSES[name]
        print(f"  {name}...", end=" ", flush=True)
        result = benchmark_detector(name, cls, features, close, n_states=args.n_states)
        if result is None:
            print("SKIPPED")
            continue
        results[name] = result
        if result["passed"]:
            n_reg = result["n_regimes"]
            avg_dur = result["stability"]["avg_regime_duration_bars"]
            tpy = result["stability"]["transitions_per_year"]
            print(f"PASS - {n_reg} regimes, avg {avg_dur:.0f} bars, {tpy:.0f} transitions/yr")
            # Collect labels for correlation
            corr_detector = _create_detector(name, DETECTOR_CLASSES[name], args.n_states)
            corr_detector.fit(features)
            labels_dict[name] = corr_detector.predict(features)
        else:
            print(f"FAIL - {result['error']}")

    if len(labels_dict) >= 2:
        print("\nDetector correlation (Cramer's V):")
        corr = compute_detector_correlation(labels_dict)
        print(corr.to_string(float_format=lambda x: f"{x:.3f}"))
        results["_correlation"] = {k: v.to_dict() for k, v in corr.items()}

    # Summary report
    print("\n" + "=" * 70)
    print("REGIME-CONDITIONAL FORWARD RETURNS (20d)")
    print("=" * 70)
    for name, result in results.items():
        if name.startswith("_"):
            continue
        print(f"\n{name} ({result.get('n_regimes', '?')} regimes):")
        for regime, rets in result.get("regime_returns", {}).items():
            f20 = rets.get("fwd_20d", float("nan"))
            n = rets.get("n_signals", 0)
            print(f"  {regime:20s} | 20d ret={f20:+.2f}% | n={n}")

    out_path = args.output or f"reports/regimes/benchmark_{args.symbol}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            serializable[k] = v.to_dict()
        else:
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
