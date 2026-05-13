# %% [markdown]
# # Cross-Asset Feature Experiment — H14 Validation
#
# Runs a controlled experiment: train all 6 instruments (SPY, QQQ, CRVL, KODK, HIFS, JOE)
# **with** and **without** cross-asset features, then compare results.
#
# **Controls:**
# - Start/Stop button — run the full batch or cancel mid-experiment
# - Live progress per instrument
# - Side-by-side comparison table after completion
#
# **What to look for:**
# | Scenario | Test AUC | Overfit Gap | Cross-Asset in Top 10 | Interpretation |
# |----------|----------|-------------|------------------------|----------------|
# | A (Strong) | > 0.60 | < 0.15 | Yes | Learnable alpha found |
# | B (Honest) | Similar | Drops | Yes | Features reduce memorization |
# | C (No change) | Same | Same | No | Market context already captured |
# | D (Worse) | Drops | Any | Any | Cross-asset adds noise |

# %% [markdown]
# ## Configuration

# %%
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ipywidgets as widgets
import pandas as pd
from IPython.display import HTML, display

def _safe_display(*args, **kwargs):
    """display() wrapper that handles UnicodeEncodeError on Windows console."""
    try:
        display(*args, **kwargs)
    except UnicodeEncodeError:
        print("(Display skipped — Unicode characters not supported on this console)")

# ── Config ─────────────────────────────────────────────────────────────
INSTRUMENTS = [
    {"symbol": "data/raw/SPY_daily.csv", "label": "SPY", "horizon": 5},
    {"symbol": "data/raw/QQQ_daily.csv", "label": "QQQ", "horizon": 5},
    {"symbol": "data/raw/CRVL_daily.csv", "label": "CRVL", "horizon": 5},
    {"symbol": "data/raw/KODK_daily.csv", "label": "KODK", "horizon": 5},
    {"symbol": "data/raw/HIFS_daily.csv", "label": "HIFS", "horizon": 5},
    {"symbol": "data/raw/JOE_daily.csv", "label": "JOE", "horizon": 5},
]

# Experiment modes
MODES = [
    {"name": "With Cross-Asset", "flag": None, "suffix": "v3_ca"},
    {"name": "Without Cross-Asset", "flag": "--no-cross-asset", "suffix": "v3_baseline"},
]

TRAIN_SCRIPT = "scripts/train_ml_model_v2.py"
SUMMARY_DIR = Path("reports/ml_training")

# ── Shared State ────────────────────────────────────────────────────────
_run_lock = threading.Lock()
_running = False
_pending_stop = False
_results: List[Dict] = []
_status_widget: Optional[widgets.HTML] = None
_progress_widget: Optional[widgets.IntProgress] = None

# %% [markdown]
# ## Helper: Parse Summary JSON

# %%
def _load_latest_summaries() -> pd.DataFrame:
    """Scan reports/ml_training/ for training_summary_*.json files."""
    if not SUMMARY_DIR.exists():
        return pd.DataFrame()
    records = []
    for p in sorted(SUMMARY_DIR.glob("training_summary_*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            records.append({
                "file": p.name,
                "symbol": data.get("symbol", "?"),
                "suffix": _extract_suffix(data.get("run_id", "")),
                "model_type": data.get("model_type", ""),
                "horizon": data.get("horizon", ""),
                "n_samples": data.get("n_samples", 0),
                "n_features": data.get("n_features", 0),
                "ic_filtered": data.get("ic_filtered", False),
                "cross_asset": data.get("cross_asset_enabled", False),
                "train_auc": data.get("final_metrics", {}).get("train_auc"),
                "test_auc": data.get("final_metrics", {}).get("test_auc"),
                "overfit_gap": data.get("final_metrics", {}).get("overfit_gap"),
                "calibration_error": data.get("final_metrics", {}).get("calibration_error"),
                "cv_mean_test_auc": data.get("cv_metrics", {}).get("mean_test_auc"),
                "cv_std_test_auc": data.get("cv_metrics", {}).get("std_test_auc"),
                "cv_mean_overfit_gap": data.get("cv_metrics", {}).get("mean_overfit_gap"),
                "top_features": data.get("top_features", {}),
                "label_type": data.get("label_type", ""),
            })
        except Exception:
            pass
    return pd.DataFrame(records)


def _extract_suffix(run_id: str) -> str:
    """Extract suffix from run_id like '20260509_031858_v3_ca_test_catboost'."""
    parts = run_id.split("_")
    for p in parts:
        if p.startswith("v"):
            idx = parts.index(p)
            return "_".join(parts[idx:-1]) if len(parts) > idx + 1 else p
    return ""

# %% [markdown]
# ## Experiment Runner

# %%
def _run_single_training(
    instrument: Dict,
    mode: Dict,
    status_widget: widgets.HTML,
) -> Optional[Dict]:
    """Run train_ml_model_v2.py for one instrument x mode combination."""
    global _pending_stop

    cmd = [
        sys.executable, "-m", "uv", "run", TRAIN_SCRIPT,
        "--symbol", instrument["symbol"],
        "--horizon", str(instrument["horizon"]),
        "--suffix", f"{mode['suffix']}_{instrument['label']}",
    ]
    if mode["flag"]:
        cmd.append(mode["flag"])

    label = f"{instrument['label']} ({mode['name']})"
    start_time = datetime.now()
    _update_status(status_widget, f"Running: {label} ...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max per run
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        if _pending_stop:
            _update_status(status_widget, f"Stopped during: {label}")
            return None

        if result.returncode != 0:
            _update_status(status_widget, f"FAILED: {label} (exit code {result.returncode})")
            # Print tail of stderr for debugging
            tail = result.stderr.strip().split("\n")[-5:]
            for line in tail:
                print(f"  [stderr] {line}")
            return None

        # Find the most recently created summary JSON
        summaries = sorted(
            SUMMARY_DIR.glob("training_summary_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if summaries:
            with open(summaries[0]) as f:
                data = json.load(f)
            entry = {
                "instrument": instrument["label"],
                "mode": mode["name"],
                "suffix": mode["suffix"],
                "elapsed_sec": round(elapsed, 1),
                "train_auc": data.get("final_metrics", {}).get("train_auc"),
                "test_auc": data.get("final_metrics", {}).get("test_auc"),
                "overfit_gap": data.get("final_metrics", {}).get("overfit_gap"),
                "calibration_error": data.get("final_metrics", {}).get("calibration_error"),
                "cv_mean_test_auc": data.get("cv_metrics", {}).get("mean_test_auc"),
                "cv_std_test_auc": data.get("cv_metrics", {}).get("std_test_auc"),
                "cv_mean_overfit_gap": data.get("cv_metrics", {}).get("mean_overfit_gap"),
                "n_features": data.get("n_features", 0),
                "n_samples": data.get("n_samples", 0),
                "top_features": list(data.get("top_features", {}).keys())[:5],
                "summary_file": summaries[0].name,
            }
            _update_status(status_widget, f"Done: {label} ({elapsed:.0f}s)")
            return entry
        else:
            _update_status(status_widget, f"No summary found for: {label}")
            return None

    except subprocess.TimeoutExpired:
        _update_status(status_widget, f"TIMEOUT: {label} (>30 min)")
        return None
    except Exception as e:
        _update_status(status_widget, f"ERROR: {label}: {e}")
        return None


def _update_status(w: widgets.HTML, msg: str) -> None:
    w.value = f"<pre style='color:#e0e0e0;'>{msg}</pre>"


def _run_experiment(
    start_btn: widgets.Button,
    stop_btn: widgets.Button,
    status: widgets.HTML,
    progress: widgets.IntProgress,
    results_table: widgets.Output,
) -> None:
    """Main experiment loop — runs in background thread."""
    global _running, _pending_stop, _results

    with _run_lock:
        if _running:
            return
        _running = True
        _pending_stop = False

    try:
        start_btn.disabled = True
        stop_btn.disabled = False
        _results = []

        total = len(INSTRUMENTS) * len(MODES)
        progress.max = total
        progress.value = 0
        results_table.clear_output()
        _update_status(status, "Starting experiment...")

        for instrument in INSTRUMENTS:
            if _pending_stop:
                _update_status(status, "Experiment stopped by user.")
                break

            for mode in MODES:
                if _pending_stop:
                    break

                entry = _run_single_training(instrument, mode, status)
                if entry is not None:
                    _results.append(entry)

                progress.value += 1
                # Brief pause to let UI breathe
                time.sleep(0.5)

        if not _pending_stop:
            _update_status(status, "Experiment complete!")
        _display_results(results_table)

    finally:
        _running = False
        start_btn.disabled = False
        stop_btn.disabled = True


def _start_experiment(
    start_btn: widgets.Button,
    stop_btn: widgets.Button,
    status: widgets.HTML,
    progress: widgets.IntProgress,
    results_table: widgets.Output,
) -> None:
    """Launch experiment in background thread."""
    thread = threading.Thread(
        target=_run_experiment,
        args=(start_btn, stop_btn, status, progress, results_table),
        daemon=True,
    )
    thread.start()


def _stop_experiment(
    start_btn: widgets.Button,
    stop_btn: widgets.Button,
    status: widgets.HTML,
) -> None:
    """Signal the experiment loop to stop after current run."""
    global _pending_stop
    _pending_stop = True
    _update_status(status, "Stop requested — finishing current instrument...")
    stop_btn.disabled = True

# %% [markdown]
# ## Results Display

# %%
def _display_results(results_table: widgets.Output) -> None:
    """Render comparison table from experiment results."""
    results_table.clear_output()
    with results_table:
        if not _results:
            print("No results yet.")
            return

        df = pd.DataFrame(_results)

        # Build comparison: each instrument gets one row with CA / no-CA columns
        instruments = sorted(df["instrument"].unique())
        rows = []
        for inst in instruments:
            sub = df[df["instrument"] == inst]
            ca = sub[sub["mode"] == "With Cross-Asset"]
            no = sub[sub["mode"] == "Without Cross-Asset"]

            row = {"Instrument": inst}
            for prefix, data in [("CA", ca), ("NoCA", no)]:
                if data.empty:
                    continue
                d = data.iloc[0]
                row[f"{prefix}_Train AUC"] = _fmt(d.get("train_auc"))
                row[f"{prefix}_Test AUC"] = _fmt(d.get("test_auc"))
                row[f"{prefix}_Overfit Gap"] = _fmt(d.get("overfit_gap"))
                row[f"{prefix}_CV AUC (±)"] = (
                    f"{_fmt(d.get('cv_mean_test_auc'))} ± {_fmt(d.get('cv_std_test_auc'))}"
                )
                row[f"{prefix}_Top Features"] = ", ".join(d.get("top_features", [])[:3])

            rows.append(row)

        comparison = pd.DataFrame(rows)

        # Calculate deltas
        deltas = []
        for inst in instruments:
            sub_ca = df[(df["instrument"] == inst) & (df["mode"] == "With Cross-Asset")]
            sub_no = df[(df["instrument"] == inst) & (df["mode"] == "Without Cross-Asset")]
            if sub_ca.empty or sub_no.empty:
                continue
            ca_auc = sub_ca.iloc[0].get("test_auc", 0)
            no_auc = sub_no.iloc[0].get("test_auc", 0)
            ca_gap = sub_ca.iloc[0].get("overfit_gap", 0)
            no_gap = sub_no.iloc[0].get("overfit_gap", 0)
            auc_delta = ca_auc - no_auc if ca_auc and no_auc else None
            gap_delta = ca_gap - no_gap if ca_gap and no_gap else None

            scenario = "—"
            if auc_delta is not None and gap_delta is not None:
                if auc_delta > 0.02 and ca_gap < 0.15:
                    scenario = "A (Strong)"
                elif gap_delta < -0.02:
                    scenario = "B (Honest)"
                elif abs(auc_delta) < 0.02 and abs(gap_delta) < 0.02:
                    scenario = "C (No change)"
                elif auc_delta < -0.02:
                    scenario = "D (Worse)"

            deltas.append({
                "Instrument": inst,
                "AUC Δ": f"{auc_delta:+.3f}" if auc_delta is not None else "N/A",
                "Gap Δ": f"{gap_delta:+.3f}" if gap_delta is not None else "N/A",
                "Scenario": scenario,
            })

        delta_df = pd.DataFrame(deltas)

        _safe_display(HTML("<h3>Full Comparison</h3>"))

        _safe_display(comparison)

        _safe_display(HTML("<h3>Delta Summary</h3>"))

        _safe_display(delta_df)

        _safe_display(HTML("<h3>Raw Results</h3>"))

        _safe_display(df.drop(columns=["top_features", "summary_file"], errors="ignore"))

        # Save to CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = SUMMARY_DIR / f"cross_asset_experiment_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

# %% [markdown]
# ## Experiment Dashboard
#
# **How to use:**
# 1. Click **▶ Start Experiment** to begin
# 2. Training runs sequentially: 6 instruments × 2 modes = 12 runs (~60-90 min total)
# 3. Click **■ Stop** to cancel after the current run finishes
# 4. Results appear in the table below automatically
# 5. To view results from a previous run, use the "Load Previous" tab

# %%
# ── Build UI ─────────────────────────────────────────────────────────────
start_btn = widgets.Button(
    description="▶ Start Experiment",
    button_style="success",
    icon="play",
)
stop_btn = widgets.Button(
    description="■ Stop",
    button_style="danger",
    icon="stop",
    disabled=True,
)

status = widgets.HTML(
    value="<pre style='color:#888;'>Ready. Click Start to begin.</pre>"
)

progress = widgets.IntProgress(
    value=0,
    min=0,
    max=len(INSTRUMENTS) * len(MODES),
    description="Progress:",
    bar_style="info",
    style={"bar_color": "#4a9eff"},
)

results_output = widgets.Output()

# Wire up callbacks
start_btn.on_click(
    lambda _: _start_experiment(start_btn, stop_btn, status, progress, results_output)
)
stop_btn.on_click(
    lambda _: _stop_experiment(start_btn, stop_btn, status)
)

# ── Display ──────────────────────────────────────────────────────────────
try:
    _safe_display(
        widgets.HBox([start_btn, stop_btn]),
        status,
        progress,
        results_output,
    )
except (UnicodeEncodeError, RuntimeError):
    print("(Widgets skipped — running outside Jupyter/IPython)")
    print("Results output available in `results_output` variable")

# %% [markdown]
# ## Load Previous Results
#
# Use this cell to review results from a prior experiment without re-running.

# %%
def show_previous_summaries() -> pd.DataFrame:
    """Display a table of all training summary JSONs in reports/ml_training/."""
    df = _load_latest_summaries()
    if df.empty:
        print("No training summaries found.")
        return df

    # Show recent runs
    recent = df.sort_values("file", ascending=False).head(20)
    cols = ["symbol", "suffix", "cross_asset", "train_auc", "test_auc",
            "overfit_gap", "cv_mean_test_auc", "n_features"]
    _safe_display(recent[cols].style.format({
        "train_auc": "{:.4f}",
        "test_auc": "{:.4f}",
        "overfit_gap": "{:.4f}",
        "cv_mean_test_auc": "{:.4f}",
    }))
    return df


_prev_df = show_previous_summaries()
