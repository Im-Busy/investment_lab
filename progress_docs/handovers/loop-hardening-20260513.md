# Handover: Autonomous Loop Hardening — 5-Point Implementation

**Source:** Architecture re-evaluation 2026-05-13 after implementing + bugfixing the autonomous training loop (`scripts/autonomous_train_loop.py`, 1418 lines).

**Context:** The loop wraps `train_ml_pipeline_v3.py`, `run_ml_backtest.py`, and `tune_model.py` into a 5-phase orchestration with independence clustering, consecutive confirmation, cross-group OOS testing, and Optuna Bayesian sweep. The re-evaluation found the architecture is sound — no redesign needed. 5 hardening items remain (all additive, ~200 lines total, 2 files).

---

## LH-1: Checkpointing + Resume (`--resume`)

**Priority:** P0
**Problem:** If loop crashes at iteration 15/20 or `--timeout-hours` expires mid-iteration, all progress is lost. Each iteration (train + backtest) takes ~30-60s.
**Impact:** Prevents compute waste. ~60 lines.

### Implementation Steps

**File:** `scripts/autonomous_train_loop.py`

1. Add checkpoint constant near line 54 (after `LOOP_LOG_PATH`):
```python
CHECKPOINT_PATH = Path("reports/autonomous_loop/checkpoint.json")
```

2. Add `save_checkpoint()` function after `_finish_mlflow_run()`:
```python
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
```

3. Add `load_checkpoint()`:
```python
def load_checkpoint() -> dict | None:
    """Load saved loop state. Returns None if no checkpoint exists."""
    if not CHECKPOINT_PATH.exists():
        return None
    return json.loads(CHECKPOINT_PATH.read_text())
```

4. Call `save_checkpoint(state)` inside `run_phase_4_refinement_loop()` — insert after line 920 (`_write_loop_log(log_entry)`):
```python
save_checkpoint(state)
```

5. At start of `run_phase_4_refinement_loop()`, check for resume (needs new `resume: bool = False` parameter):
```python
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
        # Skip iterations already completed
        while state.iteration < max_iterations:
            # ... existing iteration loop ...
            state.iteration += 1
        return state
```

6. Add `--resume` CLI argument and wire to Phase 4 call. After `--skip-phase-4` (line 1312):
```python
parser.add_argument(
    "--resume", action="store_true",
    help="Resume Phase 4 from last checkpoint",
)
```

7. Wire to Phase 4 call (line 1373 area):
```python
resume=args.resume,
```

8. Clear checkpoint on successful lock (inside the `action == "locked"` block, before `break`):
```python
if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
```

### Test Command
```bash
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --fast --max-iterations 5 --trail-stop --start 2020-01-01 --end 2023-12-31
# Ctrl+C mid-run, then:
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --resume --fast --max-iterations 5 --trail-stop
# Should resume from where it stopped
```

---

## LH-2: Multi-Objective Pareto Optimization (`--pareto`)

**Priority:** P0
**Problem:** Sharpe-only optimization converges to a single local optimum regardless of operator risk preference. Pareto gives the operator a frontier to choose from.
**Impact:** ~40 lines in `_optuna_sweep()`.

### Implementation Steps

**File:** `scripts/autonomous_train_loop.py`

1. Modify `_optuna_sweep()` — replace single-objective study with multi-objective. The key changes are in `objective()` and `optuna.create_study()`:

Replace the `create_study()` call:
```python
# OLD (single objective):
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
)

# NEW (multi-objective):
study = optuna.create_study(
    directions=["maximize", "minimize", "maximize"],  # Sharpe, |MaxDD|, WinRate
    sampler=optuna.samplers.TPESampler(seed=42),
)
```

Replace the return value in `objective()`:
```python
# OLD:
return float(sharpe)

# NEW:
return (
    float(sharpe),                             # obj 1: maximize Sharpe
    float(abs(agg.get("mean_max_dd", 0))),     # obj 2: minimize drawdown magnitude
    float(agg.get("mean_win_rate", 0) / 100.0), # obj 3: maximize win rate
)
```

Update penalty return for `< 20 trades`:
```python
return (-999.0, 999.0, 0.0)
```

2. After `study.optimize()`, extract Pareto frontier:
```python
# Replace the single best_params extraction with Pareto extraction
pareto_front = []
for trial in study.best_trials:  # Optuna returns non-dominated trials for multi-objective
    if trial.values[0] > -999:  # skip failed trials
        pareto_front.append({
            "params": trial.params,
            "sharpe": trial.values[0],
            "max_dd": -trial.values[1],  # negate back
            "win_rate": trial.values[2] * 100,
        })

# Log Pareto frontier
logger.info(f"Pareto frontier: {len(pareto_front)} non-dominated solutions")
for i, sol in enumerate(sorted(pareto_front, key=lambda x: x["sharpe"], reverse=True)[:5]):
    logger.info(f"  #{i+1}: {sol['params']} → Sharpe={sol['sharpe']:.3f}, DD={sol['max_dd']:.1f}%, WR={sol['win_rate']:.1f}%")
```

3. Add `--pareto` CLI flag:
```python
parser.add_argument(
    "--pareto", action="store_true",
    help="Use multi-objective Pareto optimization (Sharpe + MaxDD + WinRate)",
)
```

4. Wire to `_optuna_sweep()` call with a new `pareto: bool = False` parameter.

### Test Command
```bash
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 5 --fast --model models/pattern_classifier_v3_SPY_20260513_121802.pkl --optuna-trials 10 --pareto --start 2020-01-01 --end 2023-12-31
```

---

## LH-3: Auto ETF-Component Cross-Asset Exclusion

**Priority:** P1
**Problem:** Current ETF leakage detection only logs a warning. If operator runs `SPY,MSFT`, cross-asset features embed look-ahead (SPY is ~7% MSFT). The feature columns leak future information.
**Impact:** ~40 lines in loop + small pipeline patch. Closes the biggest look-ahead gap.

### Implementation Steps

**File 1:** `scripts/autonomous_train_loop.py`

1. Add `build_exclusion_pairs()` function after `check_etf_leakage()`:
```python
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
```

2. In `run_phase_1_ticker_selection()`, after the leakage warnings, build exclusion pairs and pass them forward. Add to the return or save to a file:

```python
# After leakage_warnings loop:
exclusion_pairs = build_exclusion_pairs(all_tickers)
if exclusion_pairs:
    logger.info(f"Cross-asset exclusions: {exclusion_pairs}")
    # Save to file for pipeline to read
    ex_path = Path("reports/autonomous_loop/exclusion_pairs.json")
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ex_path, "w") as f:
        json.dump([[a, b] for a, b in exclusion_pairs], f)
```

**File 2:** `scripts/train_ml_pipeline_v3.py`

3. In `run_pipeline()`, after feature engineering (the cross-asset feature merge section, approximately line 830-870), add code to read exclusion pairs and drop the leaking feature columns:

```python
# After cross-asset feature computation, before IC filter
exclusion_path = Path("reports/autonomous_loop/exclusion_pairs.json")
if exclusion_path.exists():
    with open(exclusion_path) as f:
        exclusion_pairs = json.load(f)
    for ta, tb in exclusion_pairs:
        # Drop cross-asset features between the pair
        # Feature naming convention: e.g., "rel_ret_{tb}_5d", "beta_{tb}_20d"
        for t in [ta, tb]:
            cols_to_drop = [c for c in X_combined.columns if f"_{t}_" in c and c.startswith(("rel_ret_", "beta_", "corr_"))]
            if cols_to_drop:
                logger.warning(f"Dropping {len(cols_to_drop)} cross-asset features for {ta}-{tb} exclusion")
                X_combined = X_combined.drop(columns=cols_to_drop)
```

### Test Command
```bash
uv run scripts/autonomous_train_loop.py --tickers "SPY,MSFT,QQQ,AAPL" --phase 1 --start 2020-01-01 --end 2023-12-31
# Check reports/autonomous_loop/exclusion_pairs.json for [["SPY","MSFT"],["QQQ","MSFT"],["SPY","AAPL"],["QQQ","AAPL"]]
```

---

## LH-4: Next-Bar-Direction Label (`--label-type next_bar`)

**Priority:** P1
**Problem:** Triple-barrier labels embed `horizon` bars of future price data. Even PurgedKFold can't eliminate the leakage when ticker returns are correlated. A pure next-bar-direction label has ZERO look-ahead and provides a clean baseline.
**Impact:** ~30 lines. If comparable OOS to triple-barrier, the leakage was material.

### Implementation Steps

**File 1:** `scripts/train_ml_pipeline_v3.py`

1. Add `--label-type` parameter to `run_pipeline()`:
```python
def run_pipeline(
    ...
    label_type: str = "triple_barrier",  # "triple_barrier" or "next_bar"
    ...
) -> dict[str, Any]:
```

2. In the label generation section (approximately line 820-840, where triple-barrier labels are created), add the alternative:
```python
if label_type == "next_bar":
    # Pure next-bar direction: 1 if tomorrow's close > today's, else 0
    # ZERO look-ahead — label uses only t+1 data
    y = (df_pooled["Close"].shift(-1) > df_pooled["Close"]).astype(int)
    y = y.dropna()
    X_filtered = X_filtered.loc[y.index]
    logger.info(f"Next-bar labels: {len(y)} samples, {y.mean():.1%} positive")
else:
    # Existing triple-barrier code
```

3. Log label type in config dict (line 953 area):
```python
"label_type": label_type,
```

**File 2:** `scripts/autonomous_train_loop.py`

4. Add `--label-type` CLI argument. After `--entry-threshold` (line 1305):
```python
parser.add_argument(
    "--label-type", type=str, default="triple_barrier",
    choices=["triple_barrier", "next_bar"],
    help="Label type: triple_barrier (forward horizon) or next_bar (no look-ahead)",
)
```

5. Pass `label_type` through Phase 4 and Phase 3 `run_pipeline()` calls.

### Test Command
```bash
# Quick test of next-bar labels vs triple-barrier on SPY
uv run python -c "
from scripts.train_ml_pipeline_v3 import run_pipeline
r1 = run_pipeline(['SPY'], start='2020-01-01', end='2023-12-31', fast=True, label_type='triple_barrier')
r2 = run_pipeline(['SPY'], start='2020-01-01', end='2023-12-31', fast=True, label_type='next_bar')
print(f'Triple-barrier CV AUC: {r1[\"cv_results\"][\"mean_test_auc\"]:.4f}')
print(f'Next-bar CV AUC:      {r2[\"cv_results\"][\"mean_test_auc\"]:.4f}')
"
```

---

## LH-5: Phase Completion Tracking (`phase_state.json`)

**Priority:** P2
**Problem:** Running `--phase 5` alone requires manually passing `--model` and `--tickers` that Phases 1-4 produced. No metadata tracking.
**Impact:** ~30 lines. Pure QoL.

### Implementation Steps

**File:** `scripts/autonomous_train_loop.py`

1. Add constant (near line 53):
```python
PHASE_STATE_PATH = Path("reports/autonomous_loop/phase_state.json")
```

2. Add `save_phase_state()` function:
```python
def save_phase_state(phase: int, **kwargs) -> None:
    """Track which phases completed with which outputs."""
    PHASE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if PHASE_STATE_PATH.exists():
        state = json.loads(PHASE_STATE_PATH.read_text())
    state[f"phase_{phase}"] = {"timestamp": datetime.now().isoformat(), **kwargs}
    with open(PHASE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
```

3. Call after each phase completes in `main()`:
```python
# After Phase 1:
save_phase_state(1, groups=groups, tickers=list(dict.fromkeys(tickers)))

# After Phase 3:
save_phase_state(3, model_path=cross_group_result.get("model_path", ""))

# After Phase 4:
save_phase_state(4, model_path=state.best_model_path, best_config=state.best_config,
                 best_sharpe=state.best_sharpe, iterations=state.iteration)

# After Phase 5:
save_phase_state(5, sweep_results=sweep_results[-1] if sweep_results else {})
```

4. Auto-load model from phase state when `--model` not provided:
```python
# In main(), before Phase 4:
if not args.model and not cross_group_result.get("model_path"):
    phase_state = json.loads(PHASE_STATE_PATH.read_text()) if PHASE_STATE_PATH.exists() else {}
    if "phase_3" in phase_state:
        initial_model = phase_state["phase_3"].get("model_path", "")
```

### Test Command
```bash
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 1,2,3 --fast --start 2020-01-01 --end 2023-12-31
cat reports/autonomous_loop/phase_state.json
```

---

## Dependency Order

```
LH-1 (checkpointing) → first, everything else depends on it for resilience
LH-2 (Pareto)        → independent, can implement in parallel
LH-3 (ETF exclusion)  → independent, can implement in parallel
LH-4 (next-bar label) → independent, can implement in parallel
LH-5 (phase tracking) → last, depends on all phases being stable
```

Recommended execution: LH-1 → LH-2 → LH-3 → LH-4 → LH-5.

## File Map

| File | Lines affected | What changes |
|------|---------------|-------------|
| `scripts/autonomous_train_loop.py` | ~185 total across all 5 items | Checkpointing functions, Pareto Optuna, ETF exclusion builder, next-bar label flag, phase state functions, CLI arguments |
| `scripts/train_ml_pipeline_v3.py` | ~15 in LH-3, ~15 in LH-4 | Cross-asset exclusion drop + next-bar label generation |

All changes are additive — no existing behavior is modified, only new code paths activated by CLI flags (`--resume`, `--pareto`, `--label-type`).

## Pre-Flight Checklist

Before implementing, verify the baseline still works:
```bash
uv run scripts/autonomous_train_loop.py --tickers "SPY,QQQ" --phase 4 --fast --max-iterations 1 --trail-stop --start 2020-01-01 --end 2023-12-31
# Should complete ~30s with 303+ trades, Sharpe ~0.20
```
