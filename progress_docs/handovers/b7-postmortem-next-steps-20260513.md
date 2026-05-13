# Handover: B7.5 Post-Mortem — Qlib Evaluation Results & Next-Step Investigation

**Source:** Phase 6a — B7 (Qlib ADARNN/ADD concept-drift evaluation) completed 2026-05-13.
**Decision gate:** "Neither works on OOS" → re-evaluate feature set and approach.

---

## B7 Recap: What Was Done

### B7.1-B7.2 (Prior session): Infrastructure
- Cloned Qlib to `useful_resources/useful_repos/Qlib/` (pyqlib 0.9.8.dev31, editable install)
- Converted SPY CSV → Qlib bin format at `data/qlib_bin/`
- Verified Alpha158 handler: 158 features, 2515 bars (2015-2024), labels compute correctly

### B7.3: CatBoost Qlib Baseline
- **Config:** `useful_resources/useful_repos/Qlib/examples/benchmarks/SPY/workflow_config_catboost_Alpha158_SPY.yaml`
- Train: 2015-2019, Valid: 2020, Test: 2021-2024
- **Result:** Trained successfully (16 iters, RMSE=0.021, early-stopped). BUT:
  - Model predictions are near-zero (0.0004 range) — label is raw next-day return (mean ~0.0006)
  - `TopkDropoutStrategy(topk=1, n_drop=0)` generates **0 positions** (PA=0.0, POS=0.0)
  - Excess return vs SPY B&H: -16.1% annualized, IR=-1.42
  - Qlib cross-sectional IC is NaN (cross-sectional correlation on 1 stock = undefined)

### B7.4: ADARNN / ADD
- Both models **fail at data-loading stage** — internal dataloaders hardcoded for Alpha360 format
- **ADARNN** (`pytorch_adarnn.py:347`): `reshape(array, (6, 60))` — expects 6 OHLCV features × 60 time steps
- **ADD** (`pytorch_add.py:517`): `x[:, 1, :]` dimension mismatch — expects multi-dimensional Alpha360 sequences
- Both architectures initialize correctly (0.94MB AdaRNN, 0.41MB ADDModel) but crash during training
- **Root cause:** ADARNN/ADD are specifically designed for Alpha360 (sequential OHLCV: last 60 days × 6 features). Alpha158's flat 158 cross-sectional features are incompatible.

### Bugs Fixed During B7
| File | Fix |
|------|-----|
| `useful_resources/useful_repos/Qlib/qlib/workflow/recorder.py:376` | `out.decode()` → `out.decode("utf-8", errors="replace")` — binary git diffs crashed Unicode |
| CatBoost YAML | Removed `grow_policy: Lossguide` and `bootstrap_type: Poisson` (GPU-only, CPU incompatible) |
| All 3 YAMLs | `CSZScoreNorm` → removed. Cross-sectional Z-score on single stock (mean=x, std=0) = NaN |
| All 3 YAMLs | Added `benchmark: SPY` — default `SH000300` doesn't exist in US data |
| ADARNN/ADD YAMLs | `lr: 1e-3` → `lr: 0.001` — YAML scientific notation parsed as string by qlib's ruamel.yaml |

### Files to Reference
| File | Purpose |
|------|---------|
| `useful_resources/useful_repos/Qlib/examples/benchmarks/SPY/workflow_config_catboost_Alpha158_SPY.yaml` | Working CatBoost Qlib config |
| `useful_resources/useful_repos/Qlib/examples/benchmarks/SPY/workflow_config_adarnn_Alpha158_SPY.yaml` | ADARNN config (data-load-fail) |
| `useful_resources/useful_repos/Qlib/examples/benchmarks/SPY/workflow_config_add_Alpha158_SPY.yaml` | ADD config (data-load-fail) |
| `scripts/run_qlib_workflows.py` | Runner script — uses `qlib.model.trainer.task_train()` |
| `reports/qlib/comparison.json` | B7.5 comparison report (json) |
| `reports/qlib/qlib_results.json` | Full results from all 3 models |
| `reports/qlib/qlib_catboost_result.json` | CatBoost-specific metrics |
| `progress_docs/current.md:7` | B7.3-B7.5 session log entry |
| `progress_docs/handovers/b6-b7-retrain-qlib-20260513.md` | Original B6/B7 implementation spec (decision gate at line 168) |

---

## Key Technical Findings

### 1. Qlib is a Multi-Stock Framework — Not Single-Stock Friendly
Every component assumes multiple stocks:
- **Cross-sectional IC** (`calc_ic`): groups by date, computes corr across stocks. 1 stock = NaN
- **CSRankNorm/CSZScoreNorm**: normalize labels across all stocks at each time step. 1 stock = degenerate
- **TopkDropoutStrategy**: ranks stocks by prediction, picks top/bottom N. 1 stock = always 0 or 1 position
- **Benchmarks**: default to `SH000300` (CSI 300 index), US data has no equivalent

### 2. ADARNN/ADD Are Alpha360-Only
Their internal dataloaders hardcode:
- Feature reshape: `(n_stocks × 6, 60)` — requires exactly 6 OHLCV features × 60 time steps
- The `d_feat` parameter changes model input dim but NOT the dataloader reshape logic
- To use ADARNN/ADD: must switch data handler from `Alpha158` → `Alpha360`

### 3. CatBoost Predictions Near-Zero (0.0004 Range)
The label is raw next-day return `Ref($close, -2)/Ref($close, -1) - 1` with mean ~0.0006.
CatBoost with RMSE loss trains to predict these tiny values. The model achieved RMSE=0.021,
meaning predictions cluster tightly near zero. The strategy needs a stronger signal to trigger.

---

## Next-Step Investigation Options

### Option A: Switch to Alpha360 for ADARNN/ADD (Pursue Concept-Drift Models)
```
# Create new Alpha360 YAML configs with d_feat=6
# Copy from existing ADARNN/ADD Alpha360 templates, adapt region/dates/market
```
- **Pros:** Tests whether temporal distribution matching (ADARNN) improves over CatBoost
- **Cons:** Alpha360 uses only raw OHLCV (6 features × 60 lags) — no engineered factors. Different feature set from our V3 model (88 features).

### Option B: Diagnose CatBoost Near-Zero Predictions
Investigate why Qlib CatBoost predictions are near-zero:
```bash
# Check raw label distribution in Qlib data
uv run python -c "
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
qlib.init(provider_uri='data/qlib_bin', region='us')
handler = Alpha158(instruments=['SPY'], start_time='2015-01-02', end_time='2024-12-31')
dataset = DatasetH(handler=handler, segments={'train': ['2015-01-02', '2019-12-31']})
df = dataset.prepare('train', col_set=['feature', 'label'])
print(df['label'].describe())
"
```
Possible causes:
- Label values are intrinsically small (daily returns ~0.06%), making predictions naturally near-zero
- RMSE loss optimizes for mean prediction; the model converges to near-mean prediction
- Feature normalization (RobustZScoreNorm) may compress signal

### Option C: Re-evaluate Qlib Fit for This Project
Qlib is designed for multi-stock quantitative portfolios (CSI 300, CSI 500 backtests).
This project is a single-ticker pattern-detection system. Fundamental questions:
- Does the cross-sectional factor suite (Alpha158) add value for single-ticker prediction?
- Are temporal distribution methods (ADARNN/ADD) applicable to single-ticker?
- Should we borrow Qlib's Alpha158 factors and feed them to our CatBoost V3 pipeline instead?

### Option D: Patch ADARNN/ADD Dataloaders to Accept Alpha158
Modify `pytorch_adarnn.py:347` and `pytorch_add.py` internal dataloaders to handle arbitrary `d_feat`:
- Change hardcoded `(6, 60)` reshape to `(d_feat, len_seq)`
- Fix ADD's dimension indexing to handle variable feature counts
- **Risk:** The models' architecture (gate layers, BN layers) may assume specific dimensions

---

## Recommended Path (My Assessment)

**Priority: B → A → C → D**

1. **First, understand the CatBoost prediction issue (Option B).**
   If we can't get useful predictions from Qlib's own CatBoost on Alpha158, there's no reason
   to invest in more complex Qlib models. Check label distribution, prediction distribution,
   and whether the model is learning anything.

2. **If CatBoost actually has signal but the strategy can't use it, try Option A.**
   Alpha360 + ADARNN/ADD may produce different predictions that the strategy can act on.
   Create Alpha360 configs using the existing ADARNN/ADD templates (only adapt region/dates/market).

3. **If Qlib fundamentally doesn't fit, pivot (Option C).**
   Extract Alpha158 factors into our project's FeatureExtractor, run through our existing
   CatBoost V3 pipeline, and compare to our 88-feature model.

### Pre-Flight Check (for next session)
```bash
# Verify Qlib is still functional
uv run python -c "import qlib; qlib.init(provider_uri='data/qlib_bin', region='us'); print('OK')"

# Re-run CatBoost to confirm reproducibility
uv run python scripts/run_qlib_workflows.py --models catboost

# Check label distribution
uv run python -c "
import qlib; from qlib.contrib.data.handler import Alpha158; from qlib.data.dataset import DatasetH
qlib.init(provider_uri='data/qlib_bin', region='us')
h = Alpha158(instruments=['SPY'], start_time='2015-01-02', end_time='2024-12-31')
d = DatasetH(handler=h, segments={'train': ['2015-01-02', '2019-12-31'], 'test': ['2021-01-04', '2024-12-31']})
for seg in ['train', 'test']:
    df = d.prepare(seg, col_set=['feature', 'label'])
    print(f'{seg}: {len(df)} rows, label mean={df[\"label\"].iloc[:,0].mean():.6f}, std={df[\"label\"].iloc[:,0].std():.6f}')
"
```

---

## Qlib Pipeline to Retain (Even If We Pivot)

The Qlib data pipeline (`data/qlib_bin/`, Alpha158 handler) is working and valuable:
- 158 Alpha158 factors are richer than our 88 FeatureExtractor features
- The Qlib data format supports multi-instrument (add more stocks to `instruments/all.txt`)
- `scripts/run_qlib_workflows.py` runs any Qlib YAML config end-to-end

To add a new stock:
```bash
python useful_resources/useful_repos/Qlib/scripts/dump_bin.py dump_all ...
# Or manually extend data/qlib_raw/ and re-convert
```
