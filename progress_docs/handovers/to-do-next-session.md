# Handover Prompt: Overfitting Fixes (B11-B14)

## Current State (2026-05-13 21:44)

**B9 (Stability Selection) COMPLETE.** Replaced ARO collapse (5 features) with Meinshausen & Buehlmann stability selection.
**B10 (Per-Sector Models) COMPLETE.** 4 sub-steps done:
- `src/ml/sector_map.py` — SECTOR_MAP (44 tickers → 7 sectors) + SECTOR_NAMES
- `scripts/train_ml_pipeline_v3.py` — `--sector` CLI flag (`choices=SECTOR_NAMES + ["all"]`), filters tickers, disables cross-asset, names model `pattern_classifier_v3_{sector}_{timestamp}.pkl`
- `src/strategies/ml_strategy.py` — added `ticker: str = ""` param, `init()` resolves sector model via `glob("models/pattern_classifier_v3_{sector}_*.pkl")`, falls back to `model_path`
- `scripts/run_ml_backtest.py` — passes `ticker=symbol` to `bt.run()`
- `docs/COMMAND_CHEATSHEET.md` — added `--sector` and `--sector all` examples

**Problem:** ML model is overfit. Train Sharpe 0.73 → OOS Sharpe -0.27. Three root causes diagnosed:

| Root Cause | Evidence | Status |
|------------|----------|--------|
| Feature Selection Collapse | ARO selects only 5 features from 88+ pool | **FIXED (B9)** |
| Cross-Asset Feature Gap | 2 of 5 selected features are cross-asset (unavailable at single-ticker inference → filled with 0.0) | **FIXED (B10)** |
| Regime Shift / No Adaptation | 18/57 features shift distribution OOS (ATR doubled, KS=0.62 p=10^-109) | B12 |

**Remaining** (implementation graph):
```
B9 (done) ──→ B10 (done) ──→ B11 (CPCV) ──→ B12 (Dynamic Ensemble)
                                                    │
B13 (Meta-Labeling) ←───────────────────────────────┘
                                                    │
B14 (Production Hardening) ←────────────────────────┘
```

---

## What To Do: B11 — Combinatorial Purged Cross-Validation (CPCV)

### Why This Matters

PurgedKFold tests only a **single chronological path**. CPCV generates φ(N,k) backtest path combinations — each path tests the model against different regime sequences. Paper evidence (ScienceDirect 2024):

- CPCV has **lower PBO** (Probability of Backtest Overfitting) than both PurgedKFold and Walk-Forward
- CPCV has **higher DSR** (Deflated Sharpe Ratio)
- Bagged CPCV (aggregating predictions across paths) outperforms single-path CV

### Reference Implementations

- **mlfinlab**: `github.com/hudson-and-thames/mlfinlab` — canonical López de Prado implementation (complex, heavy dependency)
- **markmipt/fast_combinatorial_cv**: `github.com/markmipt/fast_combinatorial_cv` — NumPy-based, scikit-learn compatible, simpler

**Recommendation:** Implement from scratch using the combinatorial math directly (no external dependency). The algorithm is straightforward:

1. Partition N time periods into k groups
2. Select k-1 groups for training, 1 for testing — this gives k combinations
3. Repeat for all k=1..N → generates φ(N,k) = N! / (k!·(N-k)!) paths
4. Purge overlapping labels between train/test (same logic as existing PurgedKFold)
5. Add embargo buffer after test period
6. Aggregate metrics across all paths

### Implementation Plan (4 sub-steps)

#### B11.1: Implement `src/ml/cross_validation/cpcv.py`

Create a standalone CPCV module. The class should mirror PurgedKFold's API so it can drop in:

```python
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation for financial time series.

    Generates phi(N,k) = N! / (k! * (N-k)!) backtest paths. Each path:
    - Partitions total samples into N groups (chronological)
    - Combines N-k groups for training, k groups for testing
    - Purges train samples whose label window overlaps with test
    - Applies embargo buffer between test and next train

    Key difference from PurgedKFold: PurgedKFold splits into k folds and
    tests each sequentially (single path). CPCV tests ALL possible
    combinations of k test groups from N total groups (multiple paths).
    This generates many regime-conditional paths and reduces PBO.

    Args:
        n_groups: Number of chronological groups to partition data into.
        n_test_groups: Number of groups to hold out for testing each path.
        pct_embargo: Fraction of test span to embargo.
        label_span: Forward return horizon.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        pct_embargo: float = 0.05,
        label_span: int = 5,
    ) -> None: ...

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) for each combinatorial path."""
        ...

    def get_n_splits(self) -> int:
        """Return total number of combinatorial paths: C(n_groups, n_test_groups)."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)
```

**Algorithm** (3 steps):
1. Partition indices chronologically into `n_groups` equal-sized groups
2. Generate all combinations of `n_test_groups` groups: `itertools.combinations(range(n_groups), n_test_groups)`
3. For each combination: train = all groups NOT in test combo, test = groups in test combo. Purge train samples within `label_span` of test window. Apply embargo.

**Helper functions** (same as purged_cv.py):
- `_purge_train_indices(train_idx, test_idx, label_span, embargo)` — removes train indices that leak into test period
- `_compute_embargo(test_span, pct_embargo)` — embargo buffer size

#### B11.2: Integrate into `train_ml_pipeline_v3.py`

**Files/lines to modify:**

| File | Line(s) | Change |
|------|---------|--------|
| `scripts/train_ml_pipeline_v3.py:48` | import | Add `from src.ml.cross_validation.cpcv import CombinatorialPurgedCV` |
| `scripts/train_ml_pipeline_v3.py:192-300` | `train_with_nested_purged_cv()` | Add `--cv-method cpcv` CLI flag. When `cpcv`, replace inner/outer PurgedKFold calls with CPCV. |
| `scripts/train_ml_pipeline_v3.py:1206-1210` | CLI | Add `--cv-method` argument (choices: `purged`, `cpcv`, default: `cpcv`) |

**Integration strategy:**

The `train_with_nested_purged_cv()` function should be refactored to accept a `cv_method` parameter:

```python
def train_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    model_type: str = "catboost",
    cv_method: str = "cpcv",
) -> dict[str, Any]:
```

When `cv_method == "cpcv"`:
- Replace `cv_outer = PurgedKFold(...)` with `cv_outer = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, ...)`
- Replace `cv_inner = PurgedKFold(...)` with `cv_inner = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, ...)`
- Log total paths: `C(6,2)=15 outer paths, C(5,1)=5 inner paths`

For the inner loop: CPCV with `n_test_groups=1` is equivalent to standard PurgedKFold (each group tested once). So inner can stay as PurgedKFold for simplicity — or replace both.

**Simplest approach:** Only replace the outer CV loop with CPCV. Inner loop stays as PurgedKFold. This gives us 15 OOS paths (C(6,2)) with 5-fold inner CV for each path.

Add `--cv-method` to CLI in `main()`:
```python
parser.add_argument("--cv-method", type=str, default="purged", choices=["purged", "cpcv"],
                    help="CV method (default: purged). Use 'cpcv' for lower PBO.")
```

#### B11.3: Add Bagged CPCV variant

Bagged CPCV: train one model per CPCV path, then ensemble predictions via mean.

Add to `train_ml_pipeline_v3.py`:
- After CPCV completes, train a **final model for each path**
- Save all models with path index: `pattern_classifier_v3_{sector}_path{i}_{run_id}.pkl`
- Add `model_paths` list to metadata

Add to `src/strategies/ml_strategy.py`:
- When `model_paths` is provided (list of paths instead of single path), load all models
- `predict()` returns mean probability across all path models
- This gives the Bagged CPCV behavior at inference time

```python
# In MLStrategy.init():
if isinstance(self.model_path, list):
    self._models = []
    for mp in self.model_path:
        model = PatternClassifier()
        model.load(mp)
        self._models.append(model)
elif self.model_path:
    self._model = PatternClassifier()
    self._model.load(self.model_path)
```

#### B11.4: Validate CPCV vs PurgedKFold

**Validation workflow:**
1. Train SPY model with `--cv-method purged` (baseline)
2. Train SPY model with `--cv-method cpcv` (CPCV, 15 paths)
3. Compare metrics:

```bash
# Baseline
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method purged

# CPCV
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv
```

**Success criteria:**

| Metric | PurgedKFold | CPCV Target |
|--------|-------------|-------------|
| CV stability (coefficient of variation across paths) | ~0.3 | < 0.2 |
| Train AUC mean | ? | lower than PurgedKFold (less overfit) |
| Test AUC mean | ? | same or better |
| Overfit gap | > 0.15 | < 0.10 |
| Number of CV paths | 5 | 15 (or φ(N,k)) |

### Key Files (B11 Context)

| File | Line(s) | Purpose | Change |
|------|---------|---------|--------|
| `src/ml/cross_validation/cpcv.py` | — | **NEW** — CombinatorialPurgedCV class | Implement CPCV math + split() |
| `src/ml/purged_cv.py` | 1-194 | Existing PurgedKFold | Refactor _purge helpers into shared utils |
| `scripts/train_ml_pipeline_v3.py` | 48 | Import PurgedKFold | Add CPCV import |
| `scripts/train_ml_pipeline_v3.py` | 192-300 | `train_with_nested_purged_cv()` | Add cv_method param, CPCV logic |
| `scripts/train_ml_pipeline_v3.py` | 943-947 | Stage 6 call site | Pass `cv_method` |
| `scripts/train_ml_pipeline_v3.py` | 1206-1210 | CLI args | Add `--cv-method` |
| `src/strategies/ml_strategy.py` | 113-116 | Model loading in `init()` | Handle `model_paths` list for bagged CPCV |

### Validation Commands

```bash
# Unit test: CPCV split integrity (no overlap)
uv run python -c "
from src.ml.cross_validation.cpcv import CombinatorialPurgedCV
import numpy as np
cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, pct_embargo=0.05, label_span=5)
print(f'Total paths: {cv.get_n_splits()}')  # Should be C(6,2) = 15
X = np.arange(1000)
paths = list(cv.split(X))
print(f'Generated {len(paths)} paths')
for train, test in paths[:3]:
    overlap = len(set(train) & set(test))
    print(f'  train={len(train)}, test={len(test)}, overlap={overlap}')
"

# Train with CPCV (fast mode)
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv

# Compare CPCV vs PurgedKFold
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method purged
uv run scripts/train_ml_pipeline_v3.py --symbol SPY --fast --cv-method cpcv

# With sector model training
uv run scripts/train_ml_pipeline_v3.py --sector tech --fast --cv-method cpcv
```

### Anti-Patterns (B11-specific)

1. **Don't shuffle/randomize group boundaries** — groups must be chronological. CPCV is NOT random CV.
2. **Don't use CPCV as a hyperparameter tuner** — it evaluates CV methodology stability. Inner CV still handles HP selection.
3. **Don't aggregate predictions across all paths naively** — bagged CPCV requires per-path models, not just per-path metrics.
4. **Don't delete PurgedKFold** — keep it as a `--cv-method purged` baseline for comparison.

### CPCV Math Reference

```
Given N chronological groups and k test groups per path:

φ(N,k) = C(N,k) = N! / (k! · (N-k)!)

Example: N=6, k=2 → 15 paths
Each path: train on 4 groups, test on 2 groups

Purging:
- For test group spanning indices [t1, t2]:
  - Train sample at index i uses forward return over [i, i+label_span]
  - If i+label_span >= t1, sample i leaks test info → purge from training
  - Embargo: skip t2+1 through t2+embargo_days after test period

Compared to PurgedKFold (N=5 folds):
- PurgedKFold: 5 paths (each fold tested exactly once, 4 train + 1 test)
- CPCV: 15 paths (each group of 2 tested once, 4 train + 2 test)
- More paths → lower variance in CV metric estimates
```

### B11.4 Backtest Validation (After B11.1-11.3 Complete)

```bash
# Run with bagged CPCV model (after training with --cv-method cpcv)
uv run scripts/run_ml_backtest.py SPY --trail-stop \
    --model models/pattern_classifier_v3_SPY_bagged_cpcv.pkl

# Compare against PurgedKFold baseline
uv run scripts/run_ml_backtest.py SPY --trail-stop \
    --model models/pattern_classifier_v3_SPY_20260511_224704.pkl
```

### Expected Outcome

- CPCV OOS Sharpe should be **closer to train Sharpe** (less overfit)
- CV metric standard deviation should be **lower** (more stable)
- Single-ticker backtests should show **fewer false positives** (trades only in regime-conditions that passed more test paths)
