# Handover — Phase 17 P1 (R5-R8) Next Session

> **State:** Phase 17 P0 (R1-R4) COMPLETE. 6 new files, 3 modified. All ruff clean.
> **Next target:** R5-R8 (P1 tier): collinearity analysis, liquidity factor, multi-dim scoring, MAD pipeline.
> **Data available:** FMP free tier (API key in MEMORY.md), yfinance (installed), SPY_daily.csv, 184 ETF/stock CSVs in data/raw/.

---

## What Was Done (R1-R4)

### R1: IR-Weighted Pattern Synthesis
- **New:** `src/signals/ir_weighting.py` — IRWeighting class, look-ahead safe rolling IR from close prices
- **Modified:** `src/strategies/rules_first_strategy.py` — `use_ir_weights`, `ir_weighting_window` params, `_init_ir_weights()`, `_compute_score()` updated
- **Modified:** `scripts/backtest_rules_first.py` — `--ir-weights`, `--ir-weighting-window` flags
- **Verified:** `uv run scripts/backtest_rules_first.py SPY --ir-weights` runs, produces trades

### R2: Factor Purification Module
- **New:** `src/signals/factor_purification.py` — FactorPurifier (OLS regression), PurificationReport dataclass
- **Verified:** purity_ratio=0.997 for random signal (clean), 0.143 for sector-dominated signal (contaminated)

### R3: 4-Step Pattern Evaluation Gate
- **New:** `src/signals/evaluation_gate.py` — PatternEvaluationGate (4 sequential steps)
- **New:** `scripts/evaluate_pattern.py` — CLI with `--ticker` or `--signals-file` input, `--json-output`
- **Verified:** synthetic predictive signal passes all 4 steps (tstat=29.76, IC=0.667, type=return, spread=0.019)

### R4: HP Filter Return Forecasting
- **New:** `src/ml/expected_returns.py` — hp_filter(), hp_forecast(), hp_expected_return(), HPFilter online class
- **Verified:** HP trend computed on 500-bar synthetic cumulative return, gradient correct

### Documentation Updated
- `docs/COMMAND_CHEATSHEET.md` — Phase 17 section with all 4 command groups + key file table
- `MEMORY.md` — Phase 17 progress table, new files, completed tasks
- `progress_docs/plans/full.md` — P0 complete status
- `progress_docs/plans/17-resource-enhancements.md` — P0 status → complete
- `progress_docs/current.md` — session entry

---

## Next: P1 Tier (R5-R8) — High Impact, Light Dependencies

### R5: Collinearity Analysis for Pattern Overlap ★★★★

**File:** `src/signals/collinearity.py`

**What:** Compute VIF (Variance Inflation Factor) matrix across all 45 pattern signal series. Flag redundant patterns. Rules: within-category correlation → IR-weight synthesize; across-category correlation → discard weaker (lower IC).

**Approach:**
```python
# VIF = 1 / (1 - R²_i) where R²_i is regression of pattern_i on all others
# Use statsmodels.stats.outliers_influence.variance_inflation_factor
# Input: DataFrame (N_bars × M_patterns) of precomputed pattern signals from RulesFirstStrategy
# Output: VIF matrix, redundancy pairs (VIF > 5), synthesis/discard recommendations
```

**Key integration:** Load precomputed signals from `RulesFirstStrategy._signals_cache` (dict of pattern_name → int8 array). The strategy already precomputes all signals in `init()`.

**Deps:** `statsmodels` (already installed — used by Phase 16 Phase S pairs trading)

**Verification:**
```bash
uv run python -c "
from src.signals.collinearity import analyze_collinearity
import pandas as pd, numpy as np
# Synthetic test: two highly correlated signals
n = 500
s1 = np.random.choice([-1,0,1], n)
s2 = s1.copy(); s2[np.random.choice(n, 50)] *= -1  # 90% correlated
s3 = np.random.choice([-1,0,1], n)
df = pd.DataFrame({'p1': s1, 'p2': s2, 'p3': s3})
result = analyze_collinearity(df)
print(result['vif'])  # p1 and p2 should have high VIF
"
```

---

### R6: Liquidity Factor (CEI) ★★★★

**File:** `src/ml/factor_features.py` (append to existing, don't create new per project rules)

**What:** Compute Acharya-Pedersen Capital Efficiency Index: CEI = 12-month change in market equity − 12-month cumulative return.

**Formula:** `CEI = (ME_t / ME_{t-252}) - 1 - (P_t / P_{t-252} - 1)` where ME = price × shares_outstanding.

**Dependencies:**
- **shares outstanding** — Available via FMP API (free tier, 250 req/day). Key: `cv5v6VVC1p6ZAunPjWwwfMXQ0cvslEYI`. Endpoint: `https://financialmodelingprep.com/stable/historical-shares-outstanding?symbol=SPY&apikey=KEY`
- OR use yfinance: `yf.Ticker('SPY').get_shares_full()` for historical shares — may be simpler for backtesting

**Integration:**
- Compute CEI for each bar as new feature column
- Add to `FUNDAMENTAL_FACTOR_CONFIG` dict in `fundamental_features.py`
- Add to `FACTOR_NAMES` list
- Wire into `FundamentalFeatureExtractor.extract()` alongside existing 14 factors

**Relevant existing code:**
- `src/ml/fundamental_features.py:24-39` — `FUNDAMENTAL_FACTOR_CONFIG` dict (14 factors, add `cei`)
- `src/data_ingestion/fmp_client.py` — may exist for FMP API access; check before creating

**Verification:**
```bash
uv run python -c "
from src.ml.factor_features import compute_liquidity_factor
import pandas as pd
df = pd.read_csv('data/raw/SPY_daily.csv', index_col=0)
cei = compute_liquidity_factor(df['Close'])
print(f'CEI range: {cei.min():.4f} to {cei.max():.4f}')
"
```

---

### R7: Multi-Dimensional Signal Scoring ★★★★

**File:** `src/signals/scoring.py`

**What:** Replace single `confidence` score with 5-axis scoring:
1. IC (predictive accuracy) — rank correlation signal vs forward return
2. IR (stability) — IC mean / IC std over rolling window
3. Turnover impact — how often does the signal change direction
4. Diversity — correlation with other active signals (want low)
5. Overfitting risk — IS IC / OOS IC ratio

**Existing integration points:**
- `src/signals/signal_generator.py` — `SignalGenerator` class, `AggregatedSignal` dataclass (line 65)
- `src/strategies/rules_first_strategy.py` — `_compute_score()` — the sigmoid/tanh scoring
- `src/signals/ir_weighting.py` — IRWeighting can feed step 2 (IR stability)

**Approach:**
```python
@dataclass
class MultiAxisScore:
    ic: float          # predictive accuracy
    ir: float          # information ratio stability
    turnover: float    # 1 - turnover rate (higher = more stable)
    diversity: float   # 1 - max_correlation with other signals
    overfit_risk: float  # 1 - IS_IC/OOS_IC ratio
    composite: float   # geometric mean (all axes 0-1)

class MultiAxisScorer:
    def score(self, signal: np.ndarray, returns: np.ndarray,
              other_signals: dict[str, np.ndarray]) -> MultiAxisScore: ...
```

**Deps:** 0 (numpy + scipy, both already available)

**Verification:**
```bash
uv run python -c "
from src.signals.scoring import MultiAxisScorer
# ... test with synthetic signal ...
"
```

---

### R8: MAD + Rank Data Standardization Pipeline ★★★

**File:** `src/ml/preprocessing.py`

**What:** Preprocessing pipeline with two steps:
1. Median-based outlier removal: `|x - median| > n × MAD` → clip to boundary or remove
2. Rank standardization (non-parametric): replace values with percentile rank → map to normal distribution (or just use rank / N)

**Why rank standardization:** Handles crypto's fat tails. Non-parametric, no assumption of normality. 华泰 recommends it for broader applicability.

**Integration:** sklearn `TransformerMixin` pipeline that can be inserted into `FeatureExtractor` or `train_ml_pipeline_v3.py`.

**Existing preprocessing:** Check `src/ml/feature_engineering.py` for existing normalization steps before adding.

**Approach:**
```python
from sklearn.base import BaseEstimator, TransformerMixin

class MADOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, n_mad=5.0): ...
    def fit(self, X, y=None): ...
    def transform(self, X): ...

class RankStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, output_distribution='uniform'): ...  # or 'normal'
    def fit(self, X, y=None): ...
    def transform(self, X): ...
```

**Deps:** sklearn (already installed)

**Verification:**
```bash
uv run python -c "
from src.ml.preprocessing import MADOutlierClipper, RankStandardizer
import numpy as np
x = np.random.standard_t(3, size=1000)  # fat-tailed
x[0] = 100  # outlier
clipper = MADOutlierClipper()
clipped = clipper.fit_transform(x.reshape(-1,1))
print(f'Max before: {x.max():.1f}, after: {clipped.max():.1f}')
stand = RankStandardizer()
ranked = stand.fit_transform(clipped)
print(f'Ranked range: [{ranked.min():.3f}, {ranked.max():.3f}]')
"
```

---

## Execution Order

```
R6 (Liquidity CEI) → R7 (Multi-Dim Scoring) → R8 (MAD Pipeline) → R5 (Collinearity)
```

Rationale: R6 adds alpha (new feature), R7 improves signal quality (directly improves backtest metrics), R8 improves data pipeline (foundational), R5 identifies redundancy (informs pattern pruning).

---

## Files Created/Modified This Session (for git context)

**New:**
| File | Purpose |
|------|---------|
| `src/signals/ir_weighting.py` | IRWeighting class |
| `src/signals/factor_purification.py` | FactorPurifier + PurificationReport |
| `src/signals/evaluation_gate.py` | PatternEvaluationGate (4-step) |
| `src/ml/expected_returns.py` | HP filter + HPFilter class |
| `scripts/evaluate_pattern.py` | CLI for pattern evaluation |

**Modified:**
| File | Change |
|------|--------|
| `src/strategies/rules_first_strategy.py` | `use_ir_weights`, `ir_weighting_window`, `_init_ir_weights()` |
| `scripts/backtest_rules_first.py` | `--ir-weights`, `--ir-weighting-window` flags |
| `src/signals/__init__.py` | Exports IRWeighting, FactorPurifier, PatternEvaluationGate |
| `src/ml/__init__.py` | Exports hp_filter, HPFilter, lambda constants |
| `docs/COMMAND_CHEATSHEET.md` | Phase 17 section |
| `MEMORY.md` | Phase 17 progress + new files |
| `progress_docs/plans/full.md` | P0 complete |
| `progress_docs/plans/17-resource-enhancements.md` | P0 status → complete |
| `progress_docs/current.md` | Session entry |

---

## Quick Reference: Data Available

| Resource | Location | Notes |
|----------|----------|-------|
| SPY daily OHLCV | `data/raw/SPY_daily.csv` | 2015-2026, backtesting.py compatible |
| 183+ other instruments | `data/raw/*_daily.csv` | ETFs, stocks, crypto |
| FMP API key | `cv5v6VVC1p6ZAunPjWwwfMXQ0cvslEYI` | 250 req/day, append `?apikey=KEY` |
| yfinance | installed | `yf.Ticker('SPY').info` for fundamentals |
| scipy | 1.16.3 | Already available |
| statsmodels | installed | Phase 16 Phase S used it for cointegration |
| sklearn | installed | Pipeline, CalibratedClassifierCV, etc. |
| numpy | installed | Core computation |

---

## Anti-Patterns to Avoid

1. **Don't create new files when appending to existing.** R6 goes in `src/ml/factor_features.py` (append, don't create `src/ml/factor_features_v2.py`). R8 goes in `src/ml/preprocessing.py` (create new only because no existing preprocessing module).

2. **Don't forget to update exports.** Every new public class/function must be in `src/signals/__init__.py` or `src/ml/__init__.py`.

3. **Don't skip COMMAND_CHEATSHEET.md.** Every new CLI flag or script gets documented.

4. **Use `uv run`, never bare `python`.**

5. **Ruff clean before marking done:** `uv run ruff check <files>`.

6. **Look-ahead bias:** Always check that features use only data available at decision time. R6 CEI at bar T uses Close[T] and shares_outstanding[T] — both known at T.

7. **FMP API rate limit:** 250 req/day. Cache results locally. Don't fetch on every script run — use `data/raw/` or `data/cache/` for persistence.
