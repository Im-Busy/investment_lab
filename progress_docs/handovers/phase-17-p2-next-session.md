# Handover — Phase 17 P2 (R9-R12) Next Session

> **State:** Phase 17 P0 (R1-R4) + P1 (R5-R8) COMPLETE. 10 new files, 3 modified. All ruff clean.
> **Next target:** R9-R12 (P2 tier): return/risk classification, attribution, default risk DtD, QRAFTI.
> **Data available:** FMP free tier (API key in MEMORY.md), yfinance (installed), SPY_daily.csv, 184 ETF/stock CSVs in data/raw/.

---

## What Was Done (R5-R8)

### R5: Collinearity Analysis
- **New:** `src/signals/collinearity.py` — VIF matrix via statsmodels, `RedundancyPair`, `CollinearityReport`, synthesize/discard recommendations
- **Verified:** r=0.99 pair → VIF=57/54, same-category → synthesize, different-category → discard lower IC

### R6: Liquidity Factor CEI
- **New:** `src/ml/factor_features.py` — `compute_cei()` (Acharya-Pedersen), `compute_amihud_illiquidity()`, `compute_roll_spread()`, `add_liquidity_features()`
- **Verified:** CEI=0 with constant shares, ~0.22 with 20% share increase. Uses yfinance for shares outstanding.

### R7: Multi-Dimensional Signal Scoring
- **New:** `src/signals/scoring.py` — `MultiAxisScore`, `MultiAxisScorer`, `quick_5axis_report()`, 5-axis scoring (IC/IR/turnover/diversity/overfit)
- **Verified:** Predictive signal composite=0.71, noise composite=0.21, high-turnover composite=0.002

### R8: MAD + Rank Standardization Pipeline
- **New:** `src/ml/preprocessing.py` — `MADOutlierClipper`, `RankStandardizer` (sklearn transformers), `mad_rank_pipeline()`
- **Verified:** Outlier 100→5.4, rank output ~N(0,1) for fat-tailed t(3) data

### Exports Updated
- `src/signals/__init__.py` — +9 exports (scoring + collinearity)
- `src/ml/__init__.py` — +13 exports (factor_features + preprocessing)
- `docs/COMMAND_CHEATSHEET.md` — Phase 17 P1 section with all 4 command groups + key file table

---

## Next: P2 Tier (R9-R12) — Moderate Impact/Complexity

### R9: Return Factor vs Risk Factor Classification ★★★
**File:** `src/signals/classification.py`
**What:** Classify each pattern as "return factor" (directional, IC significant) or "risk factor" (variance explanatory). Return factors → entry signals. Risk factors → position sizing modifiers.
**Deps:** statsmodels (t-test). ~100 lines.

### R10: Performance Attribution Decomposition ★★★
**File:** `scripts/attribution.py`
**What:** Decompose strategy returns: r_P = β_market + β_sector + Σ β_style + α_specific. Post-hoc CLI analysis, not real-time.
**Deps:** statsmodels (OLS). ~180 lines.

### R11: Default Risk Factor (Merton DtD) ★★★
**File:** `src/ml/factor_features.py` (append)
**What:** Merton Distance-to-Default: DtD = (ln(V/D) + (r − σ²/2)T) / (σ√T). CatBoost feature.
**Deps:** FMP (market cap, total debt, risk-free rate). ~80 lines.

### R12: QRAFTI Standardized Evaluation Protocol ★★
**File:** `scripts/evaluate_qrafti.py`
**What:** 14-test diagnostic suite (Novy-Marx/Velikov 2023) covering factor construction, signal quality, implementation feasibility.
**Deps:** 0 (self-contained tests). ~250 lines.

---

## Files Created/Modified This Session

**New:**
| File | Purpose |
|------|---------|
| `src/ml/factor_features.py` | CEI + Amihud illiquidity + Roll spread liquidity factors |
| `src/signals/scoring.py` | MultiAxisScorer, 5-axis signal quality |
| `src/ml/preprocessing.py` | MADOutlierClipper + RankStandardizer sklearn transformers |
| `src/signals/collinearity.py` | VIF collinearity analysis + redundancy detection |

**Modified:**
| File | Change |
|------|--------|
| `src/signals/__init__.py` | +9 exports (scoring + collinearity) |
| `src/ml/__init__.py` | +13 exports (factor_features + preprocessing) |
| `docs/COMMAND_CHEATSHEET.md` | +Phase 17 P1 section (~130 lines) |
| `MEMORY.md` | Updated phase progress, completed tasks, key files, next session |
| `progress_docs/plans/full.md` | P1 complete status |
| `progress_docs/plans/17-resource-enhancements.md` | P1 status → complete |
| `progress_docs/current.md` | Session entry |

---

## Anti-Patterns to Avoid (continued from P0)

1. **Don't create new files when appending to existing.** R11 goes in `src/ml/factor_features.py` (append, don't create new).
2. **Don't forget to update exports.** Every new public class/function must be in the appropriate `__init__.py`.
3. **Don't skip COMMAND_CHEATSHEET.md.** Every new CLI flag or script gets documented.
4. **Use `uv run`, never bare `python`.**
5. **Ruff clean before marking done:** `uv run ruff check <files>`.
6. **FMP API rate limit:** 250 req/day. Cache results locally.
