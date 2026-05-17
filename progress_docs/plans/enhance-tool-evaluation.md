---
project: investment_trying
plan_name: Tool Evaluation Additions
created: 2026-05-12
source: Online research evaluation of 18 open-source tools vs project stack
scope: |
  Add 6 new open-source tools to the project: Optuna (HP tuning), PyPortfolioOpt
  (portfolio optimization), Bandit (security linting), CCXT (crypto API), FinGPT
  (LLM sentiment), aeon (time-series ML). Only HIGH/MEDIUM priority items get
  immediate implementation. LOW items are deferred until project direction warrants.
status: ✅ Complete — T10a-1 through T10a-6 done (2026-05-16 verification). T10b-1 ✅. T10c deferred.
tasks_total: 10
tasks_active: 0
tasks_complete: 7
---

# Tool Evaluation Additions

## Background

Evaluated 18 open-source tools against the existing project stack:
- **12 tools** already covered (Freqtrade, Jesse, yfinance, PurgedKFold, XGBoost, SHAP, MLflow, Ruff, ta-lib, scikit-learn — all have existing equivalents)
- **6 tools** represent real gaps

### Gap → Tool Mapping

| Gap | Tool | Priority | Why |
|-----|------|----------|-----|
| No hyperparameter optimization framework | **Optuna** | HIGH | Existing GWO/GA/WOA tuners are custom metaheuristics. Optuna provides Bayesian (TPE), CMA-ES, grid, random, and pruning — integrates natively with CatBoost/LightGBM. Replaces manual `optimize_rsi.py`/`optimize_macd.py` scripts. |
| Basic portfolio optimization only (scipy.minimize) | **PyPortfolioOpt** | HIGH | Existing `portfolio_optimizer.py` uses vanilla scipy. PyPortfolioOpt adds: efficient frontier, HRP (hierarchical risk parity), CVaR optimization, CLA (critical line algorithm), proper Black-Litterman blending with market-implied returns. Complements existing `black_litterman.py`. |
| No Python security linting | **Bandit** | MEDIUM | Ruff covers style/logic. Bandit catches Python-specific security: hardcoded passwords, unsafe deserialization (pickle), subprocess injection, SQL injection vectors. Important given Google Cloud dependencies. |
| No crypto exchange API | **CCXT** | LOW | Current system targets equities. Only relevant if crypto trading is added. |
| No LLM sentiment signals | **FinGPT** | LOW | Existing `paper2md` pipeline + CHART_PATTERN_KNOWLEDGE_BASE already handle NLP. FinGPT adds live sentiment trading signals — a separate project direction. |
| Time-series ML not yet used | **aeon** | LOW | Already available as a skill. Existing 34+ pattern detectors + CatBoost/LightGBM cover classification. aeon adds shapelet discovery, TS clustering, TS regression — niche use cases. |

---

## Tasks

### Phase 10a: HIGH Priority (Implement Now)

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **T10a-1** | Install dependencies | — | `uv add optuna PyPortfolioOpt bandit` |
| **T10a-2** | Optuna hyperparameter tuning for CatBoost/LightGBM | T10a-1 | Replace `scripts/tune_model.py` GWO/GA/WAO calls with Optuna study. Implement: (a) `src/ml/tuning/optuna_tuner.py` — CatBoost/LightGBM/LGBM tuning with PurgedKFold CV, (b) TPE sampler with median pruner, (c) CLI integration in `scripts/tune_model.py` via `--algo optuna`, (d) MLflow logging of Optuna trials. |
| **T10a-3** | Optuna strategy parameter tuning | T10a.1 | Replace manual `scripts/optimize_rsi.py`/`optimize_macd.py` grid searches. Implement Optuna study per strategy: RSI thresholds, MACD params, EMA ribbon lengths, Keltner channel multipliers. Backtesting.py eval per trial. |
| **T10a-4** | PyPortfolioOpt integration | T10a-1 | Replace `src/optimizer/portfolio_optimizer.py` scipy.minimize with PyPortfolioOpt: (a) EfficientFrontier + HRP + CVaR methods, (b) integrate with existing BlackLittermanOptimizer for posterior blending, (c) add to MultiStrategyEngine weight allocation. |
| **T10a-5** | Test + validate tool additions | T10a-2, T10a-3, T10a-4 | (a) unit tests for optuna_tuner.py (≥5 tests), (b) unit tests for PyPortfolioOpt integration (≥5 tests), (c) run full Optuna study on CatBoost (SPY, 20 trials, verify improvement over GWO), (d) run HRP vs equal-weight comparison on 33-ticker basket. |
| **T10a-6** | Update documentation | T10a-5 | Update COMMAND_CHEATSHEET.md with Optuna + PyPortfolioOpt commands. Add entries to `.useful_commands/ml_training_commands.txt`. Update ML_TRAINING_GUIDE.md Section 7 (Tool Inventory) with new entries. |

### Phase 10b: MEDIUM Priority (Implement Soon)

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **T10b-1** | Add Bandit to pre-commit hooks | T10a-1 | Configure `.pre-commit-config.yaml` to run `bandit -c pyproject.toml` on every commit. Add `[tool.bandit]` config to pyproject.toml with project-appropriate exclusions (tests, notebooks). Run initial full scan, fix findings. |

### Phase 10c: LOW Priority (Deferred)

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **T10c-1** | CCXT crypto exchange integration | T10a-1 | Install `ccxt`. Implement `src/data_ingestion/crypto_fetcher.py` with CCXT wrapper. Add `--asset-class crypto` flag to `scripts/run_ml_backtest.py`. Only if/when crypto trading direction is confirmed. |
| **T10c-2** | FinGPT sentiment signal integration | T10a-1 | Install FinGPT. Implement `src/signals/sentiment_signal.py` with FinGPT sentiment scores. Feed into EventWeightedAggregator as a signal modifier. Requires API key (OpenAI). |
| **T10c-3** | aeon time-series ML integration | T10a-1 | Install `aeon`. Implement shapelet discovery for price patterns (`src/ml/shapelet_discovery.py`). Compare learned shapelets vs existing 34+ hand-crafted pattern detectors. LGBM-first principle: only use aeon if shapelet classification beats CatBoost AUC by ≥5%. |

---

## Impact Assessment

| Tool | Expected Impact | Risk |
|------|----------------|------|
| **Optuna** | +0.02–0.05 Sharpe from better hyperparameters across all ML models. Replaces ad-hoc grid search with principled Bayesian optimization. | Low — well-established library, ~10K GitHub stars, native scikit-learn/CatBoost integration. |
| **PyPortfolioOpt** | Lower portfolio volatility via HRP, better risk-adjusted returns via CVaR optimization. Complements existing BL implementation. | Low — 4.5K GitHub stars, actively maintained, pure Python. |
| **Bandit** | Prevents security regressions (no runtime impact). Catches pickle deserialization, hardcoded keys, subprocess injection. | None — static analysis only, runs as pre-commit hook. |
| **CCXT** | Enables crypto backtesting (BTC, ETH, SOL + 100+ altcoins). Only valuable if project expands beyond equities. | Medium — dependency on exchange API stability. Zero value if crypto never added. |
| **FinGPT** | Adds NLP sentiment signals as alpha source. 22 papers in knowledge graph mention sentiment but no implementation exists. | Medium — requires LLM API costs (~$0.01–0.10 per signal). May not add edge beyond existing price-based features. |
| **aeon** | Shapelet discovery could find novel price subsequences missed by hand-crafted patterns. TS-specific ML (BOSS, ROCKET, HIVE-COTE) vs generic tree boosters. | Medium-High — TS-specific classifiers rarely beat CatBoost/LightGBM on financial tabular data. Benchmark before committing. |

---

## Execution Order

```
Phase 10a: HIGH (Implement Now)
  ├── T10a-1: uv add optuna PyPortfolioOpt bandit
  ├── T10a-2: Optuna HP tuning for CatBoost/LightGBM ← FILLS BIGGEST GAP
  ├── T10a-3: Optuna strategy parameter tuning
  ├── T10a-4: PyPortfolioOpt integration
  ├── T10a-5: Test + validate
  └── T10a-6: Documentation

Phase 10b: MEDIUM (After 10a)
  └── T10b-1: Bandit pre-commit hook

Phase 10c: LOW (Deferred — gate on project direction)
  ├── T10c-1: CCXT (gate: crypto trading confirmed)
  ├── T10c-2: FinGPT (gate: sentiment alpha proven)
  └── T10c-3: aeon (gate: shapelets beat CatBoost by ≥5% AUC)
```

## Dependencies Added

```
optuna>=4.0.0          # Hyperparameter optimization
PyPortfolioOpt>=1.5.0  # Portfolio optimization methods
bandit>=1.8.0          # Python security linter
ccxt>=4.0.0            # Crypto exchange API (deferred)
finGPT>=1.0.0          # LLM financial sentiment (deferred)
aeon>=1.0.0            # Time-series ML toolkit (deferred)
```

## Relationship to Existing Plans

- **Optuna** replaces `scripts/optimize_rsi.py` + `scripts/optimize_macd.py` (manual grid search). Complements GWO/GA/WOA tuners in `src/ml/tuning/` — Optuna becomes the default with GWO/WOA as fallback/specialized alternatives.
- **PyPortfolioOpt** enhances `src/portfolio/black_litterman.py` + `src/optimizer/portfolio_optimizer.py`. Does not replace — provides additional methods (HRP, CVaR) that existing code doesn't implement.
- **Bandit** adds to existing Ruff + mypy pre-commit hooks. No overlap.
- **CCXT/FinGPT/aeon** are deferred — no conflict with any existing code.
