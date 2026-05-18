---
project: investment_trying
plan_name: Overfitting Fixes — Stability Selection, Per-Sector Models, CPCV, Dynamic Ensembles
created: 2026-05-13
source: |
  Web research synthesis of 32 papers and production resources addressing:
  (1) Model overfit: train Sharpe 0.73 → OOS -0.27
  (2) ARO collapses to 5 features (from 88+)
  (3) Cross-asset features filled with 0.0 at single-ticker inference
  (4) Regime shift: 18/57 features shifted, ATR doubling
  (5) Weak signal strength (predictions in 0.0004 range)
scope: |
  Six evidence-backed fixes from published research, prioritized by impact:
  P0: Stability Selection + Per-Sector Models (eliminate two root causes)
  P1: CPCV + Dynamic Ensemble Learning (prevent overfit + adapt to regime shift)
  P2: Meta-Labeling + Production Hardening (signal filtering + deployment guards)
status: 🔄 In Progress
tasks_total: 16
tasks_active: 0
tasks_complete: 10
---

# Overfitting Fixes — Production-Grade Solutions

## Background

After B1 (OOS validation FAILED: Sharpe -0.27 vs +0.73) and B6 (retrain with normalized ATR: 2/5 pass, OOS -3.22%), we diagnosed three root causes:

1. **Feature selection instability** — ARO selects only 5 features (from 88+), and those 5 include cross-asset features unavailable at single-ticker inference (filled with 0.0)
2. **CV methodology gap** — PurgedKFold tests only a single time path, inflating IS performance
3. **Regime shift** — 18/57 features shift distribution OOS, no adaptive mechanism exists

Web research across 32 papers and production resources identified six proven solutions.

### Evidence Base

| Problem | Solution | Evidence |
|---------|----------|----------|
| Feature selection collapse | Stability Selection (Meinshausen & Bühlmann 2010, *JRSS-B*) | Bootstrap-aggregated feature selection; `scikit-learn-contrib/stability-selection` library |
| Cross-asset features unavailable | Per-sector models (QuantPedia 2024) | Group-specific ML models significantly outperform full cross-section models |
| Single-path CV | Combinatorial Purged CV (López de Prado 2017; ScienceDirect 2024) | CPCV > PurgedKFold > Walk-Forward in PBO/DSR benchmarks |
| Regime shift / non-stationarity | Dynamic Ensemble Learning (Edinburgh 2025; DriftMoE arXiv 2025; OneNet NeurIPS 2023) | EGD-weighted ensemble with periodic re-initialization; mixture-of-experts |
| Weak signal strength | Meta-Labeling (JFDS 2022; hudson-and-thames open-source) | Secondary ML filters primary signals using regime/volatility features (no cross-asset dependency) |
| Production evaluation gaps | DSR, PBO, strict walk-forward invariant (arXiv 2604.15531) | Walk-forward enforced at preprocessing, feature construction, AND HP tuning stages |

---

## Tasks

### Phase B9: Stability Selection — Replace Single-Run ARO

**Why:** ARO collapses to 5 features because single-run selection is unstable in small-N, large-P settings. Stability Selection bootstraps the data N times, runs selection on each, and keeps only features appearing in ≥π% of runs. This is the published solution for feature selection instability.

**Goal:** Feature set expands from 5 → 40+ stable features, all computable at single-ticker inference (no cross-asset features unless sector-appropriate).

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B9.1** | Install `stability-selection` | — | `uv add stability-selection` (scikit-learn-contrib package). Compatible with CatBoost, LASSO, or any sklearn-compatible estimator. |
| **B9.2** | Implement `src/ml/tuning/stability_selector.py` | B9.1 | Wrapper around `StabilitySelection`: (a) bootstraps training data N=100 times, (b) runs CatBoost feature importance on each bootstrap, (c) aggregates selection probabilities across runs, (d) selects features with stability_score ≥ π_threshold (default 0.6). Complementary-pairs bootstrapping (Shah & Samworth 2013) for variance reduction. |
| **B9.3** | Integrate into `train_ml_pipeline_v3.py` | B9.2 | Replace `aro_selector.select()` call with `stability_selector.select()`. Add `--stability-threshold` CLI flag (default 0.6). Log stability scores per feature to MLflow. |
| **B9.4** | Verify: re-run SPY feature selection | B9.3 | Train on SPY 2016-2024. Verify ≥40 features selected, <5% cross-asset features, stability distribution logged. Compare vs ARO 5-feature result. |

### Phase B10: Per-Sector Models — Eliminate Cross-Asset Feature Gap

**Why:** Basket-trained model includes `beta_QQQ_60d`, `resid_vol_SPY_60d`, `rel_ret_TLT_5d` — none available at single-ticker inference (MLStrategy fills with 0.0). The "Less is More" paper (QuantPedia 2024) shows per-sector models significantly outperform models trained on the full cross-section. By training one model per sector (tech, financials, energy, etc.), cross-asset features are intra-sector only and always available.

**Goal:** 7 sector models, each with reliable features computable at single-ticker inference. No 0-filled features.

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B10.1** | Sector classification + mapping | B9.4 | Hardcode sector map for 33-ticker basket: Tech (XLK, QQQ, AAPL, MSFT, NVDA, AVGO, AMD, ADBE, CRM, CSCO, INTC, IBM), Financials (XLF, JPM, BAC, WFC, GS, MS, C), Energy (XLE, XOM, CVX, COP, SLB), Healthcare (XLV, JNJ, UNH, PFE, ABBV), Consumer (XLP, PG, KO, PEP, WMT, COST), Industrials (XLI, UNP, CAT, GE, BA), Utilities/REITs (XLU, AVB, D, SO). |
| **B10.2** | Per-sector training pipeline | B10.1 | Modify `train_ml_pipeline_v3.py`: (a) accept `--sector` flag, (b) filter feature pool to intra-sector only (no cross-sector features), (c) train CatBoost per sector, (d) save as `pattern_classifier_v3_SPY_20260513_tech.pkl` etc. |
| **B10.3** | Per-sector inference in MLStrategy | B10.2 | `ml_strategy.py`: load correct sector model based on ticker. Sector lookup via mapping dict. Fallback: load "general" model for unclassified tickers. |
| **B10.4** | Backtest all sectors | B10.3 | Run `scripts/run_ml_backtest.py` per ticker using correct sector model. Compare per-sector vs basket model on 33-ticker basket. Expect: all tickers produce trades (no zero-signal failures), mean Sharpe > basket baseline (0.15). |

### Phase B11: Combinatorial Purged Cross-Validation (CPCV)

**Why:** PurgedKFold tests only a single chronological path. CPCV generates hundreds of train/test path combinations — each tests the model against different regime sequences. Paper (ScienceDirect 2024) shows CPCV has lower PBO (Probability of Backtest Overfitting) and higher DSR (Deflated Sharpe Ratio) than both PurgedKFold and Walk-Forward.

**Goal:** Replace PurgedKFold with CPCV in training pipeline. Lower PBO, stable CV metrics across paths. **Status: B11.1-B11.3 done, B11.4 pending.**

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B11.1** | `src/ml/combinatorial_purged_cv.py` | — | ✅ Done — CombinatorialPurgedCV class with split(), get_path_summary(), get_test_coverage(). C(6,2)=15 paths. 32 tests pass. |
| **B11.2** | Integrate into `train_ml_pipeline_v3.py` | B11.1 | ✅ Done — `--cv-method cpcv` CLI flag. Replaces outer PurgedKFold with C(6,2)=15 CPCV paths. Inner loop stays as PurgedKFold. Logs per-path metrics. |
| **B11.3** | Bagged CPCV variant | B11.1 | ✅ Done — `train_bagged_cpcv()` trains 15 models (one per CPCV path). `ml_strategy.py` supports `model_paths` list for ensemble mean prediction. |
| **B11.4** | Validate CPCV vs PurgedKFold | B11.3 | Compare on SPY: (a) PBO via `src/analysis/deflated_sharpe.py`, (b) CV metric stability (coefficient of variation across paths), (c) OOS Sharpe difference. Expect: PBO < 0.3, CV stability < 0.2. |

### Phase B12: Dynamic Ensemble Learning (DEL) — Regime Adaptation

**Why:** 18/57 features shifted distribution OOS (KS=0.62 for ATR, p=10^-109). Single model degrades as regime changes. Literature survey (IJIMAI 2023, 223 papers) finds no single model dominates across regimes. Solution: maintain ensemble of models, dynamically weight by recent performance.

**Goal:** Ensemble of 5 CatBoost models (different hyperparameters + training periods), weighted via exponential gradient descent on recent OOS error. Re-initialize weights every K=60 bars.

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B12.1** | Implement `src/ml/dynamic_ensemble.py` | B10.4 | ✅ Done — `DynamicEnsemble` class with 5 CatBoost variants, EGD weight optimizer (η=0.1, reinit K=60), save/load. |
| **B12.2** | EGD weight optimizer + MLStrategy integration | B12.1 | ✅ Done — `ml_strategy.py` supports `use_dynamic_ensemble` + `dynamic_ensemble_path`. Weight update per bar via bar return. `run_ml_backtest.py --use-dynamic-ensemble` flag. |
| **B12.3** | Backtest DEL vs single model | B12.2 | ✅ Done — DEL Sharpe 0.91 vs single 0.69 (+32%). Return 78.1% vs 66.4%. DD -20.8% vs -25.7%. PF 1.74 vs 1.45. |
| **B12.4** | Document pipeline integration | B12.3 | ✅ Done — Pipeline trains DEL via `--cv-method cpcv` (Stage 6c). Standalone script: `scripts/train_dynamic_ensemble.py`. |

### Phase B13: Meta-Labeling Secondary Filter

**Why:** Primary model signal is weak (predictions in 0.0004 range). Meta-labeling (López de Prado / JFDS 2022) adds a secondary model trained to predict: "given a primary signal, will this trade be profitable?" The meta-model uses regime/volatility/signal-clustering features — no cross-asset dependency. Full open-source implementation available at `hudson-and-thames/meta-labeling`.

**Goal:** Meta-model filters primary signals. Only trades where primary AND meta-model agree execute.

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B13.1** | Implement `src/ml/meta_labeler_v2.py` | — | Train CatBoost classifier on: (a) primary model probability, (b) ADX trend strength, (c) vol_regime_ratio, (d) signal_density (signals in last 20 bars), (e) days_since_signal, (f) prob_rolling_mean/std. Target: was_next_trade_profitable (binary, horizon=N bars). |
| **B13.2** | Integrate into MLStrategy | B13.1 | `ml_strategy.py`: add `--use-meta-label` flag. After primary `predict()`, run `meta_labeler.predict()`. Trade only if both > threshold. Default: primary > 0.45 AND meta > 0.5. |
| **B13.3** | Backtest meta-labeling on SPY | B13.2 | Compare: primary-only vs primary+meta. Expect: fewer trades, higher win rate, Sharpe ≥ primary-only. |

### Phase B14: Production Hardening — Evaluation Guards

**Why:** The "Spurious Predictability" paper (arXiv 2604.15531) shows random CV can inflate performance even when models are well-specified. Strict walk-forward must be enforced as a workflow invariant at ALL stages — preprocessing, feature construction, HP tuning, model selection — not just final fit.

**Goal:** Enforce walk-forward invariant. DSR + PBO as hard gates. Trainer warns/fails if violated.

| # | Task | Depends On | Description |
|---|------|------------|-------------|
| **B14.1** | Walk-forward invariant enforcement | B11.4 | Add `--strict-wf` flag to `train_ml_pipeline_v3.py`. Verifies: (a) no future data in feature computation (label span + embargo respected), (b) HP tuning uses chronological PurgedKFold only, (c) feature selection done within each walk-forward fold (not globally). Raises `RuntimeError` on violation. |
| **B14.2** | PBO + DSR gates in training | B14.1 | After CV evaluation, compute PBO and DSR via `src/analysis/deflated_sharpe.py`. Gate: PBO must be < 0.3, DSR must be > 1.0. Warn if near boundary, fail if below. |
| **B14.3** | Final untouched hold-out validation | B14.2 | Reserve 2025-01-01 → 2026-05-13 as final hold-out (never used in training, CV, or HP tuning). Only evaluated ONCE, after all development complete. Compare with B1 baseline (Sharpe -0.27). |
| **B14.4** | Production health dashboard | B14.3 | Script `scripts/model_health.py`: monitors (a) feature drift (KS test vs training distribution), (b) prediction distribution stability (KL divergence), (c) recent Sharpe vs expected, (d) flag retrain triggers. |


## Implementation Order

```
B9  (Stability Selection) ──┐
                             ├──→ B10 (Per-Sector Models) ──→ B11 (CPCV) ──→ B12 (Dynamic Ensemble)
                             │                                                    │
                             └────────────────────────────────────────────────────┘
                                                                                   │
B13 (Meta-Labeling) ←─────────────────────────────────────────────────────────────┘
                                                                                   │
B14 (Production Hardening) ←──────────────────────────────────────────────────────┘
```

### Dependency Rationale

- B9 must run before B10: B10 needs stable feature sets per sector
- B10 must run before B11: B11 evaluates cross-validation methodology on final model architecture
- B11 must run before B12: B12 needs CPCV-validated base models
- B12 + B13 are independent; both depend on B9+B10+B11
- B14 gates on everything; final validation + monitoring

### Success Criteria

| Criterion | Current | Target |
|-----------|---------|--------|
| Feature count (stable) | 5 (ARO) | ≥ 40 |
| Features available at inference | 3/5 (60%) | 100% per sector |
| PBO | Not measured | < 0.3 |
| OOS Sharpe (final hold-out) | -0.27 | > 0.0 |
| Trades with zero-signal failures | Basket model only | All 33 tickers |
| Mean 33-ticker Sharpe | 0.15 (basket) | > 0.25 |
| Regime drift detection | None | KS alarm + auto retrain trigger |

### References

| Paper / Tool | URL |
|-------------|-----|
| Stability Selection (Meinshausen & Bühlmann 2010) | `github.com/scikit-learn-contrib/stability-selection` |
| Feature Selection with Annealing (financial TS) | `arxiv.org/abs/2303.02223` |
| CPCV (López de Prado) + Bagged/Adaptive variants | `sciencedirect.com/science/article/abs/pii/S0950705124011110` |
| mlfinlab CPCV implementation | `github.com/hudson-and-thames/mlfinlab` |
| Fast CPCV + Optuna | `github.com/markmipt/fast_combinatorial_cv` |
| "Less is More" — per-sector models | `quantpedia.com/less-is-more-reducing-biases...` |
| "Overhyped? Can ML Models Predict Returns?" | `yankikalfa.com/research/ml/Hyped_ML_paper-2.pdf` |
| Dynamic Ensemble Learning (DEL) framework | `crc.business-school.ed.ac.uk/sites/crc/files/2025-11/...` |
| OneNet: EGD-weighted online ensemble (NeurIPS 2023) | `papers.neurips.cc/paper_files/paper/2023/...` |
| DriftMoE: Mixture of Experts for concept drift | `arxiv.org/abs/2507.18464` |
| ML for Financial Prediction Under Regime Change (survey) | `ijimai.org/index.php/ijimai/article/view/281` |
| Meta-Labeling (JFDS paper + open-source code) | `github.com/hudson-and-thames/meta-labeling` |
| Spurious Predictability in Financial ML | `arxiv.org/abs/2604.15531` |
