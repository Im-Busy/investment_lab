---
type: enhancement
name: ML Capability Enhancements (GWO, ARO, GA, WOA, InterpretML, AutoGluon + wider gaps)
status: planned
created: 2026-05-05
updated: 2026-05-05
depends_on:
  - Phase 04 (ML Foundation) ✅
  - D1-D5 data leak fixes ✅
blocks: Phase 05 (ML Advanced) — these are CPU-safe ML improvements that don't need GPU
rationale: |
  Survey of 10 ML areas revealed critical gaps: no hyperparameter tuning,
  no AutoML, limited interpretability, missing advanced backtest stats.
  Multi-optimizer strategy (ARO→GWO→GA→WOA) covers feature selection,
  supervised tuning, unsupervised clustering, and pattern optimization.
  ARO/InterpretML/AutoGluon are CPU-safe and address top-3 gaps.
  Beyond regime classification, ML can enhance 15+ trading pipeline stages.
---

# ML Capability Enhancement Plan

## Optimizer Strategy — Multi-Agent Tuning Pipeline

Instead of a single optimizer, use different metaheuristics for different tasks based on
their mathematical strengths on OHLCV data:

| Optimizer | Best Task | Noise Handling | Speed | Mechanism |
|-----------|-----------|---------------|-------|-----------|
| **ARO** | Feature Selection | High | Very Fast | Detour foraging (broad search) + random hiding (fine tuning) |
| **GWO** | CatBoost HP Tuning | Excellent | Medium | Wolf pack hierarchy: alpha→beta→delta guides convergence |
| **GA** | Unsupervised Regime Clustering | Moderate | Slow | Population diversity prevents cluster collapse |
| **WOA** | Volatility/Breakout Detection | High | Medium | Spiral search logic captures non-linear price patterns |

### Pipeline Flow

```
OHLCV Data
  │
  ├─► ARO (Feature Selection) ────► Select predictive indicators from 132+ features
  │
  ├─► GWO (Supervised Tuning) ───► Tune CatBoost learning_rate, depth, l2_leaf_reg
  │
  ├─► GA  (Unsupervised) ─────────► Cluster market into n regimes (Bull/Bear/Sideways/...)
  │
  └─► WOA (Pattern Optimization) ─► Tune entry/exit thresholds for breakout patterns
```

## Priority Tiers

### Tier 1 — Critical Gaps (CPU-safe, high ROI)

| # | Task | Library | Effort | Impact |
|---|------|---------|--------|--------|
| T1a | ARO-based feature selection engine | custom ARO + `numpy` | Medium | High — 132 features → ~15 predictive ones |
| T1b | GWO CatBoost hyperparameter tuner | custom GWO | Medium | High — all ML models currently use hardcoded defaults |
| T1c | GA unsupervised regime optimizer | custom GA | Medium | High — finds optimal n_regimes + cluster centroids |
| T1d | WOA breakout/volatility threshold tuner | custom WOA | Low | Medium — optimizes pattern entry confidence thresholds |
| T2 | Glassbox regime classifier with EBM | `interpret` (InterpretML) | Medium | High — explain *why* model says "bear market" |
| T3 | AutoML baseline with AutoGluon | `autogluon.tabular` | Low | Medium — establishes honest baseline for all ML work |
| T4 | Advanced backtest validation (DSR, PSR, FDR) | `numpy`, `scipy` | Medium | High — protects against backtest overfitting |
| T5 | Ensemble methods (Stacking + Voting + Blending) | `sklearn`, `vecstack` | Medium | High — model combination can boost accuracy 5-15% |

### Tier 2 — Important but Specialized

| # | Task | Library | Effort | Impact |
|---|------|---------|--------|--------|
| T6 | Monte Carlo VaR + CVaR risk modeling | `numpy`, `scipy` | Medium | Medium — complements existing binomial VaR |
| T7 | Black-Litterman portfolio optimization | `PyPortfolioOpt` or custom | Medium | ✅ Done — `src/portfolio/black_litterman.py` |
| T8 | SHAP visualization dashboard (beeswarm, waterfall, force) | `shap` (already installed) | Low | Medium — SHAP library already integrated, just needs plot code |
| T9 | Signal meta-labeling (López de Prado triple-barrier) | Custom | High | High — determines when to trade, not just what |

### Tier 3 — Lower Priority or GPU-Dependent

| # | Task | Library | Effort | Impact |
|---|------|---------|--------|--------|
| T10 | XGBoost integration in PatternClassifier | `xgboost` | Low | Low — CatBoost already covers GBDT well |
| T11 | TabNet deep tabular model | `pytorch-tabnet` | High | Medium — needs GPU, good for regime detection |
| T12 | Temporal Fusion Transformer | `pytorch-forecasting` | High | High — needs GPU, state-of-art time series DL |
| T13 | Hierarchical Risk Parity (HRP) | `PyPortfolioOpt` | Low | Medium — modern alternative to Markowitz |

---

## Task Details — Tier 1

### T1a: ARO Feature Selection Engine

**ARO (Artificial Rabbit Optimization):** Two-phase metaheuristic for feature selection:
- **Detour Foraging:** Rabbits explore away from nest (other rabbits) to find food — translates to broad exploration of 132+ technical indicators to find the most predictive subset
- **Random Hiding:** Rabbits dig burrows randomly to evade predators — translates to fine-grained selection of regularization-sensitive features
- **Why ARO for features:** Faster convergence than GWO or PSO on high-dimensional indicator spaces. Recent benchmarks show 20-40% fewer iterations to reach equivalent AUC.

**Implementation:**
```python
# src/ml/tuning/aro_selector.py
class AROFeatureSelector:
    """Select predictive features from 132+ technical indicators."""

    def __init__(self, n_rabbits=30, max_iter=100):
        self.population = n_rabbits
        self.max_iter = max_iter

    def select(self, X, y, base_estimator=None) -> list[str]:
        # Each "rabbit" = binary vector of feature inclusions
        # Fitness = PurgedKFold AUC with selected features
        # Detour phase: perturb inclusion vector substantially
        # Hiding phase: fine-tune near current best
        ...
```

**Commands:**
```bash
uv run scripts/select_features.py --symbol SPY --method aro --rabbits 30 --iterations 100
```

### T1b: GWO CatBoost Hyperparameter Tuner

**GWO (Grey Wolf Optimizer):** Four-level hierarchy for hyperparameter search:
- **Alpha (α):** Best solution found — lowest validation loss
- **Beta (β):** Second-best — explores nearby region
- **Delta (δ):** Third-best — scouts wider for escape from local minima
- **Omega (ω):** Rest of pack — follows alpha/beta/delta toward convergence

Best for CatBoost tuning because wolves naturally balance exploration (staying away from each other) with exploitation (converging toward alpha). Handles mixed int/float/categorical parameters without gradient requirements.

**Implementation:**
```python
# src/ml/tuning/gwo_tuner.py
class GWOTuner:
    """GWO-based CatBoost hyperparameter optimizer."""

    PARAM_SPACE = {
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'l2_leaf_reg': (1, 30),
        'random_strength': (0.5, 5.0),
        'bagging_temperature': (0.0, 2.0),
        'border_count': (32, 255),
    }

    def optimize(self, X, y, n_wolves=20, max_iter=50) -> dict:
        # Wolf positions = hyperparameter vectors
        # Fitness = PurgedKFold mean AUC
        # A = 2*a*r1 - a  (coefficient vector, decays linearly)
        # C = 2*r2         (stochastic coefficient)
        # D = |C*X_prey - X_wolf|  (distance to prey)
        # X_next = X_prey - A*D    (position update)
        ...
```

**Commands:**
```bash
uv run scripts/tune_model.py --model catboost --method gwo --wolves 20 --iterations 50
```

### T1c: GA Unsupervised Regime Optimizer

**Genetic Algorithm for Regime Discovery:** Market regimes are clusters of similar OHLCV behavior. GA prevents cluster collapse because:
- Population diversity ensures multiple regime hypotheses coexist
- Crossover combines features of two good regime definitions
- Mutation introduces novel regime boundaries (e.g., splitting "bull" into "steady bull" vs "parabolic bull")
- Selection pressure converges toward regimes with highest silhouette scores

**Implementation:**
```python
# src/ml/tuning/ga_regime.py
class GARegimeOptimizer:
    """Genetic Algorithm for optimal market regime clustering."""

    def optimize(self, X, n_regimes_range=(2, 10), pop_size=50, generations=100):
        # Each chromosome = (n_clusters, feature_weights, method)
        # method ∈ {kmeans, gmm, agglomerative}
        # Fitness = silhouette_score + regime_separability
        # Tournament selection with elitism
        # Two-point crossover on feature weight vectors
        # Gaussian mutation on weights, discrete mutation on n_clusters
        ...
```

**Commands:**
```bash
uv run scripts/optimize_regimes.py --symbol SPY --method ga --pop 50 --generations 100
```

### T1d: WOA Pattern Threshold Tuner

**Whale Optimization Algorithm for Pattern Detection:** WOA's spiral bubble-net hunting mimics how breakouts form:
- Whales spiral around prey before attacking — analogous to price coiling before breakout
- The spiral equation `X(t+1) = D' * e^(bl) * cos(2πl) + X*(t)` naturally captures the non-linear geometry of head-and-shoulders, wedges, and flags
- 50% probability of spiral (local exploitation) vs encircling (global exploration) balances precision vs robustness

**Implementation:**
```python
# src/ml/tuning/woa_thresholds.py
class WOAPatternTuner:
    """WOA-based optimization of pattern detection thresholds."""

    PARAMS = {
        'confidence_threshold': (0.3, 0.9),
        'volume_confirmation_mult': (0.5, 3.0),
        'min_body_ratio': (0.01, 0.5),
        'atr_stop_multiplier': (1.0, 4.0),
        'min_bars_since_signal': (1, 20),
    }

    def optimize(self, pattern_name, X, y, n_whales=30, max_iter=50) -> dict:
        ...
```

**Commands:**
```bash
uv run scripts/tune_patterns.py --pattern "VWAP Bounce" --method woa --whales 30
```

### T2: Glassbox Regime Classifier (InterpretML EBM)

**Why:** Regime predictions ("bear market") require trust. EBM is a GAM (Generalized Additive Model) that:
- Achieves accuracy close to GBDT but is fully interpretable
- Shows exactly how each feature contributes to regime prediction
- No black-box SHAP post-hoc explanations needed — it's natively transparent

**Implementation:**
```python
# src/ml/ebm_classifier.py
from interpret.glassbox import ExplainableBoostingClassifier

class EBMRegimeClassifier:
    def fit(self, X, y):
        self.model = ExplainableBoostingClassifier()
        self.model.fit(X, y)

    def explain(self, X) -> EBMExplanation:
        # Returns feature-wise contribution curves
        return self.model.explain_global()
```

**Commands:**
```bash
uv run scripts/train_ebm_regime.py --symbol SPY
```

### T3: AutoML Baseline (AutoGluon)

**Why:** AutoGluon automatically stacks CatBoost + LightGBM + XGBoost + neural nets,
tunes them, and ensembles the best. This gives an honest performance ceiling for
any hand-tuned model.

**Implementation:**
```python
# src/ml/automl_baseline.py
from autogluon.tabular import TabularPredictor

class AutoMLBaseline:
    def fit(self, train_data, label, time_limit=300):
        self.predictor = TabularPredictor(label=label).fit(
            train_data, time_limit=time_limit
        )
```

**Commands:**
```bash
uv run scripts/automl_baseline.py --symbol SPY --time-limit 600
```

### T4: Advanced Backtest Validation

**Why:** Without DSR/PSR/FDR, a strategy that appears to have Sharpe 2.0 could be
pure overfit. These stats are standard in quantitative finance literature
(López de Prado, Bailey).

**Implementation:**
```
src/analysis/
├── deflated_sharpe.py     # DSR + PSR computation
├── false_discovery.py     # Benjamini-Hochberg FDR control
└── cpcv.py                # Combinatorial Purged Cross-Validation
```

**Key metrics:**
- Deflated Sharpe Ratio (DSR): Sharpe adjusted for multiple testing
- Probabilistic Sharpe Ratio (PSR): Probability Sharpe > 0 is real
- FDR: Controls false discoveries when testing 34+ patterns simultaneously

### T5: Ensemble Methods

**Why:** Combining CatBoost + LightGBM + RF predictions via stacking meta-learner
typically improves AUC by 3-8% over any single model.

**Implementation:**
```python
# src/ml/ensemble.py
from sklearn.ensemble import StackingClassifier, VotingClassifier

class EnsembleClassifier:
    def __init__(self):
        self.base_models = [
            ('catboost', CatBoostClassifier(...)),
            ('lightgbm', LGBMClassifier(...)),
            ('random_forest', RandomForestClassifier(...)),
        ]
        self.stacking = StackingClassifier(
            estimators=self.base_models,
            final_estimator=LogisticRegression(),
            cv=PurgedKFold(n_splits=5),
        )
```

---

## Task Details — Tier 2 (selected highlights)

### T6: Monte Carlo VaR + CVaR
Replace the binomial approximation in `src/risk/position_probability.py` with:
- Historical VaR (resampling actual returns)
- Monte Carlo VaR (parametric with fat tails — Student's t)
- CVaR (Expected Shortfall) — average loss in worst 5% of cases

### T8: SHAP Visualization Dashboard
SHAP is already installed and integrated. Just add visualization code:
- `beeswarm_plot()` — which features matter most across all predictions
- `waterfall_plot()` — how each feature pushed this specific prediction
- `force_plot()` — interactive single-prediction explanation
- `dependence_plot()` — how a feature's impact changes with its value

---

## Broader ML Use Cases — Beyond Regime Classification

The project's pipeline has 8 stages where ML directly improves outcomes.
Regime classification is only one of them.

### Pipeline Stage Map

```
Data → Features → Pattern Detection → Signals → Risk → Portfolio → Execution → Backtest
  ↑       ↑            ↑              ↑       ↑        ↑          ↑          ↑
 ML    ML feat     ML pattern       ML     ML     ML port   ML exec    ML validation
prep  selection   quality score   weighting sizing allocation timing    stats
```

### 1. Data Preparation

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Missing data imputation | Forward fill | GBM-based imputation with regime awareness | `src/ml/data_imputer.py` |
| Outlier detection | None | Isolation Forest / LOF for anomalous bars | `src/ml/outlier_detector.py` |
| Tick quality scoring | None | ML scores bar reliability (gaps, volume anomalies) | `src/ml/bar_quality.py` |

### 2. Feature Selection

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Indicator selection | 132+ features, manual selection | **ARO** picks ~15 predictive features per regime | `src/ml/tuning/aro_selector.py` (T1a) |
| Feature importance tracking | Static FeatureSelector | Time-varying MI scores — which indicators matter *now*? | `src/ml/adaptive_features.py` |

### 3. Pattern Detection Quality

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Pattern confidence scoring | Rule-based (confidence_base hardcoded) | ML scores each detection's probability of being a true pattern | `src/ml/pattern_scorer.py` |
| False positive filtering | None | Binary classifier: real pattern vs noise | Already partially in `PatternClassifier` |
| Pattern template matching | 34 rule-based detectors | Siamese network for pattern similarity (GPU) | `src/ml/pattern_embedding.py` |

### 4. Signal Aggregation

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Optimal signal weights | 7 fixed aggregation methods | Learned weights via time-decayed regression | `src/ml/signal_weights.py` |
| Pattern confluence scoring | Fixed rules in EventWeightedAggregator | GBDT learns which pattern combinations predict profitable moves | `src/ml/confluence_scorer.py` |
| Signal timing optimization | Pattern fires → trade | ML predicts optimal delay (wait 1 bar? 2 bars?) | `src/ml/signal_timer.py` |

### 5. Risk Management

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Dynamic position sizing | 6 fixed methods | ML selects best sizing method per regime/volatility | `src/ml/adaptive_sizing.py` |
| Stop-loss optimization | Fixed ATR multiplier | GBDT predicts optimal stop distance per trade context | `src/ml/stop_loss_optimizer.py` |
| Drawdown prediction | Reactive (circuit breaker trips after loss) | ML predicts probability of hitting drawdown limit in next n bars | `src/ml/drawdown_predictor.py` |
| Monte Carlo VaR | Binomial approximation only | Full MC VaR with fat tails | `src/risk/mc_var.py` (T6) |

### 6. Portfolio Allocation

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Regime-specific allocation | Manual | ML learns allocation weights per regime | `src/ml/regime_allocator.py` |
| Strategy weight optimization | Static equal/sharpe/inv_vol | Reinforcement learning for dynamic strategy weighting | `src/ml/rl_allocator.py` |
| Correlation regime detection | None | HMM on rolling correlation matrix — when do diversifiers stop working? | `src/ml/correlation_regime.py` |

### 7. Trade Execution

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Exit timing | Fixed hold period or SL/TP | ML predicts probability of further favorable move vs reversal | `src/ml/exit_predictor.py` |
| Pre-trade success probability | `CrashFactorModel` (binary) | Gradient-boosted probability of profitable trade before entry | `src/ml/pre_trade_scorer.py` |
| Meta-labeling | None | López de Prado triple-barrier: should we take this signal? | `src/ml/meta_labeler.py` (T9) |

### 8. Backtest Validation

| Use Case | Current State | ML Enhancement | New Module |
|----------|--------------|----------------|------------|
| Overfit detection | WalkForwardValidator (4 levels) | DSR + PSR + FDR for rigorous multiple-testing control | `src/analysis/deflated_sharpe.py` (T4) |
| Strategy comparison | Manual | ML ranks strategies by probability of OOS outperformance | `src/ml/strategy_ranker_ml.py` |
| Synthetic data testing | None | GAN-generated synthetic price series for stress testing | `src/ml/synthetic_markets.py` |

### Quick Wins — Top 5 by Impact/Effort Ratio

| # | Use Case | Why High Impact | Prerequisites |
|---|----------|----------------|--------------|
| 1 | **Pattern confidence scoring** | Filters false positives from 34 detectors — directly improves win rate | PatternClassifier exists, just needs scoring mode |
| 2 | **Stop-loss optimization** | Even 0.2 ATR improvement on stops adds 5-10% to total return | Requires trade history data |
| 3 | **Confluence scoring** | Learns which pattern combos work (e.g., doji + VWAP bounce) — 34 patterns = 561 pairs | EventWeightedAggregator already has mappings |
| 4 | **Pre-trade success probability** | Score every signal before execution — reject low-probability trades | FeatureExtractor already provides features |
| 5 | **Dynamic position sizing** | Kelly during low-vol regimes, fixed-fractional during high-vol — adapts to conditions | PositionSizer supports multiple methods already |

### Interaction Matrix

Which ML enhancements amplify each other:

| Base Enhancement | Amplified By | Mechanism |
|-----------------|-------------|-----------|
| Pattern scoring | T1a (ARO features) | Better features → better pattern classification |
| Stop-loss optimization | T6 (MC VaR) | Accurate risk estimates → optimal stop placement |
| Confluence scoring | T5 (Ensemble) | Multi-model consensus on pattern combinations |
| Pre-trade scoring | T1b (GWO tuning) | Tuned CatBoost → calibrated probabilities |
| Dynamic sizing | T1c (GA regimes) | Correct regime identification → correct sizing method |

---

## Implementation Structure

```
src/ml/tuning/
├── __init__.py
├── aro_selector.py        # T1a: ARO feature selection
├── gwo_tuner.py            # T1b: GWO CatBoost HP tuning
├── ga_regime.py            # T1c: GA regime optimization
├── woa_thresholds.py       # T1d: WOA pattern threshold tuning
└── tuning_pipeline.py      # Unified API: pipeline.optimize(X, y, task='catboost')

src/ml/
├── pattern_scorer.py       # Quick win #1: ML pattern confidence
├── confluence_scorer.py    # Quick win #3: learned pattern pair scoring
├── pre_trade_scorer.py     # Quick win #4: pre-entry success probability
├── stop_loss_optimizer.py  # Quick win #2: optimal stop distance
├── adaptive_sizing.py      # Quick win #5: regime-aware position sizing
├── signal_weights.py       # Learned aggregation weights
├── exit_predictor.py       # Optimal exit timing
├── meta_labeler.py         # T9: triple-barrier meta-labeling
└── regime_allocator.py     # ML portfolio allocation per regime
```

---

## Dependency Graph

```
ARO (T1a: feature selection)
 ├── Enables GWO (T1b: smaller HP search space)
 ├── Enables Pattern Scoring (fewer features = faster training)
 └── Enables Ensemble (T5: tuned features)

GWO (T1b: CatBoost tuning)
 ├── Enables Pre-Trade Scoring (calibrated probabilities)
 ├── Enables Confluence Scoring (tuned base models)
 └── Enables Ensemble (T5: tuned base models)

GA (T1c: regime optimization)
 ├── Enables Dynamic Sizing (correct regime → correct method)
 ├── Enables Regime Allocator (portfolio weights per regime)
 └── Enables EBM (T2: glassbox needs well-defined regimes)

WOA (T1d: pattern tuning)
 └── Enables Pattern Scoring (optimized thresholds → better signals)

InterpretML EBM (T2)
 └── Complementary to SHAP (T8) — glassbox vs post-hoc

AutoGluon (T3)
 └── Sets performance ceiling for T1-T5

DSR/PSR/FDR (T4)
 └── Validates everything — prevents false discoveries

Ensemble (T5)
 └── Depends on T1a + T1b for tuned base models
```

## Integration Points

| New Module | Touches Existing | Risk |
|------------|-----------------|------|
| `src/ml/tuning/` | `signal_scorer.py`, `regime_model.py`, `pattern_classifier.py` | Low — wrapper layer, doesn't modify existing |
| `src/ml/ebm_classifier.py` | `regime_model.py` | Low — new class, same `RegimeDetectorBase` interface |
| `src/ml/automl_baseline.py` | `feature_engineering.py` | Low — uses same FeatureExtractor |
| `src/analysis/deflated_sharpe.py` | `metrics.py`, walk-forward validators | Low — pure computation |
| `src/ml/ensemble.py` | `pattern_classifier.py`, `regime_model.py` | Low — wraps existing models |
| `src/ml/pattern_scorer.py` | `PatternClassifier`, 34 detectors | Medium — touches signal pipeline |
| `src/ml/pre_trade_scorer.py` | `CrashFactorModel`, `PositionManager` | Medium — gate before execution |
| `src/ml/adaptive_sizing.py` | `PositionSizer` (6 methods) | Low — method selector, not replacer |

---

## Friend's Pipeline Recommendations — External Review Additions

> Source: Friend's AI-assisted OHLCV pipeline analysis (2026-05-07). 20 supervised + unsupervised guesses evaluated against actual codebase. Items marked 🔄 already partially exist. Items marked ⭐ are net-new additions.

### Tier 1 — High Impact, Moderate-Low Effort (Build Now)

| # | Task | Source Guess | Library | Effort | Impact |
|---|------|-------------|---------|--------|--------|
| FS1 | Survival Analysis for Time-to-Target | Supervised #8 | `scikit-survival` (skill available) | Medium | High — predicts *when* TP/SL hits, not just if. Natively fits multi-horizon labeling. Censored regression handles trades still open at window end. |
| FS2 | Historical Analog Matching (k-NN) | Unsupervised #5 | `sklearn.NearestNeighbors` or `faiss` (skill available) | Low | High — "when did market look like this before?" Shows what happened next in 10 most similar historical segments. Directly useful for traders. |
| FS3 | MAE/Drawdown as Primary Target | Supervised #10 | Existing `feature_engineering.py` (already computes `max_drawdown_N`) | Low | High — predict worst-case drawdown before pattern resolves. Feeds into position sizing: dynamic stops instead of fixed ATR multiples. |
| FS4 | Triple Barrier Labeling (elevated from T9) | Friend's pipeline recommendation | Custom (`PositionManager` already has TP/SL levels) | Medium | High — make labels match actual trade mechanics. Current: "profitable within N bars?" Real: "hits TP before SL within time limit?" This is the biggest disconnect between ML training and backtest execution. |
| FS5 | GMM Soft Regime Assignments | Friend's pipeline recommendation | `sklearn.mixture.GaussianMixture` | Low | Medium — swap KMeans→GMM in `pca_kmeans_regime.py`. Soft regime probabilities (proba_0..K) become features for PatternClassifier. KMeans forces hard boundaries; GMM gives regime confidence. |
| FS6 | Combinatorial Purged CV | Friend's pipeline recommendation | Extend existing `src/ml/purged_cv.py` | Medium | Medium — current PurgedKFold removes label overlap at boundaries. Combinatorial variant removes ALL training samples whose label windows overlap with ANY test sample. Stricter anti-leakage. |

### Tier 2 — Medium Impact, Medium Effort (Build After Tier 1)

| # | Task | Source Guess | Library | Effort | Impact |
|---|------|-------------|---------|--------|--------|
| FS7 | Change-Point Detection for Regimes | Unsupervised #6 | `ruptures` (pure Python) | Medium | Medium — catch regime shifts in real-time instead of waiting for rolling windows. Plugs into `RegimeDetectorBase` architecture. |
| FS8 | Volatility Forecasting Model | Supervised #3 | `catboost` (existing) | Medium | Medium — predict realized vol over next N bars. Feeds dynamic position sizing (wider stops in high-vol, tighter in low-vol) and option-like strategy triggers. |
| FS9 | Volume-Price Profile Clustering | Unsupervised #7 | `sklearn.cluster` | Medium | Medium — detect accumulation vs distribution phases from volume-at-price distributions. Complements existing price-based regime detection with volume intelligence. |
| FS10 | HDBSCAN Anomaly Detection | Unsupervised #2 | `hdbscan` | Low | Medium — auto-detect flash crashes, liquidity gaps, spoofing without forcing K clusters. Feed anomaly_score into existing circuit breakers (`src/risk/circuit_breakers.py`). |
| FS11 | Fractional Differentiation | Friend's pipeline recommendation | `statsmodels` (already installed) | Low | Medium — apply `fracdiff(d≈0.3-0.5)` to price-derived features. Preserves memory while achieving stationarity. Current StandardScaler leaks future regime information. |
| FS12 | Cross-Symbol Dynamic Clustering | Unsupervised #10 | `sklearn.cluster` + existing `CrossAssetFeatures` | Medium | Medium — cluster instruments by rolling OHLCV behavior to detect lead-lag relationships and correlation regime shifts. Builds on existing `cross_asset_features.py`. |
| FS13 | Breakout Probability ML | Supervised #6 | `catboost` (existing) + Donchian detector | Low | Medium — ML score on Donchian breakouts: is this a true breakout or a false one? Use volume confirmation, preceding volatility, and regime context as features. |

### Tier 3 — Lower Priority / Research / Defer

| # | Task | Source Guess | Library | Effort | Rationale for Deferral |
|---|------|-------------|---------|--------|----------------------|
| FS14 | Volume/Dollar/Tick Bars | Friend's pipeline recommendation | Custom bar builder | High | High effort: requires re-basing all 34 pattern detectors to alt bars. Defer until pattern system stabilizes. |
| FS15 | Latent Market Embeddings (VAE) | Unsupervised #4 | `torch` (GPU) | High | GPU-dependent. Interesting research but unclear path to trading edge. Defer to Phase 05 (ML Advanced). |
| FS16 | Unsupervised Pattern Discovery (Shapelets/DTW) | Unsupervised #3 | `aeon` (skill available) | High | Competes with 34 proven rule-based detectors. Research value only — not production trading. Defer to Phase 06/auto-research. |
| FS17 | Volume Anomaly Forecasting | Supervised #9 | `catboost` (existing) | Low | Low standalone value. Volume anomalies are features for other models, not a standalone target. Absorb into FS10 (HDBSCAN anomaly detection). |
| FS18 | Synthetic OHLCV via GANs/VAEs | Unsupervised #9 | `torch` (GPU) | High | Already noted in broader use cases. GANs are unstable, hard to validate realism. Defer indefinitely. |
| FS19 | Gap-Fill Prediction | Supervised #7 | `catboost` (existing) | Low | Niche case. Gaps are rare in crypto, infrequent in equities. Very small edge. |
| FS20 | Heikin-Ashi Integration | Friend's pipeline recommendation | Custom bar converter | Medium | Already listed as feature request FR-005. Smoothes noise but loses OHLC precision needed for pattern detection. Defer until after Tier 1 bar improvements. |
| FS21 | Sparse PCA for Dimension Reduction | Friend's pipeline recommendation | `sklearn.decomposition.SparsePCA` | Low | Low — current feature selector (variance→correlation→mutual info→SFI) is already more effective than PCA for interpretability. SparsePCA helps visualization, not prediction. |

### Tier Adjustments to Existing Tasks

| Existing # | Original Tier | New Tier | Reason |
|-----------|---------------|----------|--------|
| T9 (Meta-Labeling / Triple Barrier) | Tier 2 | **Tier 1** | Friend's external review confirms this is the single biggest gap. Labels currently don't match trade mechanics. Now tracked as FS4 (implementation) with T9 kept as the broader meta-labeling research task. |
| T6 (MC VaR) | Tier 2 | Tier 2 | No change. Complements FS3 (MAE prediction) — MC VaR is distribution-level, MAE is observation-level. Both needed. |

### New Dependency Graph (Friend Additions)

```
Survival Analysis (FS1)
 ├── Uses existing multi-horizon labeling (horizon=1,3,5,10,20)
 ├── Uses PositionManager TP/SL levels for censorship
 └── Unblocks: better exit timing, dynamic stop optimization

Historical Analog Matching (FS2)
 ├── Uses rolling window features (already computed in FeatureExtractor)
 └── Unblocks: deployment confidence tool, trader-facing dashboard

MAE/Drawdown Target (FS3)
 ├── Uses existing max_drawdown_N computation
 ├── Feeds into PositionSizer (dynamic stops)
 └── Amplified by T6 (MC VaR) — distribution + observation-level risk

Triple Barrier Labeling (FS4)
 ├── Uses PositionManager TP/SL levels
 ├── Replaces current close.shift(-N) labeling in ALL scripts
 └── Unblocks: FS1 (survival analysis), FS3 (MAE), better win rate calibration

GMM Regimes (FS5)
 └── Feeds regime_proba columns into PatternClassifier, FS1, FS3

Combinatorial Purged CV (FS6)
 └── Validates FS1-FS4 — prevents inflated scores from remaining label overlap
```

### Execution Order (Friend Additions Integrated)

```
Phase 1: Foundation Fixes (enable everything below)
 ├── T4 (DSR/PSR/FDR) — P1, unblocks validation
 ├── FS4 (Triple Barrier) — P1, fixes label/trade disconnect
 └── FS6 (Combinatorial Purged CV) — P1, strict anti-leakage

Phase 2: High-Impact Quick Wins
 ├── FS2 (Historical Analog Matching) — lowest effort, high trader value
 ├── FS3 (MAE/Drawdown Target) — uses existing data, feeds position sizing
 ├── FS5 (GMM Regimes) — 1-line swap, soft probabilities
 └── FS11 (Fractional Differentiation) — 1 library call, stationarity

Phase 3: New Models
 ├── FS1 (Survival Analysis) — highest technical value, requires FS4 completion
 ├── FS8 (Volatility Forecasting) — feeds dynamic sizing
 └── FS13 (Breakout Probability ML) — low effort, high false-positive payoff

Phase 4: Advanced Pipeline
 ├── FS7 (Change-Point Detection) — better regime timing
 ├── FS9 (Volume-Price Profile Clustering) — accumulation/distribution
 ├── FS10 (HDBSCAN Anomalies) — risk overlay
 └── FS12 (Cross-Symbol Clustering) — portfolio-level

Phase 5: Deferred / GPU
 └── FS14-FS21 — revisit after Phase 05 (ML Advanced) unblocked
```
