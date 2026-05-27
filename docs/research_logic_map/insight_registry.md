# Insight Registry

**Last Updated:** 2026-05-26
**Total Insights:** 116
**Papers Analyzed:** 33 (30 papers + 3 new resources) + 66-paper Master Comparison Report Phase 25

---

## Tag Legend

### Topics
- **Risk**: Risk management, position sizing, drawdown control
- **Friction**: Transaction costs, turnover, slippage
- **Regime**: Market regime detection, adaptive strategies
- **Signal.Quality**: Signal aggregation, confluence scoring, filtering
- **Event.Type**: Event-type extraction, sentiment analysis
- **Position.Sizing**: Capital allocation, Kelly criterion
- **Portfolio**: Portfolio construction, diversity, correlation
- **Strategy.Lifecycle**: Strategy decay, retirement, validation
- **Market.Efficiency**: Alpha decay, arbitrage dynamics, instrument selection
- **Instrument.Selection**: Universe construction, efficiency screening, asset class choice
- **ML.Training**: Training data strategy, transfer learning, multi-asset generalization

### Impact Levels
- 🔴 **Critical**: Must implement; alpha-destroying if ignored
- 🟠 **High**: Significant performance impact
- 🟡 **Medium**: Nice to have; incremental improvement
- 🟢 **Low**: Marginal benefit

### Status
- ✅ **Implemented**: Code exists in `src/`
- 🔄 **In-Progress**: Currently being implemented
- ⏳ **Deferred**: Backlogged; not yet started
- ❌ **Rejected**: Explicitly decided against

---

## Insights by Paper

### P1: OOM-RL (arXiv:2604.11477)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I1.1 | Turnover penalty as hard constraint | Friction | 🔴 Critical | ✅ | `src/backtest/friction_scoring.py:461-510` |
| I1.2 | Dynamic rebalancing frequency (daily→weekly) | Signal.Decay | 🟠 High | ✅ | Phase 11 validation |
| I1.3 | Structured drawdown diagnostics ("Epistemic Autopsy") | Risk | 🔴 Critical | ⏳ | Backlog #3 |
| I1.4 | Liquidity loading filter for position sizing | Friction | 🟡 Medium | ⏳ | Backlog #8 |
| I1.5 | Immutable test suites for strategy code | Strategy.Lifecycle | 🟡 Medium | ❌ | Rejected (overhead > benefit) |

### P2: Investing Is Compression (arXiv:2604.10758)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I2.1 | Divergence-in-bits metric for strategy comparison | Signal.Quality | 🟠 High | ⏳ | Backlog #11 |
| I2.2 | Winner-fraction position sizing heuristic | Position.Sizing | 🟠 High | ⏳ | Backlog #12 |
| I2.3 | MDL (Minimum Description Length) for feature pruning | Signal.Quality | 🟡 Medium | ⏳ | Backlog #13 |

### P3: Risk Management for Event-Driven Funds (SSRN-1018281)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I3.1 | Per-position success/failure probability (binary outcomes) | Risk | 🔴 Critical | ✅ | `src/risk/position_sizing.py` |
| I3.2 | Diversity score for portfolio construction (BET formula) | Portfolio | 🟠 High | ✅ | `src/risk/diversity_score.py` |
| I3.3 | Market regime → break probability linkage | Regime | 🟡 Medium | ⏳ | Backlog #14 |
| I3.4 | Economic capital sizing at 99.9% VaR | Risk | 🟠 High | ✅ | `src/risk/daily_limits.py` |
| I3.5 | Optimal portfolio size tradeoff (N vs. expertise dilution) | Portfolio | 🟡 Medium | ⏳ | Backlog #15 |

### P4: Against a Universal Trading Strategy (arXiv:2604.13334)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I4.1 | Regime declaration mandatory per strategy | Regime | 🔴 Critical | ✅ | `src/indicators/regime_detector.py` |
| I4.2 | Failure-set analyzers (time-reversal, counter-trend, fat-tail) | Risk | 🟠 High | ⏳ | Backlog #9 |
| I4.3 | Strategy decay monitoring & automatic retirement | Strategy.Lifecycle | 🟠 High | ⏳ | Backlog #16 |
| I4.4 | Cascade risk prevention (portfolio-level circuit breakers) | Risk | 🔴 Critical | ✅ | `src/risk/daily_limits.py:308-411` |
| I4.5 | Undecidability humility → defense in depth | Risk | 🟡 Medium | ✅ | Position limits, stop-losses, drawdown halts |

### P5: Survey of Statistical Arbitrage Pair Trading (WNE_WP485)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I5.1 | Add pairs trading as 8th pattern category | Strategy.Lifecycle | 🟡 Medium | ⏳ | Backlog #17 |
| I5.2 | Cointegration + ML hybrid for pair selection | Signal.Quality | 🟡 Medium | ⏳ | Backlog #18 |
| I5.3 | Volatility-adaptive thresholds (dynamic z-scores) | Signal.Quality | 🟠 High | ⏳ | Backlog #19 |

### P6: Event-Based Trading: IE Tools (SSRN-2907600)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I6.1 | Event-type signals outperform aggregated sentiment | Event.Type | 🟠 High | ⏳ | Backlog #6 |
| I6.2 | Holding-period alignment per event type | Event.Type | 🟡 Medium | ⏳ | Backlog #20 |
| I6.3 | Event-type taxonomy for signal classification | Event.Type | 🟡 Medium | ⏳ | Backlog #21 |

### P7: Multimodal Event-driven LSTM (IEEE Access)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I7.1 | Tensor-based signal fusion (not vector concatenation) | Signal.Quality | 🟡 Medium | ⏳ | Research phase |
| I7.2 | Event-driven memory updates (not fixed time steps) | Signal.Quality | 🟡 Medium | ❌ | Rejected (complexity > benefit) |
| I7.3 | Co-movement signals (sector peer correlations) | Portfolio | 🟡 Medium | ⏳ | Backlog #22 |

### P8: Algorithmic Trading & AI Review (WJAETS-2024-0054)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I8.1 | Portfolio-level circuit breakers essential | Risk | 🔴 Critical | ✅ | `src/risk/daily_limits.py` |
| I8.2 | Adaptive parameter tuning by market conditions | Regime | 🟠 High | ✅ | `src/strategies/adaptive_router.py` |
| I8.3 | Stress scenario testing (flash crash, liquidity drought) | Risk | 🟡 Medium | ⏳ | Backlog #23 |

### P9: Calendar of Events Database (IEEE Access)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I9.1 | Event database as feature store | Event.Type | 🟡 Medium | ⏳ | Backlog #24 |
| I9.2 | Spike detection for signal triggers | Signal.Quality | 🟡 Medium | ⏳ | Backlog #25 |
| I9.3 | Keyword-signal mapping by sector | Event.Type | 🟢 Low | ⏳ | Backlog #26 |

### P10: Emergence of Statistical Financial Factors (arXiv)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I10.1 | Network-derived factor model (alternative to PCA) | Portfolio | 🟡 Medium | ⏳ | Research phase |
| I10.2 | Coupling matrix for stock relationships | Portfolio | 🟢 Low | ⏳ | Research phase |
| I10.3 | Optimal regime for factor emergence | Regime | 🟢 Low | ⏳ | Research phase |

### P13: Crash-Based Quantitative Trading Strategies (Finance Research Letters, 2022)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I13.1 | Crash factor as pre-trade risk filter | Risk | 🟠 High | ⏳ | Backlog #27 |
| I13.2 | Fixed take-profit (3%) > RSI exits | Signal.Quality | 🟠 High | ✅ | `src/signals/position_manager.py` |
| I13.3 | CMRS scoring for signal ranking | Signal.Quality | 🟡 Medium | ⏳ | Backlog #28 |
| I13.4 | Momentum crash survivability via crash-factor adjustment | Regime | 🟠 High | ⏳ | Backlog #29 |

### P14: Global Market Inefficiencies (Bartram & Grinblatt, 2019)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I14.1 | Alpha 40-70 bps/month higher in emerging vs. developed markets | Market.Efficiency | 🟠 High | 🔬 | Research note: market_efficiency_and_instrument_selection.md |
| I14.2 | Country's pre-cost alpha positively correlated with trading costs | Market.Efficiency | 🟠 High | 🔬 | Friction deters arbitrageurs → alpha persists |
| I14.3 | Global equity markets are inefficient, especially where frictions exist | Market.Efficiency | 🟡 Medium | 🔬 | Justifies instrument universe expansion |

### P15: When Systematic Strategies Decay (Falck, Rej, Thesmar, 2021)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I15.1 | Published anomalies lose ~5ppt Sharpe per year post-publication | Strategy.Lifecycle | 🔴 Critical | 🔬 | Applies to our pattern catalog — need decay monitoring |
| I15.2 | Year of publication alone explains 30% of Sharpe decay variance | Strategy.Lifecycle | 🟠 High | 🔬 | Newer signals decay faster |
| I15.3 | Formula complexity and sensitivity to outliers predict decay | Signal.Quality | 🟠 High | ⏳ | Simpler signals may be more robust |

### P16: ML on Trades & Holdings (DeMiguel, Sang, Zhang, 2024)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I16.1 | ML predictability stronger for smaller/illiquid stocks | Market.Efficiency | 🔴 Critical | 🔬 | Directly supports expanding to mid/small-caps |
| I16.2 | Predictability stronger with lower analyst coverage | Market.Efficiency | 🟠 High | 🔬 | Use analyst coverage as universe filter |
| I16.3 | Nonlinear interactions in participant trades reveal price discovery info | ML.Training | 🟡 Medium | 🔬 | Use nonlinear models (XGBoost, NNs) to capture interactions |

### P17: Factor Investing with Delays (Dickerson, Robotti, Nozawa, 2024)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I17.1 | ML strategies beat corporate bonds before costs, fail after delay costs | Friction | 🔴 Critical | 🔬 | Illiquidity is double-edged — signal exists but execution kills it |
| I17.2 | Transaction delays in illiquid securities can fully erase alpha | Friction | 🟠 High | 🔬 | Need delay-aware backtesting for illiquid instruments |
| I17.3 | Large number of bond factors outperform before costs, not after | Strategy.Lifecycle | 🟡 Medium | 🔬 | Cost adjustment is essential for strategy evaluation |

### P18: Beyond Fama-French — Integrating Default, Liquidity, Momentum Factors (AI Survey, 2025)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I18.1 | Default Risk (Merton DtD) + Liquidity (Acharya-Pedersen CEI) as alpha sources | Factor.Modeling | 🟠 High | 🔬 | Factor Engine library (`github.com`) available for computation |
| I18.2 | LLM + MCTS for automated alpha factor mining (IC 0.055 vs 0.046 GP baseline) | Factor.Modeling | 🟠 High | 🔬 | Multi-dimensional scoring: IC, RankIR, turnover, diversity, overfitting risk |
| I18.3 | QRAFTI standardized evaluation protocol (Novy-Marx/Velikov 2023) for factor validation | Factor.Modeling | 🟡 Medium | 🔬 | Standardized diagnostics pipeline — adopt for model validation |
| I18.4 | Non-linear factor architectures (MFIN, KAN Autoencoders, Tensor Factor Models) outperform linear | Factor.Modeling | 🟢 Low | 🔬 | Gated on GPU >= 8GB |

### P19: 华泰多因子系列1 — Multi-Factor Model System (Huatai Securities, 2016)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I19.1 | 12-category factor taxonomy (74 style factors) for systematic feature engineering | Factor.Modeling | 🟠 High | 🔬 | Value, Growth, Quality, Leverage, Size, Momentum, Volatility, Turnover, Modified Momentum, Sentiment, Shareholder, Technical |
| I19.2 | 4-phase construction pipeline: Preparation → Return Model → Risk Model → Optimization | Factor.Modeling | 🟠 High | 🔬 | Mirrors this project's ML pipeline conceptually |
| I19.3 | Factor purification — regress out sector/size before IC computation eliminates confounding | Factor.Modeling | 🟠 High | ⏳ | Adopt for signal quality assessment in pattern detectors |
| I19.4 | HP filter for factor return forecasting (extract trend from cumulative return) outperforms ARIMA/EWMA | Factor.Modeling | 🟡 Medium | ⏳ | Replace simple mean with HP filter for expected return estimation |
| I19.5 | IR-weighted category factor synthesis (accounts for return AND volatility) superior to equal-weight/PCA | Factor.Modeling | 🟡 Medium | ⏳ | Apply to pattern category aggregation in confluence scoring |

### P20: FMZ Strategies Repository — 5,807 Trading Strategies (Crowd-Sourced, 2017-2025)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I20.1 | Only 1 ML strategy in 5,807 files — confirms ML is genuine differentiator, not commodity | Strategy.Conversion | 🔴 Critical | ✅ | This project's CatBoost pipeline has no equivalent in the FMZ corpus |
| I20.2 | Multi-confirmation designs (triple/quad indicator) survive crowd-testing; single-indicator strategies dominate but are noise | Strategy.Conversion | 🟠 High | ✅ | `src/patterns/fmz/alpha_beast.py`, `multi_factor_trend.py` |
| I20.3 | ATR-based risk management is universal standard across 5,807 strategies — no better alternative found | Strategy.Conversion | 🟠 High | ✅ | Already used in RulesFirstStrategy; confirmed by corpus evidence |
| I20.4 | Zero OOS validation in most strategies — confirms this project's walk-forward/IS-OOS split is genuine alpha protection | Strategy.Conversion | 🔴 Critical | ✅ | IS/OOS split in all backtest scripts |
| I20.5 | PineScript → Python manual conversion is feasible when pattern abstraction (BasePattern) exists; ~17 helper functions cover 90%+ of tactics | Strategy.Conversion | 🟡 Medium | ✅ | `src/indicators/pinescript_helpers.py`, `src/patterns/fmz/` |
| I20.6 | Force detection (RSI momentum validation on prior swing legs) improves reversal signal quality by filtering weak reversals after strong momentum | Strategy.Conversion | 🟠 High | ✅ | `src/patterns/fmz/momentum_zigzag.py` — QQE/MACD/MA modes |

### P21: Constrained LLM Agents for Factor Discovery (arXiv:2604.26747v1)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I21.1 | Constrained Factor DSL for reproducible signal definitions — 4 operator families over point-in-time variables | Signal.Quality | 🔴 Critical | ✅ | `src/patterns/dsl/` (grammar, executor, validator) |
| I21.2 | Agentic sequential hypothesis search with deterministic evaluation engine separation | Signal.Quality | 🟠 High | ✅ | `src/patterns/dsl/trace.py` (FactorTrace, append_round) |
| I21.3 | IC-based selection gates (mean IC ≥ 0.02, IC t-stat ≥ 2.0, coverage ≥ 0.70) on training window only | Signal.Quality | 🔴 Critical | ✅ | `src/signals/ic_gate.py` (ICGate) |
| I21.4 | Append-only experiment trace for full auditability — hypothesis→recipe→metrics→gate→interpretation | Strategy.Lifecycle | 🟠 High | ✅ | `src/patterns/dsl/trace.py` (verify_integrity) |
| I21.5 | Pool governance: hold pool (pass gate) → good pool (curated for mechanism diversity, corr < 0.7) | Signal.Quality | 🟠 High | ✅ | `src/patterns/dsl/trace.py` (curate_good_pool) |
| I21.6 | Ridge-regularized factor aggregation (α=1.0, cross-sectional standardization) outperforms complex composites | Signal.Quality | 🟠 High | ✅ | `src/signals/ridge_combiner.py` (RidgeSignalCombiner) |
| I21.7 | Range-persistence (hl_range MA crossover) as speculative attention proxy — paper's top-performing factor family | Signal.Quality | 🟠 High | ✅ | `src/patterns/range_persistence.py` (3 detectors) |
| I21.8 | Capacity analysis: equal-weight vs market-cap-weight divergence reveals alpha concentration in small assets | Friction | 🟡 Medium | ⏳ | Backlog #23 |
| I21.9 | Protocol immutability enforcement (config hash, frozen params after session start) | Strategy.Lifecycle | 🟡 Medium | ⏳ | Backlog #24 |
| I21.10 | Mechanical vs hypothesis candidate balance (60/30/10 mix) to balance exploitation/exploration | Signal.Quality | 🟡 Medium | ⏳ | Backlog #25 |
| I21.11 | Failure interpretation framework — 6 categories (NOISE, REGIME_DEPENDENT, CAPACITY_LIMITED, REDUNDANT, DATA_ISSUE, HYPOTHESIS_INVALID) | Signal.Quality | 🟡 Medium | ✅ | `src/patterns/dsl/trace.py` (FailureCategory enum) |
| I21.12 | Small-cap + liquidity-scarcity + intraday range crypto factor convergence across 5 search rounds | Signal.Quality | 🟠 High | ✅ | `src/patterns/range_persistence.py` |

---

## Phase 25: Master Comparison Report — 66 Papers (2026-05-22)

All missing implementable items from `useful_resources/papers_md/MASTER_COMPARISON_REPORT_2026-05-21.md` now implemented.

### P25: Lock Box + Infrastructure Anti-Overfitting (C1, C2, C13)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| P25.1 | Lock Box methodology — blind holdout accessed exactly once after all decisions final | ML.Training | 🔴 Critical | ✅ | `src/ml/lock_box.py` (LockBox, create_lock_box, create_lock_box_chronological) |
| P25.2 | Nested cross-validation — inner loop tunes, outer loop evaluates, never mix | ML.Training | 🔴 Critical | ✅ | `src/ml/nested_cv.py` (NestedPurgedCV, leave_one_group_out_cv) |
| P25.3 | Blind analysis protocol — optimize on scrambled labels, evaluate once on true | ML.Training | 🟠 High | ✅ | `src/ml/blind_analysis.py` (run_blind_analysis) |
| P25.4 | Label-shuffling baseline test — verify model doesn't exceed random on shuffled targets | ML.Training | 🟠 High | ✅ | `src/ml/label_shuffling.py` (run_label_shuffling_test) |

### P25: SVM Regime + Dual Alpha/Beta (B34, E9)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| P25.5 | SVM market classifier using raw price sequences (82% precision) vs indicators (64%) | Regime | 🟠 High | ✅ | `src/ml/svm_regime.py` (SVMRegimeClassifier) |
| P25.6 | Dual alpha/beta — separate bull/bear alpha+beta, Chow test for structural breaks | Portfolio | 🟡 Medium | ✅ | `src/analysis/dual_alpha_beta.py` (compute_dual_alpha_beta) |

### P25: NLP Sentiment Pipeline (D1-D4)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| P25.7 | Distant supervision via emoticon labeling (80%+ accuracy, no hand-labeling) | Event.Type | 🟠 High | ✅ | `src/nlp/sentiment_pipeline.py` (label_via_emoticons, apply_distant_supervision) |
| P25.8 | SVM + TF-IDF sentiment (82-94% accuracy across 4 papers, ngram(1,2)) | Event.Type | 🟠 High | ✅ | `src/nlp/sentiment_pipeline.py` (SVMTfidfSentiment) |
| P25.9 | BiLSTM+LR architecture (128d embed→BiLSTM(64u)→dropout(0.25)→LR(C=10), 82.4%) | Event.Type | 🟡 Medium | ✅ | `src/nlp/sentiment_pipeline.py` (BiLSTMSentiment) |
| P25.10 | Ensemble RF+SVM+DT via AdaBoost (93.4% accuracy) | Signal.Quality | 🟡 Medium | ✅ | `src/nlp/sentiment_pipeline.py` (SentimentEnsemble) |

### P25: Fuzzy System + NSGA-II + Dynamic GA (B32, B33, B35)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| P25.11 | Fuzzy rule system with 5-state trapezoidal membership (Buy/Sell/Hold) | Signal.Quality | 🟠 High | ✅ | `src/signals/fuzzy_system.py` (FuzzyInferenceSystem) |
| P25.12 | NSGA-II multi-objective optimization (Pareto front for conflicting objectives) | Portfolio | 🟡 Medium | ✅ | `src/optimization/nsga2_optimizer.py` (NSGA2Optimizer) |
| P25.13 | Dynamic GA with associative memory per regime (hyper-mutation on regime shift) | Regime | 🟡 Medium | ✅ | `src/optimization/dynamic_ga.py` (DynamicGAOptimizer) |

### P26: TimeGAN — Volatility & Irregularity Capturing on DAX (Mushunje, Allen, Peiris — Columbia Univ/USYD)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I26.1 | TimeGAN outperforms LSTM/GRU/WGAN on shock-perturbed financial data (DAX 2010-2022, COVID shocks) | TimeSeries.Forecast | 🟠 High | ⏳ | GAN-based synthetic data generation |
| I26.2 | GANs capture stylized facts (fat-tails, kurtosis, long-range dependence) that traditional models miss | TimeSeries.Generation | 🟠 High | ⏳ | Synthetic OOS stress testing |
| I26.3 | Hybrid GAN+sequential models capture both distribution learning and temporal dependencies | ML.Architecture | 🟡 Medium | ⏳ | Combined GAN+LSTM for feature generation |

### P27: Wasserstein GAN-GP — Bitcoin Financial Time Series Generation (Pfenninger, Bigler, Rikli, Osterrieder — ZHAW/UTwente)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I27.1 | WGAN-GP with LSTM generator/discriminator generates visually indistinguishable Bitcoin price series | TimeSeries.Generation | 🟠 High | ⏳ | Synthetic crypto data augmentation |
| I27.2 | Wasserstein distance + gradient penalty stabilizes GAN training on volatile financial data | ML.Training | 🟡 Medium | ⏳ | Stable GAN training loss function |
| I27.3 | Generated data statistically close but distinguishable from real — QQ-plot + ACF evaluation framework | Validation | 🟡 Medium | ⏳ | Synthetic data quality metrics |

### P28: TTS-GAN — Transformer-Based GAN for Financial Time Series Augmentation (Podobinski, Chudziak — Warsaw UT)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I28.1 | **Transformer-based GAN (TTS-GAN) augments scarce financial data → improves LSTM forecasting accuracy** on BTC + S&P500 | TimeSeries.Augmentation | 🔴 Critical | ⏳ | `src/ml/data_augmentation.py` |
| I28.2 | Novel DTW+DeD-iMs convergence metric for monitoring GAN training quality on time series | ML.Training | 🟠 High | ⏳ | GAN training quality monitoring |
| I28.3 | Regime-shift shortens relevant data horizon → DL overfits; GAN augmentation bridges the gap | ML.Data | 🔴 Critical | ⏳ | Anti-overfitting via augmentation |

### P29: TsLLM — LLM Augmented for Time Series Understanding & Prediction (Parker, Chan, Zhang, Ghobadi — JHU)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I29.1 | Patch-based VAE encoder-decoder bridges LLMs to time series; scale-aware encoding decouples shape from magnitude | LLM.Architecture | 🟠 High | ⏳ | LLM-based contextual forecasting |
| I29.2 | Contextual forecasting conditioned on unstructured text (news/sentiment) via interleaved token sequences | NLP.Integration | 🔴 Critical | ⏳ | `src/nlp/contextual_forecast.py` |
| I29.3 | TsLLM achieves zero-shot/few-shot time series tasks (forecasting, classification, anomaly detection) without retraining | ML.Transfer | 🟡 Medium | ⏳ | Zero-shot regime detection |
| I29.4 | Text tokenization inflates numeric values to multiple tokens — raw LLMs blind to time series patterns | NLP.Limitation | 🟡 Medium | ✅ | Documented antipattern |

### P30: ALGAN — Adjusted-LSTM GAN for Anomaly Detection (Bashar, Nayak — QUT)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I30.1 | Attention-adjusted LSTM hidden states reduce information loss in long-sequence anomaly detection | Anomaly.Detection | 🟡 Medium | ⏳ | GAN-based regime detection |
| I30.2 | 46 univariate + 1 multivariate dataset benchmark; outperforms traditional + NN + GAN baselines | Anomaly.Detection | 🟡 Medium | ⏳ | Anomaly detection benchmark harness |

### P31: MIM-GAN — Message Importance Measure GAN for Multivariate Anomaly Detection (Lu, Dong, Cai, Fang, Zhao — Tibet Univ/JNU/Western)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I31.1 | Exponential information measure loss function avoids mode collapse in GAN training | ML.Training | 🟡 Medium | ⏳ | Stable GAN loss functions |
| I31.2 | Combined discriminator+reconstruction score for robust anomaly scoring | Anomaly.Detection | 🟡 Medium | ⏳ | Hybrid anomaly scoring |

### P32: TadGAN — Cycle-Consistent GAN for Time Series Anomaly Detection (Geiger, Liu, Alnegheimish, Cuesta-Infante, Veeramachaneni — MIT)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I32.1 | **Cycle-consistent GAN with LSTM generator/critic achieves highest averaged F1 across 11 benchmark datasets** (NASA, Yahoo, Numenta, Amazon, Twitter; 492 signals) | Anomaly.Detection | 🟠 High | ⏳ | `src/ml/anomaly_detection.py` |
| I32.2 | Dual anomaly score: reconstruction error + critic output combined via novel weighting schemes | Anomaly.Detection | 🟡 Medium | ⏳ | Anomaly scoring methodology |
| I32.3 | Open-source benchmarking system for time series anomaly detection with 9 pipelines + 13 datasets | Validation | 🟡 Medium | ⏳ | Anomaly detection test harness |

### P33: TSI-GAN — Unsupervised Time Series Anomaly Detection via Convolutional Cycle-Consistent GAN (Saravanan, Luo, Ngo — Missouri S&T/SUTD)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I33.1 | Time series → 2D image encoding enables convolutional GANs for anomaly detection; 13% over MERLIN on 250 datasets | Anomaly.Detection | 🟠 High | ⏳ | Image-based anomaly detection |
| I33.2 | Hodrick-Prescott filter post-processing reduces false positives in anomaly detection | Signal.Quality | 🟡 Medium | ⏳ | False positive reduction |
| I33.3 | Real-time inference via encoder-decoder (no latent optimization at inference time) | ML.Performance | 🟡 Medium | ⏳ | Production anomaly detection |

### P34: WaveletDiff — Multilevel Wavelet Diffusion for Time Series Generation (Wang, Milenkovic — UIUC)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I34.1 | Diffusion on wavelet coefficients preserves multi-resolution structure; 3× better discriminative scores than baselines | TimeSeries.Generation | 🟠 High | ⏳ | Wavelet-based synthetic data |
| I34.2 | Cross-level attention with adaptive gating enables selective information exchange between temporal/frequency scales | ML.Architecture | 🟡 Medium | ⏳ | Multi-scale feature fusion |
| I34.3 | Parseval's theorem energy preservation constraints maintain spectral fidelity during diffusion | TimeSeries.Generation | 🟡 Medium | ⏳ | Spectral fidelity verification |

### P35: WDformer — Wavelet-Based Differential Transformer for Time Series Forecasting (Wang, Zhang, Zheng, Jiang — Zhejiang Normal Univ)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I35.1 | **Differential attention mechanism** (difference of two softmax matrices) filters noise without signal loss | ML.Architecture | 🟠 High | ⏳ | `src/ml/models/differential_attention.py` |
| I35.2 | Wavelet transform + inverted-dimension attention captures multi-variate correlations better than standard attention | TimeSeries.Features | 🟡 Medium | ⏳ | Wavelet feature extraction pipeline |
| I35.3 | SOTA on multiple real-world datasets — demonstrated accuracy and effectiveness for financial forecasting | TimeSeries.Forecast | 🟡 Medium | ⏳ | Wavelet forecasting model |

### P36: AWEMixer — Adaptive Wavelet-Enhanced Mixer Network for Long-Term Forecasting (Li, Zhang, Tao, Wang, Pan, Wei — Xi'an Jiaotong)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I36.1 | Frequency Router: FFT finds global periodicities → adaptively weights localized wavelet subbands | Signal.Processing | 🟡 Medium | ⏳ | Frequency-aware feature engineering |
| I36.2 | Coherent Gated Fusion: cross-attention+gating integrates frequency features with multi-scale temporal representations | ML.Architecture | 🟡 Medium | ⏳ | Multi-scale feature fusion |
| I36.3 | Key insight — "when" matters more than "what frequency": FFT tells which frequencies exist, not when they appear; wavelets solve time-frequency localization | Signal.Processing | 🟠 High | ⏳ | Wavelet-based regime detection |
| I36.4 | 7 public benchmarks; beats transformer-based + MLP-based SOTA for long-sequence forecasting | TimeSeries.Forecast | 🟡 Medium | ⏳ | Long-horizon forecasting model |

### P37: DB2-TransF — Learnable Daubechies Wavelets Replace Self-Attention (Gupta, Tripathi — IIT Dharwad)

| ID | Insight | Topic | Impact | Status | Linked Feature |
|---|---------|-------|--------|--------|----------------|
| I37.1 | **Learnable Daubechies wavelet module replaces O(n²) self-attention** with linear complexity while preserving multi-scale pattern capture | ML.Architecture | 🟠 High | ⏳ | `src/ml/models/wavelet_attention.py` |
| I37.2 | 13 benchmark datasets; comparable/better accuracy at substantially lower compute vs transformers | ML.Performance | 🟡 Medium | ⏳ | Efficient wavelet transformer |
| I37.3 | Wavelet coefficients capture both noise components and smooth temporal trends simultaneously | TimeSeries.Features | 🟡 Medium | ⏳ | Dual-component feature extraction |

---

## Cross-Cutting Themes

| Theme | Related Insights | Implementation Status |
|-------|------------------|----------------------|
| **No Free Lunch** (every strategy has failure set) | I4.1, I4.2, I4.3, I5.3 | ✅ Partially implemented (regime detector, adaptive router) |
| **Friction > Signal Strength at Scale** | I1.1, I1.2, I8.1, I13.2, I17.1, I17.2 | ✅ Implemented (friction_scoring.py) |
| **Event Granularity Beats Aggregation** | I6.1, I6.2, I6.3, I9.1 | ⏳ Deferred (needs event taxonomy) |
| **Position-Level Risk > Portfolio Metrics** | I3.1, I3.2, I3.4, I13.1 | ✅ Implemented (position_sizing.py, daily_limits.py) |
| **Crash Factors + Timing > Risk Filters Alone** | I13.1, I13.3, I13.4 | ⏳ Partially implemented |
| **Market Efficiency Inversion** (less traded = more alpha) | I14.1, I14.2, I14.3, I16.1, I16.2, I17.1 | 🔬 Research phase — instrument universe expansion |
| **Alpha Decay Is Universal** (all signals decay post-discovery) | I15.1, I15.2, I15.3, I4.3 | 🔬 Research phase — needs decay monitoring |
| **Systematic Factor Construction** (standardize → purify → synthesize → optimize) | I19.1, I19.2, I19.3, I19.4, I19.5 | 🔬 Research phase — 华泰 4-phase pipeline |
| **Crowd-Sourced Strategy Patterns** (multi-confirmation + ATR risk = survival) | I20.1, I20.2, I20.3, I20.4, I20.5, I20.6 | ✅ 6 FMZ detectors converted and integrated |
| **Non-Linear Factor Discovery** (ML + MCTS for alpha mining) | I18.1, I18.2, I18.3, I18.4 | 🔬 Long-term research (GPU-gated for DL) |
| **Agentic Factor Discovery** (LLM-guided DSL search with deterministic evaluation) | I21.1, I21.2, I21.3, I21.4, I21.5, I21.6, I21.7, I21.11, I21.12 | ✅ Core modules built (DSL, trace, IC gate, ridge combiner, range patterns) |
| **GAN-Based Data Augmentation** (synthetic financial data for scarce regimes) — NEW | I26.1, I26.2, I27.1, I28.1, I28.3 | ⏳ Phase 27 planned — TTS-GAN + TimeGAN for augmentation |
| **Wavelet Frequency Localization** (time-frequency decomposition beats FFT for non-stationary signals) — NEW | I34.1, I35.1, I36.3, I37.1 | ⏳ Phase 27 planned — wavelet feature preprocessor |
| **Contextual Time Series Forecasting** (LLM + time series for news-conditioned predictions) — NEW | I29.1, I29.2, I29.3 | ⏳ Phase 27 planned — TsLLM-inspired contextual forecasting |

---

## Insights by Topic

### Risk (16 insights)
I1.1, I1.3, I3.1, I3.4, I4.2, I4.4, I4.5, I8.1, I8.3, I13.1, I13.2, I3.3, I3.5, I4.3, I13.3, I13.4

### Friction (6 insights)
I1.1, I1.4, I8.1, I13.2, I17.1, I17.2

### Regime (7 insights)
I3.3, I4.1, I8.2, I10.3, I13.4, I5.3, I8.2

### Signal.Quality (25 insights) — UPDATED
I2.1, I2.3, I5.2, I5.3, I7.1, I7.2, I9.2, I13.2, I13.3, I6.2, I6.3, I15.3, I21.1, I21.2, I21.3, I21.5, I21.6, I21.7, I21.10, I21.11, I21.12, I20.1, I20.2, I20.5, I33.2

### Event.Type (5 insights)
I6.1, I6.2, I6.3, I9.1, I9.3

### Portfolio (6 insights)
I3.2, I3.5, I7.3, I10.1, I10.2, I10.3

### Position.Sizing (3 insights)
I2.2, I3.1, I3.4

### Strategy.Lifecycle (9 insights)
I1.5, I4.3, I5.1, I13.4, I15.1, I15.2, I17.3, I21.4, I21.9

### Factor.Modeling (9 insights)
I18.1, I18.2, I18.3, I18.4, I19.1, I19.2, I19.3, I19.4, I19.5

### Agentic.Discovery (12 insights)
I21.1, I21.2, I21.3, I21.4, I21.5, I21.6, I21.7, I21.8, I21.9, I21.10, I21.11, I21.12

### Market.Efficiency (5 insights)
I14.1, I14.2, I14.3, I16.1, I16.2

### ML.Training (4 insights) — UPDATED
I16.3, I27.2, I28.2, I31.1

### Strategy.Conversion (6 insights)
I20.1, I20.2, I20.3, I20.4, I20.5, I20.6

### Phase 25 Cross-Paper (13 insights)
P25.1, P25.2, P25.3, P25.4, P25.5, P25.6, P25.7, P25.8, P25.9, P25.10, P25.11, P25.12, P25.13

### TimeSeries.Generation (6 insights) — NEW
I26.1, I26.2, I27.1, I28.1, I34.1, I34.3

### Anomaly.Detection (8 insights) — NEW
I30.1, I30.2, I31.2, I32.1, I32.2, I33.1, I33.3

### ML.Architecture (6 insights) — NEW
I26.3, I29.1, I34.2, I35.1, I36.2, I37.1

### TimeSeries.Features (3 insights) — NEW
I35.2, I37.3

### Signal.Processing (2 insights) — NEW
I36.1, I36.3

### LLM.Integration (3 insights) — NEW
I29.1, I29.2, I29.3

---

## Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Implemented | 48 | 44% |
| 🔄 In-Progress | 0 | 0% |
| ⏳ Deferred/Backlog | 59 | 54% |
| ❌ Rejected | 2 | 2% |
| 🔬 Research Phase | 0 | 0% |

| Impact | Count | Percentage |
|--------|-------|------------|
| 🔴 Critical | 19 | 17% |
| 🟠 High | 46 | 42% |
| 🟡 Medium | 43 | 39% |
| 🟢 Low | 6 | 5% |

---

*Registry updated 2026-05-26: Phase 26 added 28 new insights from 12 papers (TimeGAN, WGAN-GP, TTS-GAN, TsLLM, ALGAN, MIM-GAN, TadGAN, TSI-GAN, WaveletDiff, WDformer, AWEMixer, DB2-TransF) covering GAN financial augmentation, wavelet forecasting, anomaly detection, and LLM-time-series integration. Total: 116 insights from 34 sources.*
