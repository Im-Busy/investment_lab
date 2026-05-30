# Investment Trying Lab (QuantSteps)

**Production-grade, rule-first multi-pattern trading system.** 17-instrument verified basket, 100% OOS positive (mean Sharpe 1.135), Q2 2026. 27 completed phases, 300+ files, 35K+ GitNexus-indexed symbols.

---

## OOS Performance (Q2 2026 Re-Run)

| Tier | Instruments | Allocation | Mean OOS Sharpe | Positive |
|------|------------|------------|-----------------|----------|
| **S** | XLK, XLE, GLD, SPY, SLV, QQQ | 60% | **+0.949** | 6/6 (100%) |
| **A** | NUE, STLD, HAL, MPC, EOG | 25% | **+1.350** | 5/5 (100%) |
| **B** | INTC, AMD, LMT, JNJ, MRK, NEM | 15% | **+1.143** | 6/6 (100%) |
| **All** | 17 instruments | 100% | **+1.135** | **17/17 (100%)** |

IS→OOS correlation: **-0.388**. IS is anti-predictive — phoenix plays (IS negative → OOS positive) win. Bear market (2022-2026) survival: 12/18 (67%) OOS positive. See [BESTS.md](BESTS.md) for the full leaderboard.

---

## Architecture

```
src/
├── strategies/    59 files   Primary strategies (RulesFirst, Combined, ML, RegimeRouter, SMC) + 50+ indicator-based
├── ml/            91 files   CatBoost/LightGBM pipeline, regime models (HMM/GMM/SVM/CNN), ensembles, GANs, overfitting guards
├── signals/       41 files   Signal scoring, confluence, pattern reliability registry, factor purification
├── indicators/    34 files   Numba-accelerated technicals, PineScript→Python helpers, SMC/ICT liquidity analysis
├── patterns/      55 files   54 chart pattern detectors across 10 categories
├── risk/          23 files   Position sizing, Kelly, circuit breakers, binomial VaR, diversity constraints
├── optimization/   6 files   NSGA2, Two-Phase GA, Dynamic GA, skfolio (MV/CVaR/HRP)
├── data/          10 files   Survivorship-free universe, options Greeks, congressional signals, futures inventory
├── analysis/      12 files   Dual alpha/beta, divergence-in-bits, deflated Sharpe, post-backtest validation
├── nlp/            6 files   SVM+TF-IDF/BiLSTM sentiment, Adanos cross-source API, ensemble
├── features/       3 files   Wavelet feature preprocessor (3 SHAP Top-20 features)
├── cli/            1 file    Option discovery CLI
├── mcp/            1 file    MCP stock data server (3 tools for AI agents)
├── backtest/       3 files   Custom event-driven engine
├── portfolio/      2 files   Black-Litterman allocation, signal aggregation
├── per_instrument/ 4 files   Config, timezone registry, performance tracker, strategy selector
├── config/         4 files   System configuration
└── utils/          6 files   Shared utilities
```

**Key design principle: Rules-First over ML.** Pure rule-based production config (mr=0.70, et=0.55) delivers +1.135 mean OOS Sharpe. Every ML-augmented variant underperforms. ML is secondary validation only.

---

## Core Principles (Immutable)

1. **Trail stop is non-negotiable.** Every config with trail stop outperforms no-trail by 2-4×.
2. **Rules-First > ML for OOS robustness.** Rules survive regime shifts. ML degrades when stacked.
3. **Mid-cap alpha.** $10-50B market cap is the sweet spot — above noise, below quant competition.
4. **Calibration over model retraining.** Isotonic calibration (+44% Sharpe) and normalized ATR > new models.
5. **IS is not predictive.** 9/11 instruments improved OOS vs IS. Never discard a strategy on IS alone.
6. **min_confluence = 0.** Pattern agreement gating reduces trades without improving quality. Irrefutable: 100% of sweep chose 0.
7. **ALL-ON signal stacking = saturation.** 13+ enhancers degraded Sharpe from 1.135 → 0.195. Add one flag at a time and validate OOS.

See [BESTS_INSIGHTS.md](docs/BESTS_INSIGHTS.md) for the full distilled knowledge base.

---

## System Capabilities

### Chart Pattern Detection
54 detectors across 10 categories: basic, breakout, candlestick, classic, complex, continuation, harmonic, fmz, technical, range-persistence. Plus Faiss GPU similarity search and Patternity deterministic recognition wrapper.

### ML Infrastructure
- **Training pipeline:** 9-stage pipeline with triple-barrier labeling, PurgedKFold, NestedPurgedCV, walk-forward optimization
- **Overfitting guards:** LockBox blind holdout, Blind Analysis (scrambled-label tuning), Label Shuffling, Huber loss, MRE-gap analysis
- **Models:** CatBoost, LightGBM, EBM, AutoGluon, pattern classifier (37-ticker trained), meta-labeler (AUC 0.633)
- **Explainability:** SHAP dashboards, feature importance, stability selection (38/98 features pass)
- **Regime detection:** HMM, GMM, HDBSCAN, SVM, CNN, change-point, ensemble, macro, path-signature (11 models)
- **Generative:** TTS-GAN OHLCV augmentation, TadGAN anomaly detection, Chronos-2 LoRA fine-tuning, WaveletDiff

### Risk & Position Sizing
Kelly allocator, signal-strength dynamic sizing, VIX regime sizing, circuit breakers, daily loss limits, binomial VaR/CVaR, copula risk, diversity constraints, strategy-aware sizing, market impact estimation.

### Optimization
NSGA2 multi-objective, Two-Phase GA (per-rule then ensemble), Dynamic GA (per-regime memory), skfolio (MV/CVaR/HRP/RiskBudget/InvVol).

### Data & Signals
- **Alt data:** Congressional trade signals (House/Senate APIs), futures inventory, options Greeks, fund flow (MFM/OBV/A/D/EoM)
- **Sentiment:** SVM+TF-IDF (82-94%), BiLSTM (128d embed), Adanos cross-source (Reddit/X/news/Polymarket), ensemble (93.4%)
- **Fundamental:** 60+ ratios via FinanceDatabase+FinanceToolkit, ROCE/earnings-yield Magic Formula screening
- **Cross-asset:** Sector map, VAR/Granger causality, Kalman hedge ratios, wavelet/FFT transforms
- **Survivorship-bias-free:** Historical universe retains delisted tickers per date
- **MCP server:** 3-tool stock data API for AI agents (get_price, get_technicals, get_multi)

### Stock Selection
11 hard filters (F1-F11) via centralized `docs/stock_selection_criteria.md`: market cap, daily volume, price, history, exchange, ETF AUM, leveraged products, ROCE, sector exclusions (Financials/Utilities). 9 desirability dimensions. 3 context tiers.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run rules-first backtest (production config)
uv run scripts/backtest_rules_first.py SPY --start 2025-01-01 \
    --entry-threshold 0.55 --min-reliability 0.70

# OOS re-run for 17-instrument basket
$env:TQDM_DISABLE = "1"; uv run scripts/backtest_all_comprehensive.py \
    --tickers XLK,XLE,GLD,SPY,SLV,QQQ,NUE,STLD,HAL,MPC,EOG,INTC,AMD,LMT,JNJ,MRK,NEM \
    --period oos --use-best

# Screen & backtest new ticker candidates
uv run scripts/screen_and_backtest_10.py

# Daily paper trading signals
$env:PYTHONIOENCODING = "utf-8"; uv run scripts/paper_trade_daily.py --basket XLK,XLE,GLD,SPY,SLV,QQQ,NUE,STLD,HAL,MPC,EOG,INTC,AMD,LMT,JNJ,MRK,NEM,CN_CATL

# Generate daily market report
$env:PYTHONIOENCODING = "utf-8"; uv run scripts/daily_report_agent.py
```

Full command reference: [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md)

---

## Documentation

| Document | Purpose |
|----------|---------|
| [BESTS.md](BESTS.md) | Backtest leaderboard — all best results by configuration |
| [BESTS_INSIGHTS.md](docs/BESTS_INSIGHTS.md) | Distilled knowledge: core principles, factor rankings, tier list, anti-patterns |
| [COMMAND_CHEATSHEET.md](docs/COMMAND_CHEATSHEET.md) | Every CLI command with flags and examples |
| [stock_selection_criteria.md](docs/stock_selection_criteria.md) | 11 hard filters + 9 desirability dimensions for ticker selection |
| [ML_TRAINING_GUIDE.md](docs/ML_TRAINING_GUIDE.md) | ML components, decision tree, data flow, file location reference |
| [guide-ml-pipeline.md](docs/guide-ml-pipeline.md) | Standard 9-stage ML training pipeline |
| [guide-anti-overfitting.md](docs/guide-anti-overfitting.md) | LockBox, Nested CV, Blind Analysis, Label Shuffling |
| [GPU_TASK_QUEUE.md](docs/GPU_TASK_QUEUE.md) | GPU-blocked tasks with self-contained tutorials |
| [reference_phase28_external_tools.md](docs/reference_phase28_external_tools.md) | Indicators comparison, OpenStock architecture, WFGY evaluation |

---

## Research & Knowledge Management

- **Insight registry:** 116 insights from 34 sources in [insight_registry.md](docs/research_logic_map/insight_registry.md)
- **Web source registry:** Non-paper web sources with W-IDs in [web_source_registry.md](docs/research_logic_map/web_source_registry.md)
- **Knowledge graph:** Auto-cross-references papers with project modules via `useful_resources/_knowledge_analysis.py`
- **Paper2md pipeline:** Batch PDF→Markdown conversion + LLM summarization
- **Source Attribution Protocol:** All external code/concepts carry mandatory `Source:`/`Reference:` docstrings (see AGENTS.md)

---

## Progress & Planning

| Phase | Status |
|-------|--------|
| 01-08 | ✅ Complete (patterns, optimization, regime, ML, research, paper trading prep, attribution) |
| 09-20 | ✅ Complete (backtest engine, advanced strategies, SMC/ICT, PineScript conversion) |
| 21-25 | ✅ Complete (ML pipeline v3, anti-overfitting infra, signal enhancers, Perf tracking, 66-paper comparison) |
| 26 | ✅ Complete (bear market validation, paper trading launch, 18-instrument basket) |
| 27 | ✅ Complete (TTS-GAN, wavelet features, TadGAN, Chronos-2, WaveletDiff) |
| 28 | 🔄 Active (22/27 tasks done, 33 insights from 4 external repos, ~6,480 LOC) |

See [progress_docs/plans/full.md](progress_docs/plans/full.md) for the master plan (28 phases, 1,092 lines). Session log in [progress_docs/current.md](progress_docs/current.md).

---

## Requirements

- Python 3.13+
- Package management via [uv](https://github.com/astral-sh/uv)
- ~600MB disk (models + data cache)

## Development

```bash
uv sync                 # Install all deps
uv run pytest           # Run test suite
uv run ruff check .     # Lint
uv run ruff format .    # Format
npx gitnexus analyze    # Re-index code intelligence graph
```

This project uses **GitNexus** (35K symbols, 56K relationships, 300 execution flows) and **CodeGraphContext** (Cypher graph) for code intelligence. Consult AGENTS.md for the code intelligence protocol.
