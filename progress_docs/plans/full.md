---
project: investment_trying
last_updated: 2026-05-18 (Phase 21 EXPANDED — 41/46 items. D5+D12 RL done. ALL IMPLEMENTABLE ITEMS COMPLETE. 5 FPGA out of scope.)
summary: |
  Rule-based multi-pattern trading system with 47+ chart pattern detectors, ML-enhanced
  regime detection, Numba-accelerated indicators, PPO/SAC/CQL RL trade execution, and
  event-driven backtesting engine. All 21 phases complete. Phase 21: 41/46 items.
  Phase 07 paper trading running (14-day protocol).
phases_total: 21
phases_complete: 21
phases_active: 0
phases_deferred: 3
enhancement_tracks: 4
---

# Master Plan

## Phase Status

| # | Phase | Plan | Log | Status |
|---|-------|------|-----|--------|
| 01 | Pattern Detection & Strategies | [plan](01-patterns.md) | [log](../logs/01-patterns.md) | ✅ Complete |
| 02 | Performance Optimization | [plan](02-optimization.md) | [log](../logs/02-optimization.md) | 🔄 Numba done, vectorbt deferred (Windows) |
| 03 | Regime Detection & Adaptation | [plan](03-regime.md) | [log](../logs/03-regime.md) | ✅ Complete |
| 04 | ML Foundation | [plan](04-ml-foundation.md) | [log](../logs/04-ml-foundation.md) | ✅ Complete |
| 05 | ML Advanced | [plan](05-ml-advanced.md) | [log](../logs/05-ml-advanced.md) | ⏸️ Deferred |
| 06 | Research-Based Enhancements | [plan](06-research.md) | [log](../logs/06-research.md) | ✅ Complete |
| 07 | Paper Trading & Live Readiness | [plan](07-paper-trading.md) | [log](../logs/07-paper-trading.md) | ⏸️ Deferred |
| 08 | Contribution & Attribution | [plan](08-attribution.md) | [log](../logs/08-attribution.md) | ✅ Complete |
| 10 | Tool Evaluation Additions | [plan](enhance-tool-evaluation.md) | [log](../logs/10-tool-evaluation.md) | ✅ Complete |
| 10b | Autonomous Loop Hardening | [plan](enhance-loop-hardening.md) | [log](../logs/10b-loop-hardening.md) | ✅ Complete — 5/5 done |
| 11 | Overfitting Fixes (B9-B14) | [plan](fix-overfitting.md) | [log](../logs/11-overfitting-fixes.md) | ✅ Complete — B9-B14 done, model retrained |
| 12 | Post-Retrain Validation & Next Steps | [plan](post_retrain_next_steps.md) | — | ✅ CONCLUDED — All 15 items (12a/12b/12c/12d) complete. 5/15 gates PASS, 10/15 FAIL. |
| 13 | **NEW — Direction C: Research-Driven Signal Sources** | [plan](direction-c-research-signals.md) | — | ✅ Complete — C1-C6 done (all 6 phases). Direction C CONCLUDED. |
| 14 | **NEW — Direction E: Foundation Model Integration** | [plan](direction-e-foundation-models.md) | — | ✅ CONCLUDED — E1+E2+E5 done. Chronos-2 degraded, cross-asset feature fix works (+0.40 Sharpe). |
| 15 | **NEW — Direction A+B: Regime-Adaptive ML + Rules-First** | [plan](direction-ab-regime-rules.md) | — | ✅ CONCLUDED — All phases complete. Sub-track A: A1-A5 ✅. Sub-track B: B1-B5 ✅. AB1-AB2 ✅. AB4: Rules-First is production system (OOS Sharpe +0.76). Combined ML+Rules degrades OOS. All three directions now concluded.
| 16 | Direction N: NLP + Quant Schools | [plan](nlp-and-quant-schools-implementation.md) | — | ✅ CONCLUDED — N+P+O+Q+R+S+V done (7/9). T+U permanently gated. ACCEPT LIMITS. |
| 17 | **Resource-Driven Enhancements** | [plan](17-resource-enhancements.md) | — | ✅ COMPLETE — P0 (R1-R4) + P1 (R5-R8) + P2 (R9-R12) + P3 (R13-R14) all done. Phase 17 concluded. |
| 18 | **Useful Repos Integration (Wave 1)** | [plan](18-useful-repos-integration.md) | — | ✅ DONE — All 7 items complete (3 P0 + 2 P2). |
| 19 | **New Repos Wave 2** | [plan](19-new-repos-wave2.md) | — | ✅ DONE — All 9 items complete (3 P0 + 3 P1 + 3 P2). |
| **20** | **System Hardening & Signal Quality** | [plan](20-system-hardening.md) | — | ✅ Complete — H1-H7 all implemented (2026-05-17). Multi-TP default, pattern registry, sector scoring, IR scalar, GA post-step, BTC audit. |
| **21** | **Quant-Resources Signal Enhancers** | [plan](21-quant-resources-insights.md) | — | ✅ Complete — 41/46 items. ALL gates open. Q1-Q8 + D3/D5/D6/D7a-d/D9/D10/D11/D12 + B1/B2/B4/B5/B6/B7/B8/B9/B10/B11 + C10 + A1-A7/A11/A12/A13. 5 FPGA out-of-scope (C1-C7). |
## Three-Direction Execution Roadmap

```
NOW ──────────────────────────────────────────────────────► FUTURE
 │
 ├─ Phase 13: Direction C (Research Signals) ✅ CONCLUDED
 │   C1: Overfitting detection ✅ | C2: Sentiment ✅ | C3: Events ✅
 │   C4: RL execution ✅ | C5: Kelly+Circuit+Crash ✅ | C6: Adversarial+Defensive+Production ✅
 │
 ├─ Phase 14: Direction E (Foundation Models) [gate on C → NOW OPEN]
 │   E1: Chronos installation + smoke test
 │   E2: Chronos signal generator + MLStrategy integration
 │   E3: Precomputation pipeline (batch forecasts → DuckDB cache)
 │   E4: Cross-asset foundation model ensemble
 │   E5: Cross-asset feature pipeline upgrade (fallback)
 │
 └─ Phase 15: Direction A+B (Regime + Rules) [gated on C+E]
      Sub-track A: Regime-Adaptive ML ✅ DONE
        A1: Benchmark 8 regime detectors → select top 2-3 ✅
        A2: Build RegimeRouter (per-regime model dispatch) ✅
        A3: Train per-regime CatBoost models ✅
        A4: Backtest RegimeRouter on OOS (2025-2026) ✅
        A5: Remove flipped features from training ✅
      Sub-track B: Rules-First Pattern System ✅ DONE
        B1: Build RulesFirstStrategy from knowledge base ✅
        B2: Wire into backtesting.py ✅
        B3: Backtest rules-first on SPY (2016-2024) ✅
        B4: Optimize pattern weights from ablation results ✅
        B5: Backtest rules-first on OOS (2025-2026) ✅ — Sharpe +0.76
      Convergence: ✅ DONE
        AB1: CombinedStrategy (dynamic ML+Rules weights) ✅
        AB2: Backtest combined system ✅
        AB3: Portfolio-level backtest ⏸️ (deferred — rules-first confirmed dominant, multi-asset expansion is future work)
      AB4: Production decision ✅ — Rules-First is primary system
```

---

## Phase 16: Direction N — NLP + Quant Schools Expansion (NEW 2026-05-15)

**Source:** NLP in Quant 101 course + Chinese 5 Quant Schools framework
**Plan:** [nlp-and-quant-schools-implementation.md](nlp-and-quant-schools-implementation.md) — 9 sub-phases, 81 ideas

```
NOW ──────────────────────────────────────────────────────────► FUTURE
 │
 ├─ Phase N (P0) — NLP Foundation: Loughran-McDonald Dictionary Scorer
 │   N1: Load LM dictionary → N2: LM scorer class → N3: Integrate into MLStrategy
 │   N4: Degree adverb weighting
 │   Impact: Replaces synthetic sentiment with real financial-domain signal. ~300 lines.
 │
 ├─ Phase P (P0) — Dedicated Mean Reversion System
 │   P1: MeanReversionStrategy → P2: RegimeRouterStrategy → P3: MR-specific stops
 │   P4: Backtest + compare trend-only vs reversion-only vs routed
 │   Impact: Covers ranging-market blind spot. Indicators already exist. ~400 lines.
 │
  ├─ Phase O (P1) — Social Media Validation (gate on lead/lag)
  │   O1: Sentiment lead/lag analysis rho(tau) → O2: Noise processing (sarcasm/emoji/bot)
  │   O3: Data ingestion (Twitter/Reddit/StockTwits) — ONLY if O1 gate passes
  │   Impact: Validation before investment. Don't build social infra if sentiment doesn't predict.
  │   **Status: ✅ COMPLETE — Gate BLOCKED. No social media infra.**
  │
  ├─ Phase Q (P1) — Multi-Factor Fundamental Factors
  │   Q1: Fundamental factor extractor (Value/Quality/Size/Growth) → Q2: Pipeline integration
  │   Q3: Multi-factor scoring backtest
  │   Impact: New feature dimension for CatBoost. Complements 88 technical features.
  │   **Status: ✅ COMPLETE — 14 factors, backtest, ML pipeline integration.**
 │
  ├─ Phase R (P2) — Advanced NLP: FinBERT + SEC Filings (builds on N)
  │   R1: FinBERT deployment → R2: SEC filing miner → R3: Multi-source fusion
  │   R4: NLP+Financial fusion CatBoost model
  │   Impact: Industry-standard financial NLP. GPU optional (CPU-viable).
  │   **Status: ✅ COMPLETE — FinBERT, SEC scraper, filing analyzer, fusion. Gate FAIL (AUC -0.019). NLP redundant with price.**
  │
 ├─ Phase V (P2) — Strategy Architecture v2: Short-Side, Crypto, Portfolio
 │   V1: Short-side activation → V2: Strategy-aware position sizing
 │   V3: Correlation-aware multi-asset portfolio → V4: Crypto expansion
 │   Impact: Production-grade multi-strategy, multi-asset system.
 │   **Status: ✅ COMPLETE — Short-side `use_short=True`, Kelly sizing, correlation-clustered allocator, CCXT crypto provider.**
 │
 ├─ Phase S (P2) — Statistical Arbitrage: Pairs Trading
 │   S1: Pairs trading engine (cointegration + rolling hedge) → S2: Sector pairs universe
 │   S3: Pairs backtest
 │   Impact: Orthogonal alpha source, market-neutral.
 │   **Status: ✅ COMPLETE — CVX-XOM Sharpe 0.40. Energy/utilities work. ETF pairs fail.**
 │
  ├─ Phase T (P3) — Deep Learning (GPU gate, same as Phase 05) ⏸️ PERMANENTLY GATED
  │   T1: LSTM/GRU → T2: Transformer → T3: CNN patterns → T4: TFT → T5: GAN market data
  │   Impact: Only if DL beats CatBoost AUC by ≥ 0.02.
  │
  └─ Phase U (P3) — Alternative Data (paid subscription gate) ⏸️ PERMANENTLY GATED
      U1-U5: Satellite, credit card, supply chain, job postings, Google Trends
      Impact: High-cost experiments. Gate on data budget.
```

## Enhancements

| Name | Plan | Status |
|------|------|--------|
| **Direction C — Research-Driven Signal Sources** (6 phases: overfitting detection, sentiment, events, RL, Kelly, adversarial) | [plan](direction-c-research-signals.md) | ✅ CONCLUDED — C1-C6 complete. 12 papers mapped to 6 phases. |
| **Direction E — Foundation Model Integration** (5 phases: Chronos zero-shot, signal gen, precomputation, cross-asset ensemble) | [plan](direction-e-foundation-models.md) | ✅ CONCLUDED — E1+E2+E5 done. Chronos-2 degraded, cross-asset fix works. |
| **Direction A+B — Regime-Adaptive + Rules-First** (Sub-track A: 5 phases per-regime ML. Sub-track B: 5 phases rules-first. Convergence: combined system) | [plan](direction-ab-regime-rules.md) | ✅ CONCLUDED — Rules-First is production system (OOS Sharpe +0.76). |
| **Direction N — NLP + Quant Schools Expansion** (9 sub-phases: N,P,O,Q,R,V,S,T,U. 81 ideas from NLP 101 + 5 Quant Schools, prioritized by impact/complexity) | [plan](nlp-and-quant-schools-implementation.md) | ✅ CONCLUDED (2026-05-16) — N+P+O+Q+R+S+V done. T+U permanently gated on GPU/paid data. |
| **Overfitting Fixes** (Stability Selection, Per-Sector Models, CPCV, DEL, Meta-Labeling, Production Guards) | [plan](fix-overfitting.md) | ✅ Complete — All B9-B14 done, model retrained through 2026-05-14 |
| **Post-Retrain Next Steps** (15 items: bear market validation, calibration, DSR investigation, baselines, new signal sources, production harness) | [plan](post_retrain_next_steps.md) | ✅ CONCLUDED — All 15 items complete across 12a/12b/12c/12d |
| ML Capability Enhancements (GWO, InterpretML, AutoGluon + wider gaps) | [plan](enhance-ml-capabilities.md) | ✅ Complete — All 9 items done (T1a-T1d, T2-T4, T0, QW1) |
| External Review Additions (21 friend-suggested items) | [plan](enhance-ml-capabilities.md#friends-pipeline-recommendations--external-review-additions) | 🔴 Planned — 6 Tier 1, 7 Tier 2, 8 Tier 3 |
| Notebook Audit Fixes (F1-F4, ML1-ML4, S1-S4) | [plan](notebook-audit-fixes.md) | 🔄 In Progress — F1 ✅, F2 ✅ |
| Knowledge Graph Insights (58 papers cross-referenced) | [plan](enhance-knowledge-graph.md) | ✅ Complete — 50 connections, 12 actionable recommendations |
| Paper-to-Module Gap Closure (overfitting, sentiment, RL, events) | [plan](enhance-knowledge-graph.md) | ✅ Complete — Absorbed into Direction C (CONCLUDED) |
| **Tool Evaluation Additions** (Optuna, PyPortfolioOpt, Bandit, CCXT, FinGPT, aeon) | [plan](enhance-tool-evaluation.md) | ✅ Complete — T10a-1 through T10a-6 done. T10b-1 done. Verified 2026-05-16 (24 tests pass). |

---

## Pending (Ready to Start)

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| P1 | T1a | ARO feature selection engine | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/tuning/aro_selector.py`, 7 tests |
| P1 | T1b | GWO CatBoost hyperparameter tuner | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/tuning/gwo_tuner.py`, 7 tests, CLI `scripts/tune_model.py` |
| P1 | T1c | GA unsupervised regime optimizer | Phase 03 (Regime) ✅ | ✅ Done — `src/ml/tuning/ga_tuner.py`, 4 tests, CLI |
| P1 | T1d | WOA pattern threshold tuner | Phase 01 (Patterns) ✅ | ✅ Done — `src/ml/tuning/woa_tuner.py`, 4 tests, CLI |
| P1 | T2 | InterpretML EBM regime classifier | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/ebm_classifier.py`, 10 tests, glassbox explainability |
| P1 | T0 | Unify classifier defaults to CatBoost | Phase 04 ✅ | ✅ Done — RegimeClassifier + SignalRegressor + PatternClassifier all default to catboost |
| P1 | T3 | AutoGluon AutoML baseline | ML Foundation (Phase 04) ✅ | ✅ Done — `src/ml/automl.py`, 8 tests, CLI `scripts/run_automl.py` |
| P1 | T4 | Advanced backtest validation (DSR, PSR, FDR) | Phase 04 ✅, Phase 08 ✅ | ✅ Done — `src/analysis/deflated_sharpe.py`, 43 tests |
| P1 | QW1 | Pattern confidence scoring | PatternClassifier exists | ✅ Done — `src/ml/pattern_scorer.py`, 6 tests |

| **MR0** | C1 | Paper trade re-run with fixed drawdown (12 tickers) | PatternClassifier V3 model, paper_trade_v3.py ✅ | ✅ Done — `reports/paper_trading/20260511_044647/` |
| **MR0** | C2 | Walk-forward chronological validation (per-ticker) | C1 | ✅ Done — 15 bugs fixed, 5/12 pass (XLK/QQQ/SPY/KODK/GLD) |
| **MR1** | C3 | Basket vs single-ticker ablation study | C2 | ✅ Done — 5-winner model = worse IC everywhere (-0.107→-0.02). 12-all model = +0.048→+0.112 across winners. Adding diverse tickers IMPROVES generalization. GATE OPEN. |
| **MR1** | C4 | Model prediction correlation decomposition | C2 | ✅ Done — SPY/QQQ/XLK r>0.82 (tech cluster, concentrated). KODK r≈0.27, GLD r≈0.30 vs tech (independent). 3 distinct signals, not 1. |
| **MR1** | C6a | Ticker screening pipeline (auto) | C3, C4 passing | ✅ Done — `scripts/screen_tickers.py` + `.kilo/skills/ticker-screener/`. Screened 75 tickers across 8 sectors (v2 relaxed criteria: MC>$200M, vol>12%, inst>25%). 22 passed walk-forward IC > 0.03. See `docs/ticker-test-log.md`. |
| **MR2** | C6 | Expand basket to 30+ tickers | C6a | ✅ Done (2026-05-11). 33-ticker model beats 12-ticker model on all 3 gate criteria: (1) 91% tickers improved IC, (2) max drop 0.012, (3) 20/21 new tickers pass IC>0.03. Mean IC: 0.038→0.067 (+75%). Model: `models/pattern_classifier_v3_SPY_20260511_164601.pkl`. See `experiments/c6_comparison.csv`. |
| **MR2** | C6b | Edge relaxation analysis (what drives usefulness) | C6 | ✅ Done (2026-05-11). Tested 81 tickers against 33-ticker model. Volatility is the binding constraint (r=−0.44, p<0.0001). Market cap irrelevant (r=+0.02). Volume weakly negative (r=−0.31). Sweet spot: 12–25% ann. vol (mean IC 0.056, 73% pass, 0% negative). See `docs/guide-edge-characterization.md`. |
| **MR2** | C5 | ML + pattern detector integration (`ml_strategy.py`) | C6 ✅ | ✅ Done (2026-05-11). `src/strategies/ml_strategy.py` with backtesting.py integration. No-cross-asset model: `models/pattern_classifier_v3_SPY_20260511_224704.pkl` (57 features, CV AUC 0.546, gap −0.002). All 33 tickers tested. Best: WMT +175% (Sharpe 0.76), UNP +146% (0.66), D +134% (0.62). 19/33 profitable. Mean Sharpe 0.15. Results map to vol-based edge characterization: utilities/REITs/staples strong, energy/miners/EM weak. 2933 total trades. See `reports/ml_backtest/comparison.csv`. |

### Phase 18: Useful Repos Integration (NEW — 2026-05-16)

**Source:** 7 repos in `C:\Dev\useful_repos` evaluated for integration into investment_trying. See [full plan](18-useful-repos-integration.md).

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **R18a** | eiten: Portfolio optimization integration (Eigen/MVP/MSR/GA) | — | ✅ DONE — 6 files (~500 loc): `src/portfolio/eiten_adapters/` (5 strategies + RMT), `src/portfolio/eiten_builder.py`, `scripts/optimize_portfolio.py` CLI with --compare/--from-signals. |
| **P0** | **R18b** | Scrapling: Financial data scraping pipeline | — | ✅ DONE — 2 files (~350 loc): `src/data_ingestion/financial_scraper.py` (insider/news/SEC/earnings), `scripts/scrape_financial_data.py` CLI. |
| **P1** | **R18c** | agent-skills: Adopt 23 engineering workflows into `.kilo/skills/` | — | ✅ DONE — 6 skills copied to `.kilo/skills/engineering/`, anti-rationalization table in `.kilo/global-rules.md`. |
| **P2** | **R18d** | Qbot: Study 25+ DL model implementations + 30+ indicator strategies | ✅ | 2 DL strategies found (LSTM, RL). Documented in `docs/reference-qbot-ml-patterns.md`. |
| **P2** | **R18e** | qmd: Index project knowledge base for semantic search | ✅ | @tobilu/qmd 2.1.0 installed. 118 files indexed, 900 chunks embedded. 2 collections created. |
| **P3** | **R18f** | CLI-Anything: Adopt agent-friendly CLI conventions | — | Study CLI generation methodology. Apply `--json` output + SKILL.md pattern to existing scripts (backtest_rules_first.py, run_ml_backtest.py, sweep_entry_thresholds.py). Convention doc only. |
| **P3** | **R18g** | local-deep-research: Evaluate MCP-server research pattern | R2, Q2 | Study LangGraph agentic search architecture. Evaluate exposing backtesting/ML capabilities via MCP. Deferred until Phase 12 stabilization. |

### Phase 18 Cross-Project Utility (C:\Dev\projects) — 2026-05-16

Evaluated 4 other projects for utility from the same 7 repos:

| Project | agent-skills | CLI-Anything | eiten | local-deep-research | Qbot | qmd | Scrapling |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| albion_get_rich | MED | — | — | LOW | — | LOW | **HIGH** |
| data_and_stat_analysis | MED | LOW | — | MED | — | MED | MED |
| personal_website | **HIGH** | — | — | LOW | — | LOW | LOW |
| skills_arsenal | **HIGH** | **HIGH** | — | MED | — | **HIGH** | MED |

Key findings:
- **Scrapling → albion_get_rich**: Directly replaces 4 existing scraping deps (seleniumbase, nodriver, camoufox, curl_cffi). Built-in Cloudflare Turnstile bypass. Adaptive element tracking survives AO Data API website changes.
- **agent-skills → personal_website**: 23 web-dev lifecycle skills (frontend engineering, browser testing, performance optimization, shipping/launch) for Next.js 16 project.
- **agent-skills + qmd + CLI-Anything → skills_arsenal**: Canonical skill format reference + semantic search across 564 files + 40+ community skills to add.

### Phase 19: New Repos Wave 2 (NEW — 2026-05-16)

**Source:** 9 new repos cloned to `C:\Dev\useful_repos` beyond the 7 from Phase 18. See [full plan](19-new-repos-wave2.md).

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **R19a** | graphify: Build project knowledge graph | — | ✅ DONE — installed, graph built in `src/graphify-out/` (7068 nodes, 11170 edges, 498 communities). |
| **P0** | **R19b** | lean-ctx: Install MCP server for context compression | — | ✅ DONE — installed via npm (lean-ctx 3.6.0), setup complete (18/18 checks). Configured for Kiro MCP. |
| **P0** | **R19c** | skills (mattpocock): Port 8 highest-value skills | — | ✅ DONE — copied to `.kilo/skills/engineering/mattpocock/`. |
| **P1** | **R19d** | dictionary-of-ai-coding: Integrate into skills_arsenal | — | ✅ DONE — 62-term glossary. Created `skills_arsenal/.../ai-coding-dictionary/SKILL.md`, updated CATALOG/README/SUMMARY (45 universal, 92 total). Created `docs/glossary-ai-coding.md` with trading adaptations. |
| **P1** | **R19e** | GitNexus: Install and index project | ✅ | 28,316 symbols, 43,736 edges, 300 flows. 16 MCP tools ready. `.gitnexus/` in gitignore. |
| **P1** | **R19f** | BettaFish: Study ForumEngine debate pattern | ✅ | Documented in `docs/reference-bettafish-patterns.md` (5 patterns, structured debate format). |
| **P2** | **R19g** | three-geospatial: N/A for inv | — | Only relevant for personal_website's Three.js/R3F atmosphere/cloud effects. |
| **P2** | **R19h** | maigret: Extract async patterns | — | Async queue executor, layered settings loading. Low urgency — covered by Scrapling (Phase 18). |
| **P2** | **R19i** | MinerU: Deferred | — | PDF→MD already covered by paper2md. Only adopt if scanned Chinese PDFs appear. |

### Phase 19 Cross-Project Utility

| Repo | inv | albion | data | personal_website | skills_arsenal |
|------|:---:|:---:|:---:|:---:|:---:|
| **graphify** | **HIGH** | MED | MED | MED | MED |
| **lean-ctx** | **HIGH** | **HIGH** | **HIGH** | **HIGH** | LOW |
| **skills** (mattpocock) | **HIGH** | HIGH | HIGH | HIGH | **HIGH** |
| **dictionary-of-ai-coding** | HIGH | MED | MED | MED | **HIGH** |
| **GitNexus** | **HIGH** | HIGH | MED | LOW | MED | LOW |
| **BettaFish** | MED | LOW | — | — | — |
| **three-geospatial** | — | — | LOW | **HIGH** | — |
| **maigret** | LOW | LOW | — | — | — |
| **MinerU** | LOW | — | LOW | MED | MED |

Key cross-project actions:
- **lean-ctx → ALL projects**: Universal token savings. Install once, configure per-project.
- **skills (mattpocock) → skills_arsenal**: 8 highest-value skills fill gaps in engineering section.
- **dictionary-of-ai-coding → skills_arsenal**: 62-term glossary as standalone skill.
- **skills (mattpocock) → personal_website**: diagnose/tdd/architecture directly applicable to Next.js dev.
- **three-geospatial → personal_website**: Drop-in atmosphere/clouds/stars components.

### Phase 6c: Strategy Refinement (post-C5)

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P1** | **C7** | Strategy execution improvements | ✅ DONE | (1) Volatility gate — implemented, no benefit (Sharpe unchanged). (2) Consecutive confirmation — implemented, reduced trades (Sharpe 0.33). (3) **Trailing stop — WINNER**: Sharpe 0.36→0.47 (+31%). (4) Position scaling by conviction — marginal (Sharpe 0.38). Best config: `--trail-stop` alone. All 4 are toggleable in ml_strategy.py. |
| **P1** | **C8** | Pattern detector confluence (PatternBoostFilter) | C7 ✅ | ✅ DONE (2026-05-12). `src/signals/pattern_boost.py` — 35 pattern detectors with reliability weights from NCFE/Duddella research (H&S=0.87, Gartley=0.85, C&H=0.80, etc.). PatternBoostFilter precomputes vectorized signals, computes directional bull/bear boost (0.0–0.10) at each bar. Integrated into MLStrategy via `use_pattern_boost=True`. |
| **P1** | **B1** | OOS validation on 2025-2026 data | C5 ✅, BESTS.md | ✅ Done — **FAILED**: Sharpe -0.27 vs train +0.73. Model overfit. Does not generalize past 2024. |
| **P1** | **B2** | Fix `backtest_ml_enhanced.py` (broken ML comparison baseline) | — | ✅ Done — 3 bugs fixed: (1) Signal gen now uses entry points, (2) replaced local training with pre-trained PatternClassifier V3, (3) backtesting.py handles pos sizing. Ruff clean. |
| P2 | **B3** | Model probability calibration audit | B1 | ✅ Done (2026-05-13) — Rewrote `scripts/model_calibration.py` with triple-barrier labels. IS ECE=0.128, OOS ECE=0.197. P=0.45 wins 39.5% IS, 33.1% OOS. Model overconfident. Degraded OOS — regime shift, not probability drift. |
| P1 | **B4** | Regime shift root cause investigation | B1 | ✅ Done (2026-05-13) — `scripts/investigate_regime_shift.py`. KS tests: raw ATR doubled (3.97→8.06, KS=0.62 p=10^-109). 18/57 shifted, 4 flipped, 13 weakened. Fix: normalized ATR=ATR/Close in `feature_engineering.py:173-174`. |
| P1 | **B5** | Walk-forward optimization comparison | B4 | ✅ Done (2026-05-13) — `scripts/backtest_wfo.py`. WFO OOS +1.7% vs single-split -1.3%. WFO total Sharpe 1.76. |
| — | **B8** | Agent-centric workflow infrastructure | B3-B5 | ✅ Done (2026-05-13) — `.kilo/agent/{model-doctor,backtest-runner,ml-trainer}.md` + `.kilo/command/{model-diagnose,backtest,train-ml}.md`. AGENTS.md updated. |
| **P0** | **B6** | Retrain basket model with normalized ATR | B5, ATR fix | ✅ Done (2026-05-13) — 2/5 pass. SPY-only model: CV AUC=0.595, WF IC=0.182, OOS Sharpe -0.44 (worse). Basket model: ARO collapsed to 5 features (2 cross-asset → 0 trades). Root cause not fixed — deferred to B9-B14. |
| **P0** | **B7** | Evaluate Qlib concept-drift models (ADARNN/ADD) | B6 | ✅ Done (2026-05-13) — CatBoost trained but 0 positions (weak signal). ADARNN/ADD fail: Alpha360-only architecture incompatible with Alpha158. Qlib IC NaN for single-stock. |
| **P2** | **C9** | Honest walk-forward paper trading | B9 ✅ | Re-run `scripts/paper_trade_v3.py` with stability-selected + per-sector model. Avoids backtesting.py precomputation — recomputes features on expanding windows only. |
| **P2** | **C10** | Portfolio-level backtest | B10 ✅ | Equal-weight portfolio of profitable tickers, monthly rebalancing. Threshold: Sharpe > 0.5, DD < 15%, 50+ total trades. |

### Phase 11: Overfitting Fixes (B9-B14) — NEW (2026-05-13)

**Source:** Web research synthesis of 32 papers and production resources (2026-05-13). Six evidence-backed fixes targeting the three root causes of model overfit.

See [full plan](fix-overfitting.md) for detailed task breakdown, implementation steps, and references.

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **B9** | Stability Selection (replace ARO) | — | Replace single-run ARO with bootstrapped Stability Selection (Meinshausen & Bühlmann 2010). Runs CatBoost feature importance on N=100 bootstrap samples, keeps features with stability_score ≥ 0.6. Target: 5 → 40+ stable features. Library: `scikit-learn-contrib/stability-selection`. |
| **P0** | **B10** | Per-Sector Models (eliminate cross-asset gap) | B9 | Train 7 sector-specific models (Tech, Financials, Energy, Healthcare, Consumer, Industrials, Utilities/REITs). Intra-sector features only — no cross-asset features → no 0-filled gaps at inference. Evidence: "Less is More" paper (QuantPedia 2024) — group-specific models outperform full cross-section models. |
| **P1** | **B11** | Combinatorial Purged CV (replace PurgedKFold) | B10 | ✅ Done (2026-05-14) — C(6,2)=15 paths, bagged ensemble, 32 tests pass. Integrated via `--cv-method cpcv`. |
| **P1** | **B12** | Dynamic Ensemble Learning (regime adaptation) | B11 | ✅ Done (2026-05-14) — `src/ml/dynamic_ensemble.py`. 5 CatBoost variants, EGD weight optimizer. Integrated into MLStrategy + pipeline. DEL Sharpe 0.91 vs single 0.69 (+32%). |
| **P2** | **B13** | Meta-Labeling secondary filter | B10 | ✅ Done (2026-05-14) — CatBoost classifier on regime/volatility/signal-clustering features. AUC 0.633, 14.6% signal reduction. Sharpe +2% (1.01→1.03), PF +11%. |
| **P2** | **B14** | Production hardening (DSR/PBO/walk-forward gates) | B11, B12, B13 | ✅ Done (2026-05-14) — PBO gate (0.061 PASS), DSR gate (0.946 FAIL), hold-out validation (Sharpe 1.27), health dashboard (KL 0.016 OK, 1 correlation_flip trigger). |

### Phase 6d: PDF Insight Integration

**Source:** `useful_resources/CHART_PATTERN_KNOWLEDGE_BASE.md` (12 parts, consolidated 2026-05-12 from 5 PDFs: Duddella 366pp, Harmonic Guide, NCFE, Fidelity/Kirkpatrick, 151 Trading Strategies). 8 insights extracted.

| Priority | # | Task | Insight Source | Depends On | Notes |
|----------|---|------|---------------|------------|-------|
| **P1** | **C11** | Volume/OI pattern validation layer | Insight #3 — 4 independent sources converge (NCFE, Fidelity, Duddella, Warrior Trading) on volume/OI rules | C10 | Rules: (1) High volume on breakout = confirm (+0.02 bonus already in C8). (2) Declining volume during formation = normal (no penalty). (3) Volume dissipating on Right Shoulder (H&S) = required validation for signal generation. (4) OI declining at Head (H&S) = required validation. Touches: `src/patterns/complex/head_shoulders.py`, `src/patterns/classic/double_top.py`, `src/patterns/classic/ascending_triangle.py`. |
| **P1** | **C12** | Multi-TP exit logic (partial take-profit) | Insight #5 — Universal practice in harmonic trading guides. 2-3 TP levels per trade. | C10 | Add partial TP to MLStrategy: TP1 at 50% of ATR target (exit 50% position), TP2 at 100% target (exit remainder). Move SL to breakeven after TP1 hit. Mechanical improvement — unchanged signals, only exit logic. Estimated +0.05–0.15 Sharpe from volatility drag reduction. |
| **P1** | **C13** | Gap pattern hierarchy + size filter | Insight #6 — Duddella 4-type gap classification (Common/Breakaway/Continuation/Exhaustion) | C10 | Enhance `src/patterns/breakout/gap.py`: (1) Classify gaps by type. (2) Breakaway → trade direction (almost never fills). (3) Exhaustion → fade (reversal). (4) Gap size > 2.5× ATR(10) → skip bar (noise filter). (5) Common gaps → skip entirely (low reliability). |
| **P2** | **C14** | Missing harmonic pattern detectors | Insight #2 | C8 ✅ | ✅ DONE (2026-05-16) — Butterfly, Bat, Crab, Cypher, Shark in `src/patterns/harmonic/extended.py`. Registered in RulesFirstStrategy + paper_trade_wf_honest.py. |
| **P2** | **C15** | Pipe pattern detector | Insight #8 | C8 ✅ | ✅ DONE (2026-05-16) — `src/patterns/complex/pipe.py`. 2-bar mechanical reversal, zero parameters. |
| **P2** | **C16** | Dead Cat Bounce ≥15% threshold fix | Insight #4 | C8 ✅ | ✅ DONE (2026-05-16) — threshold already enforced (default 0.15), bounce 50-62%, target 100% gap range. |
| **P3** | **C17** | Empirical pattern reliability calibration | Insight #1 | C10 | ✅ DONE (2026-05-16) — `scripts/calibrate_pattern_reliability.py`. Infrastructure built; solo data quality limited (many patterns need confluence to fire). Recalibrate with better solo backtest params later. |
| — | **V1** | Design validation: 151 Strategies confirms multi-condition approach | Insight #7 — 18 of 151 strategies are TA-based; 3-MA (3.13) and dual-momentum (4.1.2) use filter-on-filter logic | C3 ✅, C4 ✅ | Validates existing MLStrategy design (vol gate + confirm bars + trail stop). No code changes. Document in `docs/guide-pdf-insights.md`. |

### Insight → Implementation Mapping

| Insight | Plan Item | Priority | Rationale |
|---------|-----------|----------|-----------|
| #1 Pattern reliability rankings (empirical) | C17 | P3 | Already wired in C8 via literature defaults. Calibrate empirically later. |
| #2 Harmonic patterns have real structure | C14 | P2 | 5 missing detectors = uncovered alpha. |
| #3 Volume/OI missing validation | C11 | P1 | 4 independent sources converge. Low-hanging fruit for signal quality. |
| #4 Dead Cat Bounce ≥15% threshold | C16 | P2 | Existing detector may lack threshold. Quick fix. |
| #5 Multi-TP exit = free Sharpe boost | C12 | P1 | Mechanical exit improvement, no signal changes. |
| #6 Gap hierarchy underutilized | C13 | P1 | Breakaway vs Exhaustion = opposite trades. Critical classification. |
| #7 151 Strategies validates design | V1 | Validated | No code. Documentation only. |
| #8 Pipe pattern simplest formula | C15 | P2 | Two-bar, zero-parameter = most testable pattern. |

### Phase 10: Tool Evaluation Additions (NEW)

**Source:** Online research evaluation of 18 open-source tools vs project stack (2026-05-12). 12 already covered, 6 represent real gaps. [Full plan](enhance-tool-evaluation.md).

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P1** | **T10a-1** | Install dependencies (Optuna + PyPortfolioOpt + Bandit) | — | `uv add optuna PyPortfolioOpt bandit` |
| **P1** | **T10a-2** | Optuna hyperparameter tuning for CatBoost/LightGBM | T10a-1 | Replace GWO/GA/WOA with Bayesian (TPE) tuning. Create `src/ml/tuning/optuna_tuner.py`. Integrate into `scripts/tune_model.py --algo optuna`. MLflow logging of trials. |
| **P1** | **T10a-3** | Optuna strategy parameter tuning | T10a-1 | Replace manual `scripts/optimize_rsi.py`/`optimize_macd.py` grid searches with Optuna studies. Backtesting.py eval per trial. |
| **P1** | **T10a-4** | PyPortfolioOpt integration | T10a-1 | Replace `scipy.minimize` in `src/optimizer/portfolio_optimizer.py` with PyPortfolioOpt (EfficientFrontier, HRP, CVaR). Integrate with existing `black_litterman.py`. |
| **P1** | **T10a-5** | Test + validate tool additions | T10a-2,3,4 | Unit tests (≥5 each for Optuna tuner + PyPortfolioOpt). Full Optuna study on CatBoost (SPY, 20 trials). HRP vs equal-weight comparison on 33-ticker basket. |
| **P1** | **T10a-6** | Update documentation | T10a-5 | COMMAND_CHEATSHEET.md + `.useful_commands/` + ML_TRAINING_GUIDE.md Section 7. |
| **P2** | **T10b-1** | Add Bandit to pre-commit hooks | T10a-1 | ✅ Done — `.pre-commit-config.yaml`, 13 issues fixed, 0 medium+ |
| **P3** | **T10c-1** | CCXT crypto exchange integration | T10a-1 | Deferred — gate on crypto trading direction. |
| **P3** | **T10c-2** | FinGPT sentiment signal integration | T10a-1 | Deferred — gate on sentiment alpha proven via KG insights. |
| **P3** | **T10c-3** | aeon time-series ML integration | T10a-1 | Deferred — gate on shapelets beating CatBoost AUC by ≥5%. |

### Phase 10b: Autonomous Loop Hardening (NEW — 2026-05-13)

**Source:** Architecture re-evaluation after autonomous loop implementation (5 phases, 1418 lines). Loop wraps `train_ml_pipeline_v3.py`, `run_ml_backtest.py`, `tune_model.py` with independence clustering, consecutive confirmation, cross-group OOS, and Optuna Bayesian sweep. 5 hardening items identified.

| Priority | # | Task | Depends On | Notes |
|----------|---|------|------------|-------|
| **P0** | **LH-1** | Add checkpointing + resume (`--resume`) | Autonomous loop Phase 4 ✅ | ✅ Done — `save_checkpoint()` + `load_checkpoint()` + `--resume` flag. ~60 lines in `autonomous_train_loop.py`. |
| **P0** | **LH-2** | Multi-objective Pareto optimization (`--pareto`) | Autonomous loop Phase 5 ✅ | ✅ Done — `_optuna_sweep()` multi-objective branch. Pareto frontier from `study.best_trials`. ~55 lines. |
| **P1** | **LH-3** | Auto ETF-component cross-asset exclusion | Autonomous loop Phase 1 ✅, `train_ml_pipeline_v3.py` ✅ | ✅ Done — `build_exclusion_pairs()` + `exclusion_pairs.json` + pipeline column drop. ~30 lines. |
| **P1** | **LH-4** | Next-bar-direction label option (`--label-type next_bar`) | `train_ml_pipeline_v3.py` ✅, `autonomous_train_loop.py` ✅ | ✅ Done — `label_type` param in `run_pipeline()` + `generate_labels()`. `--label-type` CLI flag. ~25 lines. |
| **P2** | **LH-5** | Phase completion tracking (`phase_state.json`) | Autonomous loop all phases ✅ | ✅ Done — `save_phase_state()` after phases 1,3,4,5. Auto-load model from state. ~35 lines. |

### Implementation Order

```
LH-1 (Checkpointing) → LH-2 (Pareto) → LH-3 (ETF exclusion) → LH-4 (Next-bar label) → LH-5 (Phase tracking)
```

All 5 items are additive (~200 lines total, 2 files) — no architectural changes.

### Phase 12: Post-Retrain Validation & Next Steps (NEW — 2026-05-14)

**Source:** Analysis of retrained model state post B9-B14. 15 items identified across 4 priority tiers covering validation, signal quality, architecture, and new signal sources. [Full plan](post_retrain_next_steps.md).

| Priority | # | Task | Category | Notes |
|----------|---|------|----------|-------|
| **P0** | P0-1 | 2022 Bear Market Backtest | Validation | Gates all subsequent work. Does the strategy protect capital in a real bear market? |
| **P1** | P1-1 | Probability Calibration Audit | Signal Quality | Run `model_calibration.py` on retrained model. Apply Platt/Isotonic if overconfident. |
| **P1** | P1-2 | DSR Investigation | Validation | Is DSR=0.946 a signal failure or sample-size artifact? MC simulation of Sharpe=1.01 with N=19. |
| **P1** | P1-3 | Simpler-Than-ML Baseline | Validation | Compare CatBoost vs RSI(14) / 50-200 MA crossover on same engine. |
| **P1** | P1-4 | Regime-Conditional Returns | Signal Quality | Decompose backtest returns by Trending/Ranging/Bull/Bear regimes. |
| **P2** | P2-1 | Flipped Feature Analysis | Features | Which 3 features flipped correlation? Structural or noise? |
| **P2** | P2-2 | Feature Importance Comparison | Features | Stability-selected vs SHAP-important — do they agree? |
| **P2** | P2-3 | Dynamic Ensemble Fix | Architecture | Investigate why 5-model EGD ensemble produced Sharpe 0.26. |
| **P2** | P2-4 | Walk-Forward Cadence | Production | Simulate retraining at 6mo/12mo/24mo intervals. Find optimal cadence. |
| **P2** | P2-5 | Entry Threshold Sweep | Signal Quality | Sweep 0.35-0.55 on retrained model. Is 0.45 still optimal? |
| **P3** | P3-1 | Paper-Trading Harness | Production | Daily signal generation script for 2026-05-15 onward. Zero-risk OOS accumulation. |
| **P3** | P3-2 | Kelly Position Sizing | Portfolio | Compute Kelly fraction. Half-Kelly allocation. Minimum capital estimate. |
| **P3** | P3-3 | Survival Analysis | New Signal | scikit-survival Cox/RandomSurvivalForest for time-to-exit prediction. |
| **P3** | P3-4 | Regression Labels | New Signal | Predict 5-day forward return instead of binary TP/SL. CatBoostRegressor. |
| **P3** | P3-5 | HMM Regime Detection | New Signal | hmmlearn GaussianHMM for latent regime detection vs rule-based ADX/ATR. |

### Implementation Order

```
Phase 12a: Quick Validation (P0-1 → P1-1 → P1-3 → P2-5)
Phase 12b: Signal Understanding (P1-2 → P1-4 → P2-1 → P2-2)
Phase 12c: Architecture Improvements (P2-3 → P2-4 → P3-1 → P3-2)
Phase 12d: New Signal Sources (P3-3 || P3-4 || P3-5)  # parallelizable
```

 |


| P2 | T5 | Ensemble methods (Stacking, Voting, Blending) | T1b (tuned base models) | Combines CatBoost + LightGBM + RF |
| P2 | T6 | Monte Carlo VaR + CVaR risk modeling | Risk module ✅ | Replace binomial with proper VaR |
| P2 | T8 | SHAP visualization dashboard | SHAP already integrated | Waterfall, beeswarm, force plots |
| P2 | QW2 | Stop-loss optimization | Trade history data | GBDT predicts optimal stop distance |
| P2 | RS1 | EBM shape function alpha research pipeline | T2 (EBM) ✅ | Extract per-feature contribution curves → quantifiable alpha signals. Run on all 34 pattern detectors |
| P2 | RS2 | Dream team stacking ensemble (CatBoost + LightGBM) | T1b (GWO) ✅, T0 (unified) ✅ | Literature-validated: R² 0.815 vs 0.788 single-model. 3-5% accuracy gain |
| P2 | FS1 | Survival Analysis for Time-to-Target | scikit-survival (skill), existing multi-horizon labels | Predicts *when* TP/SL hits via censored regression. Requires FS4 completion. |
| P2 | FS2 | Historical Analog Matching (k-NN) | sklearn.NearestNeighbors or faiss | "When did market look like this before?" Trader-facing confidence tool. |
| P2 | FS3 | MAE/Drawdown as Primary Target | Existing max_drawdown_N computation | Predict worst-case drawdown → dynamic stops instead of fixed ATR. |
| P2 | FS5 | GMM Soft Regime Assignments | sklearn.mixture.GaussianMixture | Swap KMeans→GMM. Feed regime_proba to PatternClassifier. |
| P2 | FS8 | Volatility Forecasting Model | catboost (existing) | Predict realized vol over N bars → dynamic position sizing. |
| P2 | FS11 | Fractional Differentiation | statsmodels (installed) | Apply fracdiff(d≈0.3-0.5) to price features. Replace leaking StandardScaler. |
| P3 | T7 | Black-Litterman portfolio optimization | Portfolio module ✅ | ✅ Done — `src/portfolio/black_litterman.py` |
| P3 | T9 | Signal meta-labeling (kept as broader research task) | Signals module ✅ | López de Prado triple-barrier research task. Implementation now tracked as FS4. |
| P3 | RS3 | TabNet evaluation for regime detection | T2 (EBM) ✅ | Evaluate TabNet on high-dimensional feature sets. Only if >10K samples and GPU available |
| P3 | FS7 | Change-Point Detection for Regimes | ruptures (pure Python) | Real-time regime shift detection. Plugs into RegimeDetectorBase. |
| P3 | FS9 | Volume-Price Profile Clustering | sklearn.cluster | Accumulation vs distribution detection from volume-at-price. |
| P3 | FS10 | HDBSCAN Anomaly Detection | hdbscan | Auto-detect flash crashes, gaps. Feed anomaly_score → circuit breakers. |
| P3 | FS12 | Cross-Symbol Dynamic Clustering | sklearn + CrossAssetFeatures | Lead-lag relationships, correlation regime shifts. |
| P3 | FS13 | Breakout Probability ML | catboost + Donchian detector | ML score on breakouts: true breakout vs false one. |
| **KG1** | KG-H1 | Training-history overfitting detection | 5520 paper (training history), existing ML pipeline | Monitor loss curves for overfit in PurgedKFold. Source: Knowledge Graph Insights #1. |
| **KG1** | KG-H2 | Synthetic OOS comparison framework | Backtest Overfitting paper, `src/ml/` | Comprehensive OOS testing: combinatorial CV + synthetic controls. Source: KG Insights #1. |
| **KG1** | KG-H3 | Sentiment scores as signal weight modifier | 4+ sentiment papers, `src/signals/` | Feed Twitter/news sentiment into `EventWeightedAggregator`. Source: KG Insights #3. |
| **KG1** | KG-H4 | Event-driven pattern category | Building Calendar paper, Event-Based Trading paper, `src/patterns/` | New pattern category for event-based signals. Source: KG Insights #5. |
| **KG2** | KG-M1 | `src/rl/` module with trade execution env | OOM-RL paper, Adaptive RL paper, Deep Portfolio RL paper | New module: RL environment for trade execution + portfolio optimization. Source: KG Insights #2. |
| **KG2** | KG-M2 | Kelly criterion allocator | Investing Is Compression paper, `src/portfolio/` | Entropy/divergence-based position sizing. Source: KG Insights #4. |
| **KG2** | KG-M3 | AutoAlpha factor mining pipeline | AutoAlpha paper, `src/ml/` | Hierarchical evolutionary algorithm for formulaic alpha generation. Source: KG Insights #6. |
| **KG2** | KG-M4 | Circuit-based overfitting detection | Circuit Intrinsic Methods paper, `src/ml/` | Perturb rare patterns through model circuits. Source: KG Insights #1. |
| **KG2** | KG-M5 | Behavioral crash regime detection | Crash-based trading paper, `src/risk/` | Herding/overconfidence indicators for crash timing. Source: KG Insights #7. |
| **KG3** | KG-L1 | Adversarial overfitting detection | advrisk_neurips2019 paper, `src/ml/` | Use adversarial examples to expose overfit boundaries. Source: KG Insights #1. |
| **KG3** | KG-L2 | Financial event calendar database | 2 event papers, `src/data_ingestion/` | Build event DB from price spikes + news. Source: KG Insights #5. |
| **KG3** | KG-L3 | Defensive backtesting with time-reversal | Against Universal Trading paper, `src/backtest/` | Time-reversal heuristic for strategy validation. Source: KG Insights #7. |
| Research | T9 | Signal Meta-Labeling | LGBM default (CatBoost if ≥10% win) | Phase 6b Tier 1. "Should I take this signal?" |
| Research | FS19 | Gap-Fill Prediction | LGBM default | Phase 6b Tier 1. Predict gap fill within N bars. |
| Research | Ablation | Pattern Detector Audit | Backtest engine | Phase 6b Tier 1. Which 34 patterns produce edge? |
| Research | FS16 | Shapelets Discovery | aeon (CPU) | Phase 6b Tier 2. Learn discriminative price subsequences. |
| Research | FS15-lite | VAE Latent Embeddings [CPU] | torch (CPU, 32GB RAM ok) | Phase 6b Tier 2. Small VAE discovers market structure. |
| Research | FS20 | Heikin-Ashi Bars | Custom converter | Phase 6b Tier 2. Test smoothed bars on detectors. |
| Research | FS14 | Volume/Dollar/Tick Bars | Custom bar builder | Phase 6b Tier 3. Gate on Tier 1/2 positive results. |
| Deferred | FS17 | Volume Anomaly Forecasting | catboost | Absorbed into FS10 (HDBSCAN). |
| Deferred | FS18 | Synthetic OHLCV via GANs | torch (GPU) | GANs unstable. Defer indefinitely. |
| Deferred | FS21 | Sparse PCA | sklearn.decomposition | Feature selector more effective. |

## Deferred

| # | Phase/Item | Reason | Since | Revisit When |
|---|-----------|--------|-------|-------------|
| 05 | ML Advanced (full phase) | GPU >=16GB required for CNN training, autoencoder sweeps | 2026-04-30 | WSL2 with GPU passthrough or cloud GPU |
| 02 | vectorbt integration (sub-phase) | C++ compilation fails on Windows | 2026-04-20 | Linux/macOS environment |
| 07 | Paper Trading (full phase) | Optional — depends on Phase 06 completion | 2026-04-19 | Stages 01-06 stable |

### Phase 03 Detail: Regime Classification Table

*(Merged from 03-regime.md)*

| Regime | Condition | Enabled Strategies |
|--------|-----------|-------------------|
| Trending | ADX(14) > 25 | EMA Ribbon, SMA Crossover, ADX, Parabolic SAR, TSI |
| Ranging | ADX(14) < 20, low ATR | RSI Divergence, Williams %R, Stoch RSI, CCI, MFI |
| Volatile | ATR > 80th %ile | Chandelier Exit, Bollinger, Keltner, VWAP Bounce |
| Transition | ADX 20-25 | Maintains previous regime |

### Phase 05 Held Items (ML Advanced — Deferred for GPU)

*(Merged from 05-ml-advanced.md)*

| Item | Reason | Blocked By |
|------|--------|------------|
| CNN regime full training | >16GB VRAM needed | Hardware |
| Autoencoder risk factor sweep | GPU + 32GB RAM | Hardware |
| Ensemble XGB+LGBM+CatBoost | Training time prohibitive on CPU | Hardware |
| Walk-forward 5yr rolling windows | Compute bound | Hardware |

### Phase 07 Go/No-Go Criteria (Paper Trading)

*(Merged from 07-paper-trading.md)*

| Metric | Threshold |
|--------|-----------|
| Duration | >=14 calendar days |
| Signal match rate vs backtest | >=90% |
| Live Sharpe | >=0.5 |
| Live win rate | >=45% |
| Live profit factor | >=1.2 |
| Max drawdown | <=15% |
| Slippage tolerance | <=0.2% avg |
| Critical bugs | Zero |

**Risk limits:** Daily 3%, weekly 6%, monthly 10% loss halt. Kelly-capped at 2%/trade, max 5 open positions. Circuit breaker: 3 consecutive >2% losses → pause 24h.

### Phase 08 Detail: Verified Files

*(Merged from 08-attribution.md)*

| Layer | Component | File | Tests |
|-------|-----------|------|-------|
| 1 | Signal Event Log | `src/analysis/signal_event_log.py` | 7/7 |
| 2 | Trade Attributor | `src/analysis/trade_attributor.py` | 4/4 |
| 3 | Ablation Engine | `src/analysis/ablation_engine.py` | 3/3 |
| 4 | Synergy Analyzer | `src/analysis/synergy_analyzer.py` | 2/2 |
| — | Contribution Report | `src/analysis/contribution_report.py` | 2/2 |

### Pending Items Tracker

*(Merged from pending_items.md — last updated 2026-04-30. Full status is now tracked in the master task tables above.)*

Key completed items:
- Auto-Research Tools: 9 repos evaluated, 17 skills installed, ARCHITECTURE_ANALYSIS.md created. RD-Agent + OpenScholar blocked (Docker/Linux).
- ML Phase A (Infrastructure): All 6 components verified (experiment_logger, purged_cv, feature_store, metrics, registry, backtest_bridge).
- ML Phase B (Enhancement): All 7 components complete (features, regime_model, signal_scorer, feature_selector, cnn_regime, risk_factors, B7 backtest).
- Framework migration: B7 scripts run, remaining tasks (notebooks, optimization examples) deferred.
- Pattern Selection Pipeline: 7 modules verified.
- 10 items completed in 2026-04-26 session, 6 items in 2026-04-30 session.

## Execution Order

```
Phase 01 (Patterns) ✅
  → Phase 02 (Perf) 🔄 Numba done / vectorbt deferred
  → Phase 03 (Regime) ✅
  → Phase 04 (ML Foundation) ✅
      → Phase 06 (Research Enhancements) ✅
          → Phase 08 (Attribution) ✅
          → Phase 07 (Paper Trading) ⏸️
  Phase 05 (ML Advanced) ⏸️ — wait for GPU

Enhancement Execution Order:
  Enhancement Phase 1: Foundation Fixes (P1) ✅ DONE
    ├── T4 (DSR/PSR/FDR) ✅
    ├── FS4 (Triple Barrier Labeling) ✅
    └── FS6 (Combinatorial Purged CV) ✅
  Enhancement Phase 2: Quick Wins (P2) ✅ DONE
    ├── FS2 (Historical Analog Matching) ✅
    ├── FS3 (MAE/Drawdown Target) ✅
    ├── FS5 (GMM Regimes) ✅
    └── FS11 (Fractional Differentiation) ✅
  Enhancement Phase 3: New Models (P2) ✅ DONE
    ├── FS1 (Survival Analysis) ✅
    ├── FS8 (Volatility Forecasting) ✅
    ├── FS13 (Breakout Probability ML) ✅
    ├── T5 (Ensemble Methods) ✅
    └── T8 (SHAP Dashboard) ✅
  Enhancement Phase 4: Advanced Pipeline (P3) ✅ DONE
    ├── FS7 (Change-Point Detection) ✅
    ├── FS9 (Volume-Price Profile Clustering) ✅
    ├── FS10 (HDBSCAN Anomalies) ✅
    ├── FS12 (Cross-Symbol Clustering) ✅
  Enhancement Phase 4+: Risk, Alpha & Ensemble (T6, QW2, RS1, RS2) ✅ DONE
    ├── T6 (MC VaR + CVaR) ✅
    ├── QW2 (Stop-Loss Optimization) ✅
    ├── RS1 (EBM Alpha Pipeline) ✅
    └── RS2 (Dream Team Ensemble) ✅
   Enhancement Phase 6a: Model Robustness & Production Readiness ← IN PROGRESS
      │  └──  Why first: edge found on 5/12 tickers only. Must understand scope before
      │      building on top of it. Research/audit work is noise until the edge is clear.
      ├── MR0.1 C1 (Paper trade re-run) ✅
      ├── MR0.2 C2 (Walk-forward validation) ✅ — 15 bugs fixed, 5/12 pass
      ├── MR1 C3 (Basket vs single-ticker ablation) ✅ — 12-all > 5-winner. More diversity = better IC.
      ├── MR1 C4 (Prediction correlation decomposition) ✅ — Tech cluster r>0.82, KODK/GLD independent.
      ├── MR1 C6a (Auto ticker screening pipeline) ✅ — 75 screened, 22 passed. docs/ticker-test-log.md
       ├── MR2 C6 (Expand basket to 30+ tickers) ✅ (2026-05-11) — 33-ticker model. Mean IC +75%. All gates pass.
       ├── MR2 C6b (Edge relaxation analysis) ✅ (2026-05-11) — Volatility is the binding constraint. Sweet spot ≤25% ann. vol.
       ├── MR2 C5 (ML + pattern detector integration) ✅ (2026-05-11) — ml_strategy.py, 33-ticker backtest, 19/33 profitable.
       ├── P1  C7 (Strategy execution improvements) ✅ DONE (trail-stop winner)
       ├── P2  C8 (Pattern detector confluence) ✅ DONE (PatternBoostFilter)
      ├── P2  B1 (OOS validation) ✅ — FAILED: Sharpe -0.27 vs +0.73. Model overfit.
      ├── P1  B2 (Fix backtest_ml_enhanced.py) ✅ — 3 bugs fixed.
      ├── P2  B3 (Calibration audit) ✅ — ECE 0.128 IS, 0.197 OOS. Overconfident.
      ├── P1  B4 (Regime shift investigation) ✅ — Root cause: raw ATR scaling.
      ├── P1  B5 (WFO comparison) ✅ — WFO OOS +1.7% vs single-split -1.3%.
       ├── —   B8 (Agent infrastructure) ✅ — 3 agents + 3 commands + AGENTS.md update.
       ├── P0  B6 (Retrain with normalized ATR) ✅ — 2/5 pass. OOS worse. Root cause not fixed.
       ├── P0  B7 (Evaluate Qlib ADARNN/ADD) ✅ — ADARNN/ADD incompatible. CatBoost 0 positions.
        ├── P0  B11 (Combinatorial Purged CV) ✅ DONE
        ├── P0  B12 (Dynamic Ensemble Learning) ✅ DONE — Sharpe +32%
        ├── P1  B13 (Meta-Labeling) ← NEXT
       ├── P2  B14 (Production Hardening)
       └── P2  C9, C10, C11-C16 (gated on B9-B14)

  Enhancement Phase 6b: Pioneer Research (prove-or-discard) ← DEFERRED until MR passes
    ├── P1.1 T9 (Meta-Labeling) ✅
    ├── P1.2 FS19 (Gap-Fill Prediction) ✅
    ├── P1.3 Ablation (Pattern Detector Audit) ✅
    ├── P2.1 FS16 (Shapelets)
    ├── P2.2 FS15-lite (VAE on CPU)
    ├── P2.3 FS20 (Heikin-Ashi)
    └── P3.1 FS14 (Alt Bars — gate on Tier 1/2)
  Enhancement Phase 7: Notebook Audit Fixes ← DEFERRED until MR passes
    ├── F1 (Fix ablation/synergy metric extraction) ✅
    ├── F2 (Fix PatternSelector pattern discovery) ✅
    ├── F3 (Re-run notebooks 06, 07 to validate)
    ├── ML1 (Fix signal scorer)
    ├── ML2 (Reduce regime classifier overfit)
    └── ML3-4, S1-S4 (Feature, ensemble, meta-labeling improvements)
  Enhancement Phase 5: Deferred / GPU
    └── T7 (Black-Litterman) ✅, FS17, FS18, FS21, Phase 05 (ML Advanced)

   Phase 6c: Strategy Refinement ← ACTIVE
     ├── P1  C7 (Strategy execution improvements) ✅ DONE (trail-stop winner)
     ├── P1  C8 (Pattern detector confluence — PatternBoostFilter) ✅ DONE
    ├── P2  C9 (Honest walk-forward paper trading)
       ├── P2  C10 (Portfolio-level backtest)
       └── Phase 6d: PDF Insight Integration ← NEW
           ├── P1  C11 (Volume/OI validation)
           ├── P1  C12 (Multi-TP exit logic)
           ├── P1  C13 (Gap hierarchy + size filter)
           ├── P2  C14 (Missing harmonic detectors: Butterfly/Bat/Crab/Cypher/Shark)
           ├── P2  C15 (Pipe pattern detector)
           ├── P2  C16 (Dead Cat Bounce threshold fix)
           └── P3  C17 (Empirical reliability calibration)

   Phase 10: Tool Evaluation Additions ← NEW (2026-05-12)
     ├── P1  T10a-1 (Install Optuna + PyPortfolioOpt + Bandit)
     ├── P1  T10a-2 (Optuna HP tuning for CatBoost/LightGBM)
     ├── P1  T10a-3 (Optuna strategy parameter tuning)
     ├── P1  T10a-4 (PyPortfolioOpt integration)
     ├── P1  T10a-5 (Test + validate)
     ├── P1  T10a-6 (Documentation)
     ├── P2  T10b-1 (Bandit pre-commit hook)
     └── P3  T10c-1/2/3 (CCXT, FinGPT, aeon — deferred)

   Phase 10b: Autonomous Loop Hardening ← DONE (2026-05-13)
     ├── P0  LH-1 (Checkpointing + resume) ✅
     ├── P0  LH-2 (Multi-objective Pareto optimization) ✅
     ├── P1  LH-3 (Auto ETF cross-asset exclusion) ✅
     ├── P1  LH-4 (Next-bar-direction label) ✅
     └── P2  LH-5 (Phase completion tracking) ✅

   Phase 11: Overfitting Fixes ← COMPLETE (2026-05-14)
      ├── P0  B9  (Stability Selection) ✅
      ├── P0  B10 (Per-Sector Models) ✅
      ├── P1  B11 (CPCV) ✅ — C(6,2)=15 paths, bagged ensemble
      ├── P1  B12 (Dynamic Ensemble Learning) ✅ — Sharpe +32% (0.69→0.91)
      ├── P2  B13 (Meta-Labeling) ✅ — AUC 0.633, Sharpe +2% (1.01→1.03)
      └── P2  B14 (Production Hardening) ✅ — PBO 0.061 PASS, hold-out Sharpe 1.27, KL 0.016 OK

   Phase 12: Post-Retrain Next Steps ← IN PROGRESS
       ├── P0-1: 2022 Bear Market Backtest ✅ — beats SPY (-8.68% vs -11.27%) but Sharpe -1.5. GATE FAIL.
       ├── P1-1: Probability Calibration Audit ✅ — ECE OOS=0.197. GATE FAIL. Overconfident.
       ├── P1-3: Simpler-Than-ML Baseline ✅ — ML 0.60 < MA 0.70. GATE FAIL.
       ├── P2-5: Entry Threshold Sweep ✅ — et=0.35 trail new best (Sharpe 0.60).
       ├── P1-2: DSR Investigation ✅ — proxy not real DSR. IC=0.189 is moderate.
       ├── P1-4: Regime-Conditional Returns ✅ — profitable in ALL regimes, +0.25% in Bear.
       ├── P2-1: Identify Flipped Features ✅ — 3 flipped, vol_regime KS=0.81 worst.
       ├── P2-2: Feature Importance Comparison ✅ — MDI vs SHAP rho=0.675. Top 5 Jaccard 43%.
        ├── P2-3: Dynamic Ensemble Collapse ✅ — Stacking meta-model replaces EGD. Rerouted to LogisticRegression ensembling.
         ├── P2-4: Walk-Forward Cadence ✅ — 12mo best Sharpe 1.62, 24mo best return 9.1%. Frequent (≤4mo) degrades. scripts/backtest_wf_cadence.py
        ├── P3-1: Paper-Trading Harness ✅ — scripts/paper_trade_daily.py with backfill + status
        ├── P3-2: Kelly Position Sizing ✅ — estimate_minimum_capital() + format_kelly_report() in kelly_allocator.py
         ├── P3-3: Survival Analysis ✅ — GBSA C-index OOS 0.684, RSF 0.673. Gate PASSES. scripts/train_survival_exit.py
         ├── P3-4: Regression Labels ✅ — CatBoostRegressor DirAcc OOS 62.6%, IC 0.107. Gate PASSES. scripts/train_regression_labels.py
         └── P3-5: HMM Regime Detection ✅ — HMM Ensemble AUC 0.558 < Single 0.610. Gate FAILS. Simple 200MA rule superior. scripts/train_hmm_regime.py

   Phase 16: Direction N — NLP + Quant Schools Expansion ← ACTIVE
       │  └── Source: NLP in Quant 101 course + Chinese 5 Quant Schools. 81 ideas → 9 phases.
       ├── P0  Phase N (NLP Foundation): LM Dictionary Scorer ......... ✅ DONE
       ├── P0  Phase P (Mean Reversion System): Dedicated engine ..... ✅ DONE (RegimeRouter 209% ret, Sharpe 0.68)
       ├── P1  Phase O (Social Media Valid): Lead/lag gate first ..... Validation before infra
       ├── P1  Phase Q (Multi-Factor Fund): Fundamental factors ...... New feature dimension
       ├── P2  Phase R (Advanced NLP): FinBERT + SEC filings ......... Builds on Phase N
       ├── P2  Phase V (Strategy Arch v2): Short/crypto/portfolio.... Structural improvements
       ├── P2  Phase S (Stat Arb): Pairs trading engine .............. Orthogonal alpha
       ├── P3  Phase T (Deep Learning): GPU gate ...................... LSTM/Transformer/CNN
        └── P3  Phase U (Alternative Data): Paid data gate ............. Satellite/credit/supply

   Phase 17: Resource-Driven Enhancements — Factor + Conversion ← NEW (2026-05-16)
       │  └── Source: 华泰多因子, Beyond Fama-French, FMZ Strategies (5,807 files). 14 additions.
       ├── P0  R1  (IR-Weighted Synthesis): Replace static reliability → rolling IR ....... 80 loc, 0 deps
       ├── P0  R2  (Factor Purification): Regress out sector/size before IC ................ 120 loc, 0 deps
        ├── P0  R3  (4-Step Eval Gate): t-test → IC → quantile backtest for new patterns ... 200 loc, statsmodels
        ├── P0  R4  (HP Filter Forecasting): Replace mean → HP trend extraction .............. 30 loc, 0 deps
        ├── P1  R5  (Collinearity Analysis): VIF matrix + synthesize/discard redundancy ..... 100 loc, statsmodels ✅
        ├── P1  R6  (Liquidity Factor CEI): Acharya-Pedersen, Δmkt_equity − return .......... 60 loc, FMP ✅
        ├── P1  R7  (Multi-Dim Scoring): IC + IR + turnover + diversity + overfit axes ...... 150 loc, 0 deps ✅
        ├── P1  R8  (MAD+Rank Pipeline): Median outlier removal + non-parametric std ......... 100 loc, 0 deps ✅
       ├── P2  R9  (Return/Risk Classification): Directional vs variance factor tagging...... 100 loc, statsmodels
       ├── P2  R10 (Attribution Decomp): Marketβ + sector + style + specific alpha ......... 180 loc, statsmodels
       ├── P2  R11 (Default Risk DtD): Merton model — ln(V/D) + (r−σ²/2)T / σ√T .......... 80 loc, FMP
       ├── P2  R12 (QRAFTI Protocol): 14-test standardized diagnostic suite ................ 250 loc, 0 deps
   ├── P3  R13 (Full Optimization Pipeline): IR→HP→risk→QP ............................. 400 loc, scipy ✅
         └── P3  R14 (Factor Engine Wrapper): Try-install open-source library ................. 40 loc, factor-engine pkg ✅

   Phase 18: Useful Repos Integration ← ACTIVE (2026-05-16)
        │  └── Source: 7 repos in C:\Dev\useful_repos evaluated. Priority-ranked.
        ├── P0  R18a (eiten: Portfolio Optimization): Eigen/MVP/MSR/GA + RMT + Monte Carlo .. 300 loc, scipy/sklearn
        ├── P0  R18b (Scrapling: Web Scraping): Financial data pipeline + anti-bot ........... 200 loc, lxml/curl_cffi
        ├── P1  R18c (agent-skills: Workflow Skills): Port 23 skills → .kilo/skills/.......... 0 loc (markdown only)
        ├── P2  R18d (Qbot: Quant Reference): Study 25+ DL models + 30+ indicators............ 0 loc (study only)
        ├── P2  R18e (qmd: Knowledge Base Search): Index docs + papers_md .................... 0 loc (external tool)
        ├── P3  R18f (CLI-Anything: Script Patterns): Agent-friendly script conventions ...... 0 loc (convention doc)
        └── P3  R18g (local-deep-research: MCP Pattern): Research server architecture ........ 0 loc (study only)

    Cross-Project Repo Utility (C:\Dev\projects)
         │  └── Evaluated 4 other projects for utility from same 7 repos.
         ├── albion_get_rich: Scrapling HIGH (replace 4 scraping deps), agent-skills MED, qmd LOW
         ├── data_and_stat_analysis: Scrapling MED, local-deep-research MED, qmd MED, agent-skills MED
         ├── personal_website: agent-skills HIGH (web dev lifecycle), qmd LOW
         └── skills_arsenal: agent-skills HIGH (skill format), CLI-Anything HIGH, qmd HIGH, Scrapling MED

   Phase 20: System Hardening & Signal Quality ← PLANNED (2026-05-17)
        │  └── Source: Empirical session testing across all modules.
        ├── P0  H1  (Multi-TP Default): use_multi_tp=True as system default ............... 3 loc, 0 deps
        ├── P0  H2  (Pattern Quality Registry): Gate 45 patterns, FAIL→0.5x/confluence .... 150 loc, 0 deps
        ├── P1  H3  (Sector-Factor Scoring): Per-sector factor models → signal modifier .... 200 loc, statsmodels
        ├── P1  H4  (IR Scalar Mode): 0.3x–0.7x multiplier instead of binary zeroing ...... 30 loc, 0 deps
        ├── P2  H5  (GA Post-Step): Auto-optimize portfolio after batch backtest ........... 50 loc, scipy
        ├── P2  H6  (BTC Config Audit): Systematic sweep → crypto preset ................... 30 loc, 0 deps
   └── P3  H7  (Pattern Gate Sweep): evaluate_pattern.py on all 45 detectors ......... 50 loc, 0 deps

    Phase 21: Quant-Resources Signal Enhancers ← EXPANDED — ALL BLOCKS COMPLETE (2026-05-18)
        │  └── Source: Quant-Developers-Resources repo analysis (52 md files, 12 PDFs). 46 ideas extracted (4 blocks) → 8 prioritized + 38 deferred/planned.
        │
        ├── Implementation Queue (P0-P3, 8 items)
        │   ├── P0  Q1  (VIX Regime Gate): Wire RegimeFolio VIX slope → RegimeGate features ... 80 loc, 0 deps (75% built)
        │   ├── P0  Q2  (Yield Curve Inversion): 2s10s/10y3m binary flag → macro_regime ........ 120 loc, FRED/yfinance
        │   ├── P1  Q3  (GARCH Vol Forecast): Forward-looking vol beats reactive ATR ........... 200 loc, `uv add arch`
        │   ├── P1  Q4  (PC Ratio + GEX): Options flow signals → SentimentProvider protocol .... 250 loc, yfinance/FMP
        │   ├── P2  Q5  (Model Validation): PSI/KS/Gini/SHAP monitoring suite .................. 300 loc, sklearn (0 deps)
        │   ├── P2  Q6  (Copula Tail-Risk): Joint extreme risk for basket strategies ........... 300 loc, `uv add copulae`
        │   ├── P3  Q7  (Market Impact): Almgren-Chriss backtest→live cost bridge .............. 200 loc, numpy/scipy
        │   └── P3  Q8  (Order Book Features): Extend order_flow.py with bid-ask/imbalance .... 300 loc, 0 deps
        │
        ├── Block A: Options Pricing & Volatility (13 ideas — gated on Q4)
        │   ├── → Q1 (A10: VIX term structure), → Q4 (A8: PC ratio, A9: GEX)
        │   ├── A1  (Black-Scholes pricing) 🔴 gated on Q4
        │   ├── A2  (Binomial Tree) 🔴 gated on Q4
        │   ├── A3  (Monte Carlo pricing) 🔴 gated on Q4
        │   ├── A4  (Heston stochastic vol) 🔴 gated on Q4 — stub in overfitting_detectors.py:131
        │   ├── A5  (SABR stochastic vol) 🔴 gated on Q4+A4
        │   ├── A6  (Volatility Surface) 🔴 gated on Q4+A1-A4
        │   ├── A7  (Greeks calc) 🔴 gated on Q4+A1
        │   ├── A11 (Delta hedging) 🔴 gated on Q4+A7
        │   ├── A12 (Options payoff viz) 🔴 gated on Q4+A6
        │   └── A13 (Vol trading strategies) 🔴 gated on Q4+A1-A7 — highest-value, highest-dependency
        │
        ├── Block B: Fixed Income & Macro (11 ideas — gated on Q2)
        │   ├── → Q2 (B3: yield spread inversion), B4 (credit spread ≈ gate) 🔴 planned Q2+
        │   ├── B1  (Nelson-Siegel yield curve) 🔴 gated on Q2 + Treasury data
        │   ├── B2  (Duration/convexity for bonds) 🔴 gated on Q2 — TLT/IEF in cross_asset_features.py
        │   ├── B5  (Vasicek rate model) 🔴 gated on Q2+B1
        │   ├── B6  (CIR rate model) 🔴 gated on Q2+B5
        │   ├── B7  (Rate derivative pricing) 🔴 gated on Q2+B5+B6 — P3
        │   ├── B8  (Treasury auction cycles) 🔴 gated on Q2
        │   ├── B9  (TLT/IEF rate proxies) 🔴 partial — _add_bond_features exists
        │   ├── B10 (Real yield TIPS analysis) 🔴 gated on Q2+B1
        │   └── B11 (CDS pricing) 🔴 gated on Q4 — P3
        │
        ├── Block C: FPGA / HFT Systems (10 ideas)
        │   ├── → Q8 (C8: order book dynamics), → Q7 (C9: market impact)
        │   ├── C1-C2 (FPGA feed handlers / matching engine) 🔴 Out-of-Scope — hardware only
        │   ├── C3-C4 (Kernel bypass / PTP timing) 🔴 Concept-Only — educational for architecture
        │   ├── C5 (HFT risk checks) 🔴 Concept-Only — design patterns applicable to software
        │   ├── C6 (Exchange protocols: ITCH/OUCH/FIX) 🔴 Concept-Only — external docs
        │   ├── C7 (Latency profiling) 🔴 Concept-Only — instrument backtest engine design pattern
        │   └── C10 (HF signal processing: wavelet/FFT) 🔴 gated on Q8 — tick data required
        │
        └── Block D: Original Recommendations (12 ideas)
            ├── → Q3 (D1: GARCH), → Q6 (D2: copula), → Q5 (D8: model validation), → Q7 (D4: Almgren-Chriss)
            ├── D3  (Structural Break / Unit-Root tests) 🔴 gated on Q1+Q2 — ~150 loc, statsmodels
            ├── D5  (PPO/SAC RL) 🔴 gated on RL infra stable — ~500 loc, stable-baselines3
            ├── D6  (VAR + Granger Causality) 🔴 planned P3 — gated on basket strategies
            ├── D7a-d (Ichimoku/Keltner/Williams %R/CCI) 🔴 deferred — 45+ patterns, diminishing returns
            ├── D9  (Kalman Filter hedge ratios) 🔴 partial — kalman_hedge.py exists, extend to portfolio
            ├── D10 (Rolling ARIMA+GARCH) 🔴 gated on Q3 — ~250 loc
            ├── D11 (State Space Models) 🔴 gated on Q3 — ~200 loc
            └── D12 (Offline RL CQL) 🔴 gated on D5 — ~400 loc
```
```
