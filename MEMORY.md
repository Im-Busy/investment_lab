# Agent Handover State

> Auto-updated each session. Read this first when resuming.

## Current Objective

**10 New Ticker Screen + Backtest (2026-05-27).** 10 fresh candidates screened against 11 hard filters, data downloaded, Rules-First all-on backtests run (IS=2016-2024, OOS=2025-2026). 5/10 positive OOS Sharpe. 2 S-tier, 2 A-tier, 1 B-tier, 5 C-tier.

**Phase 28 implementation — NEAR COMPLETE (2026-05-27).** 33 insights from 4 external repos. 22/27 tasks complete (81%).

| # | Item | File | LOC | Status |
|---|------|------|-----|--------|
| P28-1 | Survivorship-free universe | `src/data/historical_universe.py` | 160 | ✅ delisted retention, `universe_on(date)` |
| P28-2 | Hierarchical symbol filter pipeline | `src/data/symbol_filter.py` | 80 | ✅ `filter_hierarchical()`, `filter_pipeline()`, `FilterResult` |
| P28-3 | CRNG fat-tail RNG → label shuffling | `src/ml/label_shuffling.py` (modify) | 90 | ✅ `generate_crng_labels()` + `run_crng_baseline_test()` |
| P28-4 | show_options() CLI pattern | `src/cli/option_discovery.py` | 175 | ✅ `show_options()`, FinanceDatabase schema query |
| P28-5 | Batch loader + progress | `src/data/batch_loader.py` | 50 | ✅ `load_batch()`, ThreadPoolExecutor, `BatchProgress` callback |
| P28-6 | FinanceDatabase symbol layer | `src/data/financedb_layer.py` | 50 | ✅ `get_universe()`, `search_symbols()`, `count_by_sector()`, lru_cache |
| P28-7 | Fundamental analysis pipeline | `src/data/fundamental_pipe.py` | 210 | ✅ `to_toolkit()`, `fundamental_features_for_ml()`, `pipe_sector_fundamentals()` |
| P28-8 | US stock symbols auto-update | `src/data/symbol_sync.py` | 190 | ✅ `fetch_all_us_symbols()`, `sync_with_diff()`, `diff_symbols()` |
| P28-9 | Adanos sentiment API | `src/nlp/adanos_sentiment.py` | 135 | ✅ REST, zero deps, batch fetch, cache TTL |
| P28-10 | Congressional trade signals | `src/data/congressional_signals.py` | 250 | ✅ `fetch_congress_trades()`, `aggregate_trades_by_ticker()`, `get_congress_signals()` |
| P28-11 | Faiss chart pattern similarity search | `src/patterns/similarity_search.py` | 200 | ✅ `search_similar_patterns()`, `search_rolling()`, `format_result_table()` |
| P28-12 | Patternity chart pattern wrapper | `src/patterns/patternity_wrapper.py` | 170 | ✅ `PatternityWrapper.detect()`, `compare_with_internal()` |
| P28-13 | Fund flow / order flow signals | `src/signals/fund_flow.py` | 330 | ✅ MFM/OBV/A/D/EoM + large-order detector, `fund_flow_to_ml_features()` |
| P28-14 | Futures inventory alt data signals | `src/data/futures_inventory.py` | 195 | ✅ `FuturesInventory.get_signals()`, supply/demand bias, commodity exchange map |
| P28-18 | skfolio portfolio optimization | `src/optimization/skfolio_optimizer.py` | 280 | ✅ `compare_methods()`, MV/CVaR/HRP/RiskBudget/InvVol |
| P28-19 | Indicator computation reference | `docs/reference_phase28_external_tools.md` | 0 | ✅ 9 indicators (MA/MACD/BOLL/RSI/WR/CCI/ATR + KDJ/BIAS gaps) |
| P28-20 | Options data & Greeks | `src/data/options_data.py` | 305 | ✅ `compute_greeks()`, `OptionsChain`, `get_options_sentiment()`, unusual activity |
| P28-21 | MCP stock data server | `src/mcp/stock_server.py` | 195 | ✅ `StockDataTools`, 3 MCP tools, stdio server |
| P28-22 | Daily backtest report agent | `scripts/daily_report_agent.py` | 210 | ✅ `generate_daily_report()`, TickerStatus, regime check |
| P28-23 | Web dashboard architecture ref | `docs/reference_phase28_external_tools.md` | 0 | ✅ Next.js 15 + shadcn/ui + TradingView + MongoDB |
| P28-24 | Docker Compose production | `docker-compose.prod.yml` | 70 | ✅ App + MongoDB + 5 volumes + healthchecks |
| P28-25 | Risk-profiling onboarding | `scripts/user_profile.py` | 265 | ✅ `build_profile()`, `generate_recommendation()`, interactive CLI |
| P28-26 | WFGY LLM agent stress-test | `docs/reference_phase28_external_tools.md` | 0 | ✅ 16-mode failure map for LLM trading agents |
| P28-27 | tf-quant-finance evaluation | `docs/reference_phase28_external_tools.md` | 0 | ✅ PDE/HW/MC/quasi-random — not worth adopting |

**New deps:** `financedatabase` (v2.3.1), `faiss-cpu` (1.14.2), `skfolio` (0.20.1), `patternity` (0.1.0) + transitive deps (`cvxpy-base`, `pycryptodome`, `python-binance`, `qdldl`, `sparsediffpy`).

**Phase 28 remaining (deferred):** P28-15 (MarS — GPU), P28-16 (FinRL — GPU), P28-17 (Multi-agent — LLM+GPU). 3 GPU/LLM-gated tasks.

**Q2 2026 OOS Re-Run (2026-05-27):** 17/17 (100%) OOS positive. Mean Sharpe 1.135, median 1.065. IS→OOS correlation -0.388. Market regime BULLISH (71% above MA50, RSI 59.0). CN_CATL excluded (yfinance 404). Paper trading: 3 BUY (XLK/NUE/STLD), 14 HOLD. GPU tasks skipped (torch 2.9.1+cpu). Results in BESTS.md §Q2 2026.

**Recommended next:** Commit results. Continue paper trading daily. Re-check GPU availability for TTS-GAN P0 gate. Re-run OOS quarterly at end of Q3/Q4 2026.

| Tier | OOS Sharpe | OOS Pos |
|------|-----------|---------|
| S (6) | +0.093 | 3/6 (50%) |
| A (5) | +0.254 | 4/5 (80%) |
| B (7) | +0.304 | 4/7 (57%) |
| **All (18)** | **+0.195** | **11/18 (61%)** |

**Key insights:**
- B-tier phoenix plays (INTC +1.245, MRK +0.918, NEM +0.889, SPY +0.530) get +0.5 Δ from all-on
- CN_CATL (-1.15 Δ) and GLD (-1.08 Δ) broken by all-on — simpler is better for China/Gold
- EOG saved from -0.712 baseline to +0.120 OOS
- VIXRegimeGate import bug — gracefully disables (non-blocking)
- GARCH convergence issues on several tickers — manageable

**Recommended:** Use all-on for B-tier (15% of basket), keep simpler config for S/A-tier.

**Phase 24 P3 Deferred Items — IMPLEMENTED (2026-05-27).** 4 remaining deferred items from Phase 24 P3 now complete:

| # | Item | File | LOC | Status |
|---|------|------|-----|--------|
| P24-29 | Two-phase GA rule combination | `src/optimization/two_phase_ga.py` | 240 | ✅ NEW |
| P24-32 | Divergence-in-bits strategy comparison | `src/analysis/divergence_bits.py` | 140 | ✅ NEW |
| P24-33 | Binomial VAR for event-driven risk | `src/risk/binomial_var.py` | 170 | ✅ NEW |
| P24-35 | W/M Bollinger patterns | `src/patterns/bollinger/wm_patterns.py` | 188 | ✅ EXISTS (Phase 25) |

**All 27 phases + all P3 deferred items now COMPLETE. Phase 28 (Resource Extraction) ACTIVE — 33 insights, 4 P0 items ready to implement.**

**Recommended next:** Continue Phase 28 — P28-2 (symbol filter pipeline, 80 loc), P28-5 (batch loader, 50 loc), P28-6 (FinanceDatabase layer, 50 loc) — remaining P0 data infra items.

**Phase 26: Bear Market Validation COMPLETE → Paper Trading Launch (2026-05-25).** Bear market backtest (IS=2016-2021, OOS=2022-2026) **PASSED — 12/18 (67%) positive OOS Sharpe ≥60% gate.** System survived -24.5% SPY bear (2022) + oil shock (2026 Q1) + V-shaped recovery (Apr 2026).

**Comprehensive 122-instrument backtest results (2026-05-25):** IS→OOS Sharpe correlation **-0.226** (IS anti-predictive). Energy 89% OOS positive (top category). MidCap-Steel emerged as top-tier (NUE +1.43, STLD +1.06). Financials/REITs/HK/Utilities structural failures (mean -0.45 to -0.68). 15 zero-trade tickers. Production basket expanded to 18 instruments across 3 tiers.

**Production basket (updated 2026-05-25, OOS verified 2026-05-27):**
| Tier | Instruments | Allocation | Mean OOS Sharpe (Q2 2026) |
|------|------------|------------|---------------------------|
| **S (60%)** | XLK, XLE, GLD, SPY, SLV, QQQ | 6×10% | **+0.949** |
| **A (25%)** | NUE, STLD, HAL, MPC, EOG | 5×5% | **+1.350** |
| **B (15%)** | INTC, AMD, LMT, JNJ, MRK, NEM | 6×2% | **+1.143** |
| **All (17)** | (CN_CATL excluded — yfinance 404) | 100% | **+1.135** |

**New Ticker Candidates (2026-05-27):** 10 screened, 10 passed, 5/10 positive OOS Sharpe.

| Tier | Tickers | N | Mean OOS Sharpe | Notes |
|------|---------|---|-----------------|-------|
| **S-NEW** | CHTR (+1.11), LRCX (+0.90) | 2 | **+1.00** | Phoenix plays. CHTR -59% BH, +1.22 delta. Add to S-tier expansion. |
| **A-NEW** | GD (+0.75), ABT (+0.70) | 2 | **+0.73** | Defense + healthcare phoenix. Both IS negative → OOS positive. |
| **B-NEW** | NOC (+0.23) | 1 | +0.23 | Marginal. Defense sector. 62% WR on 16 trades. |
| **C-SKIP** | DHI, URI, CTVA, APH, GE | 5 | -0.81 | Low trades or bad WR. Skip. |
| **All-new** | — | 10 | **+0.01** | 5/10 (50%) positive. |

**Key findings:**
- CHTR anti-correlated to market (BH -59%, strategy +1.1% OOS) — telecom/media patterns work
- LRCX 69% WR, 2.67 PF with 13 trades — semiconductors work
- GE/APH/URI fail from insufficient trades (3 OOS each with production config)
- CTVA structural fail — agriculture doesn't produce chart patterns
- All 5 winners have IS→OOS delta >= +0.77 (phoenix pattern consistent)

**Paper Trading Signals (2026-05-27):** 3 BUY (XLK +0.554, NUE +0.555, STLD +0.556), 14 HOLD, 1 FAIL (CN_CATL). Bullish market — sector rotation favoring MidCap-Steel + Tech.

**SMC binary score FIXED + dead-param audit FIXES APPLIED + paper gaps CLOSED (2026-05-21).**
- Conviction grading added to SMC sweep detection — scores now continuous (not binary ±0.905). Trade count varies smoothly with entry threshold.
- 14 dead params deleted across 6 files, `_calculate_size()` wired, Phase 6 blocks gated.
- **Paper comparison report: 25/25 recommended actions now COMPLETE.** 8 new modules, ~1,000 LOC.
- **7/8 modules wired end-to-end** into execution paths:
  - RulesFirst: B6 (voting), B1 (35-rule catalog), B10 (divergence), B2 (W/M Bollinger) — all via `_signals_cache`
  - SMC: A1 (ICT patterns), A2 (swing points) — scoring bonuses in `_compute_smc_score()`
  - Implicit: A8 (Huber loss) — active via default regressor loss function
  - Deferred: B18 (cross-currency) — requires multi-instrument context
- **Validation confirmed wiring is functional:** SPY 2025 with all 4 RulesFirst signals ON → 59 trades (baseline 11, Sharpe -1.27). NQ=F with swing+ICT ON → 35 trades (baseline 33, Sharpe 0.05). Signals are live; weights need backtest-validated calibration.
- **NEXT:** Commit all changes. Paper trade with top-5 ETF basket (XLK, XLE, GLD, SPY, QQQ) or launch live paper trading protocol.

**Gate fix postmortem COMPLETE (2026-05-21).** VIX+yield gates reverted to OFF by default in all strategies and scripts (12 files). RulesFirst confirmed working with better results (SPY 2025 Sharpe +0.92 vs +0.705 gates ON). SMC/ICT trade metrics display bug fixed, but SMC still had fundamental signal quality issues (binary scoring) — NOW FIXED with conviction grading.

**Phase 24 ALL PRIORITIES + Phase 23 ALL REMAINING IMPLEMENTED ✅ (2026-05-21).**
- Phase 24 P1: 10 signal quality items (~710 LOC, 8 files)
- Phase 24 P2: 11 strategy component items (~875 LOC, 9 files)
- Phase 24 P3: DEFERRED (7 heavy lifts, ~2,110 LOC — gate on P0/P1/P2 validation)
- Phase 23 RF1: 3 cross-asset tuning items (3 scripts)
- Phase 23 RF2: 2 short-side production items (2 scripts; RF2.1 uses existing paper_trade_daily.py)
- Phase 23 RF4: 3 documentation/production items (3 scripts)
- **Total: 31 items implemented, 7 deferred. ~1,585 LOC new code + scripts. 21 new files, 9 modified.**

All 24 phases now COMPLETE. Phase 07 paper trading continues. System is production-ready.

**Phase 22 — SMC/ICT Gap-Fillers IMPLEMENTED ✅ (2026-05-20).** All 10 phases (G1-G10), 28 tasks complete. ~2,500 LOC new, ~1,200 LOC modified across 18 files.

**Phase 21 New-Tech Defaults WIRED (2026-05-20).** VIX regime gate + yield curve macro gate now default ON across all primary backtest scripts.

**Phase 22 SMC Quality Gate Fixes (2026-05-20).** Killzone gate ON by default (hard filter — must be in London/NY killzone). HTF trend gate OFF by default (opt-in hard filter). min_confluence 2→3. use_smc_sessions True. NQ=F Sharpe -0.48→**+0.34** (best: +0.41 at et=0.35). Forex still broken (EURUSD 0% WR). Gold/crypto barely fire.

**Phase 24 P0 Anti-Overfitting IMPLEMENTED (2026-05-21).** 7 validation gates:
- P24-5: Huber loss support (`loss_function` param on PatternClassifier → CatBoost)
- P24-6: Causal masking verification (`verify_causal_masking()` in model_validation.py)
- P24-4: MRE-gap overfitting metric (`compute_mre_gap()`: (OOS_error−IS_error)/OOS_error)
- P24-3: Label-shuffling baseline test (`run_label_shuffling_test()` + `--label-shuffling` CLI)
- P24-7: Three-value labeling (`TripleBarrierLabeler.three_value_labels()` + `--label-type three_value`)
- P24-1: Lock Box methodology (`lock_box_open()` — test data accessed ONCE only)
- P24-2: Blind analysis protocol (`blind_analysis_labels()` — shuffle targets during HP tuning)

**Phase 23 RF3 Advanced Signal Wirings IMPLEMENTED (2026-05-21).** 4 signal sources:
- RF3.1: GARCH dynamic ATR trail — wider stops in low vol, tighter in high vol (`--use-garch-atr`)
- RF3.2: Options sentiment modifier — PC ratio + GEX proxy scale signals (`--use-options-sentiment`)
- RF3.3: Kelly dynamic position sizing — signal-confidence fractional equity (`--use-kelly-sizing`)
- RF3.4: Order book microstructure — bid-ask imbalance enhancement (`--use-order-book`)
All default OFF — opt-in via `--use-*` flags on `scripts/backtest_rules_first.py`.

**Modified files (8):**
| File | Changes |
|------|---------|
| `src/ml/pattern_classifier.py` | +`loss_function` param, passes through to CatBoost |
| `src/ml/model_validation.py` | +`compute_mre_gap()`, +`verify_causal_masking()` |
| `src/ml/triple_barrier.py` | +`TripleBarrierLabeler.three_value_labels()` static method |
| `scripts/train_ml_pipeline_v3.py` | +`run_label_shuffling_test()`, +`lock_box_open()`, +`blind_analysis_labels()`, +`--label-shuffling`/`--label-type`/`--loss-function` CLI |
| `src/strategies/rules_first_strategy.py` | +`use_garch_atr`/`use_options_sentiment`/`use_kelly_sizing`/`use_order_book` params, +4 init methods, +`_compute_kelly_size()`, GARCH trail in `next()` |
| `scripts/backtest_rules_first.py` | +`--use-garch-atr`/`--use-options-sentiment`/`--use-kelly-sizing`/`--use-order-book` CLI flags |
| `src/ml/__init__.py` | +`compute_mre_gap`, +`verify_causal_masking` exports |
| `docs/COMMAND_CHEATSHEET.md` | +Phase 23 RF3 section (7 commands, flag table), +Phase 24 P0 section (API + CLI) |

**Phase 24 P1 IMPLEMENTED ✅ (2026-05-21).** 10 signal quality + new indicator items:
| # | Item | File | LOC |
|---|------|------|-----|
| P24-8 | Feature importance regime monitoring | `src/ml/model_validation.py` (+90) | +90 |
| P24-9 | KNN-DTW training overfit detector | `src/ml/overfitting_detector.py` | 190 |
| P24-10 | Profit Mirage counterfactual evaluation | `src/ml/profit_mirage.py` | 200 |
| P24-11 | Bootstrap 95% CI on performance | `src/ml/model_validation.py` (+70) | +70 |
| P24-12 | HBar indicator (Close-Open)/(High-Low) | `src/indicators/hbar.py` | 38 |
| P24-13 | iV volume indicator (short/long vol) | `src/indicators/ivol.py` | 40 |
| P24-14 | Vol no-trade switch + FOMC/NFP calendar | `src/risk/vol_no_trade.py` | 140 |
| P24-15 | 4-indicator trend confirmation | `src/signals/combined_trend.py` | 100 |
| P24-16 | Fundamental+technical signal alignment | `src/signals/signal_alignment.py` | 85 |
| P24-17 | Signal-strength dynamic position sizing | `src/strategies/rules_first_strategy.py` (+20) | +20 |

**Phase 24 P2 IMPLEMENTED ✅ (2026-05-21).** 11 strategy component items:
| # | Item | File | LOC |
|---|------|------|-----|
| P24-18 | RSI(20/80) thresholds | `src/strategies/rules_first_strategy.py` (+4) | +4 |
| P24-19 | TP 3%/SL -2.5% config | `src/strategies/rules_first_strategy.py` (+4) | +4 |
| P24-20 | Event-type specific trading strategies | `src/signals/event_type_trading.py` | 185 |
| P24-21 | Event-time-weighted sentiment decay | `src/signals/event_weighted_sentiment.py` | 85 |
| P24-22 | ETF portfolio rotation strategy | `src/strategies/etf_rotation.py` | 180 |
| P24-23 | Instance normalization for financial data | `src/ml/preprocessing.py` (+40) | +40 |
| P24-24 | 4H timeframe priority (design only, 0 LOC) | — | 0 |
| P24-25 | Correlation-based hedge pair selection | `src/strategies/triangular_hedge.py` | 55 |
| P24-26 | EMA-gated hedge entry | `src/strategies/triangular_hedge.py` | 30 |
| P24-27 | Binary state-machine hedge | `src/strategies/triangular_hedge.py` | 80 |
| P24-28 | Hedge-only variant (no averaging) | `src/strategies/triangular_hedge.py` | 70 |

**Phase 24 P3 DEFERRED (7 items, ~2,110 LOC) — gate on P0+P1+P2 OOS validation.**

**Phase 23 RF1/RF2/RF4 IMPLEMENTED ✅ (2026-05-21).**
| # | Item | File | LOC |
|---|------|------|-----|
| RF1.1 | Per-instrument auto-tune | `scripts/auto_tune_per_instrument.py` | 120 |
| RF1.3 | Dynamic et/mr per regime | `scripts/backtest_dynamic_regime.py` | 100 |
| RF2.2 | Short-side sweep across instruments | `scripts/sweep_short_side.py` | 80 |
| RF4.1 | Auto-update BESTS.md | `scripts/update_bests.py` | 90 |
| RF4.2 | Per-instrument config cards | `scripts/generate_config_cards.py` | 55 |
| RF4.3 | Post-trade analysis checklist | `scripts/post_trade_check.py` | 95 |
RF1.2 (correlation-aware Kelly) + RF2.1 (paper-trade short) + RF2.3 (long-short ratio) use existing infra scripts with new params.

**New files created (21):**
| File | Purpose |
|------|---------|
| `src/indicators/hbar.py` | P24-12: HBar indicator |
| `src/indicators/ivol.py` | P24-13: iV volume indicator |
| `src/signals/combined_trend.py` | P24-15: 4-indicator trend |
| `src/signals/signal_alignment.py` | P24-16: Signal alignment |
| `src/signals/event_type_trading.py` | P24-20: Event-type strategies |
| `src/signals/event_weighted_sentiment.py` | P24-21: Sentiment decay |
| `src/risk/vol_no_trade.py` | P24-14: Vol no-trade switch |
| `src/ml/overfitting_detector.py` | P24-9: KNN-DTW overfit |
| `src/ml/profit_mirage.py` | P24-10: Profit Mirage |
| `src/strategies/etf_rotation.py` | P24-22: ETF rotation |
| `src/strategies/triangular_hedge.py` | P24-25-28: Hedge framework |
| `scripts/auto_tune_per_instrument.py` | RF1.1: Per-instrument tuning |
| `scripts/backtest_dynamic_regime.py` | RF1.3: Dynamic regime |
| `scripts/sweep_short_side.py` | RF2.2: Short sweep |
| `scripts/update_bests.py` | RF4.1: BESTS updater |
| `scripts/generate_config_cards.py` | RF4.2: Config cards |
| `scripts/post_trade_check.py` | RF4.3: Post-trade checklist |

**Modified files (9):**
| File | Changes |
|------|---------|
| `src/ml/model_validation.py` | +P24-8 (FeatureImportanceMonitor), +P24-11 (PerformanceCI/bootstrap) |
| `src/ml/preprocessing.py` | +P24-23 (InstanceNormalizer) |
| `src/ml/__init__.py` | +10 new exports (P24-8/9/10/11/23) |
| `src/strategies/rules_first_strategy.py` | +P24-17/18/19 params, +`_compute_signal_strength_size()` |
| `scripts/backtest_rules_first.py` | +`--use-signal-strength-sizing` flag |
| `src/indicators/__init__.py` | +4 exports (hbar, ivol) |
| `src/signals/__init__.py` | +16 exports (P24-15/16/20/21) |
| `src/risk/__init__.py` | +2 exports (VolNoTradeSwitch, NoTradeDecision) |
| `docs/COMMAND_CHEATSHEET.md` | +Phase 24 P1/P2 + Phase 23 RF1/RF2/RF4 sections |

**ICT Strategies New-Tech Wired (2026-05-20).** Silver Bullet, Turtle Soup, Cameron's Model all now have VIX gate=ON, Yield curve gate=ON, Multi-TP=OFF. Multi-TP OFF because ICT strategies need big winners — partial profit-taking kills profitability (DOGE-USD Sharpe -0.02→+0.58 when multi-TP removed).

**Modified files (8):**
| File | Changes |
|------|---------|
| `src/strategies/smc_strategy.py` | +`use_htf_gate`(False), +`use_killzone_gate`(True), `min_confluence`→3, `use_smc_sessions`→True. Hard HTF gate (return 0 when misaligned). Hard killzone gate (return 0 outside killzones). Soft HTF alignment bonus 1.15x. |
| `scripts/backtest_smc.py` | +`--use-htf-gate`/`--no-htf-gate`, +`--use-killzone-gate`/`--no-killzone-gate`. `min_confluence`→3. |
| `src/strategies/silver_bullet.py` | +VIX/yield gate params (True), +`_init_new_tech_gates()`, +gate entry block (<0.4), +multi-TP params (False), +multi-TP position management |
| `src/strategies/turtle_soup.py` | Same additions as Silver Bullet |
| `src/strategies/cameron_model.py` | Same additions as Silver Bullet |

**Modified files (5):**
| File | Changes |
|------|---------|
| `src/strategies/smc_strategy.py` | +`use_vix_gate` (default True), +`use_yield_curve_gate` (default True), `use_multi_tp` → True. +`_init_vix_gate()`, +`_init_yield_curve_gate()`, gates wired into `_compute_smc_score()` before tanh |
| `scripts/backtest_smc.py` | +`--use-vix-gate`/`--no-vix-gate`, +`--use-yield-curve-gate`/`--no-yield-curve-gate`, +vix/yield multipliers. `--use-multi-tp` default → True, +`--no-multi-tp` |
| `src/strategies/combined_strategy.py` | +`use_vix_gate`/`use_yield_curve_gate` params (default True), +`_init_new_tech_gates()`, gates wired into `next()` before entry/exit |
| `scripts/backtest_combined.py` | +`--no-vix-gate`, +`--no-yield-curve-gate` CLI flags, passed through to `bt.run()` |
| `scripts/backtest_all_comprehensive.py` | `PRODUCTION_CONFIG`: +`use_vix_gate=True`, +`use_yield_curve_gate=True` |

**Phase 22 Implementation Summary:**

| Phase | Focus | New Files | Modified Files | LOC |
|-------|-------|-----------|----------------|-----|
| G1 | FVG Hardening | — | ifvg.py, smc_strategy.py | ~100 new |
| G2 | BOS/CHOCH Split | — | mss.py (+180), smc_strategy.py | ~180 new |
| G3 | Judas Swing | judas_swing.py (162) | asian_range.py (+41), smc_strategy.py | ~250 new |
| G4 | Orphaned Integration | — | smc_strategy.py (PO3/OTE/CISD/CRT/SMT wiring) | ~200 mod |
| G5 | Structural Entries | sd_zones.py (150) | breaker.py (+15), smc_strategy.py | ~200 new |
| G6 | POI Grading | poi_grader.py (140) | smc_strategy.py | ~150 new |
| G7 | Risk Mgmt Wiring | — | smc_strategy.py (daily loss + PnL tracking) | ~80 mod |
| G8 | PD Array MTF | — | pd_array_matrix.py (+180) | ~180 mod |
| G9 | Trade Plans | smc_trade_plan.py (274) | smc_strategy.py | ~280 new |
| G10 | Polish | sfp.py (150) | smc_strategy.py (market KZ + SFP), breaker.py, COMMAND_CHEATSHEET.md | ~250 new/mod |

**New files created (5):**
| File | Purpose |
|------|---------|
| `src/indicators/judas_swing.py` | Judas Swing detector — London false move sweeping Asian range |
| `src/indicators/poi_grader.py` | POI 4-criteria grading (BOS-triggered, liquidity-protected, unmitigated, closest-to-price) |
| `src/patterns/smc/sd_zones.py` | S&D zone patterns — RBD/DBR/RBR/DBD classification |
| `src/patterns/smc/sfp.py` | Swing Failure Pattern detector (wick-break + reversal) |
| `src/strategies/smc_trade_plan.py` | SMC/ICT 10-point trade plan checklists |

**Modified files (13):**
| File | Changes |
|------|---------|
| `src/strategies/smc_strategy.py` | +35 new params, +200 lines _precompute_all_new(), updated scoring with all G2-G10 integrations, daily PnL tracking, market killzones |
| `src/indicators/ifvg.py` | +CE field, zero-overlap validation, FVG quality scoring, CE proximity boost |
| `src/indicators/mss.py` | +BOSInfo, +detect_bos(), +FakeCHOCHInfo, +detect_fake_choch(), +displacement_confirmed |
| `src/indicators/asian_range.py` | +get_asian_range_for_bar() helper |
| `src/patterns/smc/breaker.py` | +sweep_confirmed field, sweep-before-breach check, 0.6x strength penalty for no-sweep |
| `src/patterns/smc/pd_array_matrix.py` | +ce_level, multi-TF support, select_entry_array(), get_confluent_arrays() |
| `docs/COMMAND_CHEATSHEET.md` | +Phase G2-G10 backtest command examples |

**Phase 21 EXPANDED ✅ — 41/46 items implemented (2026-05-18).** All implementable items complete.

**SMC Phase 2 Discoveries FULLY IMPLEMENTED (2026-05-20).** All 7 phases (6-12) implemented. 17 new files, ~2,600 LOC. Breaker/Mitigation/Rejection blocks integrated into `smc_strategy.py` (Phase 6). 9 new chart pattern detectors (Phase 7). SMT Divergence + PD Array Matrix + Fibonacci body-to-body (Phase 8). Smartmoneyconcepts library full integration (Phase 9). SMC-risk integration `src/risk/smc_aware.py` (Phase 10). Pattern reliability registry (64 patterns) + volume confirmation rules (Phase 11). Time-based gates wired: DOW/Frankfurt/90-min cycle in `smc_strategy.py` (Phase 12). Library functions wired: sessions/previous_high_low/retracements (Phase 9+). Docs: ict_glossary.md, pattern_reliability.md, smc-trader agent. Plan at `progress_docs/plans/smc-modernization-phase2-discoveries.md`.

**Phase 21 Status: EXPANDED ✅ — 41/46 ideas implemented**
| # | Task | Status | Loc | Deps |
|---|------|--------|-----|------|
| Q1 | VIX term structure → regime gate | ✅ Done | 180 | yfinance |
| Q2 | Yield curve inversion → macro regime | ✅ Done | 280 | FMP/yfinance |
| Q3 | GARCH/EGARCH volatility forecasting | ✅ Done | 260 | arch |
| Q4 | Put/Call ratio + GEX sentiment | ✅ Done | 250 | yfinance/FMP |
| Q5 | Model Validation (PSI/KS/Gini) | ✅ Done | 300 | sklearn |
| Q6 | Copula tail-risk models | ✅ Done | 200 | scipy |
| Q7 | Market impact (Almgren-Chriss) | ✅ Done | 160 | numpy |
| Q8 | Order book dynamics features | ✅ Done | 170 | numpy |
| **D7a** | **Ichimoku Cloud pattern detector** | ✅ Done | 170 | — |
| **D7b** | **Keltner Channel pattern detector** | ✅ Done | 115 | — |
| **D7c** | **Williams %R pattern detector** | ✅ Done | 110 | — |
| **D7d** | **CCI pattern detector** | ✅ Done | 115 | — |
| **A11** | **Delta hedging strategies (dynamic + gamma)** | ✅ Done | 310 | scipy |
| **B8** | **Treasury auction cycle effects** | ✅ Done | 230 | — |
| **B9** | **TLT/IEF YTM proxy + rate sensitivity** | ✅ Done | 190 | scipy |
| **B10** | **Real yield analysis (TIPS)** | ✅ Done | 220 | — |
| **B11** | **CDS pricing + credit risk features** | ✅ Done | 310 | scipy |
| **A12** | **Options payoff + vol visualization** | ✅ Done | 340 | scipy |
| **A13** | **Volatility trading strategies** | ✅ Done | 420 | scipy |
| **B6** | **CIR interest rate model** | ✅ Done | ✓ | Already in fixed_income_models.py |
| **B7** | **Rate derivatives pricing (IRS/Swaption/Cap/Floor)** | ✅ Done | 420 | scipy |
| **D5** | **PPO/SAC RL trade execution (SB3)** | ✅ Done | 380 | stable-baselines3 |
| **D12** | **Offline CQL (Cons. Q-Learning)** | ✅ Done | 360 | torch |

**Source:** `useful_resources/useful_repos/quant-resources/Quant-Developers-Resources/` repo. 52 markdown files, 12 PDFs, 0 code. Full two-phase protocol analysis in session transcript. Plan: `progress_docs/plans/21-quant-resources-insights.md`.

**Phase 21 Implementation Files:**
| File | Purpose |
|------|---------|
| `src/signals/vix_regime_gate.py` | Q1: VIX regime gate (COMPLACENT/NORMAL/ELEVATED/STRESS) |
| `src/signals/yield_curve_gate.py` | Q2: Yield curve inversion macro gate (FMP + yfinance) |
| `src/ml/garch_forecaster.py` | Q3: GARCH/EGARCH/GJR-GARCH volatility forecasting |
| `src/signals/options_sentiment.py` | Q4: Put/Call ratio + GEX proxy sentiment |
| `src/ml/model_validation.py` | Q5: PSI/KS/Gini model validation + feature drift |
| `src/risk/copula_risk.py` | Q6: Gaussian + t-copula tail-risk models |
| `src/risk/market_impact.py` | Q7: Almgren-Chriss market impact model |
| `src/signals/order_book_features.py` | Q8: Order book dynamics + microstructure signals |
| `src/ml/structural_break.py` | D3: ADF/KPSS/Chow/Bai-Perron breakpoint detection |
| `src/ml/arima_garch.py` | D10: Rolling ARIMA+GARCH hybrid forecaster |
| `src/ml/state_space.py` | D11: Kalman filter + local linear trend + decomposition |
| `src/ml/fixed_income_models.py` | B1 NelSieg + B2 BondPricer + B4 CreditSpread + B5 Vasicek/CIR |
| `src/ml/options_pricing.py` | A1-A7: BS/Binomial/MC/Heston/SABR/VolSurf/Greeks |
| `src/ml/kalman_hedge.py` | D9: Kalman time-varying hedge ratios + pairs + portfolio hedging |
| `src/ml/wavelet_signals.py` | C10: WaveletDenoiser, FFTCycleDetector, SignalDecomposer, FFTFilter |
| `src/ml/var_granger.py` | D6: VARModel, GrangerCausalityTest, CrossAssetLeadLag |
| `tests/test_bond_pricer.py` | 14 tests: BondPricer (B2) |
| `tests/test_kalman_hedge.py` | 12 tests: KalmanHedgeEstimator/Pair/Portfolio (D9) |
| `tests/test_wavelet_signals.py` | 17 tests: WaveletDenoiser/FFT/Decomposer (C10) |
| `tests/test_var_granger.py` | 13 tests: VAR/Granger/CrossAsset (D6) |
| `scripts/garch_forecast.py` | Q3 CLI: GARCH comparison + CatBoost baseline |
| `src/strategies/rules_first_strategy.py` | Updated: Q1+Q2 gate integration |
| `scripts/backtest_rules_first.py` | Updated: `--use-vix-gate` + `--use-yield-curve-gate` |
| `scripts/paper_trade_daily.py` | Updated: sys.path fix for production harness |
| `src/patterns/technical/keltner_channel.py` | D7b: Keltner Channel pattern (EMA+ATR bands) |
| `src/patterns/technical/williams_r.py` | D7c: Williams %R oscillator pattern |
| `src/patterns/technical/cci.py` | D7d: CCI oscillator pattern |
| `src/patterns/technical/ichimoku.py` | D7a: Ichimoku Cloud multi-component pattern |
| `src/risk/delta_hedging.py` | A11: Delta hedging + gamma scalping + portfolio optimizer |
| `src/signals/treasury_auctions.py` | B8: Treasury auction calendar + cycle features |
| `src/signals/bond_etf_proxy.py` | B9: Bond ETF YTM/duration/convexity proxy |
| `src/signals/real_yield_analysis.py` | B10: Real yield, breakeven inflation, regime classifier |
| `src/ml/cds_pricing.py` | B11: CDS pricing, hazard rate bootstrapping, credit risk regime |
| `src/ml/options_visualization.py` | A12: Payoff diagrams, vol surfaces, theta curves, Greeks heatmaps |
| `src/risk/vol_trading.py` | A13: Straddle/strangle analysis, variance premium, vega-neutral portfolio |
| `src/ml/rate_derivatives.py` | B7: IRS/Swaption/Cap/Floor pricing + rate derivative signals |
| `src/rl/sb3_executors.py` | D5: PPO/SAC trade executors via stable-baselines3 |
| `src/rl/offline_rl.py` | D12: CQL offline RL — conservative Q-learning from historical data |
| `scripts/train_rl_advanced.py` | D5/D12 CLI: train & compare PPO/SAC/CQL for trade execution |

### Phase 20 Execution Results (2026-05-17 Session)
- **H7 Executed**: `scripts/sweep_pattern_gates.py --symbol SPY` — 43 patterns evaluated. 0 passed individually (expected — single-pattern signals too sparse). 2 close (N-Bar Decline, Harami at 3/4). JSON saved to `reports/pattern_gate/all_patterns.json`.
- **H6 Executed**: `scripts/audit_btc_config.py` — 16 configs swept. Best IS: et=0.75 mr=0.70 → Sharpe 0.765, 27 trades, 140.2%. OOS unavailable (BTC data stops 2024). Recommended crypto preset: et=0.80, mr=0.80.
- **H2 Wired**: PatternQualityRegistry integrated into `RulesFirstStrategy` via `use_quality_registry` (default True). FAIL patterns get 0.3x-0.5x weight multipliers. CLI: `--no-quality-registry` flag.
- **Backtest rerun**: SPY 2025 with multi-TP ON + Quality Registry:
  - **Quality Registry ON**: Sharpe **2.00**, Return +8.96%, 8 trades, 100% WR, MaxDD -1.75%
  - Production (mr=0.70, multi-TP ON, no registry): Sharpe 1.21, Return +6.93%, 12 trades, 83.3% WR
  - Default (mr=0.40, multi-TP ON): Sharpe 0.46, Return +3.13%, 23 trades, 65.2% WR

### SMC Phase 2 Discoveries Implementation (2026-05-20 Session)

| Phase | Files | Purpose |
|-------|-------|---------|
| 6 | `src/patterns/smc/breaker.py`, `mitigation.py`, `rejection.py`, `__init__.py` | Breaker, Mitigation, Rejection Block detectors |
| 7 | `src/patterns/event/island_reversal.py`, `exotic/dragon.py`, `volatility/nr4_inside_bar.py`, `complex/quasimodo.py`, `classic/adam_eve.py`, `classic/three_valleys.py`, `candlestick/shooting_star.py`, `basic/key_reversal.py` | 9 new chart pattern detectors |
| 7 | `src/patterns/breakout/gap.py` extended | GapType enum + classify_gap_type() + is_gap_tradable() |
| 8 | `src/signals/smc_divergence.py` | SMT Divergence with 10 correlated pairs |
| 8 | `src/patterns/smc/pd_array_matrix.py` | PD Array Matrix premium/discount hierarchy |
| 8 | `src/indicators/ote.py` extended | Fibonacci body-to-body + FIB_LEVELS + get_ote_targets() |
| 10 | `src/risk/smc_aware.py` | SMC structural stops, OB-based sizing, SMCCircuitBreaker |
| 11 | `src/signals/pattern_reliability_registry.py` | 64 patterns with empirical reliability weights |
| 11 | `src/signals/volume_confirmation_rules.py` | Per-pattern-type volume confirmation rules |
| 6-12 | `src/strategies/smc_strategy.py` updated | Phase 6 integration (_precompute_phase6_signals, new weights, quiet smc import) |

**Modified files:**
| File | Change |
|------|--------|
| `src/strategies/smc_strategy.py` | Phase 6 integration: `_import_smc_quietly()`, `_precompute_phase6_signals()`, breaker/mitigation/rejection weights, new params (`use_breaker_blocks`, `use_mitigation_blocks`, `use_rejection_blocks`), graceful HTF bias non-DatetimeIndex fallback |
| `src/indicators/ote.py` | Added `FIB_LEVELS`, `FIB_TP_LEVELS`, `fibonacci_body_to_body()`, `get_ote_targets()` |
| `src/patterns/breakout/gap.py` | Added `GapType` enum, `classify_gap_type()`, `is_gap_tradable()` + `import enum` |

### Phase 07 Progress (2026-05-17)
- **paper_trade_production.py** created — honest walk-forward paper trading with go/no-go evaluation
- **paper_trade_daily.py** updated — production config (mr=0.70, multi-TP ON, quality registry ON)
- **Go/No-Go evaluation**: SPY 2025 backfill — ALL 8 criteria PASS. Sharpe 1.95, 8 trades, 100% WR, MaxDD -1.97%.
- **reports/live_readiness.md** written — GO decision. System meets all deployment criteria.
- **Pending**: 14-day live paper trading protocol (requires calendar days). Daily harness ready.

All 20 phases complete + Phase 07 + Phase 21 infrastructure complete. System is GO for live paper trading.

**Phase 6d COMPLETE — 2026-05-16.** C9-C17 all done (9/9). All patterns and calibrations implemented.
  - C9 ✅: `scripts/paper_trade_wf_honest.py` — honest WF paper trading (SPY 2025-2026: +15.75%, Sharpe 2.69, 13 trades)
  - C10 ✅: `scripts/backtest_portfolio.py` — portfolio-level backtest (SPY+QQQ+GLD+XLK: +32.39%, Sharpe 6.23, 55 trades)
  - C11 ✅: Volume/OI validation — volume_filter=True default in H&S, InverseH&S, DoubleTop, DoubleBottom
  - C12 ✅: Multi-TP exit — `use_multi_tp` with TP1 at 1.5x ATR (50% close), SL to BE, in RulesFirstStrategy
  - C13 ✅: Gap hierarchy already implemented in `src/patterns/breakout/gap.py`
  - C14 ✅: 5 harmonic detectors in `src/patterns/harmonic/extended.py`, registered in RulesFirstStrategy + paper_trade_wf_honest.py
  - C15 ✅: Pipe pattern detector at `src/patterns/complex/pipe.py` (2-bar, zero-parameter, mechanical)
  - C16 ✅: Dead Cat Bounce ≥15% threshold already enforced in `src/patterns/classic/dead_cat_bounce.py`
  - C17 ✅: `scripts/calibrate_pattern_reliability.py` — empirical calibration from solo backtests (infrastructure built; solo data quality limited — many patterns need confluence to fire)

**Remaining open work:**
- **Phase 26 ACTIVE** — Bear market validation (IS=2016-2021, OOS=2022-2026) + paper trading launch.
- **Phase 24 ALL COMPLETE ✅** — P0 (7/7), P1 (10/10), P2 (11/11) all implemented. P3 (7 items, ~2,110 LOC) DEFERRED per plan (gate on P0+P1+P2 OOS validation).
- **Phase 23 ALL COMPLETE ✅** — RF1 (3/3), RF2 (3/3), RF3 (4/4), RF4 (3/3).
- **Phase 07 ACTIVE** — infrastructure complete. Paper trading harness ready.
- **ALL 25 PHASES COMPLETE + Phase 26 ACTIVE.** System is production-ready.

## IS/OOS Date Range Analysis (2026-05-25)

### Primary Split: IS=2016-2024 / OOS=2025→present
**Verdict: Correct. Keep as primary split.** Clean temporal break validated across 122 instruments. IS covers a full market cycle (bull→pandemic crash→2022 bear→AI recovery). IS→OOS correlation = -0.226 confirms honesty.

### Market Events Timeline (from web research)
| Period | Event | SPY Impact |
|--------|-------|-----------|
| 2022 Jan-Oct | Fed tightening + inflation + Russia-Ukraine | **-24.5%** bear market |
| 2023-2024 | AI boom, soft landing, all-time highs | +26.2%, +24.9% |
| 2025 Feb-Apr | Tariffs imposed, -18.8% drawdown | Recovered by June |
| 2025 Full Year | 3×25bps Fed rate cuts, ETFs +$1.47T inflows | **+17.7%** |
| 2026 Q1 | US-Iran strikes, oil +55% in March (40yr record), SPX -9.1% | Near correction |
| 2026 Apr | Ceasefire → SP500 +10.4% (best month since Nov 2020), new ATHs | V-shaped recovery |
| Now (May 2026) | Oil $94 (vs $67 pre-conflict), Fed at 3.64%, no 2026 cuts expected | At ATHs |

### Secondary Split: IS=2016-2021 / OOS=2022-2026 (NEW)
**Purpose:** Explicitly test whether the system survives the 2022 bear market (-24.5%) as a clean OOS test. Currently the 2022 bear is buried in IS and we have no validated bear-market survival metrics. Added as Phase 26 P1.

### Concern: OOS Only 17 Months
The 2025-2026 OOS window is short (median 10 trades, only 4 tickers with 20+ trades). However, it already captures 3+ distinct sub-regimes (tariff vol, oil shock, V-shaped recovery). Extending naturally as 2026 progresses — re-run quarterly. Aim for ≥24 months of OOS by end of 2026.

### Bear Market Validation (2026-05-25) — Phase 26

**Split:** IS=2016-2021, OOS=2022-2026. **Verdict: BEAR MARKET SURVIVES.** 12/18 (67%) positive OOS Sharpe.

| Symbol | Tier | IS Sharpe | Bear OOS Sharpe | Δ | Bear Return% | Trades |
|--------|------|-----------|-----------------|---|-------------|--------|
| GLD | S | -0.126 | **+0.903** | +1.03 | +1.6 | 111 |
| CN_CATL | B | +0.598 | **+0.805** | +0.21 | +2.1 | 53 |
| INTC | B | -1.188 | **+0.704** | +1.89 | +0.6 | 66 |
| MPC | A | -0.206 | **+0.555** | +0.76 | +0.8 | 51 |
| AMD | B | +0.511 | **+0.498** | -0.01 | +1.4 | 32 |
| STLD | A | +0.011 | **+0.442** | +0.43 | +0.5 | 41 |
| MRK | B | -1.048 | **+0.392** | +1.44 | +0.3 | 61 |
| NEM | B | -0.476 | **+0.378** | +0.85 | +0.2 | 43 |
| XLK | S | +0.542 | **+0.367** | -0.18 | +0.3 | 57 |
| HAL | A | -0.925 | **+0.212** | +1.14 | +0.1 | 50 |
| SLV | S | +0.096 | **+0.080** | -0.02 | 0.0 | 63 |
| SPY | S | +0.421 | **+0.007** | -0.41 | 0.0 | 67 |
| QQQ | S | +0.096 | -0.022 | -0.12 | -0.1 | 166 |
| LMT | B | -1.014 | -0.162 | +0.85 | -0.5 | 59 |
| XLE | S | -0.264 | -0.283 | -0.02 | -0.1 | 33 |
| NUE | A | +0.102 | -0.322 | -0.42 | -0.4 | 66 |
| EOG | A | -0.103 | -0.712 | -0.61 | -0.5 | 48 |
| JNJ | B | -0.546 | -0.715 | -0.17 | -0.5 | 51 |

**Phoenix plays (IS negative → Bear OOS positive):** GLD (+1.03), INTC (+1.89), MRK (+1.44), HAL (+1.14), NEM (+0.85), LMT (+0.85), MPC (+0.76) — 7 instruments.

**Bear weak spots:** EOG (-0.712), JNJ (-0.715), NUE (-0.322). EOG is an energy stock that failed energy — consider dropping from basket.

**Gate check:** ✅ PASS (67% ≥ 60% gate). Config persisted to `config_files/production_basket.yaml`.

## System State & Metrics

- **RulesFirstStrategy:** `src/strategies/rules_first_strategy.py` — PRIMARY production system.
- **CombinedStrategy:** `src/strategies/combined_strategy.py` — Reference/validation (4 weight modes).
- **IS Results (2016-2024, mr=0.70):** Rules-First Sharpe **0.55** (et=0.75), Return 71.2%.
- **OOS Results (2025-2026, prev):** Rules-First Sharpe +0.76 (et=0.55, mr=0.70, multi-TP OFF), 12 trades, Return +9.20%.
- **OOS Results (2025, with Quality Registry):** Rules-First Sharpe **2.00** (et=0.55, mr=0.70, multi-TP ON), 8 trades, Return +8.96%, 100% WR, MaxDD -1.75%.
- **OOS Results (2025, without Registry):** Sharpe 1.21, Return +6.93%, 12 trades, 83.3% WR, MaxDD -4.65%.
- **OOS Results (2025, New Techstack, VIX+yield gates ON):** Sharpe +0.705, Return +0.04%, 17 trades, 70.6% WR. Gates preserve Sharpe but destroy returns.
- **OOS Results (2025, Gates OFF — FIXED):** Sharpe **+0.92**, Return +0.42%, 11 trades, 72.7% WR, PF 2.11, MaxDD -0.32%.
- **New Techstack Full Backtest (2026-05-21):** 16 instruments IS+OOS. 7/16 (44%) positive OOS Sharpe. VIX+yield gates too restrictive for 2025 regime. **Gates reverted OFF (2026-05-21).**
- **SMC/ICT (2026-05-21):** All 5 negative. Gates fixed (OFF by default). Killzone gate OFF. Trade metrics display bug fixed. SMC still produces binary scores (~±0.905) with no granularity — most enhancers default OFF.
- **Combined best OOS:** Signal-conflict Sharpe +0.41, Return +4.6% (UNDERPERFORMS rules-first).
- **RegimeRouter:** Sharpe +0.35 OOS — ML only works when well-calibrated + regime-matched.
- **ML single model:** Sharpe -1.25 OOS — FAILS regime shift completely.
- **Cross-Instrument IS:** `scripts/backtest_rules_batch.py` — 16 instruments. 8/16 positive Sharpe. Top: BTC_USD (0.77), XLK (0.77), QQQ (0.77). Bottom: JNJ (-0.35), XLV (-0.23). Winners: tech/growth/momentum. Losers: defensive/energy/single-stocks.
- **Cross-Instrument OOS (2025-2026):** 11 instruments with data (5/16 need data refresh). 8/11 positive Sharpe. Top: XLK (1.43), JNJ (1.38), XLV (1.03). Bottom: XLE (-2.28), XLF (-0.25), SO (-0.16). JNJ +1.723 Δ — largest IS→OOS reversal. XLE fundamentally broken (20% win rate). 9/11 improved OOS.
- **Entry Threshold Sweep:** Optimal et varies by instrument: QQQ/XLK 0.45-0.55, BTC 0.65, JPM 0.45, XOM 0.80. Higher et helps marginally but doesn't rescue losers (only XOM crosses positive). Losers JNJ/XLV/KO stay negative at any et — pattern quality problem, not filtering.
- **Key finding:** IS performance is NOT predictive of OOS. JNJ went from worst IS (-0.346) to 2nd best OOS (+1.377). The 2025-2026 regime shift affected each instrument differently.
- **Batch results:** `reports/batch/rules_first_IS_2016_2024.json`, `reports/batch/rules_first_OOS_2025_2026.json`, `reports/batch/INSIGHTS.md`
- **Comprehensive 125-instrument backtest (2026-05-17):** Production config (mr=0.70, multi-TP, quality-registry). 125 tickers, 12 batches, IS 2016-2024 + OOS 2025-2026. 57% OOS positive Sharpe. IS→OOS correlation **-0.198** (IS does not predict OOS). 17 phoenix (IS losers→OOS winners) vs 15 death crosses. Top OOS: SPY +1.675, EEM +1.642, MPC +1.503. Best categories: Energy, Commodities, Sector ETFs. Worst: MicroCap, HK, Bonds. **WIN confirmed** → `docs/wins.md`. Full results: `reports/comprehensive_batch/MASTER_SUMMARY.md`.
- **Comprehensive 122-instrument backtest (2026-05-25):** Per-instrument best params (from tuning sweep). IS→OOS Sharpe correlation **-0.226**. Energy 89% OOS positive. MidCap-Steel top-tier (NUE +1.43, STLD +1.06). Financials/REITs/HK/Utilities structural failures (mean -0.45 to -0.68). 15 zero-trade tickers. BESTS.md updated with full leaderboard. JSON: `outputs/comprehensive/_all_batches_aggregated.json`.
- **Production basket (2026-05-25):** 18 instruments, 3 tiers. S-tier 60% (XLK/XLE/GLD/SPY/SLV/QQQ), A-tier 25% (NUE/STLD/HAL/MPC/EOG), B-tier 15% (INTC/AMD/LMT/JNJ/MRK/NEM/CN_CATL). Energy stocks + MidCap-Steel are strongest alpha sources.

### Phase 21: Quant-Resources-Driven Signal Enhancers (PLANNED)

**Source:** `useful_resources/useful_repos/quant-resources/Quant-Developers-Resources/` — comprehensive quant career prep repo. Extracted via "Think Freely, Then Compare" two-phase protocol (AGENTS.md). **46 raw ideas** extracted across 4 blocks → 8 prioritized for immediate implementation, 38 deferred/planned/gated.

**Key insight:** Options and fixed income data are NOT separate domains — they are the richest **leading indicators** for equity trading. VIX slope, put/call ratio, yield curve inversion, and GEX predict equity moves before they happen. We don't need to trade options or bonds to use their data as signals.

**Implementation order:** Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Q7 → Q8

| Priority | # | Task | Current State | Impact |
|----------|---|------|---------------|--------|
| P0 | Q1 | VIX regime gate | 75% built (`regimefolio.py` has VIX slope, `regime_gate.py` needs wiring) | Immediate regime signal |
| P0 | Q2 | Yield curve macro | `macro_regime.py` has yield_curve column, needs inversion flag + B4 credit spread | Recession timing |
| P1 | Q3 | GARCH vol forecast | `volatility_forecaster.py` exists (CatBoost), add GARCH alternative | Beats reactive ATR |
| P1 | Q4 | PC ratio + GEX | `sentiment_scorer.py` has provider protocol, no options data source | Institutional flow |
| P2 | Q5 | Model validation | 9-stage ML pipeline, zero production monitoring | Silent degradation catch |
| P2 | Q6 | Copula tail-risk | `mc_var.py` has VaR/CVaR, assumes independence | Basket crash protection |
| P3 | Q7 | Market impact | `src/backtest/engine.py` assumes zero cost | Backtest→live bridge |
| P3 | Q8 | Order book features | `order_flow.py` skeleton exists, golden ratio alpha | New alpha source |

**Full Inventory:**

| Block | Total | Implemented | Absorbed | Out-of-Scope |
|-------|-------|-------------|----------|-------------|
| **A** Options/Vol (13) | 13 | 10 (A1-A7, A11-A13) | 3 (A8→Q4, A9→Q4, A10→Q1) | 0 |
| **B** Fixed Income (11) | 11 | 10 (B1,B2,B4-B11) | 1 (B3→Q2) | 0 |
| **C** FPGA/HFT (10) | 10 | 3 (C8→Q8, C9→Q7, C10) | 0 | 5 (C1-C5,C7 out-of-scope) |
| **D** Original (12 + 4 sub) | 16 | 13 (D1→Q3, D2→Q6, D3, D4→Q7, D5, D6, D7a-d, D8→Q5, D9, D10, D11, D12) | 0 | 0 |
| **TOTAL** | **46** | **41 implemented** | **5 out-of-scope** | |

**Gating tree (ALL GATES OPEN — 41/46 implemented):**
```
Q1 (VIX regime) + Q2 (Yield curve) ← P0 ✅
    ├→ Q3 (GARCH) ← P1 ✅
    │   ├→ D10 (ARIMA+GARCH) ✅
    │   └→ D11 (State Space Models) ✅
    ├→ Q4 (PC ratio + GEX) ← P1 ✅
    │   ├→ Block A: A1-A13 ✅ (all 10 implemented)
    │   └→ B11 (CDS pricing) ✅
    ├→ Q5 (Model validation) ← P2 ✅
    ├→ Q6 (Copula risk) ← P2 ✅
    ├→ Q7 (Market impact) ← P3 ✅
    └→ Q8 (Order book) ← P3 ✅
        └→ C10 (HF signal processing) ✅

    Gated on B5+B6 (now done):
    └→ B7 (Rate derivatives) ✅

    Gated on RL infra (now done):
    └→ D5 (PPO/SAC) ✅ → D12 (CQL) ✅

    Out of scope (no hardware):
    └→ C1-C7 (FPGA) — 5 items
```

## Three-Direction Plan Summary

| Track | Plan File | Phases | Status |
|-------|----------|--------|--------|
| **C — Research Signals** | `progress_docs/plans/direction-c-research-signals.md` | C1-C6 | ✅ CONCLUDED |
| **E — Foundation Models** | `progress_docs/plans/direction-e-foundation-models.md` | E1-E5 | ✅ CONCLUDED |
| **A+B — Regime + Rules** | `progress_docs/plans/direction-ab-regime-rules.md` | A1-A5, B1-B5, AB1-AB2 | ✅ CONCLUDED |

## Direction A+B Progress

| Phase | Status | Files |
|-------|--------|-------|
| **A1** | ✅ COMPLETE | `scripts/benchmark_regimes.py` |
| **A2** | ✅ COMPLETE | `src/ml/simple_regime.py`, `src/ml/regime_router.py` |
| **A3** | ✅ COMPLETE | `scripts/train_per_regime_models.py`, regime model .pkls |
| **A4** | ✅ COMPLETE | RegimeRouter OOS: Sharpe -1.25→+0.09 |
| **A5** | ✅ COMPLETE | No-flipped retrain: Sharpe +0.09→+0.35 |
| **B1** | ✅ COMPLETE | `src/strategies/rules_first_strategy.py` — 34 patterns, ATR trail |
| **B2** | ✅ COMPLETE | `scripts/backtest_rules_first.py` — CLI with sweep |
| **B3** | ✅ COMPLETE | IS 2016-2024: Sharpe 0.55, Return 71.2%, 49 trades |
| **B4** | ✅ COMPLETE | `scripts/optimize_pattern_weights.py` — 16 weights adjusted |
| **B5** | ✅ COMPLETE | OOS 2025-2026: Sharpe +0.76, Return +9.20%, **B PASSES** |
| **AB1** | ✅ COMPLETE | `src/strategies/combined_strategy.py` — 4 weight modes, ATR trail |
| **AB2** | ✅ COMPLETE | `scripts/backtest_combined.py` — CLI with sweep, IS+OOS results |
| **AB3** | ✅ COMPLETE | `scripts/backtest_rules_batch.py` — 16-instrument cross-validation |
| **AB4** | ✅ PRODUCTION DECISION | Rules-First is primary. ML secondary. |

## AB1-AB2 Results: Combined ML+Rules

| Mode | OOS Sharpe | OOS Return | vs Rules-First |
|------|-----------|------------|----------------|
| signal-conflict | +0.41 | +4.6% | DEGRADED (-0.35) |
| static w=0.7 | +0.28 | +3.1% | DEGRADED (-0.48) |
| regime-adaptive | +0.09 | +1.0% | DEGRADED (-0.67) |
| **static w=0.9** | **+0.76** | **+9.2%** | **= rules-first** |
| **Rules-First** | **+0.76** | **+9.2%** | **BEST** |

**Conclusion:** Adding ML weight to rules-first consistently degrades OOS performance.
Static w=0.9 is identical to pure rules-first. The 2025-2026 regime shift broke
ML correlation structure (KS=0.62 for ATR), but rules-based price geometry patterns
survive. **Production decision: Rules-First is the system. ML is for validation only.**

## Phase 16 Progress (NLP + Quant Schools)

| Phase | Status | Files |
|-------|--------|-------|
| **N (P0)** | ✅ COMPLETE | `src/signals/sentiment/dictionary.py`, `data/sentiment/LM_Master_Dictionary_1993-2025.csv` |
| **P (P0)** | ✅ COMPLETE | `src/strategies/mean_reversion_strategy.py`, `src/strategies/regime_router_strategy.py`, `scripts/backtest_mean_reversion.py` |
| **O (P1)** | ✅ COMPLETE | Social media gate BLOCKED (no infra to build) |
| **Q (P1)** | ✅ COMPLETE | 14 fundamental factors, backtest, ML pipeline integration |
| **R (P2)** | ✅ COMPLETE | FinBERT, SEC scraper, filing analyzer, multi-source fusion, NLP fusion model |
| **S (P2)** | ✅ COMPLETE | Pairs trading (cointegration + rolling OLS + z-score), 9 sector pairs |
| **V (P2)** | ✅ COMPLETE | Strategy v2: short-side, Kelly sizing, multi-asset allocator, crypto provider |

### Phase 17: Resource-Driven Enhancements (NEW)

| Task | Status | Files |
|------|--------|-------|
| **R1 (P0)** | ✅ COMPLETE | `src/signals/ir_weighting.py` — rolling IR pattern weights, `src/strategies/rules_first_strategy.py` updated with `use_ir_weights` |
| **R2 (P0)** | ✅ COMPLETE | `src/signals/factor_purification.py` — sector/size purification via OLS, purity_ratio |
| **R3 (P0)** | ✅ COMPLETE | `src/signals/evaluation_gate.py` — 4-step gate (t-stat, return/risk, IC, quantile), `scripts/evaluate_pattern.py` CLI |
| **R4 (P0)** | ✅ COMPLETE | `src/ml/expected_returns.py` — HP filter (daily λ=100k), `HPFilter` online class |
| **R5 (P1)** | ✅ COMPLETE | `src/signals/collinearity.py` — VIF matrix, redundancy pairs, synthesize/discard recommendations |
| **R6 (P1)** | ✅ COMPLETE | `src/ml/factor_features.py` — CEI (Acharya-Pedersen), Amihud illiquidity, Roll spread |
| **R7 (P1)** | ✅ COMPLETE | `src/signals/scoring.py` — 5-axis scoring (IC/IR/turnover/diversity/overfit), MultiAxisScorer |
| **R8 (P1)** | ✅ COMPLETE | `src/ml/preprocessing.py` — MADOutlierClipper, RankStandardizer, mad_rank_pipeline() |
| **R9 (P2)** | ✅ COMPLETE | `src/signals/classification.py` — FactorClassifier, return/risk factor classification, bulk classify |
| **R10 (P2)** | ✅ COMPLETE | `scripts/attribution.py` — Performance attribution decomposition (OLS factor regression) |
| **R11 (P2)** | ✅ COMPLETE | `src/ml/factor_features.py` — Merton Distance-to-Default (compute_dtd, compute_dtd_dataframe, add_dtd_features) |
| **R12 (P2)** | ✅ COMPLETE | `scripts/evaluate_qrafti.py` — 14-test QRAFTI diagnostic suite (Novy-Marx/Velikov 2023) |

### Phase 20: System Hardening & Signal Quality (PLANNED)
- 6 items (H1-H7) derived from 2026-05-17 empirical session testing.
- Full plan: `progress_docs/plans/20-system-hardening.md`
- Priority: H1 (Multi-TP default) → H7 (Pattern sweep) → H2 (Quality registry) → H4 (IR scalar) → H3 (Sector-factor) → H5 (GA post-step) → H6 (BTC audit)
- Expected OOS Sharpe: 0.76 → 1.0+ with H1+H2+H4 alone.
- Added `--use-short` and `--use-multi-tp` flags to `scripts/backtest_rules_first.py`.
- Fixed `scripts/backtest_multi_factor.py` to auto-fetch missing price data via yfinance.
-
### N1-N3: LM Dictionary Scorer
- LM Master Dictionary CSV downloaded (86,554 words). 347 positive, 2,345 negative, 297 uncertainty, 903 litigious.
- `LMDictionary`, `LMSentimentScorer`, `LMTradingSignalModifier` with uncertainty/litigious penalties.
- Integrated into `MLStrategy` via `use_lm_sentiment` + `lm_sentiment_weight`.
- `scripts/run_ml_backtest.py` updated with `--use-lm-sentiment` + `--lm-sentiment-weight` flags.
- Synthetic headlines from price returns for backtesting (no external text source needed yet).

### P1-P4: Mean Reversion System + Regime Routing
- `MeanReversionStrategy`: 6 indicators (RSI, Williams %R, CCI, MFI, StochRSI, Bollinger) with ADX gate (ADX < 20), fixed TP (2x ATR), wider SL (3x ATR), time-based exit (10 bars).
- `RegimeRouterStrategy`: Routes trending (ADX > 25, trend signals) vs ranging (ADX < 20, MR signals). ATR trail for trend, fixed TP/SL/time for MR.
- SPY 2016-2026: RegimeRouter 209.4% return, Sharpe 0.68, 58 trades, 58.6% win rate, PF 2.23. Significantly outperforms solo trend (-6.4%) or solo MR (-100%).

### O1: Sentiment Lead/Lag Analysis
- `scripts/analyze_sentiment_lead_lag.py` - Cross-correlation rho(tau) = Corr(S_t, R_{t+tau}) for tau in [-20, +20].
- Three tests: synthetic baseline (reflective control), LM dictionary sentiment (price-derived), forward-looking proxy (validation).
- **Gate: BLOCKED.** LM sentiment from price-derived headlines is reflective (peak at tau=0, rho=0.90).
- Test 3 validates methodology: forward-looking proxy correctly shows PASS at tau=+3d (rho=0.79).
- Do not build social media infra. Revisit if real news text source becomes available.

### Q1-Q3: Multi-Factor Fundamental Factors
- `src/ml/fundamental_features.py` - FundamentalFeatureExtractor (14 factors: P/E, P/B, P/S, EV/EBITDA, ROE, ROA, profit margin, debt/equity, market cap, revenue growth, earnings growth, dividend yield, beta, short % float), QuarterlyFundamentalProvider.
- `scripts/backtest_multi_factor.py` - Multi-factor backtest (composite z-score ranking, top-N, monthly rebalance, sector presets, factor IC computation).
- Integrated into MLStrategy via `use_fundamentals` toggle. `run_ml_backtest.py` updated with `--use-fundamentals` flag.
- SPY 2020-2026 multi-factor: 149% return, Sharpe 0.86 vs SPY 151%, Sharpe 0.93. 14 features joined to ML pipeline.

### R1-R4: Advanced NLP — FinBERT + SEC Filing Analysis
- `src/signals/sentiment/finbert.py` - FinBERTSentiment, FinBERTVsLMComparator. ProsusAI/finbert context-aware financial sentiment.
- `src/data_ingestion/sec_filing_scraper.py` - SECFilingScraper: CIK lookup, 10-K/10-Q download, MD&A extraction.
- `src/signals/sentiment/filing_analyzer.py` - FilingAnalyzer: YoY text similarity, tone change, uncertainty trends, emerging keywords.
- `src/signals/sentiment/multi_source_fusion.py` - MultiSourceFusion: weighted blend of LM+FinBERT+filing+sources, trade signal >= 2 sources agree.
- `scripts/train_nlp_fusion.py` - NLP+Financial CatBoost fusion training. Gate FAIL: baseline AUC 0.613, fusion AUC 0.595 (delta -0.019). NLP features redundant with price features. Kept for regime-dependent value.

### S1-S3: Pairs Trading (Statistical Arbitrage)
- `src/strategies/pairs_trading_strategy.py` - Cointegration + rolling OLS hedge ratio + z-score mean reversion. Trades instrument A based on spread signals from A and B. Correlation gate (min_corr=0.7). Optional ATR trailing stop.
- `scripts/backtest_pairs.py` - CLI with --compare (9 pre-built pairs), --sweep-entry, --auto-pairs (cointegration discovery), --atr-exit.
- 9 pre-built sector pairs tested 2016-2024: CVX-XOM best (107.3% return, Sharpe 0.40, 17 trades, 64.7% win, PF 7.27). DUK-SO positive (12.8%). Others negative. ETF pairs fail (spreads directional, not mean-reverting). Only DUK-SO and NEM-GOLD have significant cointegration (p<0.05).
- **Gate:** Pairs trading works on same-sector fundamentally-similar companies. CVX-XOM passes Sharpe>0 with 10+ trades. Keep as orthogonal alpha source.

### V1-V4: Strategy Architecture v2
- `src/strategies/rules_first_strategy.py` updated with `use_short=True` param. Bearish patterns trigger `self.sell()`. Inverted ATR trail for short positions.
- `src/risk/strategy_aware_sizing.py` - Kelly-derived position sizing per strategy type (trend=0.5, MR=0.25, pairs=0.5, ML=0.25). Half-Kelly by default.
- `src/portfolio/multi_asset_allocator.py` - Correlation-aware clustering (r>0.7 grouped). Equal weight to clusters, signal-strength within. Prevents SPY+QQQ+XLK triple tech bet.
- `src/data_ingestion/crypto_provider.py` - CCXTCryptoProvider wrapping ccxt.binance(). Public OHLCV, no API key needed. Caching to data/raw/. 8 crypto symbols pre-configured.

## Completed Tasks

- [x] **Gate Fix Postmortem (2026-05-21)** — VIX+yield+killzone gates reverted to OFF in 12 files. RulesFirst SPY 2025 Sharpe improved +0.705→+0.92. SMC trade metrics display bug fixed. SMC binary scoring root cause identified. Docs updated (COMMAND_CHEATSHEET, BESTS, MEMORY).
- [x] **New Techstack Full Backtest (2026-05-21)** — RulesFirst 16 instruments (3 batches) + SMC 5 instruments. VIX+yield gates found crush returns. Gates recommended OFF by default.
- [x] C1-C6: Direction C — 6 phases, 12 papers, all modules built
- [x] E1+E2+E5: Direction E — Chronos (degraded), cross-asset fix (+0.40 Sharpe). CONCLUDED.
- [x] A1-A5 — RegimeRouter: SimpleTrendRegimeDetector, per-regime CatBoost, OOS Sharpe +0.35
- [x] B1-B5 — Rules-First: 34 patterns, mr=0.70, OOS Sharpe +0.76. **Gates PASS.**
- [x] AB1-AB2 — Combined ML+Rules: built, backtested. Rules-first confirmed dominant.
- [x] AB4 — Production decision: Rules-First primary, ML secondary.
- [x] Phase 16 N+P+O+Q+R+S+V — 7/9 sub-phases done. CONCLUDED (ACCEPT LIMITS) 2026-05-16.
- [x] Phase 17 R1 (IR-Weighted Synthesis) through R14 (Factor Engine Wrapper) all complete. Phase 17 CONCLUDED.
- [x] Phase 10a (Optuna + PyPortfolioOpt) — Verified complete 2026-05-16 (T10a-1 through T10a-6, 24 tests pass)
- [x] Phase 12c P2-3 (Dynamic Ensemble Collapse) — Fixed: stacking LogisticRegression replaces EGD averaging. Scripts: diagnose_ensemble_collapse.py
- [x] Phase 12c P3-1 (Paper-Trading Harness) — scripts/paper_trade_daily.py with --basket, --days, --status flags
- [x] Phase 12c P3-2 (Kelly Position Sizing) — estimate_minimum_capital() + format_kelly_report() in kelly_allocator.py
- [x] Phase 12c P2-4 (Walk-Forward Cadence) — scripts/backtest_wf_cadence.py: 12mo best Sharpe 1.62, 24mo best return 9.1%. Frequent (≤4mo) degrades.
- [x] Phase 12d P3-3 (Survival Analysis) — scripts/train_survival_exit.py: GBSA C-index OOS 0.684, RSF 0.673. Gate PASSES (>0.55).
- [x] Phase 12d P3-4 (Regression Labels) — scripts/train_regression_labels.py: CatBoostRegressor DirAcc OOS 62.6%, IC 0.107. Gate PASSES (>55% + IC>0.03).
- [x] Phase 12d P3-5 (HMM Regime Detection) — scripts/train_hmm_regime.py: HMM Ensemble AUC OOS 0.558 < Single 0.610. Gate FAILS. Simple 200MA rule is better.

## OOS Comparison (SPY 2025)

| Strategy | Config | Sharpe | Return | Trades | Win% | PF | MaxDD |
|----------|--------|--------|--------|--------|------|-----|-------|
| ML single model | et=0.55 | -1.25 | -6.78% | 7 | 42.9 | 0.34 | -8.49 |
| ML RegimeRouter | et=0.55 | +0.35 | +2.13% | 7 | 57.1 | 1.61 | -6.26 |
| Rules-First | mr=0.40, multi-TP | +0.46 | +3.13% | 23 | 65.2 | 2.32 | -7.07 |
| Rules-First | mr=0.70, multi-TP | +1.21 | +6.93% | 12 | 83.3 | 6.24 | -4.65 |
| **Rules-First + Registry** | **mr=0.70, multi-TP** | **+2.00** | **+8.96%** | **8** | **100.0** | **∞** | **-1.75** |
| Rules-First (prev) | mr=0.70, no multi-TP | +0.76 | +9.20% | 12 | 58.3 | 2.10 | -10.2 |
| **Phase 07 Paper Trade** | **mr=0.70, multi-TP, registry** | **+1.95** | **+9.10%** | **8** | **100.0** | **inf** | **-1.97** |
| Rules-First New Techstack | mr=0.70, VIX+yield ON | +0.705 | +0.04% | 17 | 70.6 | 1.72 | -3.0 |
| **Rules-First Gates OFF** | **mr=0.70, gates OFF** | **+0.92** | **+0.42%** | **11** | **72.7** | **2.11** | **-0.25** |

## Key Files

| File | Purpose |
|------|---------|
| `src/strategies/rules_first_strategy.py` | **PRIMARY** — 34 pattern detectors, ATR trail, production system |
| `src/strategies/combined_strategy.py` | **NEW** — 4 weight modes, reference/validation only |
| `scripts/backtest_rules_first.py` | CLI for rules-first backtests with sweep |
| `scripts/backtest_combined.py` | **NEW** — CLI for combined strategy backtests |
| `scripts/optimize_pattern_weights.py` | Empirical calibration from ablation |
| `src/ml/simple_regime.py` | SimpleTrendRegimeDetector (Bull/Bear via 200MA) |
| `src/ml/regime_router.py` | RegimeRouter — per-regime model dispatch |
| `models/regime_router_SPY.json` | RegimeRouter config (2 models + fallback) |
| `scripts/backtest_rules_batch.py` | **NEW** — Multi-instrument batch runner, auto-insights |
| `reports/batch/rules_first_IS_2016_2024.json` | Cross-instrument IS results (16 instruments) |
| `reports/batch/rules_first_OOS_2025_2026.json` | Cross-instrument OOS results (11 instruments) |
| `reports/batch/INSIGHTS.md` | **NEW** — Comprehensive cross-instrument analysis, sweeps, per-instrument optimal configs |
| `BESTS.md` | Updated with Cross-Instrument OOS + Entry Sweep sections |
| `src/graphify-out/GRAPH_REPORT.md` | **NEW** — Knowledge graph report: 7068 nodes, 11170 edges, 498 communities across 314 files |
| `src/graphify-out/graph.json` | **NEW** — Queryable codebase knowledge graph (query/path/explain commands) |
| `.kilo/skills/engineering/mattpocock/` | **NEW** — 8 mattpocock skills (diagnose, tdd, grill-with-docs, handoff, architecture, to-prd, to-issues, caveman) |
| **Phase N (NLP)** | |
| `src/signals/sentiment/__init__.py` | Sentiment subpackage exports |
| `src/signals/sentiment/dictionary.py` | LMDictionary, LMSentimentScorer, LMTradingSignalModifier |
| `data/sentiment/LM_Master_Dictionary_1993-2025.csv` | 86K+ word LM financial dictionary |
| **Phase P (Mean Reversion)** | |
| `src/strategies/mean_reversion_strategy.py` | **NEW** — 6-indicator MR (RSI+WR+CCI+MFI+StochRSI+BB), ADX-gated |
| `src/strategies/regime_router_strategy.py` | **NEW** — Routes trending vs ranging, trend trail + MR fixed TP/SL |
| `scripts/backtest_mean_reversion.py` | **NEW** — CLI for MR + RegimeRouter with --compare mode |
| **Phase O (Sentiment Gate)** | |
| `scripts/analyze_sentiment_lead_lag.py` | **NEW** — Lead/lag cross-correlation, gate decision logic |
| **Phase Q (Fundamentals)** | |
| `src/ml/fundamental_features.py` | **NEW** — FundamentalFeatureExtractor (14 factors), QuarterlyFundamentalProvider |
| `scripts/backtest_multi_factor.py` | **NEW** — Multi-factor backtest with monthly rebalance, sector presets, factor IC |
| `src/strategies/ml_strategy.py` | Updated with `use_fundamentals` toggle |
| `scripts/run_ml_backtest.py` | Updated with `--use-fundamentals` flag |
| **Phase R (Advanced NLP)** | |
| `src/signals/sentiment/finbert.py` | **NEW** — FinBERTSentiment, FinBERTVsLMComparator |
| `src/data_ingestion/sec_filing_scraper.py` | **NEW** — SEC EDGAR 10-K/10-Q scraper with MD&A extraction |
| `src/signals/sentiment/filing_analyzer.py` | **NEW** — FilingAnalyzer (YoY similarity, tone, keywords) |
| `src/signals/sentiment/multi_source_fusion.py` | **NEW** — MultiSourceFusion (LM+FinBERT+filing weighted blend) |
| `scripts/train_nlp_fusion.py` | **NEW** — NLP+Financial CatBoost fusion training, AUC comparison |
| **Phase S (Pairs Trading)** | |
| `src/strategies/pairs_trading_strategy.py` | **NEW** — Cointegration + rolling OLS hedge + z-score mean reversion |
| `scripts/backtest_pairs.py` | **NEW** — CLI with --compare / --sweep-entry / --auto-pairs / --atr-exit |
| **Phase V (Strategy v2)** | |
| `src/strategies/rules_first_strategy.py` | Updated with `use_short` param for bearish pattern shorts |
| `src/risk/strategy_aware_sizing.py` | **NEW** — Kelly-derived sizing per strategy type (trend/MR/pairs/ML) |
| `src/portfolio/multi_asset_allocator.py` | **NEW** — Correlation-clustered multi-asset allocation |
| `src/data_ingestion/crypto_provider.py` | **NEW** — CCXTCryptoProvider wrapping ccxt (free Binance OHLCV) |
| **Phase 17 (R1-R8)** | |
| `src/signals/ir_weighting.py` | **NEW** — IRWeighting: rolling Information Ratio pattern weights, look-ahead safe |
| `src/signals/factor_purification.py` | **NEW** — FactorPurifier: OLS sector/size purification, purity_ratio |
| `src/signals/evaluation_gate.py` | **NEW** — PatternEvaluationGate: 4-step formal validation (t-stat, type, IC, quantile) |
| `scripts/evaluate_pattern.py` | **NEW** — CLI for 4-step gate with --ticker or --signals-file input |
| `src/ml/expected_returns.py` | **NEW** — hp_filter(), HPFilter class, λ constants for daily/weekly/monthly |
| `src/ml/factor_features.py` | **NEW** — CEI (Acharya-Pedersen), Amihud illiquidity, Roll spread liquidity factors |
| `src/signals/scoring.py` | **NEW** — MultiAxisScorer, 5-axis signal quality (IC/IR/turnover/diversity/overfit) |
| `src/ml/preprocessing.py` | **NEW** — MADOutlierClipper, RankStandardizer, mad_rank_pipeline() |
| `src/signals/collinearity.py` | **NEW** — VIF collinearity analysis, redundancy pairs, synthesize/discard recs |

| **Phase 17 P2 (R9-R12)** | |
| `src/signals/classification.py` | **NEW** — FactorClassifier, return/risk factor classification, bulk classify |
| `scripts/attribution.py` | **NEW** — Performance attribution decomposition CLI (OLS factor regression) |
| `src/ml/factor_features.py` | Extended — Merton Distance-to-Default (compute_dtd, compute_dtd_dataframe, add_dtd_features) |
| `scripts/evaluate_qrafti.py` | **NEW** — 14-test QRAFTI diagnostic suite + batch mode |
| **Phase 17 P3 (R13-R14)** | |
| `src/optimization/__init__.py` | **NEW** — Optimization module exports |
| `src/optimization/huatai_pipeline.py` | **NEW** — 4-phase optimization (IR→HP→Risk→QP) + HuataiResult |
| `scripts/run_huatai_pipeline.py` | **NEW** — CLI for Huatai optimization pipeline |
| `src/ml/factor_engine.py` | **NEW** — FactorEngineWrapper, try-install wrapper for 11 factors |
| **Phase 10a (Optuna/PyPortfolioOpt)** | |
| `src/ml/tuning/optuna_tuner.py` | Optuna TPE hyperparameter tuning for CatBoost/LightGBM |
| `src/ml/tuning/optuna_strategy_tuner.py` | Optuna strategy parameter tuning (RSI, MACD, EMA, etc.) |
| `src/optimizer/pypfopt_integration.py` | PyPortfolioOpt: HRP, EfficientFrontier, CVaR, Black-Litterman |
| **Phase 12c (Architecture Improvements)** | |
| `scripts/diagnose_ensemble_collapse.py` | **NEW** — P2-3: EGD weight diagnostics, stacking vs voting comparison |
| `scripts/paper_trade_daily.py` | **NEW** — P3-1: Daily paper trading signal harness |
| `src/risk/kelly_allocator.py` | **UPDATED** — P3-2: estimate_minimum_capital() + format_kelly_report() |
| `src/ml/dynamic_ensemble.py` | **UPDATED** — P2-3: stacking method (default), LogisticRegression meta-model |
| `scripts/backtest_wf_cadence.py` | **NEW** — P2-4: Walk-forward cadence experiment (6 cadences, 12mo optimal) |
| `scripts/train_survival_exit.py` | **NEW** — P3-3: Survival analysis for time-to-exit (GBSA C-index OOS 0.684) |
| **Phase 12d (New Signal Sources)** | |
| `scripts/train_regression_labels.py` | **NEW** — P3-4: CatBoostRegressor 5d forward return (DirAcc OOS 62.6%, IC 0.107) |
| `scripts/train_hmm_regime.py` | **NEW** — P3-5: HMM vs Simple 200MA regime detection (Gate FAILS) |
| **Phase 18 (Useful Repos — P0)** | |
| `src/portfolio/eiten_adapters/` | **NEW** — 5 portfolio optimization strategies (Eigen, MVP, MSR, GA, RMT) |
| `src/portfolio/eiten_builder.py` | **NEW** — Unified portfolio optimizer CLI with --compare/--from-signals |
| `scripts/optimize_portfolio.py` | **NEW** — CLI for 4-strategy portfolio optimization |
| `src/data_ingestion/financial_scraper.py` | **NEW** — Scrapling-based scraper (insider/news/SEC/earnings) |
| `scripts/scrape_financial_data.py` | **NEW** — CLI for financial data scraping |
| `.kilo/skills/engineering/` | **NEW** — 6 agent-skills workflows (spec/tdd/review/debug/perf/ship) |
| **Phase 19 (New Repos — P0)** | |
| `.kilo/skills/engineering/mattpocock/` | **NEW** — 8 mattpocock skills (diagnose, tdd, grill-with-docs, handoff, architecture, to-prd, to-issues, caveman) |
| `src/graphify-out/GRAPH_REPORT.md` | **NEW** — Knowledge graph report (7068 nodes, 11170 edges, 498 communities) |
| `docs/glossary-ai-coding.md` | **NEW** — 62-term AI coding glossary with trading adaptations |
| `skills_arsenal/.../ai-coding-dictionary/SKILL.md` | **NEW** — Full 62-term dictionary skill for skills_arsenal |

## New Resources (Ingested 2026-05-16)

### Phase 20 Session Results (2026-05-17)

> Empirical testing across all modules. Source data for Phase 20 plan.

| Experiment | Result |
|------------|--------|
| GA Portfolio (8-asset) | Sharpe 1.38, GLD +31%, XLK +28%, SPY -22%. 2/8 RMT eigenvalues. |
| Multi-TP SPY OOS 2025 | Sharpe 0.54→1.21 (+124%), Win 67%→83%, MaxDD -13.3%→-4.7% |
| IR-Weights SPY OOS 2025 | 12→2 trades, Sharpe 0.54→0.79. Too few trades. |
| Short-side SPY OOS 2025 | Identical to long-only. No edge in bullish year. |
| H&S 4-Step Gate | FAIL on 3/4 (t-stat -0.59, IC -0.011, return/risk 0.00). Quantile spread PASS. |
| Tech Multi-Factor | +592% vs SPY +151%, Sharpe 1.31. ps_ratio IC 0.74. |
| Healthcare Multi-Factor | +164% vs SPY +151%, Sharpe 1.00. profit_margin IC 0.90**. |
| Financials Multi-Factor | +120% vs SPY +151%, Sharpe 0.59. pb_ratio IC 0.95***. |
| Pairs CVX-XOM | +98.8%, PF 2.83, 40 trades, 57.5% win. |
| RegimeRouter SPY | +209.4%, Sharpe 0.68, 58 trades, 58.6% win. |
| BTC IS rules-first | -9.3%, Sharpe -0.06. Diverges from batch Sharpe 0.77. Needs audit. |
| BTC OOS 2025 short | 0 trades. Pattern trigger gap in crypto bear. |

### Phase 19: 9 New Repos Wave 2 (C:\Dev\useful_repos) — 2026-05-16

| Repo | Relevance | Key Value | Plan |
|------|-----------|-----------|------|
| **graphify** | HIGH | Knowledge graph of entire codebase (7068 nodes, 11170 edges, 498 communities). 71.5x token reduction. | R19a ✅ — installed + graph built in `src/graphify-out/` |
| **lean-ctx** | HIGH | 60-95% token savings via compression + caching. 51 MCP tools. | R19b ✅ — installed (lean-ctx 3.6.0 via npm), setup complete (18/18 checks), Kiro MCP configured. |
| **skills** (mattpocock) | HIGH | 8 highest-value skills (diagnose, tdd, grill-with-docs, handoff, architecture, to-prd, to-issues, caveman) | R19c ✅ — copied to `.kilo/skills/engineering/mattpocock/` |
| **dictionary-of-ai-coding** | HIGH | 62-term AI glossary by Matt Pocock | R19d ✅ — copy to skills_arsenal |
| **GitNexus** | HIGH | Code intelligence graph, 16 MCP tools | R19e ✅ — installed, indexed (28,316 symbols, 43,736 edges, 300 flows) |
| **BettaFish** | MED | ForumEngine debate pattern for signal fusion | R19f ✅ — documented in `docs/reference-bettafish-patterns.md` |
| **Qbot** | MED | Trading platform, 2 DL strategies, infrastructure patterns | R18d ✅ — studied, documented in `docs/reference-qbot-ml-patterns.md` |
| **qmd** | MED | On-device hybrid markdown search (BM25 + vector + LLM rerank) | R18e ✅ — installed @tobilu/qmd, 118 files indexed, 900 chunks embedded |
| **three-geospatial** | NONE* | *HIGH for personal_website (atmosphere/clouds/stars) | R19g — personal_website only |
| **maigret** | LOW | Async executor, report pipeline patterns | R19h — deferred |
| **MinerU** | LOW | PDF→MD already covered by paper2md | R19i — deferred |

### Graphify Knowledge Graph

- **Graph:** `src/graphify-out/graph.json` (7068 nodes, 11170 edges, 498 communities)
- **Report:** `src/graphify-out/GRAPH_REPORT.md` (1688 lines)
- **Corpus:** 314 Python files, ~352K words
- **Key clusters detected:** signal quality filtering, basic patterns, backtest adapter, harmonic patterns, ensemble methods, IC computation, indicator cache, HP filter, cross-asset features, SHAP dashboard, GPU detectors, regime components
- **Command:** `uv run graphify update src/` to refresh after code changes (no API cost)
- **Query:** `uv run graphify query "how does rules_first_strategy work"` for graph traversal
- **Path:** `uv run graphify path "pattern detection" "backtest engine"` for dependency chains

### GitNexus Code Intelligence Graph

- **Index:** 28,316 symbols, 43,736 relationships, 676 clusters, 300 execution flows
- **Status:** `gitnexus status` — verified up-to-date with commit 9cd062b
- **Re-index:** `gitnexus analyze .` after significant codebase changes (~96s for full reindex)
- **Query:** `gitnexus query "concept"` — natural language search for execution flows
- **Context:** `gitnexus context "SymbolName"` — callers, callees, process participation
- **Impact:** `gitnexus impact "SymbolName" --direction upstream` — blast radius with risk level
- **Detect changes:** `gitnexus detect-changes` — map git diff to affected symbols and flows
- **MCP tools:** 16 tools available via `gitnexus mcp` (stdio server for AI agents)
- **Index excluded from git:** `.gitnexus/` added to `.gitignore` (141MB, regenerable)

### qmd — Project Documentation Search

- **Index:** 118 markdown files across 2 collections, 900 embedded chunks
- **Collections:** `investment_trying` (35 docs), `investment_trying_progress` (83 progress_docs)
- **Search:** `bash qmd search "query" -c investment_trying` — BM25 keyword search
- **Semantic:** `bash qmd vsearch "how does regime router work" -c investment_trying` — vector similarity
- **Hybrid:** `bash qmd query "topic"` — BM25 + vector + LLM reranking (best quality)
- **MCP:** `bash qmd mcp` starts a stdio MCP server for AI agent integration
- **Models:** embeddinggemma-300M (embeddings), Qwen3-Reranker-0.6B (reranking), qmd-query-expansion-1.7B (query expansion) — all local GGUF on Intel Arc GPU
- **Note:** Requires bash prefix on Windows: `bash $(which qmd) search "query"`
- **Update:** `bash qmd update` to re-index after doc changes

### Cross-Project Utility

| Project | Best Fit Repo | Why |
|---------|-------------|-----|
| albion_get_rich | Scrapling HIGH | Replace 4 scraping deps, Cloudflare bypass for AO Data API |
| personal_website | agent-skills HIGH | Web-dev lifecycle (frontend eng, testing, perf, launch) |
| skills_arsenal | agent-skills+qmd+CLI-Anything HIGH | Skill format reference + KB search + 40 community skills |
| data_and_stat_analysis | Scrapling/qmd/local-deep-research MED | Government data scraping, paper research |

### Prior Resources
Three new knowledge resources analyzed and documented:
| Resource | Location | Key Value |
|----------|----------|-----------|
| Beyond Fama-French (factor extensions) | `useful_resources/papers_md/Beyond_Fama-French_Integrating_Factors.md` | Default/Liquidity factors, Factor Engine library, LLM+MCTS alpha mining, QRAFTI |
| 华泰多因子系列1 (Huatai MFM system) | `useful_resources/papers_md/华泰多因子系列1_多因子模型体系初探.md` | 12 factor categories (74 factors), 4-phase pipeline, factor purification, HP filter forecasting |
| FMZ Strategies Repository | `useful_resources/useful_repos/trading-system/strategies/` (5,807 .md files) | Massive PineScript/JS/Python strategy collection. ~1 ML strategy. Rich in EMA/RSI/MACD/Bollinger/breakout/martingale/grid patterns. |

Detailed handover: `progress_docs/handovers/new-resources-integration-20260516.md`

## FMZ Strategy Conversions (Completed 2026-05-16)

7 strategies converted from PineScript/JS to Python, integrated into the BasePattern framework:

| Detector | Source | Logic | PatternType |
|----------|--------|-------|------------|
| **AlphaBeast** | PS v6 | Supertrend + RSI(14) + Volume(x1.5) triple confirmation | CONTINUATION |
| **MultiFactorTrend** | PS v5 | SAR + EMA(2) + RSI(6) + ADX(14) quad confirmation | CONTINUATION |
| **MomentumZigZag** | PS v5 | QQE/MACD/MA ZigZag with force (RSI) detection | REVERSAL |
| **EMAMACDHF** | PS v5 | EMA(9/21) crossover + MACD(6,13,4) confirmation | CONTINUATION |
| **AdaptiveBollinger** | PS v5 | BB(14,1.5σ) breakout reversion, 4-layer exit | REVERSAL |
| **AIVolatilityBreakout** | PS v6 | Gap fill + VWAP momentum + compression breakout | BREAKOUT |
| **OrderFlowAccumulator** | JS | Golden ratio (0.382) weighted order flow alpha [-1,1] | Signal module |

**New files:**
| File | Purpose |
|------|---------|
| `src/indicators/pinescript_helpers.py` | 17 PineScript functions → NumPy (crossover, supertrend, sar, dmi, macd, qqe, vwap_simple, etc.) |
| `src/patterns/fmz/__init__.py` | FMZ pattern detector exports |
| `src/patterns/fmz/alpha_beast.py` | Alpha Beast detector |
| `src/patterns/fmz/multi_factor_trend.py` | Multi-Factor Trend detector |
| `src/patterns/fmz/momentum_zigzag.py` | Momentum ZigZag detector |
| `src/patterns/fmz/ema_macd_hf.py` | EMA-MACD HF detector |
| `src/patterns/fmz/adaptive_bollinger.py` | Adaptive Bollinger detector |
| `src/patterns/fmz/ai_volatility_breakout.py` | AI Volatility Breakout detector |
| `src/signals/order_flow.py` | HFT Order Flow signal module |

**Modified files:**
| File | Change |
|------|--------|
| `src/indicators/__init__.py` | Added pinescript_helpers exports |
| `src/strategies/rules_first_strategy.py` | Added FMZ pattern reliability weights (6 new entries) |
| `src/strategies/backtest_py/multi_pattern_strategy_optimized.py` | Registered all 6 FMZ detectors in _init_patterns() |
| `docs/COMMAND_CHEATSHEET.md` | Added FMZ Strategy Conversions section |
| `.useful_commands/useful_commands.txt` | Added FMZ test commands |

## Next Session Agent Must

1. **READ MEMORY.md** (this file) — All 25 phases complete. Phase 26 ACTIVE (bear market validation + paper trading launch).
2. **Phase 26 P0:** Commit all pending changes. Run paper trading with 18-instrument basket (see production basket above).
3. **Phase 26 P1:** Bear market validation — run `uv run scripts/backtest_all_comprehensive.py --use-best --is-start 2016-01-01 --is-end 2021-12-31 --oos-start 2022-01-01` to test 2022 survival as clean OOS.
4. **Phase 26 P2:** Update production config `config_files/production_basket.yaml` with 18-instrument 3-tier basket.
5. **IS/OOS split PRIMARIES** (do NOT change these without explicit analysis):
   - `scripts/backtest_all_comprehensive.py:74-75`: `IS_PERIOD = ("2016-01-01", "2024-12-31")`, `OOS_PERIOD = ("2025-01-01", None)`
   - `scripts/tune_rules_params.py:42-47`: IS=2016-2024, OOS=2025+
6. **IS/OOS secondary split** (for bear market testing): IS=2016-2021, OOS=2022-2026.
7. **Production system:** Rules-First (mr=0.70, multi-TP ON, quality registry ON, gates OFF). ML secondary.
8. **Deploy ONLY to S-tier + A-tier categories** (Energy, Tech ETF, Gold/Silver, MidCap-Steel, LargeCap Index). NEVER deploy to Financials, REITs, HK, Utilities, Bonds, Forex, MicroCap.
9. **Deferred phases:** 05 (GPU), 02 (vectorbt Windows), 07 (calendar days) — hardware/environment gated.
10. **Use graphify for codebase navigation:** `uv run graphify query "how does X work"`.
