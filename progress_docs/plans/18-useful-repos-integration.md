# Phase 18: Useful Repos Integration

**Source:** 7 repositories evaluated in `C:\Dev\useful_repos` (2026-05-16).
**Priority: P0 → Portfolio optimization (eiten) + Financial data scraping (Scrapling) first.**
**Total LOC estimate: ~500 (R18a: 300, R18b: 200). No heavy new dependencies.**

---

## Repository Evaluations

### 1. eiten — Portfolio Optimization Toolkit
**Path:** `C:\Dev\useful_repos\eiten` | **Lang:** Python | **License:** GPL v3

**What it does:** Algorithmic portfolio optimization with 4 strategies:
- **Eigen Portfolios** — PCA on covariance matrix; 1st eigen = market, 2nd = highest risk/reward uncorrelated
- **Minimum Variance Portfolio** — QP minimization
- **Maximum Sharpe Ratio** — Optimization maximizing risk-adjusted return
- **Genetic Algorithm** — Custom GA with selection/mutation/crossover maximizing Sharpe
- **Random Matrix Theory filtering** — Denoises covariance by filtering eigenvalues (Marchenko-Pastur)
- **Backtesting + forward testing** with Monte Carlo simulation

**Key files:** `eiten.py` (orchestrator), `strategies/eigen_portfolio_strategy.py`, `strategies/genetic_algo_strategy.py`, `strategies/strategy_helper_functions.py` (RMT), `backtester.py`, `simulator.py`

**Why integrate:** The project has strong signal generation (45+ pattern detectors, ML models) but no portfolio construction layer. Eiten provides the allocation engine that converts top-ranked signals into risk-managed portfolios. RMT covariance denoising is a sophisticated technique directly usable for multi-asset correlation stabilization.

**Integration plan (R18a):**
1. Copy core strategy classes to `src/portfolio/eiten_adapters/`
2. Create `src/portfolio/portfolio_builder.py` wrapping eiten strategies with project data structures
3. Wire into existing signal pipeline: `RulesFirstStrategy.signal → PortfolioBuilder.allocate()`
4. Add RMT covariance denoising to `src/portfolio/covariance_denoiser.py`
5. CLI: `scripts/optimize_portfolio.py --strategy eigen|mvp|msr|ga --signals <path>`
6. Test on 33-ticker basket with backtesting.py integration

**Dependencies:** numpy, scipy, scikit-learn, yfinance, pandas, matplotlib — all already in `pyproject.toml`.

---

### 2. Scrapling — Adaptive Web Scraping Framework
**Path:** `C:\Dev\useful_repos\Scrapling` | **Lang:** Python 3.10+ | **License:** BSD-3-Clause

**What it does:** High-performance undetectable web scraping with:
- **Adaptive parsing** — Elements tracked with similarity; when websites change, elements auto-relocate
- **4 fetcher types:** HTTP, Stealthy (anti-bot+CurlImpersonate), Dynamic (Playwright browser), AsyncFetcher
- **Cloudflare Turnstile bypass** — Out-of-box solving, not just detection
- **Spider framework** — Concurrent crawling with checkpoint/resume, proxy rotation, robots.txt compliance
- **12-784x faster than BeautifulSoup** (2.02ms vs 1,584ms text extraction)
- **MCP server** for AI-assisted scraping
- **CLI** with `scrapling extract` for no-code extraction

**Why integrate:** The project currently uses only Yahoo Finance (yfinance) for data. Scrapling enables:
- SEC EDGAR filings (10-K/10-Q/8-K) — annual/quarterly reports, MD&A sections
- Earnings call transcripts from financial websites
- Insider trading data (Form 4 filings)
- News sentiment from financial news aggregators
- Options flow data from exchanges
- Economic calendar events

**Integration plan (R18b):**
1. `uv add scrapling`
2. Create `src/data_ingestion/sec_scraper_v2.py` — replaces `src/data_ingestion/sec_filing_scraper.py` with adaptive parsing
3. Create `src/data_ingestion/news_scraper.py` — financial news with Spider framework for periodic collection
4. Create `src/data_ingestion/insider_scraper.py` — Form 4 filing scraper
5. Spider for daily data collection: `scripts/scrape_daily_data.py`
6. Spider for weekly earnings calendar: `scripts/scrape_earnings_calendar.py`

**Dependencies:** lxml, cssselect, orjson, tld, w3lib (+ optional: Playwright, curl_cffi, browserforge for stealth).

---

### 3. agent-skills — AI Agent Workflow Skills
**Path:** `C:\Dev\useful_repos\agent-skills` | **Lang:** Markdown | **License:** MIT

**What it does:** 23 production-grade engineering workflows for AI coding agents (Claude Code, Cursor, Gemini CLI, etc.). Created by Addy Osmani encoding Google's engineering culture. Includes:
- 7 lifecycle phases: Define → Plan → Build → Verify → Review → Ship
- 22 lifecycle skills + 1 meta-skill
- 3 agent personas: `code-reviewer`, `test-engineer`, `security-auditor`
- **Anti-rationalization tables** countering common agent excuses ("I'll add tests later", "This is too small for a spec")
- Verification gates at each stage with evidence requirements

**Why adopt:** The project already uses `.kilo/skills/` (Kilo skills). Agent-skills fills gaps with battle-tested engineering workflows. The anti-rationalization pattern directly prevents agents from skipping critical trading-system steps (OOS validation, look-ahead bias checks, backtest verification).

**Integration plan (R18c):**
1. Copy relevant skills to `.kilo/skills/engineering/`:
   - `spec-driven-development.md` — for planning new pattern detectors or backtest features
   - `test-driven-development.md` — for validating new strategy implementations
   - `code-review-and-quality.md` — for reviewing ML pipeline changes
   - `debugging-and-error-recovery.md` — for debugging backtest look-ahead bias
   - `performance-optimization.md` — for vectorized backtest engine optimization
   - `shipping-and-launch.md` — for deploying strategy changes to production
2. Create project-specific agent personas: `.kilo/agent/backtest-auditor.md`, `.kilo/agent/ml-model-reviewer.md`
3. Add anti-rationalization table to `.kilo/global-rules.md` for trading-specific excuses

**Dependencies:** None (pure Markdown).

---

### 4. Qbot — Quantitative Trading Platform (Reference)
**Path:** `C:\Dev\useful_repos\Qbot` | **Lang:** Python 3.8-3.9 | **License:** CC BY-NC-SA 4.0

**What it does:** Chinese-language full quant trading platform (AI智能量化投研平台):
- **300+ ML/DL models:** XGBoost, LightGBM, CatBoost, LSTM, GRU, Transformer, GAT, TFT, TabNet, ADARNN, HIST, IGMTF, Localformer, DoubleEnsemble
- **30+ technical indicators:** MACD, KDJ, RSRS, RSI, BOLL, SAR, DMI, CCI, etc.
- **Multi-factor framework:** alpha-101, alpha-191, genetic programming factor mining (DEAP)
- **Multiple brokers:** CTP (futures), XTP (stocks), Binance/OKX/Huobi (crypto)
- **Notification system:** email, Feishu, WeChat, desktop popup
- **Backtesting:** Backtrader + quantstats + pyfolio integration
- **Dagster orchestration** for batch data processing

**Why study (not copy):** Python 3.8-3.9 is outdated, code is in Chinese, parts are premium. Value is in the **architecture reference** and **catalog of what's possible**:
- 25+ DL paper implementations in `qbot/strategy/benchmarks/` — reference for TFT, TabNet, HIST, etc.
- 30+ indicator strategies in `qbot/strategy/` — pattern library for potential porting
- Broker interface design pattern — reference for future live trading
- Notification system architecture — multi-channel alerts

**Integration plan (R18d):**
1. Study `qbot/strategy/benchmarks/` — catalog DL architectures with paper references
2. Study `qbot/strategy/` — evaluate which indicator strategies complement existing 45+ patterns
3. Study broker interface pattern — document for future live trading phase
4. Document findings in `docs/reference-qbot-architecture.md`

**Dependencies:** N/A (study only, no integration).

---

### 5. qmd — Query Markdown Documents
**Path:** `C:\Dev\useful_repos\qmd` | **Lang:** TypeScript/Bun | **License:** MIT

**What it does:** On-device hybrid search for markdown knowledge bases:
- **3 search modes:** BM25 keyword, vector/semantic, hybrid + LLM re-ranking
- **Local GGUF models** — embedding, re-ranker, query expansion (auto-downloaded)
- **Smart chunking** for markdown; AST-aware for code (Python, TypeScript, Go, Rust)
- **MCP server** for Claude Desktop/Code integration
- **Context metadata** — attach descriptions to paths for improved search relevance
- **Multiple output formats:** CLI, JSON, CSV, Markdown, XML

**Why use:** The project has extensive documentation: `docs/` (30+ files), `progress_docs/` (session logs, plans), `useful_resources/papers_md/` (20+ papers). Currently unsearchable. QMD provides instant semantic search across all project knowledge.

**Integration plan (R18e):**
1. Install qmd: `npm install -g qmd` or Bun
2. Index project docs: `qmd collection add investment_trying docs/ progress_docs/ useful_resources/papers_md/`
3. Add context metadata: `qmd context add docs/ --desc "Trading strategy documentation and ML guides"`
4. Configure MCP server for Kilo/Claude Desktop access
5. Document usage in `.useful_commands/useful_commands.txt`

**Dependencies:** Node.js >=22 or Bun, SQLite, ~2GB GGUF model downloads.

---

### 6. CLI-Anything — Auto-Generated CLI Harnesses
**Path:** `C:\Dev\useful_repos\CLI-Anything` | **Lang:** Python 3.10+ | **License:** Apache 2.0

**What it does:** 7-phase pipeline to auto-generate agent-friendly CLIs around any software:
- Analyze source → Design commands → Implement Click CLIs → Write tests → Document → Publish
- 40+ generated CLIs (Blender, GIMP, Zotero, Obsidian, ComfyUI, Draw.io, etc.)
- CLI-Hub meta-skill for agent-autonomous CLI discovery
- JSON + human-readable dual output
- SKILL.md + command.md pattern for agent discoverability

**Why study:** The pattern of consistent Click-based CLIs with `--json` output and SKILL.md files is directly applicable to the project's 30+ scripts. Not about generating new CLIs — about adopting the conventions.

**Integration plan (R18f):**
1. Study CLI-Anything's SKILL.md + command.md pattern
2. Document conventions in `docs/guide-cli-conventions.md`
3. Apply to existing scripts gradually: add `--json` flag, structured output
4. Update `AGENTS.md` with agent-friendly script usage patterns

**Dependencies:** N/A (convention documentation only).

---

### 7. local-deep-research — Local AI Research Assistant
**Path:** `C:\Dev\useful_repos\local-deep-research` | **Lang:** Python 3.12+/TypeScript | **License:** MIT

**What it does:** Fully local AI-powered research with web UI, REST API, MCP server:
- LangGraph agentic research with adaptive search engine switching
- 20+ search sources (arXiv, PubMed, Wikipedia, Semantic Scholar, GitHub)
- Local LLM via Ollama/LM Studio
- Encrypted knowledge base with FAISS vector search
- MCP server for AI agent integration
- 95% SimpleQA accuracy on single RTX 3090

**Why deferred:** Massive dependency footprint (LangChain, Flask, Playwright, FAISS, sentence-transformers, SQLAlchemy). The MCP server + LangGraph agentic search architecture is elegant but heavy. More valuable as a pattern reference than direct integration.

**Integration plan (R18g):**
1. Study the MCP server pattern for exposing backtesting/ML capabilities
2. Study the LangGraph agentic search architecture for financial research agent
3. Evaluate as potential skill for `skills_arsenal_for_publishing` project
4. Deferred until project infrastructure stabilizes

**Dependencies:** N/A (deferred, study only).

---

## Priority Ranking & Implementation Order

```
R18a (eiten) → R18b (Scrapling) → R18c (agent-skills) → R18d (Qbot) → R18e (qmd) → R18f (CLI-Anything) → R18g (local-deep-research)
  P0              P0                    P1                     P2               P2             P3                      P3
```

### Rationale:
- **R18a first:** Direct code reuse, same Python ecosystem, immediate value (portfolio construction layer currently missing)
- **R18b second:** Opens new data sources (SEC filings, news, insider data) — currently the project's biggest gap
- **R18c third:** Workflow improvements with zero LOC cost — immediate productivity for all agents
- **R18d-R18g:** Study/reference/convenience improvements, lower urgency

## Expected Outcomes

| Task | Expected Impact | Risk |
|------|----------------|------|
| R18a (eiten) | Portfolio-level Sharpe +0.05–0.15 from RMT denoising + optimized allocation | Low — same stack, well-tested algorithms |
| R18b (Scrapling) | New alpha sources: filing tone, insider sentiment, news flow | Medium — data quality TBD, rate limiting |
| R18c (agent-skills) | Systematic quality gates for all future work | Low — conventions only, no runtime |
| R18d (Qbot) | DL architecture catalog for future GPU gate decision | Low — study only |
| R18e (qmd) | 10x faster research lookup across 50+ project docs | Low — external tool |
| R18f (CLI-Anything) | More discoverable/agent-friendly scripts | Low — conventions only |
| R18g (local-deep-research) | Reference architecture for MCP integration | Low — deferred |
