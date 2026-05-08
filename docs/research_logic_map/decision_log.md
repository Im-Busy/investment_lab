# Decision Log

**Last Updated:** 2026-04-25
**Total Decisions:** 8

---

## Format

- **Date:** When decision was made
- **Context:** What situation prompted the decision
- **Options:** Alternatives considered
- **Decision:** What was chosen
- **Rationale:** Why this option
- **Consequences:** Expected or observed outcomes
- **Review Date:** When to revisit

---

## DEC-2026-04-25: Map-Reduce over Vector DB for Paper Insights

- **Date:** 2026-04-25
- **Context:** User asked about knowledge tracking for research insights
- **Options:**
  1. Vector database with embeddings (Chroma, Weaviate)
  2. Structured markdown with YAML frontmatter
  3. Notion/Airtable database
- **Decision:** Option 2 — Markdown files with consistent templates
- **Rationale:**
  - Git-versionable (unlike Notion)
  - No infrastructure overhead (unlike vector DB)
  - Editable in any text editor
  - Matches existing project documentation style
- **Consequences:** Query capabilities limited vs. vector DB; acceptable tradeoff
- **Review Date:** 2026-07-25 (or when docs exceed 50 pages)

---

## DEC-2026-04-20: Numba over CUDA for Pattern Detection

- **Date:** 2026-04-20
- **Context:** Phase 2 performance optimization
- **Options:**
  1. CUDA GPU acceleration
  2. Numba JIT CPU optimization
  3. Multiprocessing with Ray
- **Decision:** Option 2 — Numba JIT
- **Rationale:**
  - 50-100x speedup achieved (sufficient)
  - No GPU dependency (Windows-compatible)
  - Easier debugging than CUDA
- **Consequences:** 50-100x speedup on basic patterns; complex patterns still sequential
- **Review Date:** 2026-06-20 (if backtest time >30s)

---

## DEC-2026-04-18: backtesting.py over Vectorbt for Primary Backtesting

- **Date:** 2026-04-18
- **Context:** Phase 10 architecture decision for backtesting engine
- **Options:**
  1. backtesting.py (faster, simpler, single-threaded)
  2. vectorbt (slower, vectorized, GPU-capable)
  3. Custom event-driven engine (full control, high maintenance)
- **Decision:** Option 1 + Option 3 hybrid approach
- **Rationale:**
  - backtesting.py for rapid strategy iteration
  - Custom engine for SMC/ICT-specific logic (trade management, position scaling)
  - vectorbt kept as alternative for parameter sweeps
- **Consequences:** Two engines to maintain; clearer separation of concerns
- **Review Date:** 2026-12-31

---

## DEC-2026-04-15: 34 Patterns vs. 10 High-Quality Patterns

- **Date:** 2026-04-15
- **Context:** Pattern count debate during Phase 9 implementation
- **Options:**
  1. Implement all 34+ patterns from technical analysis literature
  2. Focus on top 10 most reliable patterns with rigorous validation
  3. Hybrid: 10 core + 20 experimental patterns
- **Decision:** Option 1 — Implement full 34-pattern catalog
- **Rationale:**
  - Pattern confluence is core to system design
  - Research papers (P4, P6) show pattern diversity improves robustness
  - Can use statistical filtering later (Phase 10 analysis components)
- **Consequences:** Higher initial implementation burden; richer signal aggregation
- **Review Date:** 2026-06-30 (after walk-forward validation results)

---

## DEC-2026-04-12: Friction as First-Class Constraint (R10)

- **Date:** 2026-04-12
- **Context:** OOM-RL paper findings on alpha destruction from friction
- **Options:**
  1. Model friction as post-hoc adjustment (add to metrics after backtest)
  2. Model friction as first-class constraint (integrate into backtest engine)
  3. Ignore friction (assume it nets out)
- **Decision:** Option 2 — First-class constraint
- **Rationale:**
  - P1 findings: friction destroyed 100% of alpha in Phase 1
  - P5, P8 confirm transaction costs are primary reason paper profits fail in practice
  - Realistic evaluation requires friction-aware backtesting
- **Consequences:** More realistic backtest results; may invalidate high-turnover strategies
- **Review Date:** 2026-09-30 (after sufficient backtest data)

---

## DEC-2026-04-10: Regime Detection per Strategy (R4)

- **Date:** 2026-04-10
- **Context:** P4 (Against Universal Trading) findings on failure sets
- **Options:**
  1. Global regime declaration (one regime for all strategies)
  2. Per-strategy regime declaration (each pattern declares its operating regime)
  3. No regime declaration (assume strategies work in all regimes)
- **Decision:** Option 2 — Per-strategy regime declaration
- **Rationale:**
  - P4 proves no strategy is universally profitable
  - Adaptive router requires per-strategy regime tags
  - Enables automatic strategy filtering by detected regime
- **Consequences:** Every pattern must implement `get_regime_appropriateness()` method; additional boilerplate
- **Review Date:** 2026-07-31

---

## DEC-2026-04-08: Confluence Scoring over Single-Pattern Trading

- **Date:** 2026-04-08
- **Context:** Signal aggregation architecture decision
- **Options:**
  1. Single-pattern trading (trade each pattern independently)
  2. Confluence scoring (aggregate multiple pattern signals)
  3. ML ensemble (train meta-model on pattern signals)
- **Decision:** Option 2 — Confluence scoring with ML enhancement
- **Rationale:**
  - System differentiator is multi-pattern confluence
  - ML ensemble adds complexity without proven benefit (Phase 12 validation pending)
  - Confluence scoring is interpretable; ML ensemble is black-box
- **Consequences:** Need robust confluence formula; ML kept as optional enhancement layer
- **Review Date:** 2026-08-31 (after ML validation results)

---

## DEC-2026-04-05: Python over Jivaro/Rust for Production

- **Date:** 2026-04-05
- **Context:** Language choice for production deployment
- **Options:**
  1. Python with Numba (current state)
  2. Jivaro (JIT-compiled Python subset)
  3. Rust bindings for performance-critical paths
- **Decision:** Option 1 — Python with Numba, defer Rust until needed
- **Rationale:**
  - Numba 50-100x speedup sufficient for current backtest needs
  - Rust integration adds build complexity
  - Jivaro ecosystem immature
- **Consequences:** Runtime dependency on Numba; may need Rust rewrite if backtest time >1 minute
- **Review Date:** 2026-12-31 (or if performance becomes inadequate)

---

## Decision Statistics

| Category | Count |
|----------|-------|
| Architecture | 4 |
| Performance | 2 |
| Documentation | 1 |
| Risk Management | 1 |

| Status | Count |
|--------|-------|
| Active | 8 |
| Reversed | 0 |
| Pending Review | 3 |

---

*Log generated from `.kilo/plans/*.md` handover files and architectural decision records.*
