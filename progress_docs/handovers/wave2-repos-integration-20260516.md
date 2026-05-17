# Handover — 9 New Repos Integration (Phase 19)

**Date:** 2026-05-16
**Context:** 9 new repos cloned to `C:\Dev\useful_repos` beyond the 7 already integrated in Phase 18.
**Goal:** Evaluate each, add integration plans to `progress_docs/plans/full.md`, create detailed plan file, update MEMORY.md, identify cross-project utility, execute high-priority integrations.

---

## New Repos Evaluated (Summary)

| # | Repo | What It Does | Language | Relevance (inv) |
|---|------|-------------|----------|:---:|
| 1 | **BettaFish** | Multi-agent public opinion analysis — 4 LLM agents + Forum debate engine, 30+ social media crawlers, sentiment models, report generation | Python/Flask/LLM | MED |
| 2 | **dictionary-of-ai-coding** | 62 AI coding terms glossary in 7-section curriculum by Matt Pocock (Total TypeScript) | Markdown | HIGH |
| 3 | **GitNexus** | Code intelligence knowledge graph — indexes every symbol, dependency, call chain into graph DB, exposes 16 MCP tools for AI agents | TypeScript | HIGH |
| 4 | **graphify** | Project knowledge graph builder — 3-pass extraction (AST+video+docs), community detection, confidence tagging, 71.5x fewer tokens | Python | HIGH |
| 5 | **lean-ctx** | Context runtime for AI agents — 60-95% token reduction via compression, caching, property graphs, 51 MCP tools | Rust | HIGH |
| 6 | **skills** (mattpocock) | 17 AI agent skills (diagnose, TDD, grill-with-docs, handoff, architecture, prototype, etc.) by Matt Pocock | Markdown | HIGH |
| 7 | **three-geospatial** | Three.js/R3F geospatial rendering — atmosphere, volumetric clouds, globe, stars, terrain | TypeScript/React | NONE |
| 8 | **maigret** | OSINT username reconnaissance — checks 3000+ sites, recursive identity discovery, report generation | Python | LOW |
| 9 | **MinerU** | Document parsing engine — PDF/DOCX/PPTX/XLSX→Markdown, 109-language OCR, formula→LaTeX, table→HTML | Python/ML | LOW |

---

## Priority Ranking for investment_trying

### P0 — Direct Code Reuse / Immediate Value

| # | Repo | Task | What to Build | Effort |
|---|------|------|---------------|--------|
| **R19a** | **graphify** | Build knowledge graph of entire project | `uv add graphifyy` → `graphify detect` on `src/` + `docs/` + `useful_resources/papers_md/`. Map 45+ pattern detectors, ML pipeline, backtest engine, cross-module dependencies. The 71.5x token reduction is immediate value for every future AI session. | 30 min setup, 0 LOC |
| **R19b** | **lean-ctx** | Install as MCP server | Install Rust binary, configure MCP server, benchmark token savings on backtest/ML sessions. The 56 shell compression patterns + 10 read modes + CCP session memory would dramatically reduce LLM costs. | 30 min setup, 0 LOC |
| **R19c** | **skills** (mattpocock) | Port 8 highest-value skills | Copy to `.kilo/skills/engineering/`: `/diagnose` (6-phase debugging), `/tdd` (red-green-refactor), `/grill-with-docs` (domain glossary), `/handoff` (session compaction), `/improve-codebase-architecture`, `/to-prd`, `/to-issues`, `/caveman` (token efficiency). These fill gaps agent-skills doesn't cover — especially debugging, architecture, and domain language. | 0 LOC (markdown only) |

### P1 — Reference + Skill Enrichment

| # | Repo | Task | What to Build |
|---|------|------|---------------|
| **R19d** | **dictionary-of-ai-coding** | Integrate into skills_arsenal + project docs | 62 terms relevant for AI agent alignment. Copy to `skills_arsenal_for_publishing/universal-skills/`. Create `docs/glossary-ai-coding.md` in investment_trying with trading-specific adaptations (e.g., "harness"→"backtest harness", "AFK"→"autonomous backtest run"). |
| **R19e** | **GitNexus** | Install and index project | `npx gitnexus` → index `src/`. Precomputed impact analysis for ML pipeline changes. MCP tools for symbol search, call chains, blast radius. |
| **R19f** | **BettaFish** | Study ForumEngine + ReportEngine patterns | ForumEngine multi-agent debate → adapt to signal confluence disagreement resolution. ReportEngine IR pipeline → adapt for automated trading reports. Document patterns in `docs/reference-bettafish-patterns.md`. |

### P2 — Low Priority / Deferred

| # | Repo | Task | Notes |
|---|------|------|-------|
| R19g | **three-geospatial** | N/A for inv | Only relevant for personal_website's Three.js/R3F atmosphere/cloud effects |
| R19h | **maigret** | Extract patterns | Async queue executor, layered settings loading, multi-format report pipeline. Low urgency — scrape patterns already covered by Scrapling (Phase 18) |
| R19i | **MinerU** | Deferred | PDF→MD conversion already handled by paper2md in project. Only adopt if DAMOXUELIE-style scanned Chinese PDFs appear. |

---

## Cross-Project Utility Matrix

| Repo | inv | albion | data | personal_website | skills_arsenal |
|------|:---:|:---:|:---:|:---:|:---:|
| **BettaFish** | MED | LOW | — | — | — |
| **dictionary-of-ai-coding** | HIGH | MED | MED | MED | **HIGH** |
| **GitNexus** | **HIGH** | MED | LOW | MED | LOW |
| **graphify** | **HIGH** | MED | MED | MED | MED |
| **lean-ctx** | **HIGH** | **HIGH** | **HIGH** | **HIGH** | LOW |
| **skills** (mattpocock) | **HIGH** | HIGH | HIGH | HIGH | **HIGH** |
| **three-geospatial** | — | — | LOW | **HIGH** | — |
| **maigret** | LOW | LOW | — | — | — |
| **MinerU** | LOW | — | LOW | MED | MED |

### Key Cross-Project Actions

1. **lean-ctx → ALL projects** (HIGH): Universal token savings. Install once, configure per-project. The 56 shell compression patterns cover git/npm/cargo/docker etc. across all 5 projects.

2. **skills (mattpocock) → skills_arsenal** (HIGH): 8 highest-value skills (diagnose, tdd, grill-with-docs, handoff, architecture, to-prd, to-issues, caveman) fill gaps in the arsenal's engineering section. Copy to `universal-skills/mattpocock-*`.

3. **dictionary-of-ai-coding → skills_arsenal** (HIGH): 62-term AI coding glossary as a standalone skill. Copy to `universal-skills/dictionary-of-ai-coding/`.

4. **skills (mattpocock) → personal_website** (HIGH): `/diagnose`, `/tdd`, `/improve-codebase-architecture`, `/grill-with-docs` directly applicable to Next.js development.

5. **three-geospatial → personal_website** (HIGH): Drop-in atmosphere/clouds/stars components for Three.js/R3F background. `npm install @takram/three-atmosphere @takram/three-clouds`.

---

## What the Next Session Must Do

### Phase 1: Setup (P0 — 60 min)

1. **graphify**: `uv add graphifyy` in investment_trying, run `graphify detect` on `src/`, review `GRAPH_REPORT.md`. This is the highest-ROI item — maps the entire codebase for all future AI sessions.

2. **lean-ctx**: Download binary from releases, configure as MCP server in kilo.json. Test on a backtest session — measure token savings with `--compress` mode.

3. **skills** (mattpocock): Copy 8 skills to `.kilo/skills/engineering/mattpocock/` in investment_trying. Copy same skills to `.claude/skills/engineering/mattpocock/` in personal_website. Copy all 17 to `skills_arsenal_for_publishing/universal-skills/mattpocock-*`.

### Phase 2: Documentation (P1 — 30 min)

4. Add Phase 19 to `progress_docs/plans/full.md` (same pattern as Phase 18).
5. Create `progress_docs/plans/19-new-repos-wave2.md` with detailed plan.
6. Update `MEMORY.md` with Phase 19 context and new resource table.
7. Add entries to `progress_docs/current.md`.

### Phase 3: Skill Enrichment (P1 — 20 min)

8. Copy `dictionary-of-ai-coding/` to `skills_arsenal_for_publishing/universal-skills/dictionary-of-ai-coding/`.
9. Update skills_arsenal `CATALOG.md`, `SUMMARY.md`, `README.md` with new counts (91 + 17 + 1 = ~109 skills).
10. Install `three-geospatial` packages in personal_website for atmosphere/cloud background exploration.

### Phase 4: Study (P2 — optional)
11. Install GitNexus, index investment_trying, evaluate impact analysis value.
12. Document BettaFish ForumEngine debate pattern in `docs/reference-bettafish-patterns.md`.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `progress_docs/plans/full.md` | Add Phase 19 section + status table entry |
| `progress_docs/plans/19-new-repos-wave2.md` | CREATE — detailed plan (model on `18-useful-repos-integration.md`) |
| `MEMORY.md` | Update Current Objective (Phase 19 ACTIVE), New Resources table |
| `progress_docs/current.md` | Add session log entry |
| `.kilo/skills/engineering/mattpocock/` | CREATE — 8 SKILL.md files |
| `.claude/skills/engineering/mattpocock/` (personal_website) | CREATE — 8 SKILL.md files |
| `skills_arsenal_for_publishing/universal-skills/mattpocock-*` | CREATE — 17 skill dirs |
| `skills_arsenal_for_publishing/universal-skills/dictionary-of-ai-coding/` | CREATE — 62 term files |
| `skills_arsenal_for_publishing/CATALOG.md` | UPDATE — new counts + entries |
| `skills_arsenal_for_publishing/SUMMARY.md` | UPDATE — new counts |
| `skills_arsenal_for_publishing/README.md` | UPDATE — new counts |
| `docs/glossary-ai-coding.md` (investment_trying) | CREATE — trading-adapted glossary |
| `docs/reference-bettafish-patterns.md` | CREATE — ForumEngine pattern doc |

---

## Memory Triggers for Next Session

- 9 new repos evaluated. P0: graphify (map the codebase), lean-ctx (reduce token costs), skills-mattpocock (8 new engineering skills).
- graphify's 71.5x token reduction is the single highest-ROI item — do this FIRST.
- lean-ctx is Rust-based, MCP stdio — configure in kilo.json, not pip.
- skills (mattpocock) fills gaps agent-skills doesn't cover: `/diagnose` (debugging), `/grill-with-docs` (domain glossary), `/handoff` (session compaction), `/improve-codebase-architecture` (debt reduction).
- mattpocock skills use `CONTEXT.md` for domain glossary — create one for investment_trying defining: barrier, label, purge, embargo, confluence, regime, walk-forward, triple-barrier.
- three-geospatial is personal_website ONLY — no relevance to trading.
- maigret and MinerU are LOW priority — scrape patterns already covered by Scrapling (Phase 18), PDF conversion already covered by paper2md.
- dictionary-of-ai-coding is a pure glossary — copy to skills_arsenal, optionally create trading-adapted version.
- BettaFish ForumEngine debate pattern is conceptually interesting for multi-agent signal fusion but complex to implement — document, don't build yet.
