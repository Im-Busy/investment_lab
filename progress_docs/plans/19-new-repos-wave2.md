# Phase 19 — New Repos Wave 2 Integration

**Date:** 2026-05-16
**Source:** 9 new repos cloned to `C:\Dev\useful_repos` beyond the 7 from Phase 18.
**Status:** 🔄 ACTIVE

---

## Repos Evaluated

| # | Repo | What It Does | Language | Relevance |
|---|------|-------------|----------|:---:|
| 1 | **graphify** | Project knowledge graph builder — 3-pass extraction, community detection, 71.5x fewer tokens | Python | HIGH |
| 2 | **lean-ctx** | Context runtime — 60-95% token reduction via compression, caching, property graphs, 51 MCP tools | Rust | HIGH |
| 3 | **skills** (mattpocock) | 17 AI agent skills (diagnose, TDD, grill-with-docs, handoff, architecture, prototype, etc.) | Markdown | HIGH |
| 4 | **dictionary-of-ai-coding** | 62 AI coding terms glossary in 7-section curriculum | Markdown | HIGH |
| 5 | **GitNexus** | Code intelligence knowledge graph — 16 MCP tools for AI agents | TypeScript | HIGH |
| 6 | **BettaFish** | Multi-agent public opinion analysis — 4 LLM agents + Forum debate engine | Python/Flask | MED |
| 7 | **three-geospatial** | Three.js/R3F geospatial rendering — atmosphere, clouds, globe, stars | TypeScript/React | NONE |
| 8 | **maigret** | OSINT username reconnaissance — 3000+ sites, report generation | Python | LOW |
| 9 | **MinerU** | Document parsing engine — PDF/DOCX→Markdown, 109-language OCR | Python/ML | LOW |

---

## Priority Ranking

### P0 — Direct Code Reuse / Immediate Value

#### R19a: graphify — Build Project Knowledge Graph

**What:** Map entire codebase into a queryable knowledge graph.
**Value:** 71.5x token reduction for every future AI session. Maps 45+ pattern detectors, ML pipeline, backtest engine, cross-module dependencies.
**Effort:** 30 min setup, 0 LOC.

**Steps:**
1. `uv add graphifyy` in investment_trying
2. `graphify detect` on `src/`
3. Add `docs/` and `useful_resources/papers_md/` to graph
4. Review `GRAPH_REPORT.md`
5. Optional: integrate `graphify report` output into `.kilo/skills/` for agent context

**Files Created:** `GRAPH_REPORT.md`, `.graphify/`

#### R19b: lean-ctx — Install Context Compression MCP Server

**What:** 60-95% token savings via compression + caching + property graphs.
**Value:** Universal across ALL projects. Dramatically reduces LLM costs on long sessions.
**Effort:** 30 min setup, 0 LOC.

**Steps:**
1. Download Rust binary from `lean-ctx` releases
2. Configure as MCP server in `kilo.json`
3. Test on a backtest session with `--compress` mode
4. Benchmark token savings

**Config:** `kilo.json` MCP stdio entry.

#### R19c: skills (mattpocock) — Port 8 Engineering Skills

**What:** Copy 8 highest-value skills to `.kilo/skills/engineering/mattpocock/`.
**Value:** Fills gaps agent-skills doesn't cover: debugging, architecture, domain language, session compaction.
**Effort:** 0 LOC (markdown only).

**Skills to port:**

| Skill | Source Dir | What It Does |
|-------|-----------|-------------|
| **diagnose** | `engineering/diagnose/` | 6-phase systematic debugging |
| **tdd** | `engineering/tdd/` | Red-green-refactor with AI |
| **grill-with-docs** | `engineering/grill-with-docs/` | Domain glossary-driven development |
| **handoff** | `productivity/handoff/` | Session compaction for next agent |
| **improve-codebase-architecture** | `engineering/improve-codebase-architecture/` | Architecture debt reduction |
| **to-prd** | `engineering/to-prd/` | Feature request → structured PRD |
| **to-issues** | `engineering/to-issues/` | PRD → GitHub issues |
| **caveman** | `productivity/caveman/` | Token-efficient prompting |

**Destinations:**
- `.kilo/skills/engineering/mattpocock/` (8 SKILL.md files)
- `C:\Dev\projects\personal_website\.claude\skills\engineering\mattpocock/` (same 8)
- `C:\Dev\projects\skills_arsenal_for_publishing\universal-skills\mattpocock-*` (all 17 skills)

### P1 — Reference + Skill Enrichment

#### R19d: dictionary-of-ai-coding — Glossary Integration

**What:** 62-term AI coding glossary as skill + trading adaptations.
**Effort:** 20 min.

**Steps:**
1. Copy `dictionary/` to `skills_arsenal_for_publishing/universal-skills/dictionary-of-ai-coding/`
2. Create `docs/glossary-ai-coding.md` with trading-specific adaptations
3. Update skills_arsenal CATALOG.md, SUMMARY.md, README.md with new counts

#### R19e: GitNexus — Code Intelligence Graph

**What:** Index `src/` for precomputed impact analysis.
**Effort:** 15 min setup.

**Steps:**
1. `npx gitnexus` → index `src/`
2. Evaluate MCP tools for symbol search, call chains, blast radius
3. Determine if value justifies permanent integration

#### R19f: BettaFish — ForumEngine Debate Pattern

**What:** Study multi-agent debate for signal confluence.
**Effort:** Study only, ~200 loc potential.

**Steps:**
1. Document ForumEngine architecture in `docs/reference-bettafish-patterns.md`
2. Identify applicable patterns for signal fusion disagreement resolution
3. Defer implementation until multi-agent signal fusion is prioritized

### P2 — Low Priority / Deferred

| # | Repo | Notes |
|---|------|-------|
| R19g | three-geospatial | Only for personal_website. `npm install @takram/three-atmosphere @takram/three-clouds` |
| R19h | maigret | Async queue executor, layered settings patterns. Low urgency. |
| R19i | MinerU | PDF→MD covered by paper2md. Deferred. |

---

## Cross-Project Actions

1. **lean-ctx → ALL projects** (HIGH): Universal token savings.
2. **skills (mattpocock) → skills_arsenal** (HIGH): 8 skills fill engineering gaps.
3. **dictionary-of-ai-coding → skills_arsenal** (HIGH): 62-term glossary.
4. **skills (mattpocock) → personal_website** (HIGH): dev lifecycle skills.
5. **three-geospatial → personal_website** (HIGH): atmosphere/clouds components.

---

## Execution Order

```
Phase 1: Setup (P0 — 60 min)
  R19a (graphify) → R19c (skills) → R19b (lean-ctx)

Phase 2: Documentation (P1 — 30 min)
  full.md update → 19-new-repos-wave2.md → MEMORY.md → current.md

Phase 3: Skill Enrichment (P1 — 20 min)
  R19d (dictionary) → R19e (GitNexus) → R19f (BettaFish)

Phase 4: Cross-Project (P2 — optional)
  lean-ctx per-project config → personal_website skills + three-geospatial
```

---

## Files Checklist

| File | Action | Status |
|------|--------|--------|
| `progress_docs/plans/full.md` | Add Phase 19 section | ✅ |
| `progress_docs/plans/19-new-repos-wave2.md` | CREATE | ✅ |
| `MEMORY.md` | Update Current Objective, New Resources | ⏳ |
| `progress_docs/current.md` | Add session log entry | ⏳ |
| `.kilo/skills/engineering/mattpocock/` | CREATE — 8 SKILL.md files | ⏳ |
| `docs/glossary-ai-coding.md` | CREATE | ⏳ |
| `docs/reference-bettafish-patterns.md` | CREATE (deferred) | ⏸️ |
| `skills_arsenal_for_publishing/universal-skills/dictionary-of-ai-coding/` | CREATE — 62 term files | ⏳ |
| `skills_arsenal_for_publishing/CATALOG.md` | UPDATE | ⏳ |
