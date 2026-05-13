---
project: investment_trying
created: 2026-05-11
source: Knowledge Graph cross-reference analysis of 48 research papers against 16 project modules
last_updated: 2026-05-11
priority_groups: 3 (KG1=HIGH, KG2=MEDIUM, KG3=LOW)
---

# Enhancement Plan: Knowledge Graph Insights

> Generated from `useful_resources/papers_md/KNOWLEDGE_GRAPH_INSIGHTS.md`
> 62 paper→module connections found across 48 papers

---

## Context

A systematic cross-reference of all 48 research papers in `useful_resources/papers_md/` against 16 project modules in `src/` revealed 7 major insight clusters. This plan converts those insights into actionable tasks.

**Top findings:**
1. Overfitting prevention is the #1 cross-cutting theme (27 papers) → partially addressed
2. Sentiment/NLP completely missing from project (22 papers) → no module
3. Reinforcement Learning absent (6 papers) → no module
4. Portfolio optimization theory outpaces implementation → partial
5. Event-driven trading lacks infrastructure (10 papers) → no module

---

## Task Priority Matrix

### KG1: HIGH Priority (implement immediately)

| ID | Task | Source Paper | Target Module | Effort |
|----|------|-------------|---------------|--------|
| KG-H1 | Training-history overfitting detection | 5520_using_the_training_history_to_ | `src/ml/` | Small |
| KG-H2 | Synthetic OOS comparison framework | Backtest Overfitting in the ML Era | `src/ml/` | Medium |
| KG-H3 | Sentiment scores as signal weight modifier | 4+ sentiment papers | `src/signals/` | Medium |
| KG-H4 | Event-driven pattern category | Building Calendar + Event-Based Trading | `src/patterns/` | Medium |

### KG2: MEDIUM Priority (plan next)

| ID | Task | Source Paper | Target Module | Effort |
|----|------|-------------|---------------|--------|
| KG-M1 | `src/rl/` module with trade execution env | OOM-RL + Adaptive RL + Deep Portfolio RL | `src/rl/` (NEW) | Large |
| KG-M2 | Kelly criterion allocator | Investing Is Compression | `src/portfolio/` | Small |
| KG-M3 | AutoAlpha factor mining pipeline | AutoAlpha | `src/ml/` | Large |
| KG-M4 | Circuit-based overfitting detection | Circuit-Based Intrinsic Methods | `src/ml/` | Medium |
| KG-M5 | Behavioral crash regime detection | Crash-based trading strategies | `src/risk/` | Medium |

### KG3: LOW Priority (defer until KG1/KG2 done)

| ID | Task | Source Paper | Target Module | Effort |
|----|------|-------------|---------------|--------|
| KG-L1 | Adversarial overfitting detection | advrisk_neurips2019 | `src/ml/` | Medium |
| KG-L2 | Financial event calendar database | 2 event papers | `src/data_ingestion/` | Large |
| KG-L3 | Defensive backtesting with time-reversal | Against a Universal Trading Strategy | `src/backtest/` | Small |

---

## Execution Order

```
KG1: Quick Wins (HIGH)
  ├── KG-H1: Training-history overfit detection → add to existing OverfittingValidator
  ├── KG-H2: Synthetic OOS framework → new class in src/ml/validation/
  ├── KG-H3: Sentiment weighting → extend EventWeightedAggregator
  └── KG-H4: Event-driven patterns → new category in src/patterns/

KG2: Structural Additions (MEDIUM)
  ├── KG-M1: src/rl/ module (new) — OpenAI Gym-like env for trading
  ├── KG-M2: Kelly allocator — new class in src/portfolio/
  ├── KG-M3: AutoAlpha pipeline — evolutionary factor mining in src/ml/
  ├── KG-M4: Circuit-based detection — perturbation analysis in src/ml/
  └── KG-M5: Crash regime — extend CrashFactorModel in src/risk/

KG3: Deep Enhancements (LOW)
  ├── KG-L1: Adversarial detection — generate adversarial price sequences
  ├── KG-L2: Event calendar DB — linked events from price spikes
  └── KG-L3: Time-reversal check — adapt EpistemicAutopsy in src/backtest/
```

---

## Module Impact Summary

| Module | New Tasks | Current State | After KG |
|--------|-----------|---------------|----------|
| `src/ml/` | 5 (KG-H1, H2, M3, M4, L1) | 108 classes | + overfit detection, factor mining, adversarial testing |
| `src/signals/` | 1 (KG-H3) | 11 classes | + sentiment weighting |
| `src/patterns/` | 1 (KG-H4) | 7 classes | + event-driven category |
| `src/rl/` | 1 (KG-M1) | **NEW** | RL env + trade execution agents |
| `src/portfolio/` | 1 (KG-M2) | 26 classes | + Kelly allocation |
| `src/risk/` | 1 (KG-M5) | 39 classes | + behavioral crash detection |
| `src/backtest/` | 1 (KG-L3) | 25 classes | + time-reversal validation |
| `src/data_ingestion/` | 1 (KG-L2) | 3 files | + event calendar DB |

---

## Dependencies

```
KG-H2 (Synthetic OOS) ← Prerequisite for KG-M3 (AutoAlpha) and KG-L3 (Time-reversal)
KG-M1 (RL module) ← Requires gymnasium or custom env
KG-H3 (Sentiment) ← Requires external API or pre-trained sentiment model
```

---

## References

- Full analysis: `useful_resources/papers_md/KNOWLEDGE_GRAPH_INSIGHTS.md`
- Skill: `.kilo/skills/knowledge-graph/SKILL.md`
- Command: `/knowledge-graph`
