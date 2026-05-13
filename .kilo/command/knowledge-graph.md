---
description: Cross-reference all research papers with project modules, find insights, gaps, and actionable recommendations. Auto-detects new papers first.
---

# Knowledge Graph — Paper Cross-Reference Analysis

Run the full knowledge graph pipeline: detect new papers → convert to markdown → cross-reference with project modules → generate insights → update plans.

## Usage

```
/knowledge-graph scan     # Detect new/unprocessed papers only (no MDs made)
/knowledge-graph convert  # Convert new PDFs to markdown
/knowledge-graph analyze  # Run cross-reference analysis → update KNOWLEDGE_GRAPH_INSIGHTS.md
/knowledge-graph full     # Scan → Convert → Analyze → Update Plans (full pipeline)
/knowledge-graph plan     # Add current insights as tasks to project plans
```

## Steps (full pipeline)

1. **Scan**: Compare `useful_resources/papers/*.pdf` against `useful_resources/papers_md/*.md`. List PDFs without matching MD files.

2. **Convert**: For each new PDF, extract text using `pdfplumber` and save as `.md` in `papers_md/`. Rename PDF and MD files to their actual paper titles (from PDF metadata or first-page text). Clean up `.txt` files, stray PDFs, and duplicate directories. Fix word-merging artifacts.

3. **Analyze**: Run keyword/topic extraction across all papers. Cross-reference with 16 project modules in `src/`. Identify connections, gaps, and opportunities. Write full report to `useful_resources/papers_md/KNOWLEDGE_GRAPH_INSIGHTS.md`.

4. **Plan**: Add HIGH and MEDIUM priority recommendations from the insights into `progress_docs/plans/full.md`. Log the analysis run in `progress_docs/current.md`.

## Topic Categories

The analysis categorizes every paper into these 10 topics:

1. Overfitting & Data Leakage
2. Reinforcement Learning
3. Sentiment Analysis & NLP
4. Portfolio Optimization
5. Machine Learning Methods
6. Backtesting & Validation
7. Risk Management
8. Event-Driven Trading
9. Factor Models & Alpha
10. Time Series Forecasting

## Output

- `useful_resources/papers_md/KNOWLEDGE_GRAPH_INSIGHTS.md` — Full cross-reference report
- `progress_docs/plans/full.md` — Updated with new action items
- `progress_docs/current.md` — Session log entry
