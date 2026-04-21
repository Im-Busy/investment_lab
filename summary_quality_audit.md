# Automated Document Extraction Pipeline: Summary Quality Audit

**Date**: 2026-04-18  
**Scope**: Three recently regenerated academic paper summaries  
**Evaluator**: NLP Pipeline Optimization Review

---

## Section 1: Per-Summary Assessment

### 1.1 Algorithmic Trading and Quantitative Strategies (Velu, Hardy, Nehren, 2020)

**Strengths:**
- Correctly identifies the source as a textbook rather than an empirical paper, establishing appropriate expectations for "findings"
- Well-structured into conceptual frameworks (7 sections) and quantified observations (4 sub-sections)
- Accurate historical dates and regulatory milestones (decimalization 2001, Reg NMS 2005)
- Proper distinction between temporary vs. permanent market impact terminology
- Correct use of domain terminology: LOB, VWAP, TWAP, GARCH, VAR, cointegration, point-in-time data

**Weaknesses:**
- **Attribution ambiguity**: The "Quantified/Empirical Observations" section presents specific percentages (HFT ~60%, dark pool 30–40%, ~70% limit orders) without chapter or page citations. These are widely known industry statistics, not necessarily unique claims from this textbook. The pipeline does not distinguish between "book claims" and "general knowledge the summarizer injected."
- **Duplicate URL**: The GitHub repository URL appears twice at the end, indicating a duplication artifact in the extraction or formatting stage.
- **Emoji inconsistency**: Uses 📘, 📊, 🔑 headers that are not present in other summaries.
- **No chapter mapping**: For a textbook, the summary lacks any reference to which chapters contain which concepts, making it less useful for navigation.

---

### 1.2 Crash-based Quantitative Trading Strategies (Fang et al., 2022)

**Strengths:**
- Excellent reproduction of tabular data with clear quintile sorting methodology
- Behavioral finance foundations correctly attributed with author names and years (Avery & Zemsky, 1998; Kim et al., 2016)
- Robustness test section validates strategies across multiple market regimes (2015 crash, 2019 up-market)
- Limitations section is comprehensive (long-only constraint, short-term focus, parameter sensitivity)
- Correct use of "monotonic pattern" to describe quintile performance gradients

**Weaknesses:**
- **Suspicious over-precision**: Values like "56.000%", "3.296%", "-10.017%" with triple-decimal precision on percentages are atypical for academic finance tables. Annualized returns are rarely reported to thousandths of a percent. This suggests either over-extraction from OCR'd tables or potential hallucination.
- **Missing transaction cost context**: The reported returns do not indicate whether transaction costs are included. For any strategy paper, this is the single most critical omission. If costs are excluded, the returns are not actionable.
- **Negative return framing issue**: The 2015 robustness test shows ALL CMRS quintiles losing money (-23.0% to -13.4%), yet the summary frames this as "outperformance" without noting that the strategy still lost capital.
- **Missing sample size**: No mention of number of stocks, date range coverage, or data source beyond "CRSP."

---

### 1.3 Why Not 100% Equities? (Asness, JPM, Winter 1996)

**Strengths:**
- Historical data alignment with Ibbotson/SBBI records (S&P 500: 10.3% compound, 20.0% std dev, 1926–1993) is verifiable and accurate
- Clear exposition of the two-step portfolio construction principle (efficient set → risk choice)
- Levered 60/40 calculation (155% leverage to match 20% volatility) is internally consistent
- Proper distinction between "realized" and "expected" returns
- Includes practical workarounds for investors who cannot use explicit leverage

**Weaknesses:**
- **Borrowing cost assumption unchallenged**: The analysis assumes borrowing at the T-bill rate, which was not available to retail investors in 1996. The summary does not flag this as a limitation.
- **Suspiciously precise consistency figures**: 10-year rolling outperformance at 76.9% vs. 76.4% (0.5 pp difference) and 30-year at 100.0% for both portfolios. The 100.0% figures suggest rounding, but are presented as exact.
- **No margin requirement discussion**: Leveraged portfolios face margin maintenance rules that can force liquidation at precisely the wrong time. This is absent from both the paper summary and practical implications.
- **Publication sourcing**: "JPM Winter 1996" is ambiguous — could be JPMorgan's publication or an academic journal abbreviation. Needs disambiguation.

---

## Section 2: Prioritized Pipeline Improvements by Processing Stage

### Stage A: Data Ingestion / Document Parsing

| Priority | Issue | Impact | Action |
|----------|-------|--------|--------|
| **P0** | No source metadata extraction (chapter numbers, page numbers, table IDs) | High — cannot verify claims or navigate sources | Add metadata capture during ingestion; require field `source_references` in output schema |
| **P1** | URL duplication artifacts | Low — cosmetic but indicates processing bug | Add deduplication pass in post-processing |
| **P1** | Publication/source disambiguation | Medium — ambiguous citations reduce trust | Add publication validation against known journal/publisher registry |

### Stage B: NLP / Summarization Model

| Priority | Issue | Impact | Action |
|----------|-------|--------|--------|
| **P0** | Numerical over-precision on percentages (triple-decimal values) | High — suggests hallucination or bad OCR | Add numerical formatting guard: cap returns at 1–2 decimal places, validate against expected table precision |
| **P0** | Attribution mixing (book claims vs. summarizer-injected general knowledge) | High — user cannot distinguish source claims from pipeline additions | Add `attribution_confidence` scoring; require explicit labels like "General industry knowledge" vs. "Source states" |
| **P1** | Emotional/semantic framing errors (negative returns described as "outperformance") | Medium — misleading conclusions | Add sentiment-direction validation: if all values are negative, use "smaller losses" not "outperforms" |
| **P2** | Missing critical context (transaction costs, sample size, data source) | Medium — incomplete summaries | Add mandatory context checklist for strategy papers: costs included? sample size? data source? time period? |

### Stage C: Post-Processing / Validation

| Priority | Issue | Impact | Action |
|----------|-------|--------|--------|
| **P0** | No numerical consistency validation | High — hallucinated numbers go undetected | Add cross-check: quintile returns should approximately decompose to portfolio averages; leverage calculations should be mathematically verified |
| **P1** | No table extraction verification | Medium — OCR errors in tables propagate | Add table-to-text consistency check: sum of weights should equal 100%, quintile counts should match sample size |
| **P1** | No stylistic consistency enforcement | Low — inconsistent emoji/formatting across summaries | Add output template enforcement with configurable style guide |

### Stage D: Output Formatting

| Priority | Issue | Impact | Action |
|----------|-------|--------|--------|
| **P1** | Inconsistent emoji/style across summaries | Low — reduces professional quality | Standardize output template; remove auto-inserted decorative emoji unless user-configured |
| **P2** | No structured metadata header | Low — harder to programmatically consume | Add JSON/YAML metadata block at start: source type, date, confidence scores, key limitations |

---

## Section 3: Testable Recommendations for Immediate Implementation

### Recommendation 1: Numerical Precision Guardrail
**Stage**: B (NLP/Summarization)  
**Test**: Run pipeline on known papers with published tables. Verify that extracted percentages match source precision within ±0.1 percentage points. Flag any values with 3+ decimal places on returns/percentages for manual review.  
**Implementation**: Add post-processing regex or numeric formatter that caps percentage outputs at 1–2 decimal places unless the source explicitly requires higher precision (e.g., interest rate basis points).

### Recommendation 2: Attribution Tagging
**Stage**: B (NLP/Summarization)  
**Test**: For each quantitative claim in a summary, require an attribution tag: `["source_claim"]`, `["general_knowledge"]`, or `["inference"]`. Measure the proportion of untagged claims (target: 0%).  
**Implementation**: Modify the summarization prompt to require explicit sourcing language. Add a post-processing validator that flags claims without attribution markers.

### Recommendation 3: Context Completeness Checklist
**Stage**: C (Post-Processing)  
**Test**: Define a mandatory checklist for strategy paper summaries:
- [ ] Transaction costs: included/excluded/not mentioned
- [ ] Sample size: N = [value] or "not stated"
- [ ] Data source: [name] or "not stated"
- [ ] Time period: [start] to [end]
Run pipeline on 10 papers; measure checklist completion rate (target: 100%, with "not stated" as valid answer).  
**Implementation**: Add a structured validation layer that checks for these fields and inserts "not stated" when absent.

### Recommendation 4: Numerical Consistency Validator
**Stage**: C (Post-Processing)  
**Test**: For levered portfolio claims, verify: levered_return = (leverage_factor × portfolio_return) - ((leverage_factor - 1) × borrowing_rate). Flag any deviations >0.5% for review.  
**Implementation**: Add a calculator module that validates common financial relationships (leverage math, weight sums, quintile decomposition).

### Recommendation 5: Sentiment-Direction Validator
**Stage**: C (Post-Processing)  
**Test**: For any comparative claim ("A outperforms B"), verify numerically that A's metric > B's metric in the extracted data. Flag mismatches.  
**Implementation**: Add a simple rule-based layer that extracts comparative statements and validates them against the numerical data in the same summary.

### Recommendation 6: Source Type Detection and Schema Adaptation
**Stage**: B (NLP/Summarization)  
**Test**: The pipeline should auto-detect source type (textbook vs. empirical paper vs. commentary) and adjust the summary schema accordingly. Textbooks should require chapter references; papers should require methodology sections; commentaries should distinguish argument from evidence.  
**Implementation**: Add a source-type classifier at the ingestion stage. Route to different summarization templates based on classification. Measure classification accuracy on a labeled set (target: >90%).

---

## Summary of Priority Actions

| Priority | Action | Stage | Effort | Impact |
|----------|--------|-------|--------|--------|
| **P0** | Numerical precision guardrail | B | Low | High |
| **P0** | Attribution tagging | B | Medium | High |
| **P0** | Numerical consistency validator | C | Medium | High |
| **P1** | Context completeness checklist | C | Low | Medium |
| **P1** | Sentiment-direction validator | C | Low | Medium |
| **P1** | Source type detection + schema adaptation | B | Medium | Medium |
| **P2** | Style consistency enforcement | D | Low | Low |
| **P2** | Structured metadata header | D | Low | Low |
