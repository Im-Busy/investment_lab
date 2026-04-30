# Awesome-Auto-Research-Tools Evaluation for investment_trying Project

**Evaluation Date:** 2026-04-26
**Evaluator Purpose:** Determine which repos can help with quantitative investment research (NOT for full auto-research workflow)
**Source:** C:\Dev\research\Researcher_Workstation\Awesome-Auto-Research-Tools\README.md

---

## Evaluation Criteria

1. **Quant Finance Relevance:** Can it directly help with factor generation, backtesting, or strategy optimization?
2. **ML Research Utility:** Can it automate ML experimentation, hyperparameter tuning, or model improvement?
3. **Paper/Lit Review:** Can it help review financial ML papers or generate research summaries?
4. **Lightweight:** Is it easy to integrate without massive dependencies?

---

## TIER 1: KEEP — HIGH VALUE (Directly Applicable)

### 1. RD-Agent (microsoft) ⭐⭐⭐⭐⭐
- **Location:** `C:\Dev\research\Researcher_Workstation\RD-Agent\`
- **Why Keep:** THE most relevant tool. Has an explicit **Quantitative Finance scenario** with Qlib integration.
  - Automated factor/model evolution for quant trading
  - Coordinated factor-model co-optimization
  - ~2× ARR than benchmark factor libraries with 70% fewer factors
  - Paper-to-code implementation capability
  - Leads MLE-Bench as top ML engineering agent
- **Use Cases:**
  - Automated alpha factor generation
  - Iterative factor optimization
  - Model architecture evolution
  - Paper reproduction to code
- **Dependencies:** LiteLLM, Docker, Qlib
- **Priority:** ⬛ IMPLEMENT FIRST

### 2. AIDE (WecoAI) ⭐⭐⭐⭐
- **Location:** `C:\Dev\research\Researcher_Workstation\AIDE\`
- **Why Keep:** LLM-guided agentic tree search for ML code optimization.
  - Writes, evaluates, and improves ML code autonomously
  - 4× more Kaggle medals than best linear agent
  - Natural-language task specification
  - Model-neutral (OpenAI, Anthropic, Gemini, local LLMs)
- **Use Cases:**
  - Automated pattern detection algorithm improvement
  - Hyperparameter optimization via tree search
  - Strategy code generation and refinement
  - ML pipeline prototyping
- **Dependencies:** OpenAI/Anthropic API
- **Priority:** ⬛ HIGH

---

## TIER 2: KEEP — MEDIUM VALUE (Research Support)

### 3. GPT-Researcher ⭐⭐⭐
- **Location:** `C:\Dev\research\Researcher_Workstation\GPT-Researcher\`
- **Why Keep:** Deep web research for market intelligence and stock analysis.
  - Generates detailed research reports with citations
  - Can research stocks, markets, trends
  - Supports multiple LLM providers
  - Export to PDF/Docx/Markdown
- **Use Cases:**
  - Market research reports
  - Company/stock analysis
  - Trend spotting and sector research
  - Competitor strategy analysis
- **Dependencies:** LLM API + search API (Tavily, SearXNG, etc.)
- **Priority:** 🟡 MEDIUM

### 4. PaperQA2 (Future-House) ⭐⭐⭐
- **Location:** `C:\Dev\research\Researcher_Workstation\PaperQA2\`
- **Why Keep:** High-accuracy RAG for scientific/financial papers. Published at ICLR.
  - Answers questions over PDFs with full-text retrieval
  - Already have papers in `useful_resources/papers/`
  - Complements existing paper2md summarization workflow
  - Supports local LLMs via Ollama
- **Use Cases:**
  - Q&A over financial ML papers
  - Literature review for quant strategies
  - Paper contradiction detection
  - Research question answering
- **Dependencies:** LiteLLM
- **Priority:** 🟡 MEDIUM

### 5. STORM (Stanford) ⭐⭐⭐
- **Location:** `C:\Dev\research\Researcher_Workstation\STORM\`
- **Why Keep:** Generates Wikipedia-like articles with citations from web search.
  - Perspective-guided question asking
  - Co-STORM for human-AI collaborative research
  - Creates structured knowledge base articles
  - Modular via DSPy/LiteLLM
- **Use Cases:**
  - Strategy documentation generation
  - Knowledge base for trading concepts
  - Research topic exploration
  - Automated topic overview generation
- **Dependencies:** LiteLLM + search API
- **Priority:** 🟡 MEDIUM (nice-to-have)

---

## TIER 3: SKIP — LOW/NONE VALUE (Not Applicable)

### Not Downloaded — Skip These

| Project | Reason to Skip |
|---------|---------------|
| **autoresearch (karpathy)** | Too specific to LLM/nanochat training experiments. Not adaptable to finance. |
| **AI-Scientist / v2** | Academic ML research (DL architectures). Not applicable to quant trading. |
| **AutoResearchClaw** | Full academic paper pipeline (idea → LaTeX). Overkill, not finance-focused. |
| **ARIS** | Claude Code skills for academic ML research. Not finance-relevant. |
| **DeepScientist** | Local-first academic research studio. Bayesian optimization for academic papers. |
| **Agent Laboratory** | Multi-agent for academic literature → report. Too academic. |
| **AI-Researcher** | NeurIPS 2025 academic research system. Not applicable. |
| **claude-scholar** | Semi-automated academic research (Zotero/Obsidian). Not finance. |
| **Biomni** | Stanford biomedical research agent. Wrong domain. |
| **DATAGEN** | LangChain/LangGraph multi-agent for general research. Too generic. |
| **Idea2Paper** | Multi-agent for research proposal generation. Academic focus. |
| **InternAgent** | ML research for physics, biology, earth science. Wrong domain. |
| **MLE-agent** | Auto-debugging + arxiv integration. Useful but AIDE/RD-Agent cover more. |
| **Tongyi DeepResearch** | Alibaba's LLM for info-seeking. Generic research tool. |
| **Open Deep Research** | LangChain deep research. Generic, overlaps with GPT-Researcher. |
| **Auto-Deep-Research** | HKU's alternative to OpenAI Deep Research. Generic. |
| **DeepResearchAgent** | Skywork's hierarchical multi-agent. Generic research. |
| **OpenResearcher** | Requires training 30B model. Too heavy infrastructure. |
| **DeerFlow** | ByteDance's SuperAgent for code gen. Too generic. |
| **ChatPaper / ChatReviewer** | arXiv paper summary tools. Too narrow; paper2md already covers this. |
| **AutoGPT** | General autonomous agent. Not research-optimized. |
| **OpenHands** | AI software dev (SWE-Bench). Too general-purpose. |
| **Aider** | AI pair programming CLI. Useful but not auto-research. |
| **SWE-agent** | GitHub issue fixer. Not applicable. |
| **PaperBanana** | Academic illustration generator. Not relevant. |
| **scientific-agent-skills** | Bioinformatics/drug discovery skills. Wrong domain. |
| **AI-Research-SKILLs** | 86 skills for AI research. Many useful patterns but too broad. |

### Downloaded But Recommend SKIP

| Project | Reason to Skip | Location to Delete |
|---------|---------------|-------------------|
| **OpenScholar** | Heavy: needs custom trained Llama-3.1-OpenScholar-8B model, Semantic Scholar API, offline retrieval data. PaperQA2 covers similar ground with much lighter setup. | `C:\Dev\research\Researcher_Workstation\OpenScholar\` |

---

## Summary: Actions to Take

### KEEP (5 repos)

```
✅ RD-Agent          -> High priority: integrate for automated factor/model optimization
✅ AIDE              -> High priority: use for ML code/tree-search optimization
✅ GPT-Researcher    -> Medium: web research for market intelligence
✅ PaperQA2          -> Medium: RAG Q&A over financial papers
✅ STORM             -> Medium: knowledge base article generation
```

### DELETE (1 repo)

```
❌ OpenScholar       -> Too heavy, PaperQA2 is better fit
```

### NEVER DOWNLOAD (26 repos)

All other repos listed in the README are either:
- Academic paper writing pipelines (not finance)
- General-purpose coding agents (not research-specific)
- Wrong domain (biomedical, physics, etc.)
- Too heavy infrastructure requirements
- Redundant with already-selected tools

---

## Integration Priority Order

1. **RD-Agent** (Week 1-2): Set up with Qlib, connect to our market data, run factor evolution
2. **AIDE** (Week 2-3): Use for automated strategy code optimization
3. **PaperQA2** (Week 3): Index existing papers in `useful_resources/papers/`, enable Q&A
4. **GPT-Researcher** (Week 4): Set up for automated market research reports
5. **STORM** (Week 4+): Use for knowledge base generation

---

## Notes

- All kept repos support LiteLLM, so they can share the same LLM API configuration (OpenRouter)
- RD-Agent already has Qlib integration which aligns with quant research needs
- AIDE and RD-Agent both excel on MLE-Bench, proven for ML engineering tasks
- PaperQA2 and GPT-Researcher are lightweight additions with clear use cases
