# Handpicked Auto-Research Tools Evaluation (Round 2)

**Evaluation Date:** 2026-04-26
**Scope:** 9 repos handpicked by user, evaluated for investment_trying project
**Previous eval:** `plans/auto_research_tools_evaluation.md`

---

## Repo-by-Repo Evaluation

### 1. DeepScientist (ResearAI)
**URL:** https://github.com/ResearAI/DeepScientist (2.4k stars, ICLR 2026 Top 10)

**What it does:** Local-first autonomous research studio with Findings Memory + Bayesian optimization. Runs baseline reproduction → branched experiments → LaTeX paper drafts on your machine. Powered by Codex/Claude Code/OpenCode as runners.

**Relevant tech for us:**
- **Findings Memory:** Persistent cross-run research memory that preserves failed paths and winning paths — conceptually valuable for our pattern backtesting (remember what worked, what didn't)
- **Research Map:** Visual research structure via branches/worktrees — good mental model for exploring multiple strategy variants
- **Bayesian optimization for hypothesis generation:** Could inspire our own hypothesis-guided pattern exploration
- **One repo per quest:** Clean project isolation per research question

**Not applicable:**
- Academic paper generation pipeline — we don't need to write papers
- Connector surfaces (WeChat, QQ, Telegram) — research infrastructure, not trading
- Paper reproduction baselines — academic ML focus, not quantitative finance

**Verdict:** **READ FOR CONCEPTS, DON'T INSTALL.** Its architectural patterns (Findings Memory, Research Map, Bayesian hypothesis selection) can inspire improvements to our existing backtest/strategy exploration workflow. The core system is not finance-oriented. The `Awesome-AI-Scientist` related repo by same authors is worth browsing for landscape awareness.

---

### 2. Idea2Paper (AgentAlphaAGI)
**URL:** https://github.com/AgentAlphaAGI/Idea2Paper (1.3k stars)

**What it does:** End-to-end research agent framework. Core submodule Idea2Story transforms raw ideas into structured scientific narratives using ICLR knowledge graph, three-path retrieval (Idea/Domain/Paper), and anchored multi-agent review.

**Relevant tech for us:**
- **Knowledge Graph construction from papers:** The Paper-KG-Pipeline could be adapted to build a KG from our financial/quant papers (currently in `useful_resources/papers/`)
- **Anchored multi-agent review:** Comparing strategies/papers against known anchors with deterministic 1-10 scoring — this concept could be adapted to evaluate our pattern detectors
- **Three-path retrieval (Idea/Domain/Paper):** Interesting retrieval architecture that could inspire our own literature search

**Not applicable:**
- Scientific narrative generation — we don't need to write academic papers
- ICLR-specific dataset — would need to rebuild KG with quant finance papers
- Story2Proposal for structured paper writing — not relevant

**Verdict:** **READ FOR CONCEPTS ONLY.** The knowledge graph pipeline and anchored review system are interesting architectural patterns. But for paper Q&A/review, PaperQA2 and OpenScholar are more mature and directly usable. The KG builder could be studied if we want to build a quant-finance knowledge graph.

---

### 3. GPT-Researcher (assafelovic) ✅ ALREADY CLONED
**URL:** https://github.com/assafelovic/gpt-researcher (already in Researcher_Workstation)

**Verdict:** **KEEP** (from previous eval). Deep web research for market intelligence. Already downloaded.

---

### 4. DeepResearchAgent (SkyworkAI)
**URL:** https://github.com/SkyworkAI/DeepResearchAgent (3.3k stars)

**What it does:** Hierarchical multi-agent system with Autogenesis self-evolution protocol. Top-level planning agent coordinates specialized lower-level agents. RSPL (Resource Substrate) for versioned prompts/tools/environments, SEPL (Self Evolution) for propose/assess/commit improvement loops.

**Relevant tech for us:**
- **Autogenesis protocol (RSPL + SEPL):** Self-evolving agent tools with versioning and rollback — highly transferable concept for our agent skill management
- **Composable architecture:** agents/tools/envs/memory/optimizers as replaceable modules — good design pattern
- **Tracing & versioning:** Structured traces for analyzing failures and improvement steps
- **Optimizers:** Reflection-based self-improvement (GRPO, Reinforce++) — ML techniques but conceptually interesting

**Not applicable:**
- Built on AutoGen (Microsoft's framework) — heavy dependency stack
- General-purpose research agent — not finance-specific
- Empirical studies focus on GAIA benchmark (general research QA)

**Verdict:** **READ FOR ARCHITECTURE CONCEPTS.** The Autogenesis protocol (RSPL/SEPL) is the key takeaway — versioned, auditable tool evolution with rollback. This directly applies to managing our agent skills under C:\Dev. Don't install as a full system.

---

### 5. OpenScholar (AkariAsai)
**URL:** https://github.com/AkariAsai/OpenScholar (Nature paper)

**What it does:** Retrieval-augmented LM searching 45M open-access papers. Published in Nature. Outperforms PaperQA2 and Perplexity Pro on ScholarQABench.

**Your point:** OpenScholar outperforms PaperQA2 — correct, by 5.5-7% on ScholarQABench.

**Why I previously said skip (and why you're right to question it):**

The original reasoning was:
1. Heavy setup: needs custom trained Llama-3.1-OpenScholar-8B model
2. Offline retrieval data required
3. Semantic Scholar API dependency
4. PaperQA2 "covers similar ground with lighter setup"

**Why OpenScholar IS worth using despite those concerns:**
- **Correctness advantage:** 5.5-7% better than PaperQA2 is significant for answering questions over scientific literature
- **Open model:** The 8B model is available on HuggingFace, can run on local GPU
- **Published in Nature:** Highest-quality research, most rigorous evaluation
- **ScholarQABench:** 2,967 queries, specifically designed for scientific paper QA
- **Web search fallback:** You.com API as additional retrieval source

**When OpenScholar wins over PaperQA2:**
- Multi-paper synthesis questions ("How do different papers approach X?")
- Questions requiring broad literature coverage (45M papers)
- When you need citations grounded in the actual corpus, not retrieved docs
- Complex technical questions where PaperQA2 might hallucinate

**When PaperQA2 still wins:**
- Local document Q&A (your own PDFs, code files)
- No GPU available (OpenScholar needs 8B model inference)
- Quick setup needed (PaperQA2 is pip install + go)
- Non-scientific documents (financial reports, strategy logs)

**Revised verdict:** **INSTALL — KEEP BOTH.** They're complementary, not redundant. Use OpenScholar for broad scientific literature synthesis, PaperQA2 for local document Q&A. Both serve different use cases. The performance gap is real and meaningful.

---

### 6. PaperBanana (dwzhu-pku)
**URL:** https://github.com/dwzhu-pku/PaperBanana (6k stars)

**What it does:** Multi-agent framework for automated academic illustration generation. Retriever → Planner → Stylist → Visualizer → Critic pipeline producing publication-quality diagrams.

**Not applicable:**
- Generates academic paper illustrations/figures — completely orthogonal to our needs
- Needs image generation model API (Gemini/DALL-E)
- Reference set built for CS papers
- Zero overlap with quantitative finance

**Verdict:** **SKIP completely.** No technology or concept worth transferring to investment research.

---

### 7. MLE-Agent (MLSysOps)
**URL:** https://github.com/MLSysOps/MLE-agent

**What it does:** ML engineering companion integrating arxiv + Papers with Code for research plans. Autonomous baselines, smart debugging, file system organization. The successor concept evolved into MLE-STAR (Google's ML engineering agent with web search + targeted code refinement).

**Relevant tech for us:**
- **Autonomous baseline creation:** Auto-generates ML baselines — conceptually relevant for our strategy baselines (buy-and-hold, SPY, etc.)
- **Arxiv + Papers with Code integration:** Literature-to-code research assistance
- **Smart debugging:** Auto-debugging ML pipelines — useful for debugging our pattern detectors
- **File system integration:** Keeps project structure organized

**Not applicable:**
- Focuses on ML model engineering, not quantitative trading
- Kaggle competition focus rather than backtesting
- Less sophisticated than AIDE or RD-Agent for our use cases

**Verdict:** **READ FOR REFERENCE.** AIDE and RD-Agent cover the ML engineering space more thoroughly. MLE-Agent's arxiv integration and baseline auto-creation patterns are worth studying, but the repo itself doesn't need to be installed.

---

### 8. scientific-agent-skills (K-Dense-AI)
**URL:** https://github.com/K-Dense-AI/scientific-agent-skills (19.4k stars)

**What it does:** 133 ready-to-use Agent Skills across 15+ scientific domains following the open Agent Skills standard (agentskills.io). Works with Claude Code, Codex, Gemini CLI, Cursor.

**RELEVANT SKILLS for investment_trying:**
- **scikit-learn:** ML model skill (used in our pattern detectors)
- **PyTorch Lightning:** Deep learning training skill
- **TimesFM:** Google's zero-shot time series forecasting — directly relevant to financial forecasting
- **Time series analysis (aeon):** Time series workflows
- **scikit-survival:** Survival analysis (could apply to trade duration analysis)
- **PyMC:** Bayesian methods — relevant for Bayesian optimization of strategy parameters
- **SHAP:** Model interpretability — useful for understanding which pattern features drive predictions
- **Optuna (if included):** Hyperparameter optimization
- **Database Lookup:** 78+ databases including **FRED** (economic data), **SEC EDGAR** (filings) — directly useful for finance
- **US Treasury Fiscal Data:** Economic indicator access
- **Scientific Writing/Peer Review:** Could help with strategy documentation
- **Statistical Analysis:** Hypothesis testing, power analysis
- **Network Analysis (NetworkX):** For market network analysis

**Question: Suitable for C:\Dev agents?**

**YES, SELECTIVE INSTALLATION recommended.** Here's the approach:

For **Kilo** (this CLI tool): Kilo uses a SKILL.md format in `.kilo/skills/`. The scientific-agent-skills repo uses the same Agent Skills standard. However, 90% of the 133 skills are bioinformatics/drug-discovery/genomics skills irrelevant to our work. Pick only:

**Directly useful for investment_trying:**
1. `scikit-learn` - our ML foundation
2. `PyTorch Lightning` - deep learning training
3. `TimesFM` - time series forecasting
4. `PyMC` - Bayesian optimization
5. `SHAP` - model interpretability
6. `FRED` (via database-lookup) - economic data
7. `SEC EDGAR` (via database-lookup) - SEC filings
8. `Statistical Analysis` - hypothesis testing
9. `Time Series Analysis` - financial time series

**Not needed:** All bioinformatics, genomics, drug discovery, clinical research, materials science, physics/astronomy skills.

**INSTALL METHOD:** Selective via `gh skill install` or copy only relevant `SKILL.md` files to `.kilo/skills/`.

**Verdict:** **SELECTIVE INSTALL — PICK 9 SKILLS.** Not the full repo, just the ML/time-series/data-analysis subset.

---

### 9. AI-Research-SKILLs (Orchestra-Research)
**URL:** https://github.com/Orchestra-Research/AI-research-SKILLs (7.4k stars)

**What it does:** 87 skills for AI agents to autonomously conduct ML research. Covers the full AI research lifecycle: training, evaluation, deployment, RAG, agents, paper writing.

**RELEVANT SKILLS for investment_trying:**
- **Autoresearch:** Autonomous research orchestration layer — could coordinate our own research loops
- **Fine-Tuning:** Axolotl, PEFT, Unsloth — relevant if we fine-tune models for signal prediction
- **Distributed Training:** DeepSpeed, FSDP, Megatron — useful for large model training
- **Optimization:** Flash Attention, quantization (GPTQ, AWQ, GGUF) — model optimization
- **Evaluation:** lm-evaluation-harness — model evaluation patterns
- **RAG:** Chroma, FAISS, Qdrant — relevant for our paper/document retrieval
- **Agents:** LangChain, LlamaIndex, CrewAI — agent framework docs
- **MLOps:** W&B, MLflow, TensorBoard — experiment tracking we could use
- **ML Paper Writing:** LaTeX templates — for strategy documentation
- **Research Ideation:** Brainstorming frameworks
- **Multimodal:** CLIP, Whisper — not relevant

**Question: Suitable for C:\Dev agents?**

**YES, SELECTIVE INSTALLATION recommended.** The coverage is broader and more ML/LLM-engineering focused than K-Dense's scientific-agent-skills. For our quantitative research workflow:

**Most useful categories:**
1. **RAG skills** (FAISS, Chroma, Qdrant) — our existing paper search/indexing
2. **MLOps skills** (W&B, MLflow) — experiment tracking for backtests
3. **Agents skills** (LangChain, LlamaIndex) — reference docs during agent-assisted development
4. **Fine-Tuning skills** (PEFT, Axolotl) — if we fine-tune models
5. **Evaluation skills** — backtest evaluation methodology
6. **Research Ideation** — strategy brainstorming frameworks
7. **Autoresearch** — research orchestration patterns

**Not needed:** Model architecture (NanoGPT, Mamba, RWKV), tokenization, mechanistic interpretability, safety & alignment, multimodal, distributed training (unless we train LLMs).

**INSTALL METHOD:** Selective install relevant categories only.

**Verdict:** **SELECTIVE INSTALL — FOCUS ON RAG + MLOps + IDEATION.** More ML-engineering focused than K-Dense's repo. Complementary to the ~9 skills from K-Dense.

---

## OpenScholar vs PaperQA2 — Reassessment

You're correct that OpenScholar outperforms PaperQA2. My original dismissal was based on setup complexity, but the performance gap is meaningful. Here's the revised comparison:

| Dimension | OpenScholar | PaperQA2 |
|-----------|------------|----------|
| **Correctness** | **Higher** (+5.5% on ScholarQABench) | Good, but not best-in-class |
| **Setup** | Heavy (8B model + offline retrieval data + S2 API) | Light (pip install + local PDF folder) |
| **Scope** | 45M open-access papers | Your local documents only |
| **Use case** | Broad scientific literature synthesis | Focused Q&A on your own papers |
| **GPU needed** | Yes (8B model inference) | No (uses API or local LLM) |
| **Hallucination** | Lower (published in Nature, rigorous eval) | Some (RAG-based, depends on retrieval) |

**Recommendation: Install both. They are NOT redundant.**
- OpenScholar for: "What does the literature say about X?" (broad synthesis)
- PaperQA2 for: "What does MY paper collection say about Y?" (local Q&A)

Both can share the same paper corpus index with different retrieval strategies.

---

## Agent Skills — C:\Dev Integration Strategy

### For C:\Dev\projects\investment_trying (Kilo agent)

**From K-Dense scientific-agent-skills (19.4k stars):**
| Skill | Priority | Reason |
|-------|----------|--------|
| scikit-learn | HIGH | Core ML library for pattern detectors |
| TimesFM | HIGH | Google time series forecasting |
| PyMC | MEDIUM | Bayesian optimization |
| SHAP | MEDIUM | Model interpretability |
| Statistical Analysis | MEDIUM | Hypothesis testing |
| FRED/SEC EDGAR (database-lookup) | HIGH | Economic data access |
| Time Series Analysis | HIGH | Financial ts workflows |
| PyTorch Lightning | LOW | If we do deep learning |

**From Orchestra AI-Research-SKILLs (7.4k stars):**
| Skill | Priority | Reason |
|-------|----------|--------|
| FAISS | HIGH | Vector search for paper retrieval |
| Qdrant | MEDIUM | Advanced vector DB |
| MLflow | MEDIUM | Experiment tracking |
| W&B | LOW | Already covered by MLflow |
| Research Ideation | MEDIUM | Strategy brainstorming |
| Autoresearch | MEDIUM | Research orchestration patterns |
| PEFT | LOW | Only if fine-tuning models |

### Installation approach:
```
C:\Dev\projects\investment_trying\.kilo\skills\
├── scikit-learn\
│   └── SKILL.md          (from K-Dense)
├── time-series\
│   └── SKILL.md          (from K-Dense)
├── timesfm\
│   └── SKILL.md          (from K-Dense)
├── pymc\
│   └── SKILL.md          (from K-Dense)
├── shap\
│   └── SKILL.md          (from K-Dense)
├── fresec-data\
│   └── SKILL.md          (from K-Dense database-lookup)
├── statistical-analysis\
│   └── SKILL.md          (from K-Dense)
├── faiss\
│   └── SKILL.md          (from Orchestra)
├── research-ideation\
│   └── SKILL.md          (from Orchestra)
└── mlops\
    └── SKILL.md          (from Orchestra)
```

---

## Summary Matrix

| Repo | Action | Reason |
|------|--------|--------|
| **RD-Agent** | ✅ KEEP | Quant finance factor/model evolution (from prev eval) |
| **AIDE** | ✅ KEEP | ML code tree-search optimization (from prev eval) |
| **GPT-Researcher** | ✅ KEEP | Web research for market intelligence (from prev eval) |
| **STORM** | ✅ KEEP | Knowledge base generation (from prev eval) |
| **PaperQA2** | ✅ KEEP | Local document Q&A (complements OpenScholar) |
| **OpenScholar** | ✅ REVERSE KEEP | Superior correctness, install alongside PaperQA2 |
| **DeepScientist** | 📖 READ ONLY | Findings Memory + Research Map architecture concepts |
| **Idea2Paper** | 📖 READ ONLY | Anchored review + KG pipeline concepts |
| **DeepResearchAgent** | 📖 READ ONLY | Autogenesis protocol (RSPL/SEPL) design patterns |
| **PaperBanana** | ❌ SKIP | Academic illustration only, zero relevance |
| **MLE-Agent** | 📖 READ ONLY | Baseline auto-creation + arxiv integration patterns |
| **scientific-agent-skills** | ⚡ SELECTIVE (9) | scikit-learn, TimesFM, PyMC, SHAP, etc. |
| **AI-Research-SKILLs** | ⚡ SELECTIVE (7) | FAISS, RAG, MLOps, ideation |
