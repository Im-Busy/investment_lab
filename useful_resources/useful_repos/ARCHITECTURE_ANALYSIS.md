# Architecture Analysis: Lessons from 4 AI Research Systems

> Analysis date: 2026-04-27
> Source repos: DeepScientist, Idea2Paper, DeepResearchAgent (Autogenesis), MLE-Agent

---

## 1. DeepScientist — Local-First Research Operating System

**Core Idea**: One quest = one Git repository. Durable state in files, not ephemeral chat.

### Key Architecture Concepts

| Concept | Description | Applicability to Our Project |
|---------|-------------|------------------------------|
| **Quest-as-Repo** | Each research quest is a full Git repo with branches, artifacts, memory | High: Could model each "strategy quest" as a Git workspace |
| **Prompt-Led Workflow** | System prompt + skill files drive behavior, not hard-coded scheduler | Medium: Our pattern detectors already follow this loosely |
| **3 MCP Namespaces** | `memory`, `artifact`, `bash_exec` — minimal surface | High: Clean abstraction for our signal/risk/backtest ops |
| **Canvas from Durable State** | UI rebuilt from Git + artifacts + events, not ephemeral | Medium: Useful for strategy visualization |
| **Findings Memory** | Cross-run reusable knowledge (paper notes, failure lessons) | High: Could store pattern failure/success history |
| **Runner Abstraction** | Pluggable runners (codex, claude, kimi, opencode) via registry | Low: Not directly applicable |
| **Connector Model** | External surfaces (chat, messaging) are adapters, not core | Low: Not applicable |

### Most Valuable Takeaway
The **quest layout contract** — `quest.yaml`, `brief.md`, `plan.md`, `status.md`, `SUMMARY.md` — provides a structured template for strategy lifecycle management. Our project could adopt similar structured files under `strategies/<name>/`.

---

## 2. Idea2Paper — Knowledge Graph + Multi-Stage Pipeline

**Core Idea**: Offline KG construction → Three-way recall → Generation with correction → RAG deduplication.

### Key Architecture Concepts

| Concept | Description | Applicability to Our Project |
|---------|-------------|------------------------------|
| **Two-Stage Recall** | Jaccard coarse filter (ms) → Embedding fine sort (secs) = 13x speedup | Medium: Could apply to pattern matching |
| **Multi-Dimensional Pattern Classification** | Stability / Novelty / Cross-Domain | High: Directly maps to our 34+ pattern categories |
| **Idea Fusion** | Conceptual fusion, not technical piling | Medium: Could inspire confluence scoring |
| **Critic Multi-Role Review** | 3 reviewers (Methodology/Novelty/Storyteller), avg ≥ 7.0 passes | High: Analogous to multi-signal confluence validation |
| **Intelligent Correction with Rollback** | Score degradation triggers rollback to best version | High: Similar concept to our circuit breakers |
| **RAG Deduplication** | Collision detection (similarity > 0.75) → pivot avoidance | Medium: Could detect redundant strategies |
| **Anchored Scoring** | Compare against real anchors, not arbitrary scores | High: Similar to our SPY/BTC baselines |

### Most Valuable Takeaway
The **generate → review → correct → rollback** loop with explicit thresholds is directly applicable to strategy validation. Our backtest engine could adopt this iterative refinement pattern:
1. Generate signals → 2. Multi-critic review (pattern confluence) → 3. Correct thresholds → 4. Rollback if degradation

---

## 3. DeepResearchAgent (Autogenesis) — Self-Evolution Protocol

**Core Idea**: RSPL (Resource Substrate) + SEPL (Self Evolution) for versioned, auditable agent improvement.

### Key Architecture Concepts

| Concept | Description | Applicability to Our Project |
|---------|-------------|------------------------------|
| **RSPL** | Prompts, agents, tools, environments, memory as versioned resources | Medium: Could version strategy configs |
| **SEPL** | Propose → Assess → Commit evolution loop with rollback | High: Similar to our optimization workflow |
| **Reflection Optimizer** | Turn feedback into updated prompts/solutions (3 rounds max) | High: Could optimize strategy parameters |
| **Config Composition** | MMEngine-style configs for agents/tools/envs/memory | Medium: Our YAML configs already do this |
| **Tool Evolution** | Retrieve → Execute → Refine on error → Synthesize new if missing | High: Applicable to indicator/tool selection |

### Most Valuable Takeaway
The **Act → Observe → Optimize → Remember** loop with versioned resources is a clean abstraction. Our backtest system could adopt this as:
- **Act**: Execute strategy signals
- **Observe**: Capture returns, drawdown, metrics
- **Optimize**: Adjust parameters via reflection
- **Remember**: Persist insights to strategy memory

---

## 4. MLE-Agent — ML Engineering Companion

**Core Idea**: Pairing LLM agent for ML tasks with baseline generation, Arxiv integration, smart debugging.

### Key Architecture Concepts

| Concept | Description | Applicability to Our Project |
|---------|-------------|------------------------------|
| **Autonomous Baseline** | Build ML baselines from vague requirements | Medium: Could generate baseline strategies |
| **Smart Debugging** | Automatic debugger-coder interaction | High: Could debug strategy signal errors |
| **Arxiv + Papers with Code Integration** | Access SOTA methods | Medium: For research paper ingestion |
| **Weekly Report Generation** | Auto-generate summaries from git history | Low: Nice-to-have for progress tracking |
| **Project Structure Management** | Organize file system efficiently | Medium: Current structure could improve |
| **Interactive CLI Chat** | Enhance projects with chat interface | Low: Not core focus |

### Most Valuable Takeaway
The **smart debugging** pattern where a debugger agent interacts with a coder agent to fix errors is applicable to our strategy development pipeline. Failed backtest signals could trigger automated diagnosis.

---

## Synthesis: Patterns to Adopt

### Priority 1 — High Impact, Low Effort

1. **Structured Strategy Lifecycle Files** (from DeepScientist)
   - Add `strategy.yaml`, `brief.md`, `plan.md`, `status.md` under each strategy dir
   - Enables durable state tracking across runs

2. **Multi-Critic Validation Loop** (from Idea2Paper)
   - Pattern confluence scoring → threshold check → correction → rollback
   - Already partially implemented; formalize the loop

3. **Versioned Resource Management** (from Autogenesis)
   - Version strategy configs and parameters
   - Enable rollback to best-performing version

### Priority 2 — Medium Impact, Medium Effort

4. **Two-Stage Pattern Matching** (from Idea2Paper)
   - Fast keyword/pattern filter → Detailed signal analysis
   - Could speed up confluence scoring

5. **Act-Observe-Optimize-Remember Loop** (from Autogenesis)
   - Formalize the optimization cycle for strategy parameters
   - Persist optimization history

6. **Smart Debugging Agent** (from MLE-Agent)
   - Automated diagnosis of failed strategies/signals

### Priority 3 — Research / Future

7. **Knowledge Graph for Patterns** (from Idea2Paper)
   - Build KG of patterns, market regimes, outcomes
   - Enable intelligent pattern recommendation

8. **Findings Memory** (from DeepScientist)
   - Cross-strategy reusable knowledge
   - Store what worked, what failed, why

---

## File Structure Recommendations

Based on analysis, suggest adding:

```
strategies/<name>/
├── strategy.yaml          # Config, params, thresholds (RSPL-style versioned)
├── brief.md               # Research goal and hypothesis
├── plan.md                # Implementation plan
├── status.md              # Current status, last run metrics
├── SUMMARY.md             # Final results and conclusions
├── versions/              # Version history of configs
│   └── v1.yaml, v2.yaml, ...
└── memory/                # Findings memory
    └── lessons.json       # What worked, what failed, why
```

---

## Conclusion

The 4 systems share a common pattern: **durable state + iterative refinement + versioned resources**. Our investment system already has strong pattern detection and backtesting; adopting these architectural patterns would improve:
- Strategy lifecycle management
- Parameter optimization workflows
- Cross-strategy knowledge reuse
- Automated validation and correction
