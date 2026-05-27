# Project Guidelines

## Python Environment

This project uses **uv** for Python package and environment management. Always use uv commands:

- `uv run <command>` - Run Python scripts/commands in the project environment
- `uv add <package>` - Add a package to the project
- `uv remove <package>` - Remove a package from the project
- `uv sync` - Sync the environment with the lock file
- `uv pip install <package>` - Install packages using uv's pip interface (if needed)

**NEVER** use `python`, `pip`, `python3`, or `pip3` directly. Always prefix commands with `uv`.

Examples:
- Instead of `python script.py`, use `uv run script.py`
- Instead of `pip install requests`, use `uv add requests`
- Instead of `pip install -r requirements.txt`, use `uv sync`

## Useful Commands Documentation

**IMPORTANT FOR ALL AIs:** When you discover or use helpful commands during your work:

1. **Add them to** `.useful_commands/useful_commands.txt` immediately
2. **Organize by category** (Training, Testing, Deployment, etc.)
3. **Include description** explaining what each command does
4. **Keep it current** - Update when commands change or become obsolete

This helps future AI sessions and human developers quickly find the right commands.

### Command Log Structure
```
## CATEGORY NAME

# Description of what the command does
actual-command-to-run
```

## AI Settings - Cross-Project Knowledge Sharing

### Command Documentation Files

**Global Command Files** (`.useful_commands/`):
- `ml_training_commands.txt` - ML model training commands
- `backtest_commands.txt` - Backtesting commands
- `quick_reference.txt` - All commands quick reference

**Distributed Command Files** (`src/*/AI_COMMANDS.txt`):
Each `src/` subdirectory contains an `AI_COMMANDS.txt` file for module-specific commands.

### Instructions for All AI Assistants

When completing tasks:

1. **Document every working command** in appropriate files:
   - General commands → `.useful_commands/[category]_commands.txt`
   - Module-specific → `src/[module]/AI_COMMANDS.txt`

2. **Use this format**:
   ```
   ## Feature Name

   # Description
   command
   ```

3. **Check existing files** before creating duplicates

4. **Update AGENTS.md** if new command categories emerge


## Command Cheatsheet - Single Source of Truth

This project maintains a centralized **Command Cheatsheet** at `docs/COMMAND_CHEATSHEET.md`.

**CRITICAL: See also `.kilo/project-rules.md#command-documentation-protocol` for the full protocol covering creation, modification, and deletion of commands.**

### Auto-Update Protocol for AI Assistants

**When implementing new features that include CLI scripts, commands, or workflows:**

1. **IMMEDIATELY UPDATE** `docs/COMMAND_CHEATSHEET.md` as part of the same commit
2. Add commands to the appropriate functional section
3. Include a brief description explaining what the command does
4. Group with related commands (ML, backtest, optimization, etc.)
5. Commit with message: "docs: update command cheatsheet for [feature-name]"

**Example Addition:**
`markdown
## New Feature Category

# Description of what the command does
uv run scripts/new_feature.py --option value

# With custom parameters
uv run scripts/new_feature.py --symbol SPY --start 2020-01-01 --end 2024-12-31
`

### Priority Order for Documentation

1. **PRIMARY:** COMMAND_CHEATSHEET.md - Centralized comprehensive guide (THIS FILE)
2. **SECONDARY:** .useful_commands/[category]_commands.txt - Detailed category-specific commands
3. **TERTIARY:** src/[module]/AI_COMMANDS.txt - Module-specific quick reference

### Validation Checklist

Before considering a feature complete, verify:
- [ ] New commands are documented in COMMAND_CHEATSHEET.md
- [ ] Commands are added to relevant .useful_commands/ file (optional, for detailed workflows)
- [ ] Module-specific AI_COMMANDS.txt updated if applicable
- [ ] All command examples are tested and working
- [ ] CLI flags and options are clearly documented
- [ ] Dependencies or prerequisites are noted if needed

### Maintenance

- Review and consolidate duplicate command documentation quarterly
- Remove outdated commands from previous implementations
- Ensure all command examples produce expected output
- Keep file structure diagram current with new scripts/directories

---

## End-to-End Wiring Protocol (CRITICAL — MANDATORY CHECK BEFORE MARKING COMPLETE)

**Every new parameter, feature flag, detector, indicator, risk module, or CLI argument MUST be wired end-to-end into the execution path.** A feature that compiles but doesn't affect scoring/entry/exit is invisible debt.

### The Rule: Trace Every Parameter

After implementing any new feature, you MUST trace its parameter from declaration → init → scoring → entry/exit gates. Use this checklist:

| Check | Question |
|-------|----------|
| **Declaration** | Is the param on the class with a default? |
| **Init gate** | Does `init()` conditionally precompute based on this param? |
| **Scoring** | Does `_compute_score()` or equivalent read this param or its arrays? |
| **Entry/Exit** | Does `next()` conditionally gate on this param's value? |
| **CLI exposure** | Is there an argparse flag that passes to this param? |
| **CLI-in-kwargs** | Does the script's kwargs dict include the flag? |

### Anti-Patterns That Passed Code Review But Were Dead

| Anti-Pattern | Real Example (2026-05-21 audit) | Consequence |
|-------------|------|-------------|
| Param declared but never read in scoring | `min_confluence=3` in SMC | Users think they're gating entries but zero effect |
| Feature precomputes data but never scored | `use_smc_phl=True` computes 4 arrays, none read | Wasted CPU + user confusion |
| Function exists but never called | `_calculate_size()` | Positions stay hardcoded 0.95 |
| CLI flag not in kwargs dict | `--no-volume-pressure` flag declared but not mapped | Flag silently ignored |
| Toggle gates precompute but not scoring | `use_breaker_blocks` blocks init but scoring always reads arrays | Misleading — appears wired when it's not |

### Mandatory End-to-End Validation Script

After implementing any new parameter, run this audit (or equivalent):

```bash
# Check if param is referenced in scoring
uv run python -c "
import inspect
from src.strategies.smc_strategy import SMCStrategy
# For each param with default, grep for usage in _compute_score and next()
# Any param not found is a wiring gap
"

# Verify CLI flag maps into kwargs dict
grep -n 'my_new_flag' scripts/backtest_smc.py  # should appear in BOTH argparse AND kwargs dict
```

**A feature is NOT complete until it produces a documented, measurable change in backtest output when toggled ON vs OFF.** If `--my-feature` and its absence produce identical backtest metrics, the feature is either dead or not contributing signal — fix or remove it.

---

## Source Attribution Protocol (CRITICAL — MANDATORY FOR ALL EXTERNAL CONCEPTS)

**Every implementation, design, algorithm, or pattern borrowed from an external source MUST carry an attribution tag.** This includes papers, web articles, blog posts, tutorials, forum posts, YouTube videos, reference repos, and documentation.

### The Rule: Cite Everything Borrowed

| Check | Question |
|-------|----------|
| **Module docstring** | Does the file have a `Source:` or `Reference:` line naming the external source? |
| **Inherited code** | Does the docstring note `Adapted from:` with URL and what was changed? |
| **Registry entry** | Is the source listed in `SOURCE_MANIFEST.md` with all files referencing it? |
| **Web source** | Is it in `docs/research_logic_map/web_source_registry.md` (W-ID assigned)? |
| **Paper source** | Is it in `docs/research_logic_map/insight_registry.md` (P-ID assigned)? |

### Standardized Docstring Format

Every Python file that borrows external concepts MUST include one of these at the top of the module docstring:

```python
"""
<brief description of what this module does>

Source: <name>, <URL/DOI>, <section/§ if applicable>
"""
```

For papers:
```python
"""
<brief description>

Paper: "<title>" (<author(s)>, <year>), <reference ID if in insight_registry.md>
"""
```

For adapted code:
```python
"""
<brief description>

Adapted from: <source name>, <URL>, <license if applicable>
Changes: <what was modified and why>
"""
```

### Aggregation Rules

| Scenario | Format |
|----------|--------|
| Multiple sources | List each on a separate `Source:` line |
| Web article | `Source: <title>, <URL>` |
| Paper with registry ID | `Paper: PXX — "<title>"` |
| Paper without registry ID | `Paper: "<title>" (<author>, <year>), <DOI>` |
| Reference implementation | `Reference: <repo name>, <URL>` |
| ICT/SMC concept | `Reference: ICT <concept name> methodology` |
| Library wrapping | `Source: <library name>, <URL>` |

### Source Registry Files

| File | Purpose | Public? |
|------|---------|---------|
| `SOURCE_MANIFEST.md` (root) | **Master ledger** — every source, every file, status. Private. | No |
| `docs/research_logic_map/insight_registry.md` | Papers only, with insight tracking | Yes |
| `docs/research_logic_map/web_source_registry.md` | Non-paper web sources, with W-IDs | Yes |

### Sync Protocol — Removing Sources

When a source should no longer be referenced:

1. Mark it `~~strikethrough~~` in `SOURCE_MANIFEST.md`
2. User invokes `/sync-attributions`
3. Agent removes all `Source:`/`Reference:` lines from listed files
4. Agent removes entries from registry files
5. Agent strips strikethrough entries from SOURCE_MANIFEST.md

### Anti-Patterns

| Don't | Because |
|-------|----------|
| Implement a paper concept without citing it | Traceability gap — future sessions can't verify origin |
| Copy code from a blog without attribution | License violation + makes auditing impossible |
| Use `Source:` without listing in SOURCE_MANIFEST.md | Manifest becomes stale; attribution drift |
| Leave orphaned references after removing code | Litters codebase with dead citations |
| Assume internal-only code doesn't need sources | All borrowed ideas need attribution, regardless of visibility |

---

This project includes a paper summarization workflow using the **paper2md** tool:

### Location
- Tool: `useful_resources/useful_repos/research-tools/paper2md/`
- Input: `useful_resources/papers_md/*.md` (markdown papers)
- Output: `useful_resources/useful_repos/research-tools/paper2md/output/SENTIMENT_ANALYSIS_SUMMARY.md`
- Final copy: `useful_resources/papers_md/SENTIMENT_ANALYSIS_SUMMARY.md`

### API Configuration
The tool uses OpenAI-compatible endpoints via OpenRouter:
- **Provider**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Model**: `deepseek/deepseek-v4-flash` (DeepSeek V4 Flash - fastest and strongest)
- Config file: `paper2md/.env`

### Usage
```bash
cd useful_resources/useful_repos/research-tools/paper2md
uv run python summarize_md_papers.py --papers-dir ..\..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md
```

### Output Format
Each paper summary includes:
- TL;DR (3 bullets)
- Problem & Motivation
- Data & Methodology
- Sentiment Analysis Approach
- ML Models & Results (with metrics)
- Trading Applications & Takeaways
- Limitations & Future Work

See `.useful_commands/paper_summarization.txt` for detailed commands and model options.

### Knowledge Graph Auto-Linking (CRITICAL)

**Whenever new papers are added to `useful_resources/papers/` or converted to `.md` in `papers_md/`, you MUST run the knowledge graph analysis to cross-reference findings with project modules:**

```bash
# Full pipeline: scan → convert → analyze → update plans
uv run useful_resources/_knowledge_analysis.py
```

This produces:
- `useful_resources/papers_md/KNOWLEDGE_GRAPH_INSIGHTS.md` — comprehensive cross-reference report
- Updated `progress_docs/plans/full.md` with new action items
- Updated `progress_docs/current.md` with session log entry

**Load the knowledge-graph skill** (`.kilo/skills/knowledge-graph/SKILL.md`) whenever:
1. New PDFs appear in `useful_resources/papers/`
2. Papers are converted to `.md`
3. User asks about paper insights, cross-references, or module gaps
4. After implementing new modules (to re-evaluate paper alignment)

**Use the `/knowledge-graph` command** for quick access to the full pipeline.

### Resource Insight Extraction Protocol — "Think Freely, Then Compare" (CRITICAL)

**Whenever processing EXTERNAL or downloaded resources (papers, strategy repos, codebases, books, courses), follow this two-phase protocol:**

**Phase 1: EXTRACT EVERYTHING (no filters)**
- Extract ALL possible insights, ideas, additions, patterns, algorithms — regardless of feasibility, impact, architecture fit, GPU/data gates.
- Do NOT self-censor. Do NOT pre-filter. List everything you find.
- Include ideas that are obvious, complex, impractical, brilliant, duplicated, trivial — all of them.
- Target: 10+ items minimum from any substantial resource.

**Phase 2: COMPARE OUT LOUD**
- For every extracted idea, state:
  - What it is (1 sentence)
  - Where it came from (resource + section)
  - Impact estimate (low / medium / high)
  - Implementation cost (lines of code + new dependencies)
  - Architecture fit (where in `src/` it would live)
- Then rank ALL ideas head-to-head in a comparison table.
- Only AFTER ranking, recommend which to implement.

**This protocol prevents:**
- Premature optimization (dismissing high-impact ideas because they seem complex)
- Blindness to simple wins (overthinking while missing 30-line quick-wins)
- Architecture lock-in (only seeing ideas that fit the current structure)
- Confirmation bias (filtering out ideas that challenge existing design decisions)

**When to use:** Processing papers/PDFs, analyzing external repos, reading course material, evaluating new libraries, or whenever the user asks "what can we learn from this resource?"

**Example output format:**
```markdown
## Raw Extract (everything I found)
- Idea 1: description (source: Paper X §2.3)
- Idea 2: description (source: Paper X §4.1)
... (10-30 items)

## Comparison (head-to-head)
| # | Idea | Impact | Cost (loc+deps) | Fit | Risk |
|---|------|--------|-----------------|-----|------|
| 1 | ...  | High   | 80 loc, 0 deps  | ... | Low  |
...

## Recommended (top N)
1. Best idea — because...
2. Second best — because...
```

---

## Session Start Protocol — CRITICAL FOR ALL SESSIONS

**Every new AI session MUST start by reading these files in order:**

1. **`MEMORY.md`** (project root) — Persistent handover state: current objective, system metrics, completed tasks, discovered issues, next session priorities. This is the "where were we" file.
2. **`progress_docs/plans/full.md`** — Master plan with phase status, pending tasks, dependencies.
3. **`progress_docs/current.md`** — Session log with timestamps and detailed action history.
4. **`docs/research_logic_map/insight_registry.md`** — All research insights (88 from 22 sources), tagged by topic/impact/status. Source of truth for what is known and what remains to implement.
5. **`docs/BESTS_INSIGHTS.md`** — Distilled knowledge from every backtest in BESTS.md. Factor rankings, strategy tier list, anti-patterns, production configs. The "what works and what to never do" file.
6. **`docs/GPU_TASK_QUEUE.md`** — Registry of GPU-blocked tasks with tutorials. **CRITICAL: Check GPU availability immediately after reading this file.** If `torch.cuda.is_available()`, attempt highest-priority pending task. See §GPU Task Protocol below.
7. **Verify indexes**: `npx gitnexus status` + `cgc stats` — reindex if stale. Ensure CGC watcher is running (`Get-Job -Name "CGCWatcher"`).

**IMPORTANT: After reading `BESTS_INSIGHTS.md`, check `BESTS.md` for its `last_updated` timestamp.** If BESTS.md changed significantly since BESTS_INSIGHTS.md was last synced (new top-3 results, new strategy categories, regime shift detected), re-evaluate the insights and update both files.

After reading state, consult `.kilo/project-loop.md` for the DEEPEN/BROADEN/PIVOT/CONCLUDE decision framework. Every phase or significant experiment must end with an explicit direction decision.

The agent updates `MEMORY.md` continuously during the session. At session end, the agent writes a detailed handover to `progress_docs/handovers/` and updates both `MEMORY.md` and `progress_docs/current.md`.

## GPU Task Protocol — MANDATORY

**Every session MUST check `docs/GPU_TASK_QUEUE.md` for pending GPU tasks.** This is step 6 of the Session Start Protocol above.

### Protocol

1. **Check GPU availability** immediately after reading the queue:
   ```bash
   uv run python -c "import torch; print('GPU:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
   ```

2. **If GPU is available:**
   a. Open `docs/GPU_TASK_QUEUE.md` and find the highest-priority pending (⏳ or ⏸️) task.
   b. Follow the **tutorial** in that task's entry exactly — do NOT re-research the paper or redesign the approach.
   c. Run the command(s) specified in the tutorial.
   d. If task completes successfully: update status to ✅, record results/metrics in the task entry, update MEMORY.md.
   e. If task is aborted (OOM, timeout, dependency error): update status to ⏸️ with reason, move to next task.
   f. Continue until all P0/P1 tasks are attempted or GPU time is exhausted.

3. **If GPU is NOT available:**
   a. Skip GPU tasks — continue with CPU-safe work.
   b. If implementing new GPU-dependent code, add it to `docs/GPU_TASK_QUEUE.md` with a self-contained tutorial.

### When to Add Tasks to the Queue

**CRITICAL — MANDATORY TASKIFICATION RULE:** Whenever you encounter, design, implement, or identify code that requires GPU (GANs, diffusion, transformers, large RL, SHAP, deep learning training) or runs >5 min/epoch on CPU, you MUST immediately add it to `docs/GPU_TASK_QUEUE.md` with a self-contained tutorial. Do NOT skip this step and do NOT defer it — even if the code hasn't been built yet. A stub task entry with planned files and estimated tutorial is sufficient. This ensures no GPU-bound work is lost across sessions.

Add a task whenever:
- Code is implemented but gate validation requires GPU training
- A model was trained on CPU with reduced parameters (note what full params would be)
- A paper's method requires GPU (GANs, diffusion, transformers, large SHAP)
- You encounter OOM or >5 min/epoch training times on CPU
- A feature requires LLM inference at scale (multi-agent frameworks, adversarial debate, LLM stress-testing) that is deferred

### Task Entry Template

```markdown
### N. Task Name (Phase Reference)

**Status:** ⏳ Not started

**What:** 1-2 sentence description.

**Why:** 1-2 sentence justification.

**Files involved:** (list existing files, files to create)

**Tutorial:**
\`\`\`bash
# Step-by-step commands with realistic flags
# Include expected output and success criteria
\`\`\`

**Expected runtime on GPU:** X-Y minutes/hours
**Paper reference:** (if applicable)
**Source:** (plan ref, awesome-ai-in-finance, paper, etc.)
```

**IMPORTANT: Even for unimplemented features** (files not yet created), add a stub task with `files to create:` and a tutorial showing planned CLI commands. The `⏳ Not started — files not built` status tag indicates the task entry is a placeholder for future implementation.

## Progress Documentation — CRITICAL FOR ALL SESSIONS

This project maintains structured progress tracking in `progress_docs/`. This is the single source of truth for project state, phase status, pending work, and session context.

[... same content until end of file ...]

---

## Tool & Concept Documentation Protocol

**When introducing any new tool, library, algorithm, or ML concept to the project, you MUST:**

1. **First, check** `docs/ML_TRAINING_GUIDE.md` — can this new item be added to an existing section?
2. **If a suitable section exists:** add a row to the relevant table, a brief description, and a code example if applicable. Update the table of contents.
3. **If no suitable document exists:** create a new doc in `docs/` with the prefix `tool-` or `guide-` (e.g., `docs/tool-purged-kfold.md`, `docs/guide-metaheuristics.md`). Link it from `docs/README.md`.
4. **Required information for each new tool/concept:**
   - **What it is** (1-2 sentences, beginner-friendly)
   - **Why it exists** (what problem it solves)
   - **Where it lives** (file path)
   - **When to use it** (decision criteria)
   - **How to use it** (code example, CLI command)
   - **Relationship to other tools** (what it depends on, what depends on it)
5. **Update `AGENTS.md`** if a new documentation category or convention is created.

---

## File Creation Guidelines — CRITICAL FOR ALL AIs

**Before creating a new file, ALWAYS ask: can this content be added to an existing file instead?**

### The Rule: Append, Don't Sprawl

When writing plans, design docs, study reports, setup instructions, or implementation reports:

1. **Find the parent file.** Every topic in this project has a canonical parent:
   - Phase plans → `progress_docs/plans/full.md`
   - Phase detail → `progress_docs/plans/0X-phase-name.md` (01-08)
   - Research studies → `progress_docs/plans/06-research.md`
   - ML implementation reports → `progress_docs/plans/04-ml-foundation.md`
   - Cross-asset experiments → `progress_docs/plans/cross_asset_features_implementation.md`
   - Tool evaluations → `progress_docs/plans/auto_research_tools_evaluation.md`
   - Setup/config guides → the relevant tools' existing plan file
   - Issue fixes → `fix-overfitting.md` or the relevant phase plan

2. **Add as a subsection**, not a new file. Append `## New Section` or `### New Subsection` to the parent. Mark merged content with `*(Added YYYY-MM-DD)*`.

3. **Only create a new file when:**
   - The topic has NO existing parent file
   - The content is fundamentally a new category (e.g., `post_retrain_next_steps.md` — entirely new domain)
   - The parent file would exceed ~800 lines (in which case split to a numbered sibling: `02-optimization.md`)

### Anti-Pattern — Do NOT Do This

Do NOT create standalone files for:
- Status summaries or completion checklists → append to the parent plan
- Sub-experiments of a larger experiment → append to the experiment's plan file
- Round 2 of an evaluation → append to Round 1's file
- Sequential phases of the same pipeline → one file per pipeline, not one per phase
- Setup status stubs → append to the setup guide they reference

**If a file exists today that should have been a subsection, fix it immediately — merge into the parent and delete the fragment.**

---

### Tool Inventory

The current tool inventory is maintained in `docs/ML_TRAINING_GUIDE.md` Section 7 (File Location Reference). Before creating new documentation, verify the tool isn't already documented there.

| Doc | Coverage |
|-----|----------|
| `docs/ML_TRAINING_GUIDE.md` | All ML components, optimizers, concepts, decision tree, data flow |
| `docs/guide-ml-pipeline.md` | Standard 9-stage ML pipeline — every AI agent MUST read before training |
| `docs/guide-anti-overfitting.md` | Lock Box, Nested CV, Blind Analysis, Label Shuffling (Phase 25) |
| `docs/research_logic_map/insight_registry.md` | 88 research insights from 22 sources, tagged by topic/impact/status |
| `COMMAND_CHEATSHEET.md` | CLI commands for every script |
| `.useful_commands/` | Detailed command workflows by category |

---

## Housekeeper Mode — File System Cleanup

Use `/housekeeper` to audit and clean the project file system. It flags misplaced files, duplicate directories, stray artifacts, and produces a safe migration plan.

### Safety Rules (CRITICAL)

When moving or renaming files, **never delete a file before the destination is verified**:

| Scenario | Safe Method |
|----------|------------|
| Git-tracked file move/rename | `git mv <source> <destination>` (atomic, revertible) |
| Same-dir rename (non-git) | `ren "old" "new"` |
| Cross-dir move (non-git) | `robocopy <srcdir> <dstdir> <file> /MOV /R:3 /W:5 /NP /LOG:move.log` |
| Bulk directory move | `robocopy <srcdir> <dstdir> /E /MOV /R:3 /W:5 /NP /LOG:move.log` |

**robocopy `/MOV`** copies then deletes source per-file — it never deletes before a successful copy. Always verify the log for `FAILED Files = 0` after each operation.

### Useful robocopy flags

| Flag | Purpose |
|------|---------|
| `/MOV` | Move (copy then delete) |
| `/E` | Include subdirectories |
| `/R:3` | Retry 3 times |
| `/W:5` | Wait 5s between retries |
| `/NP` | No progress |
| `/LOG:file` | Output to log |

---

## Agent-Centric Workflows — Slash Commands

The project exposes six primary workflows as Kilo agents and slash commands. These agents know the project's conventions, pitfalls, and baselines — they should be used instead of manual CLI commands for consistency.

### Agent Catalog

| Agent | File | Purpose |
|-------|------|---------|
| **model-doctor** | `.kilo/agent/model-doctor.md` | Runs calibration audit, regime shift investigation, WFO comparison. Produces health report with remediation playbook. |
| **backtest-runner** | `.kilo/agent/backtest-runner.md` | Executes ML strategy backtests, updates BESTS.md leaderboard, interprets results against known baselines. |
| **ml-trainer** | `.kilo/agent/ml-trainer.md` | Trains CatBoost models via 9-stage pipeline. Knows triple-barrier labels, PurgedKFold, ATR normalization, overfitting thresholds. |
| **repo-syncer** | `.kilo/agent/repo-syncer.md` | Syncs curated files from private dev repo to public-facing repo. Merges main→public, strips private data, pushes only the clean public branch. |
| **housekeeper** | `.kilo/agent/housekeeper.md` | Audits file system, flags misplaced files and duplicate dirs, produces safe migration plan. |
| **researcher** | `.kilo/agent/researcher.md` | Searches Google Scholar, ArXiv, GitHub for papers, reference implementations, and benchmarks. Cross-references findings with project modules. Auto-activates when designing new algorithms or encountering unfamiliar methods. |
| **batch-tuner** | `.kilo/agent/batch-tuner.md` | Runs parameter sweeps and backtests in parallelized batches across multiple instruments. Tunes Rules-First params, finds universal best config, auto-generates comparison reports. Use for any multi-instrument parameter optimization. |
| **sync-attributions** | `.kilo/agent/sync-attributions.md` | Purges sources marked `~~strikethrough~~` in SOURCE_MANIFEST.md from code and registries. |

### Slash Commands

| Command | Agent | Usage |
|---------|-------|-------|
| `/model-diagnose` | model-doctor | Full diagnostic suite. No arguments needed. |
| `/backtest` | backtest-runner | Run backtests with `--entry-threshold --trail-stop` etc. |
| `/train-ml` | ml-trainer | Train models with `--symbol / --basket / --fast / --walk-forward` |
| `/repo-sync` | repo-syncer | Sync curated files to public repo. `/repo-sync check` for safety-only. |
| `/housekeeper` | housekeeper | Audit file system, produce migration plan. |
| `/research` | researcher | Search for papers, repos, benchmarks. `/research regime-switching HMM` |
| `/batch-tune` | batch-tuner | Tune Rules-First params across instruments. `/batch-tune --fast` |
| `/sync-attributions` | sync-attributions | Purge strikethrough-marked sources from code. No arguments. |

### When to Use Slash Commands vs Direct CLI

| Situation | Use |
|-----------|-----|
| Training a new model | `/train-ml` — agent knows overfitting thresholds and ATR fix |
| Running a backtest | `/backtest` — agent auto-updates BESTS.md |
| Checking model health | `/model-diagnose` — agent runs all 3 diagnostics + interprets |
| Syncing to public repo | `/repo-sync` — agent handles merge, safety check, and push |
| Quick one-off script | Direct CLI — e.g., `uv run scripts/sweep_entry_thresholds.py SPY` |
| Searching for papers/implementations | `/research` — agent knows search hierarchy and ingestion pipeline |
| Tuning params across instruments | `/batch-tune` — agent batches, parallelizes, and validates OOS |

### Creating New Agents

When adding new scripts or workflows, create a `.kilo/agent/*.md` file teaching future AIs how to use them. Include:
- The exact CLI commands with realistic flags
- Known baselines and thresholds for interpreting results
- Pitfalls and anti-patterns to avoid
- When to invoke vs when to use another agent

Wrap in a `.kilo/command/*.md` for slash command access.

### Target Directory Conventions

```
root/
├── src/           # Python source
├── tests/         # All tests
├── scripts/       # CLI scripts
├── data/          # Data files
├── docs/          # Documentation
├── models/        # ML models
├── outputs/       # Generated outputs
├── logs/          # Log files
├── notebooks/     # Jupyter notebooks
  ├── reports/       # Analysis reports
  ├── pipeline/      # Pipeline definitions
  ├── plans/         # Planning docs
  ├── progress_docs/ # Progress tracking
  ├── useful_resources/useful_repos/  # Cloned reference repos
  ├── useful_resources/useful_repos/CodeGraphContext/  # CGC Python graph intelligence
  ├── useful_resources/useful_repos/GitNexus/  # GitNexus Node.js graph intelligence
  └── experiments/   # Experiment outputs
```

### Anti-Patterns

- `.py` files at root → belongs in `src/`, `tests/`, or `scripts/`
- `.md` files at root (except `AGENTS.md`, `README.md`) → belongs in `docs/`
- Duplicate directories (`logs/` + `fin_logs/`, `output/` + `outputs/`)
- Full cloned repos at root → belongs in `useful_resources/useful_repos/`
- ML artifacts at root (`catboost_info/`, `AutogluonModels/`) → belongs in `outputs/`


## Batch Backtesting & Tuning Protocol (CRITICAL — for all sessions)

When backtesting or tuning across multiple instruments, follow this protocol to avoid overcrowding and maximize throughput.

### The Rule: Small Batches, Sequentially

**NEVER** run all instruments in one monolithic command. Always split into batches of 5-6 instruments and run them one at a time.

### Three-Phase Workflow

**Phase 1 — IS Tuning (3 batches, sequential):**
```bash
# Batch 1: Major indices + sector ETFs
uv run scripts/tune_rules_params.py --symbols SPY,QQQ,IWM,XLK,XLF --fast --is-only --workers 5 \
    --json-output outputs/tune_batch1.json --md-output reports/parameter_tuning/batch1.md

# Batch 2: Sectors + commodities + bonds
uv run scripts/tune_rules_params.py --symbols XLE,XLV,GLD,TLT,KO --fast --is-only --workers 5 \
    --json-output outputs/tune_batch2.json --md-output reports/parameter_tuning/batch2.md

# Batch 3: Stocks + crypto + forex
uv run scripts/tune_rules_params.py --symbols JPM,XOM,JNJ,SO,BTC_USD,EURUSD_X --fast --is-only --workers 6 \
    --json-output outputs/tune_batch3.json --md-output reports/parameter_tuning/batch3.md
```

**Phase 2 — OOS Validation + Universal Best:**
```bash
uv run scripts/tune_rules_oos.py
```

**Phase 3 — Documentation:**
- Results auto-saved to `reports/parameter_tuning/RULES_TUNING.md`
- Update `BESTS.md` with new tuning section
- Update `docs/COMMAND_CHEATSHEET.md` with commands

### Grid Size Reference

| Flag | Combos | Per-instrument time | Use case |
|------|--------|---------------------|----------|
| `--mini` | 12 | ~50s | Quick screening |
| `--fast` | 60 | ~5 min | **Default** — good coverage |
| *(none)* | 243 | ~20 min | Final exhaustive sweep |

### Speed Settings (DO NOT REMOVE)

These are baked into `tune_rules_params.py` and `tune_rules_oos.py`:
- `TQDM_DISABLE=1` env var — eliminates progress bar I/O
- All backtesting.py loggers silenced to ERROR
- Data preloaded once before any backtest
- `ProcessPoolExecutor` with `--workers N` for parallelism
- Default 2026-05-20 baselines: universal best `et=0.60 mr=0.70 tsa=2.0 cb=0.10`

### Anti-Patterns

| Don't | Because |
|-------|----------|
| Run full grid (243) without multiprocessing | 4+ hours |
| Skip `--is-only` in Phase 1 | Doubles runtime pointlessly |
| Combine all instruments in one batch | ProcessPool saturates, harder to debug |
| Skip OOS validation | IS results are meaningless without OOS |
| Remove TQDM_DISABLE or logging suppression | Adds minutes of stderr I/O |

---
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **investment_trying_lab_private** (35,906 symbols, 54,953 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> **Auto-update:** GitNexus does NOT auto-refresh. Re-run `npx gitnexus analyze` after significant file changes. Check freshness with `npx gitnexus status`.

## Always Do

- **MUST consult `docs/stock_selection_criteria.md` whenever tasked with choosing tickers, instruments, or assembling training baskets.** The document defines 11 hard filters (F1-F11), 9 desirability scorecard dimensions (D1-D9), and context-specific override rules. Any ticker that fails hard filters must not enter the training universe or backtest pipeline.
- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/investment_trying_lab_private/context` | Codebase overview, check index freshness |
| `gitnexus://repo/investment_trying_lab_private/clusters` | All functional areas |
| `gitnexus://repo/investment_trying_lab_private/processes` | All execution flows |
| `gitnexus://repo/investment_trying_lab_private/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

---

# CodeGraphContext — Complementary Graph Intelligence (相輔相成)

This project is ALSO indexed by CodeGraphContext (CGC) — a Python-native MCP server that builds a queryable graph database of the codebase with different strengths than GitNexus. **Use BOTH together for defense-in-depth code intelligence.**

> Location: `C:\Dev\useful_repos\CodeGraphContext` | CLI: `cgc` | MCP Server: 21 tools

## GitNexus vs CGC — Complementary Roles

| Axis | **GitNexus** (Node.js) | **CGC** (Python) |
|------|----------------------|-------------------|
| **Best for** | Impact analysis, rename safety, change detection | Code search, Cypher queries, visual graph |
| **Core strength** | Execution flows (300 traced), blast radius | Multi-DB Cypher graph, 20 languages |
| **Pre-edit guard** | `impact` — tells you WHAT breaks | `analyze_code_relationships` — shows WHO calls |
| **Commit guard** | `detect_changes` — maps git diffs to symbols | N/A |
| **Safe rename** | `rename` — graph-assisted multi-file rename | Manual via Cypher + find |
| **Deep exploration** | `query` — hybrid BM25+vector semantic search | `find_code` — keyword search + fuzzy matching |
| **Schema query** | `cypher` — raw Cypher | `execute_cypher_query` — read-only Cypher |
| **Visualization** | Web UI graph explorer | Viz server + 2D/3D force graphs |
| **Portability** | LadybugDB (Kuzu) embedded | 5 DB backends (Kuzu, FalkorDB, Neo4j, Nornic, LadybugDB) |
| **Ecosystem** | npm/Node.js, TypeScript tooling | pip/Python, Typer CLI, SCIP deep indexers |
| **Graph schema** | Symbols + Processes + Clusters | Classes, Functions, Variables, Files, Imports, Inheritance |

## When to Use Each

| Situation | Use |
|-----------|-----|
| "What breaks if I change this function?" | **GitNexus** `impact` (blast radius + risk level) |
| "What files changed in the diff and which symbols are affected?" | **GitNexus** `detect_changes` |
| "I need to rename `foo()` to `bar()` across the codebase" | **GitNexus** `rename` (dry_run) |
| "Show me all callers of this function" | Either — GitNexus `context` or CGC `analyze_code_relationships` |
| "Find all functions matching this pattern" | **CGC** `find_code` (fuzzy search) |
| "Show me the class hierarchy for X" | **CGC** `analyze_code_relationships` (inheritance) |
| "Write a custom Cypher query to find patterns" | Either — same graph query language |
| "Who imports this module?" | **CGC** `analyze_code_relationships` (imports type) |
| "Find dead code across the codebase" | **CGC** `find_dead_code` |
| "Show me a visual graph of module dependencies" | **CGC** `visualize_graph_query` + viz server |
| "Pre-indexed bundles for reference repos" | **CGC** `load_bundle` / `search_registry_bundles` |

## CGC Quick Start

```bash
# Install (Python project — already in useful_repos)
cd C:\Dev\useful_repos\CodeGraphContext
pip install -e .

# Index this project
cgc index C:\Dev\projects\investment_trying

# CLI queries
cgc find "Huber"
cgc analyze callers "compute_mre_gap"
cgc analyze dead-code
cgc query complexity

# Start viz server
cgc visualize
```

## Search Productivity — Use GitNexus/CGC Instead of Grep/Glob

Both GitNexus and CGC index the entire codebase into knowledge graphs, enabling **concept-based search** that grep/glob cannot do. Agents in this project MUST prefer these tools for code discovery.

| Task | Use | Example |
|------|-----|---------|
| Find where LockBox is used | **GitNexus** `query` or `context` | `npx gitnexus query "LockBox blind holdout"` |
| Find all callers of a function | **GitNexus** `context` | `npx gitnexus context "create_lock_box"` |
| What breaks if I change X? | **GitNexus** `impact` | `npx gitnexus impact "NestedPurgedCV"` |
| Find all functions named "Huber" | **CGC** `find name` or `find pattern` | `cgc find name "Huber"` |
| Fuzzy search for "fuzzy logic" | **CGC** `find content` | `cgc find content "fuzzy logic"` |
| Find dead code | **CGC** `analyze dead-code` | `cgc analyze dead-code` |
| Show class hierarchy | **CGC** `analyze` | `cgc analyze inheritance "BaseOptimizer"` |
| which files changed and affected symbols | **GitNexus** `detect_changes` | `npx gitnexus detect_changes` |
| Raw Cypher query | Either | `npx gitnexus cypher "MATCH (f:Function) RETURN f.name LIMIT 10"` |

**This replaces:** `grep`, `rg`, `glob`, `codebase_search` for any search that involves understanding code structure, call relationships, or impact analysis.

---

## Keeping Indexes Fresh — MANDATORY Protocol

**Neither GitNexus nor CGC auto-updates by default.** After creating, modifying, or deleting files, you MUST refresh the indexes.

### Default: CGC Live Watcher (Auto)

CGC's `watch` runs as a background process, monitoring `src/` for file changes and auto-updating the graph:

```bash
# Start once per machine reboot (already running as of 2026-05-22)
cd C:\Dev\useful_repos\CodeGraphContext
$env:PYTHONIOENCODING = "utf-8"
uv run cgc watch C:\Dev\projects\investment_trying\src
```

**Agents:** verify the watcher is running with `Get-Job -Name "CGCWatcher"`. If missing, restart it.

### After Significant Changes: Reindex Both

After creating/deleting 5+ files, moving modules, or adding new packages:

```bash
# GitNexus: full reindex (~2 min)
npx gitnexus analyze

# CGC: force reindex of src/ (~1-4 min, tree-sitter parsing)
cd C:\Dev\useful_repos\CodeGraphContext
$env:PYTHONIOENCODING = "utf-8"
uv run cgc index --force C:\Dev\projects\investment_trying\src
```

### Freshness Check

```bash
npx gitnexus status          # Shows indexed commit vs HEAD
cgc stats C:\Dev\projects\investment_trying\src   # Shows file/function/class counts
```

### Auto-Refresh Checklist (Per-Session)

| Check | Command |
|-------|---------|
| GitNexus up to date? | `npx gitnexus status` — must show "✅ up-to-date" |
| CGC watcher alive? | `Get-Job -Name "CGCWatcher"` — must show "Running" |
| CGC index fresh? | `cgc stats` — function count should grow with new code |

---

## Phase 25: 66-Paper Master Comparison Report — Fully Implemented (2026-05-22)

All 42 implementable items from `useful_resources/papers_md/MASTER_COMPARISON_REPORT_2026-05-21.md` are now implemented.

### Anti-Overfitting Infrastructure

| Tool | File | Purpose | Paper |
|------|------|---------|-------|
| **LockBox** | `src/ml/lock_box.py` | Blind holdout, one-time access enforcement | C1 |
| **NestedPurgedCV** | `src/ml/nested_cv.py` | Inner=hyperparams, Outer=evaluation, never mix | C2 |
| **Blind Analysis** | `src/ml/blind_analysis.py` | Tune on scrambled labels, evaluate on true | C13 |
| **Label Shuffling** | `src/ml/label_shuffling.py` | Verify model doesn't exploit noise structure | C22 |
| **Huber Loss** | `src/ml/models/catboost_wrapper.py` + 2 others | `Huber:delta=1.0` replaces all RMSE regressors | A8 |

### New ML/Analysis Modules

| Tool | File | Purpose | Paper |
|------|------|---------|-------|
| **SVMRegimeClassifier** | `src/ml/svm_regime.py` | Raw price sequence SVM (82% precision) | B34 |
| **Dual Alpha/Beta** | `src/analysis/dual_alpha_beta.py` | Bull/bear alpha-beta + Chow test | E9 |
| **NSGA2Optimizer** | `src/optimization/nsga2_optimizer.py` | Pareto multi-objective optimization | B33 |
| **DynamicGAOptimizer** | `src/optimization/dynamic_ga.py` | Per-regime GA with associative memory | B35 |
| **FuzzyInferenceSystem** | `src/signals/fuzzy_system.py` | 5-state Mamdani fuzzy logic | B32 |

### NLP Sentiment Pipeline

| Tool | File | Purpose | Paper |
|------|------|---------|-------|
| **SVMTfidfSentiment** | `src/nlp/sentiment_pipeline.py` | SVM+TF-IDF (82-94% accuracy) | D3 |
| **BiLSTMSentiment** | `src/nlp/sentiment_pipeline.py` | 128d embed → BiLSTM(64u) → dropout(0.25) → LR(C=10) | D4 |
| **Distant Supervision** | `src/nlp/sentiment_pipeline.py` | :) / :( as noisy labels, 80%+ accuracy | D1 |
| **SentimentEnsemble** | `src/nlp/sentiment_pipeline.py` | RF+SVM+DT via AdaBoost (93.4%) | D8 |

### Mandatory Pre-Commit Checklist (UPDATED)

Every commit must now pass these additional checks:

| Check | Tool |
|-------|------|
| Blast radius for all edited symbols | `gitnexus_impact` before editing |
| Change detection before commit | `gitnexus_detect_changes` |
| GitNexus index fresh (stale → reindex) | `npx gitnexus status` |
| CGC index fresh (reindex if file count mismatch) | `cgc stats` |
| **NEW:** Label-shuffling test on new feature models | `src/ml/label_shuffling.py` |
| **NEW:** MRE-gap for any new ML model | `src/ml/model_validation.py` |
| **NEW:** Blind analysis for hyperparameter tuning | `src/ml/blind_analysis.py` |

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **investment_trying_lab_private** (36973 symbols, 56490 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/investment_trying_lab_private/context` | Codebase overview, check index freshness |
| `gitnexus://repo/investment_trying_lab_private/clusters` | All functional areas |
| `gitnexus://repo/investment_trying_lab_private/processes` | All execution flows |
| `gitnexus://repo/investment_trying_lab_private/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
