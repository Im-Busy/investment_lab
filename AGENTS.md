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

## Paper Analysis & Summarization

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
4. **`docs/research_logic_map/insight_registry.md`** — All research insights (65 from 20 sources), tagged by topic/impact/status. Source of truth for what is known and what remains to implement.
5. **`docs/BESTS_INSIGHTS.md`** — Distilled knowledge from every backtest in BESTS.md. Factor rankings, strategy tier list, anti-patterns, production configs. The "what works and what to never do" file.

**IMPORTANT: After reading `BESTS_INSIGHTS.md`, check `BESTS.md` for its `last_updated` timestamp.** If BESTS.md changed significantly since BESTS_INSIGHTS.md was last synced (new top-3 results, new strategy categories, regime shift detected), re-evaluate the insights and update both files.

After reading state, consult `.kilo/project-loop.md` for the DEEPEN/BROADEN/PIVOT/CONCLUDE decision framework. Every phase or significant experiment must end with an explicit direction decision.

The agent updates `MEMORY.md` continuously during the session. At session end, the agent writes a detailed handover to `progress_docs/handovers/` and updates both `MEMORY.md` and `progress_docs/current.md`.

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

### Slash Commands

| Command | Agent | Usage |
|---------|-------|-------|
| `/model-diagnose` | model-doctor | Full diagnostic suite. No arguments needed. |
| `/backtest` | backtest-runner | Run backtests with `--entry-threshold --trail-stop` etc. |
| `/train-ml` | ml-trainer | Train models with `--symbol / --basket / --fast / --walk-forward` |
| `/repo-sync` | repo-syncer | Sync curated files to public repo. `/repo-sync check` for safety-only. |
| `/housekeeper` | housekeeper | Audit file system, produce migration plan. |
| `/research` | researcher | Search for papers, repos, benchmarks. `/research regime-switching HMM` |

### When to Use Slash Commands vs Direct CLI

| Situation | Use |
|-----------|-----|
| Training a new model | `/train-ml` — agent knows overfitting thresholds and ATR fix |
| Running a backtest | `/backtest` — agent auto-updates BESTS.md |
| Checking model health | `/model-diagnose` — agent runs all 3 diagnostics + interprets |
| Syncing to public repo | `/repo-sync` — agent handles merge, safety check, and push |
| Quick one-off script | Direct CLI — e.g., `uv run scripts/sweep_entry_thresholds.py SPY` |
| Searching for papers/implementations | `/research` — agent knows search hierarchy and ingestion pipeline |

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
└── experiments/   # Experiment outputs
```

### Anti-Patterns

- `.py` files at root → belongs in `src/`, `tests/`, or `scripts/`
- `.md` files at root (except `AGENTS.md`, `README.md`) → belongs in `docs/`
- Duplicate directories (`logs/` + `fin_logs/`, `output/` + `outputs/`)
- Full cloned repos at root → belongs in `useful_resources/useful_repos/`
- ML artifacts at root (`catboost_info/`, `AutogluonModels/`) → belongs in `outputs/`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **investment_trying_lab_private** (28316 symbols, 43736 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
