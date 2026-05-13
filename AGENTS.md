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
- Tool: `useful_resources/useful_repos/paper2md/`
- Input: `useful_resources/papers_md/*.md` (markdown papers)
- Output: `useful_resources/useful_repos/paper2md/output/SENTIMENT_ANALYSIS_SUMMARY.md`
- Final copy: `useful_resources/papers_md/SENTIMENT_ANALYSIS_SUMMARY.md`

### API Configuration
The tool uses OpenAI-compatible endpoints via OpenRouter:
- **Provider**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Model**: `deepseek/deepseek-v4-flash` (DeepSeek V4 Flash - fastest and strongest)
- Config file: `paper2md/.env`

### Usage
```bash
cd useful_resources/useful_repos/paper2md
uv run python summarize_md_papers.py --papers-dir ..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md
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

---

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

The project exposes five primary workflows as Kilo agents and slash commands. These agents know the project's conventions, pitfalls, and baselines — they should be used instead of manual CLI commands for consistency.

### Agent Catalog

| Agent | File | Purpose |
|-------|------|---------|
| **model-doctor** | `.kilo/agent/model-doctor.md` | Runs calibration audit, regime shift investigation, WFO comparison. Produces health report with remediation playbook. |
| **backtest-runner** | `.kilo/agent/backtest-runner.md` | Executes ML strategy backtests, updates BESTS.md leaderboard, interprets results against known baselines. |
| **ml-trainer** | `.kilo/agent/ml-trainer.md` | Trains CatBoost models via 9-stage pipeline. Knows triple-barrier labels, PurgedKFold, ATR normalization, overfitting thresholds. |
| **repo-syncer** | `.kilo/agent/repo-syncer.md` | Syncs curated files from private dev repo to public-facing repo. Merges main→public, strips private data, pushes only the clean public branch. |
| **housekeeper** | `.kilo/agent/housekeeper.md` | Audits file system, flags misplaced files and duplicate dirs, produces safe migration plan. |

### Slash Commands

| Command | Agent | Usage |
|---------|-------|-------|
| `/model-diagnose` | model-doctor | Full diagnostic suite. No arguments needed. |
| `/backtest` | backtest-runner | Run backtests with `--entry-threshold --trail-stop` etc. |
| `/train-ml` | ml-trainer | Train models with `--symbol / --basket / --fast / --walk-forward` |
| `/repo-sync` | repo-syncer | Sync curated files to public repo. `/repo-sync check` for safety-only. |
| `/housekeeper` | housekeeper | Audit file system, produce migration plan. |

### When to Use Slash Commands vs Direct CLI

| Situation | Use |
|-----------|-----|
| Training a new model | `/train-ml` — agent knows overfitting thresholds and ATR fix |
| Running a backtest | `/backtest` — agent auto-updates BESTS.md |
| Checking model health | `/model-diagnose` — agent runs all 3 diagnostics + interprets |
| Syncing to public repo | `/repo-sync` — agent handles merge, safety check, and push |
| Quick one-off script | Direct CLI — e.g., `uv run scripts/sweep_entry_thresholds.py SPY` |

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
