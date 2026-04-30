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

This project maintains a centralized **Command Cheatsheet** at COMMAND_CHEATSHEET.md in the root directory.

### Auto-Update Protocol for AI Assistants

**When implementing new features that include CLI scripts, commands, or workflows:**

1. **IMMEDIATELY UPDATE** COMMAND_CHEATSHEET.md as part of the same commit
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

---

## Progress Documentation — CRITICAL FOR ALL SESSIONS

This project maintains structured progress tracking in `progress_docs/`. This is the single source of truth for project state, phase status, pending work, and session context.

### File Structure
```
progress_docs/
├── README.md           # Navigation index + type catalog + conventions
├── current.md          # Session-level live log (append-at-top) — READ FIRST
├── plans/
│   ├── full.md         # Master plan — ALL phases/studies/evaluations/setups aggregated
│   ├── 01-*.md         # Phase plans (NN-short-name.md, type: phase)
│   ├── study-*.md      # Study plans (type: study)
│   ├── eval-*.md       # Evaluation plans (type: eval)
│   └── setup-*.md      # Setup plans (type: setup)
└── logs/
    ├── full.md         # Single chronological log — ALL types interleaved
    ├── 01-*.md         # Phase-specific logs
    ├── study-*.md      # Study-specific logs
    └── ...             # Mirrors plans/ structure
```

### At Session Start (MANDATORY)
1. **Read `progress_docs/current.md`** — understand all actions from the interrupted session
2. **Read `progress_docs/plans/full.md`** — see all phases, studies, deferred items, and pending work
3. **Read `progress_docs/README.md`** — check type catalog for conventions
4. Resume from the last incomplete action in `current.md`

### During Work (MANDATORY)
1. **Log every significant action** to `progress_docs/current.md` using the table format:
   ```
   | Time | Action | Files | Result |
   ```
2. **Determine activity type** from README.md type catalog (`phase`, `study`, `eval`, `setup`, `migration`)
3. **If no matching type exists**, create one:
   - Define the `type` name
   - Create plan file with `{type}-{descriptor}.md` naming
   - Add to README.md type catalog
   - Add section in `plans/full.md`

### When Completing a Task or Activity
1. **Update frontmatter** in the plan file (`status: complete`, add `completed` date)
2. **Update aggregated status** in `progress_docs/plans/full.md`
3. **Archive** `current.md` content into both `logs/full.md` and the type-specific log file
4. **Reset** `current.md` for the next activity

### When the User Defers Work
1. Mark the phase/task as `status: deferred` in YAML frontmatter
2. Add `deferred_reason` and `revisit_when` fields
3. Add to the **Deferred** section in `progress_docs/plans/full.md`
4. Log in `current.md`: `"Deferred Phase XX per user instruction"`

### File Conventions
| Convention | Rule |
|------------|------|
| **Folder** | `progress_docs/` — single entry point |
| **Naming** | Phases: `NN-short-name.md`. Non-phases: `{type}-{descriptor}.md`. All snake_case. |
| **Plan format** | Markdown + YAML frontmatter (metadata) + Markdown tables (tasks) |
| **Log format** | Markdown tables — append-only, chronological |
| **Completion marker** | YAML `status: complete` — NEVER rename files |
| **Deferral marker** | YAML `status: deferred` + `deferred_reason` + `revisit_when` |
| **Aggregation** | `plans/full.md` has Deferred + Pending sections pulled from all files |

### Legacy Files
- `plans/pending_items.md` — superseded by `progress_docs/plans/full.md`
- `HANDOVER*.md`, `PHASE*.md` at root — content archived into `progress_docs/logs/`


