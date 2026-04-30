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

## Pending Items Tracking — CRITICAL FOR ALL FUTURE SESSIONS

This project maintains a centralized **Pending Items** file at `plans/pending_items.md`.

### Instructions for All AI Assistants

**At session start:**
1. Read `plans/pending_items.md` to understand what work is pending
2. Items are ordered by priority (P0 = critical → P3 = low)
3. Use this file to determine what to work on next when given open-ended tasks

**When completing a task:**
1. **DELETE the item** from `plans/pending_items.md`
2. **Update the status** in the original plan file (e.g., change `[ ]` to `[x]`, update progress percentage)
3. If the item was not in pending_items.md but should have been, add it before deleting

**When creating a new plan:**
1. **Add all actionable items** to `plans/pending_items.md` with appropriate priority level
2. Reference the source plan file in each item
3. Keep items specific and actionable (not vague goals)

**When planning what to work on:**
1. Check `plans/pending_items.md` first
2. Consider dependencies (some P2 items depend on P0/P1 completion)
3. If user asks "what's left to do?" — reference this file

### File Locations
- **Pending Items:** `plans/pending_items.md` (single source of truth for pending work)
- **Original Plans:** `plans/*.md` (detailed implementation plans — update status here too)
- **Completed Items Reference:** Bottom of `plans/pending_items.md` (for historical context)


