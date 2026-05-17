# Global Rules

## Behavioral Standards
- Execute the task. No preamble, no recap of what was asked.
- If a task requires multiple steps, do them all. Summarize what was done at the end.
- If something breaks during implementation, attempt the fix before reporting the error.
- State assumptions explicitly when requirements are ambiguous. Then proceed with the best assumption — don't stall.
- Never apologize. Never say "Great question." Just answer.
- Use TodoWrite for tasks requiring 3+ distinct steps. Update status in real-time. One task in_progress at a time.
- When user asks about Kilo capabilities, check `https://kilo.ai/docs` first.
- NEVER end responses with questions or offers for further assistance.

## Git Discipline
- NEVER commit changes unless the user explicitly asks you to. Only commit when explicitly asked.
- When committing: run `git status`, `git diff`, `git log` in parallel, then analyze and draft message.
- Commit messages focus on the "why" not the "what".
- NEVER run destructive/irreversible git commands (push --force, hard reset) unless explicitly requested.
- NEVER skip hooks (--no-verify, --no-gpg-sign) unless explicitly requested.
- Warn user if they request force push to main/master.
- If commit fails due to pre-commit hook: fix the issue and create a NEW commit. NEVER amend unless HEAD was created by you in this session, was NOT pushed, AND user requested amend.

## Debugging Protocol
- When an error occurs: read the full traceback, identify the root cause, attempt a fix based on context.
- Check neighboring files and existing patterns before implementing fixes.
- Use appropriate tools: `grep` for content search, `glob` for file patterns, `read` for file inspection.
- If a fix attempt fails, analyze why before trying again. Maximum 5 fix attempts before reporting.
- Never mask errors with broad try/except blocks.

## Python Tooling: `uv` (astral.sh) — MANDATORY

All Python execution and package management uses `uv`. No exceptions unless `uv` fails and the user explicitly approves a fallback.

### Execution
- `uv run <script>` — NEVER `python <script>`
- `uv run -m <module>` — NEVER `python -m <module>`
- `uv run <tool>` (e.g., `uv run pytest`, `uv run ruff`) — NEVER bare tool invocations

### Dependencies
- `uv add <package>` → project dependencies (updates `pyproject.toml`)
- `uv pip install <package>` → only for ad-hoc installs when explicitly requested
- `uv sync` → install from `pyproject.toml` / `uv.lock`
- NEVER use bare `pip install`

### Environment
- NEVER create venvs manually (`python -m venv`)
- NEVER suggest activation commands (`source .venv/bin/activate`)
- `uv run` handles isolation automatically

### Project Bootstrap
- Always prefer `pyproject.toml` over `requirements.txt`
- If `pyproject.toml` is missing → `uv init`

## Coding Conventions
- Type hints on all function signatures and return types.
- Docstrings on public functions/classes (Google style, 1-3 lines max).
- `pathlib.Path` over `os.path`.
- f-strings over `.format()` or `%`.
- Specific exception handling — never bare `except:` or `except Exception:` without re-raise.
- `logging` over `print` in any non-throwaway code.
- Constants in UPPER_SNAKE_CASE at module top.
- Private helpers prefixed with `_`.

## Prohibited Patterns
- No wildcard imports (`from module import *`).
- No mutable default arguments (`def f(x=[])`).
- No nested functions beyond 1 level deep unless it's a decorator/closure.
- No `type: ignore` without an accompanying comment explaining why.
- No hardcoded secrets, paths, or environment-specific values. Use env vars or config.

## File & Project Structure
- Snake_case for all Python files and directories.
- Keep modules under ~300 lines. Split if larger.
- Tests mirror source structure: `src/auth/service.py` → `tests/auth/test_service.py`.
- One class per file for substantial classes. Small dataclasses/enums can share a file.

## Testing Standards
- Every new public function must have at least one test.
- Test edge cases: empty inputs, boundary conditions, error paths.
- For trading strategies: test signal generation, position sizing, and exit conditions independently.
- Never assume test framework — check `pyproject.toml` or existing tests for `pytest`/`unittest` conventions.
- Run tests after implementation changes: `uv run pytest` or the project's test command.
- If tests fail due to your changes, fix them before reporting completion.

## Bash & Tool Usage
- Use `workdir` parameter instead of `cd <dir> && <command>`.
- Quote file paths with spaces: `rm "path with spaces/file.txt"`.
- Run independent commands in parallel (e.g., `git status` + `git diff` + `git log` in one message).
- For file operations: use `glob`, `grep`, `read` — NEVER `find`, `grep -r`, `cat` via bash.
- When making code changes: `read` the file first, then `edit`. Never write unless creating new.
- Use `edit` for modifications. Use `replaceAll` when renaming across a file.

## Communication & Progress
- Report task completion concisely: what was done, any assumptions made, next steps if applicable.
- When implementing multi-phase plans: report which phases were completed vs deferred.
- Ask questions via the `question` tool when choices need to be made. Batch questions together.
- For complex explorations: summarize findings in a table or structured format.
- When backtesting results are generated: always present key metrics (return, Sharpe, max drawdown, win rate) in a comparison table.

## Anti-Rationalization Guardrails (Trading System)
The following excuses are INVALID. If you think any of them apply, stop and re-evaluate.

| Rationalization | Reality |
|----------------|---------|
| "I'll add OOS validation later" | OOS is the only validation that matters. IS results without OOS are meaningless. |
| "The model's AUC is close enough" | AUC < 0.55 is random. Gap > 0.05 is overfit. There are no exceptions. |
| "One more feature will fix it" | Adding features without validation adds noise. Stability Selection > feature count. |
| "The backtest looks good in-sample" | In-sample results prove nothing. Always check OOS (2025+) performance. |
| "This is too small for a spec" | Any change touching signal generation, ML pipeline, or backtest engine needs a spec. |
| "I can skip the look-ahead check" | Look-ahead bias is the #1 source of phantom alpha. Always verify with check_lookahead(). |
| "Let me tune until it passes" | Tuning on test data = overfitting. Set parameters on train, validate once on test. |
| "The pattern worked on SPY" | Single-instrument results don't generalize. Validate across 5+ instruments. |
| "I'll add tests after it works" | Untested strategy code is production debt. At minimum: smoke test + OOS backtest. |
| "Regime shift won't happen again" | Regime shifts (like 2025) are the norm, not the exception. Every strategy must survive them. |
