# Global Rules (All Modes)

## Behavior
- Execute the task. No preamble, no recap of what was asked.
- If a task needs multiple steps, do them all. Summarize what was done at the end.
- If something breaks during implementation, attempt the fix before reporting.
- State assumptions when requirements are ambiguous. Then proceed — don't stall.
- Never apologize. Never say "Great question." Just answer.

## Python Conventions
- Type hints on all function signatures and return types.
- Docstrings on public functions/classes (Google style, 3 lines max).
- `pathlib.Path` over `os.path`.
- f-strings over `.format()` or `%`.
- Specific exception types — never bare `except:`.
- `logging` over `print` in non-throwaway code.
- Constants: `UPPER_SNAKE_CASE` at module top.
- Private helpers: prefix with `_`.

## Prohibited
- No `from module import *`.
- No mutable default arguments (`def f(x=[])`).
- No `type: ignore` without a comment explaining why.
- No hardcoded secrets, paths, or env-specific values.
- No nested functions beyond 1 level (unless decorator/closure).

## Structure
- Snake_case for all files and directories.
- Modules under ~300 lines. Split if larger.
- Tests mirror source: `src/auth/service.py` → `tests/auth/test_service.py`.
- One substantial class per file. Small dataclasses/enums can share.
