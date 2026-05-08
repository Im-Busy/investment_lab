---
type: setup
name: "Tooling Audit & Adoption"
status: complete
started: 2026-05-01
completed: 2026-05-01
completed_immediate: 2026-05-01
tools_installed: [pre-commit, jupytext, instructor, hypothesis, duckdb, pydantic-settings, mlflow, prefect (pipeline)]
configs_changed: []
reference: docs/tool-stack-audit.md
---

# Setup: Tooling Audit & Adoption

## Overview

Comprehensive audit of 18+ tools recommended for the Python backtesting + AI workflow.
Evaluated against the current codebase state (already-adopted Ruff, Mypy, Loguru, Pydantic,
PyArrow, Pytest). Result: 1 immediate action item, 7 deferred items gated on specific triggers.

## Immediate Tasks (Add Now) — ✅ COMPLETED 2026-05-01

| # | Task | Effort | Priority | Status |
|---|------|--------|----------|--------|
| T1 | Add root `.pre-commit-config.yaml` with ruff + mypy + standard file hooks | 10 min | 🔴 P0 | ✅ Done |
| T2 | Run `uv add pre-commit --group dev` (if not already present) | 2 min | 🔴 P0 | ✅ Done |
| T3 | Run `uv run pre-commit install` and verify on a test commit | 3 min | 🔴 P0 | ✅ Done |

## Deferred Tasks (Trigger-Gated) — ✅ ALL COMPLETED 2026-05-01

| # | Task | Effort | Trigger | Priority | Status |
|---|------|--------|---------|----------|--------|
| T4 | Migrate `src/config.py` to Pydantic `BaseSettings` | 2-4 hr | When Dockerizing or adding cloud deployment | 🟡 P1 | ✅ Done |
| T5 | Add Hypothesis property-based tests to top-10 pattern detectors | 2-4 hr | When pattern bugs surface or hardening for production | 🟡 P1 | ✅ Done |
| T6 | Add DuckDB for SQL feature engineering over Parquet | 2-4 hr | When feature SQL exceeds 20+ lines of pandas | 🟡 P2 | ✅ Done |
| T7 | Add MLflow experiment tracking (wrap existing ExperimentLogger) | 3-6 hr | When 20+ runs accumulated, or second person joins | 🟡 P2 | ✅ Done |
| T8 | Add Prefect workflow orchestration (replace Ploomber if needed) | 4-8 hr | When scheduled daily backtests or parallel runs needed | 🟡 P2 | ✅ Done |
| T9 | Add Jupytext for notebook version control | 15 min | When notebook diffs become unreadable | 🟡 P3 | ✅ Done |
| T10 | Add Instructor for structured LLM output | 30 min+ | When first LLM API call needs structured output | 🟡 P3 | ✅ Done |

## Rejected Tools

| Tool | Reason | Alternative |
|------|--------|-------------|
| Trigger.dev | TypeScript runtime — Python-only project | Prefect (T8) if orchestration needed |
| Mastra | TypeScript AI agent — no LLM integration exists | Instructor (T10) for structured output |
| SonarQube CE | Java server + PostgreSQL overhead. Ruff+Mypy cover 80%. | Revisit at 3+ devs |
| Polars | No usage, no bottleneck. backtesting.py requires pandas. | Keep pandas + Parquet |
| DataFusion, SQLite, ClickHouse | All alternatives for problems not present | DuckDB (T6) if SQL needed |
| DVC, Joblib/Diskcache, Sentry, LiteLLM, Phoenix | No matching need in current architecture | — |

## Completion Criteria

- [x] T1-T3: Pre-commit running on all commits
- [x] T4-T10: All deferred items implemented
- [x] Each deferred task reviewed quarterly against its trigger condition
