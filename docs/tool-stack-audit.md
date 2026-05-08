# Tool Stack Audit

**Date:** 2026-05-01
**Context:** Evaluation of Trigger.dev + Mastra + SonarQube CE stack choice, plus 15 additional tool recommendations for a Python backtesting + AI workflow.

---

## ✅ Already Adopted — No Action Required

| # | Tool | Category | How It's Used |
|---|------|----------|--------------|
| 1 | **Ruff** | Linting/Formatting | `dev` deps. Configured in `[tool.ruff]`. `line-length=100`. |
| 2 | **Mypy** | Type Checking | `dev` deps. Configured in `[tool.mypy]`. `ignore_missing_imports=true`. |
| 3 | **Loguru** | Logging | `dependencies` (v0.7.3). Used across `src/` via `from loguru import logger`. 202+ call sites. |
| 4 | **Pydantic** | Data Validation | `dependencies` (v2.12.5). Config uses `@dataclass` currently, not `BaseModel`. |
| 5 | **PyArrow** | Parquet I/O | `dependencies` (v23.0.1). Feature store writes/reads `.parquet`. `helpers.py` handles `.parquet` extensions. |
| 6 | **Pytest + pytest-cov** | Testing | `dev` deps. 30 test files covering patterns, ML, indicators, risk. |

---

## 🔴 Add Now

### 7. Pre-commit (root `.pre-commit-config.yaml`)

- **Category:** Code quality gate
- **Why:** Ruff + Mypy exist but are only enforced manually. Pre-commit runs them automatically before every `git commit`, catching issues before they reach CI. Highest-ROI quality gate at zero operational cost.
- **How:** Add `.pre-commit-config.yaml` at project root with `ruff`, `mypy`, and standard file checks. Run `uv run pre-commit install` once.
- **Effort:** 10 minutes.
- **Risk:** None — additive only. Doesn't change existing code.

---

## 🟡 Deferred — Worth Doing When Triggered

### 8. Pydantic `BaseSettings` for Config

- **Category:** Config management
- **Why:** Current `src/config.py` (397 lines) uses `@dataclass` with manual YAML/JSON I/O. No env-var overrides. Pydantic `BaseSettings` gives `BT_START_DATE=2022-01-01` for free, field validation, and `.env` support — essential for Docker/cloud.
- **How:** Replace `@dataclass` config classes with `BaseSettings` subclasses. Update consumers to use `model_dump()` instead of `vars()`.
- **Effort:** 2-4 hours. Non-trivial refactor touching `to_dict()`, `from_dict()`, `save()`, `load()`, and all callers.
- **Trigger:** When Dockerizing or adding cloud deployment.

### 9. Hypothesis for Property-Based Testing

- **Category:** Testing
- **Why:** 34 pattern detectors process price data. Edge cases (NaN gaps, flat markets, extreme volatility) can produce silent garbage. A single `@given` decorator can discover breaking inputs.
- **How:** `uv add hypothesis --group dev`. Add property tests to top ~10 pattern detectors: `@given(st.lists(st.floats(min_value=0), min_size=5))` + assert confidence is in [0,1] and no exceptions.
- **Effort:** 2-4 hours. Additive only.
- **Trigger:** When pattern detection bugs surface in backtests, or when hardening detectors for production.

### 10. DuckDB for SQL Feature Engineering

- **Category:** Data processing
- **Why:** Parquet is already used. DuckDB adds SQL querying over Parquet natively. Useful when feature engineering requires cross-ticker joins or complex rolling windows.
- **How:** `uv add duckdb`. `duckdb.query("SELECT ... FROM read_parquet('...')").df()`.
- **Effort:** 2-4 hours for initial integration. Depends on query complexity.
- **Trigger:** When feature engineering SQL exceeds 20+ lines of pandas, or joining >3 Parquet files in one query.

### 11. MLflow for Experiment Tracking

- **Category:** MLOps
- **Why:** You have `src/ml/experiment_logger.py` (476 lines) — structured JSONL logging with metadata, config, fold metrics, feature importance, and summary verdicts. MLflow adds a comparison UI and remote tracking, but your DIY logger already answers "which config produced this equity curve?"
- **How:** `uv add mlflow`. Wrap existing `ExperimentLogger` calls with `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`.
- **Effort:** 3-6 hours. Additive — doesn't replace existing logger.
- **Trigger:** When you can't remember which run produced the best result across 20+ experiments, or when a second person needs to browse experiment history.

### 12. Prefect for Workflow Orchestration

- **Category:** Workflow
- **Why:** Current pipeline is Ploomber (`pipeline.yaml`) — linear, synchronous, single-machine. Prefect adds retries, scheduling, caching, and a dashboard.
- **How:** `uv add prefect`. Convert Ploomber tasks to `@task`/`@flow` decorators.
- **Effort:** 4-8 hours. Requires refactoring pipeline tasks.
- **Trigger:** When you need scheduled daily backtests, parallel symbol runs, or webhook-triggered pipelines.

### 13. Jupytext for Notebook Version Control

- **Category:** DevEx
- **Why:** Project uses Jupyter + Papermill. Jupytext syncs `.ipynb ↔ .py:percent`, making notebook diffs readable in git.
- **How:** `uv add jupytext`. `jupytext --set-formats ipynb,py:percent path/to/notebook.ipynb`.
- **Effort:** 15 minutes. Additive only.
- **Trigger:** When notebook diffs in PRs become unreadable JSON blobs.

### 14. Instructor for Structured LLM Output

- **Category:** AI/LLM
- **Why:** Forces LLMs to return validated Pydantic models, eliminating JSON parsing bugs. Only relevant once you add LLM-powered features.
- **How:** `uv add instructor`. `client = instructor.patch(OpenAI())`. Pass `response_model=PatternAnalysis`.
- **Effort:** 30 minutes for first integration. Hours for full feature.
- **Trigger:** When you first make LLM API calls that need structured output.

---

## ⚪ Not Adding — Wrong Fit or Premature

| # | Tool | Why Not | Alternative |
|---|------|---------|-------------|
| 15 | **Trigger.dev** | TypeScript runtime. Python-only project. Requires Node.js sidecar → language boundary overhead. | Prefect (#12) if/when orchestration needed. |
| 16 | **Mastra** | TypeScript AI agent framework. Zero LLM integration exists. Requires Node.js sidecar. | Instructor (#14) for structured LLM output; LangChain/LangGraph if agent orchestration needed. |
| 17 | **SonarQube CE** | Requires Java server + PostgreSQL (~2 GB RAM). Ruff + Mypy already cover 80% of value. Operational overhead not justified for solo dev. | Revisit when 3+ developers or need PR quality gates. |
| 18 | **Polars** | No usage exists. `backtesting.py` requires pandas. No proven DataFrame bottleneck. | Keep pandas + Parquet. Revisit if profiling shows DataFrame ops >20% runtime. |
| 19 | **Apache DataFusion** | Same as Polars — alternative for a problem not present. | — |
| 20 | **SQLite** | Row-oriented, not analytical. Wrong tool for time-series backtesting queries. | DuckDB (#10) if SQL needed. |
| 21 | **ClickHouse** | Distributed server for terabytes. Overkill — you run single-ticker backtests locally. | — |
| 22 | **DVC** | Data files not bloating git. No multi-machine reproducibility needs. | — |
| 23 | **Joblib.Memory / Diskcache** | `ablation_engine.py` already has JSON-based results cache. | — |
| 24 | **Sentry** | No deployed services. All execution is local CLI scripts. | Revisit if/when deploying. |
| 25 | **LiteLLM** | No LLM calls to route. | Revisit after first LLM integration. |
| 26 | **Phoenix / LangSmith** | No LLM traces to observe. | Revisit after LLM features in production. |

---

## Summary

| Verdict | Count | Items |
|--------|-------|-------|
| Already adopted | 6 | Ruff, Mypy, Loguru, Pydantic, PyArrow, Pytest |
| **Add now** | **1** | **Pre-commit** |
| Deferred (trigger-gated) | 7 | Pydantic Settings, Hypothesis, DuckDB, MLflow, Prefect, Jupytext, Instructor |
| Not adding | 12 | Trigger.dev, Mastra, SonarQube CE, Polars, DataFusion, SQLite, ClickHouse, DVC, Joblib/Diskcache, Sentry, LiteLLM, Phoenix/LangSmith |
