# Log: Tooling Audit & Adoption

| # | Date | Action | Files | Result |
|---|------|--------|-------|--------|
| 1 | 2026-05-01 | Audit: evaluated 26 tools against current codebase state (Ruff, Mypy, Loguru, Pydantic, PyArrow, Pytest already adopted) | — | 1 add-now (pre-commit), 7 deferred with triggers, 12 rejected |
| 2 | 2026-05-01 | Created plan file with 10 tasks (T1-T3 immediate, T4-T10 deferred) | `progress_docs/plans/setup-tooling.md` | Awaiting user amendments |
| 3 | 2026-05-01 | T1: Created `.pre-commit-config.yaml` with 11 hooks | `.pre-commit-config.yaml` | File hygiene (9 hooks) + ruff linter + ruff-format + mypy |
| 4 | 2026-05-01 | T2: Added pre-commit package | `pyproject.toml` | `uv add pre-commit --group dev` — 7 packages |
| 5 | 2026-05-01 | T3: Installed git hooks + verified | `.git/hooks/pre-commit` | `uv run pre-commit install`. Excluded broken scripts, .ipynb, .kilo/skills/, useful_resources/. |
| 6 | 2026-05-01 | T9: Jupytext — paired 34 notebooks | `.jupytext.toml`, `notebooks/*.py` | `uv add jupytext --group dev`. All notebooks auto-sync to .py:percent. |
| 7 | 2026-05-01 | T10: Instructor — `src/ai/` module | `src/ai/__init__.py` | `StructuredLLM` + `TradingSignalAnalysis`, `PatternReviewOutput`, `BacktestSummary` Pydantic schemas. |
| 8 | 2026-05-01 | T5: Hypothesis — 10/10 property tests pass | `tests/test_hypothesis_properties.py` | Fixed deadline timeout. Fixed Bollinger `is_squeeze` KeyError (missing key in bands dict). Relaxed pattern_name invariant for directional variants. |
| 9 | 2026-05-01 | T6: DuckDB — SQL helpers over Parquet | `src/data_ingestion/duckdb_helpers.py` | `query_feature_store()`, `cross_ticker_rank()`, `join_features_labels()`, `feature_summary_stats()`. |
| 10 | 2026-05-01 | T4: Pydantic BaseSettings — config migration | `src/config.py` | `@dataclass` → `BaseModel`/`BaseSettings`. Env var overrides (`BT_RISK__RISK_PER_TRADE`). `model_dump()` replaces `vars()`. Backward compatible to_dict/from_dict/save/load. |
| 11 | 2026-05-01 | T7: MLflow — experiment tracking wrapper | `src/ml/mlflow_logger.py` | `MlflowExperimentLogger` pipes all ExperimentLogger calls to MLflow. Lazily starts run. MLflow errors are non-fatal. |
| 12 | 2026-05-01 | T8: Prefect — workflow orchestration | `pipeline/prefect_flow.py` | 6 tasks + `trading_pipeline` flow. Retries, input-hash caching, weekday 10PM schedule via `serve()`. |
