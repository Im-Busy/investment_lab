# Pixi to uv Migration Plan

## Overview

This document outlines the migration strategy from **pixi** to **uv** for the investment_trying project. uv is an extremely fast Python package and project manager written in Rust that can replace pip, poetry, pipx, and other tools.

## User Preferences (Confirmed)

- ✅ Use **pandas-ta** instead of TA-Lib (avoids C library dependency)
- ✅ Include **all packages** from environment_backup.yml
- ✅ **Remove requirements.txt** after migration
- ✅ Configure VS Code for uv Python interpreter
- ✅ uv installed via scoop

## Current Setup Analysis

### Existing Configuration Files

| File | Purpose | Action |
|------|---------|--------|
| [`pixi.toml`](pixi.toml) | Current pixi configuration | Delete after migration |
| [`requirements.txt`](requirements.txt) | Pip-compatible requirements | Delete after migration |
| [`environment_backup.yml`](environment_backup.yml) | Original conda environment export | Keep as reference |

### Dependencies to Include (from environment_backup.yml)

**Core Data Science:**
- Python 3.13
- pandas >= 2.0.0 (pandas 3.0.1 in backup)
- numpy >= 1.24.0 (numpy 2.4.2 in backup)
- scipy >= 1.17.0
- statsmodels >= 0.14.6

**Machine Learning:**
- scikit-learn >= 1.8.0
- lightgbm >= 4.6.0
- xgboost >= 3.1.3

**Backtesting & Trading:**
- backtesting >= 0.6.5
- yfinance >= 1.2.0
- pandas-ta >= 0.3.14b (replacing TA-Lib)
- quantstats >= 0.0.62
- mplfinance >= 0.12.10

**Visualization:**
- matplotlib >= 3.10.0
- plotly >= 5.18.0
- seaborn >= 0.13.2
- bokeh >= 3.8.2

**Jupyter & Notebooks:**
- jupyter >= 1.0.0
- jupyterlab >= 4.5.5
- ipykernel >= 7.2.0
- ipywidgets >= 8.0.0
- notebook

**Google Cloud (from pip section):**
- google-cloud-aiplatform >= 1.139.0
- google-cloud-bigquery >= 3.40.1
- google-cloud-storage >= 3.9.0
- google-genai >= 1.65.0

**Utilities:**
- loguru >= 0.7.3
- python-dotenv >= 1.2.1
- requests >= 2.32.5
- pyyaml >= 6.0.3
- tqdm >= 4.67.3
- pydantic >= 2.12.5

**Dev Dependencies:**
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- ruff (linter/formatter)
- mypy >= 1.0.0

---

## Migration Strategy

### Phase 1: Preparation

```mermaid
flowchart LR
    A[Backup current environment] --> B[Install uv]
    B --> C[Verify Python version]
    C --> D[Create pyproject.toml]
```

#### Step 1.1: Install uv

uv can be installed via PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via pip if already available:

```bash
pip install uv
```

#### Step 1.2: Verify Python Version

The project uses Python 3.13. uv can manage Python versions directly:

```bash
uv python install 3.13
uv python pin 3.13
```

This creates a `.python-version` file for consistent Python version management.

---

### Phase 2: Create pyproject.toml

uv uses the standard `pyproject.toml` format (PEP 621). Here is the target configuration with all packages from environment_backup.yml:

```toml
[project]
name = "investment-trying"
version = "0.1.0"
description = "Trading Pattern Detection System"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
readme = "README.md"
requires-python = ">=3.13,<3.14"
dependencies = [
    # Core Data Science
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.17.0",
    "statsmodels>=0.14.6",
    
    # Machine Learning
    "scikit-learn>=1.8.0",
    "lightgbm>=4.6.0",
    "xgboost>=3.1.3",
    
    # Backtesting & Trading
    "backtesting>=0.6.5",
    "yfinance>=1.2.0",
    "pandas-ta>=0.3.14b",
    "quantstats>=0.0.62",
    "mplfinance>=0.12.10",
    
    # Visualization
    "matplotlib>=3.10.0",
    "plotly>=5.18.0",
    "seaborn>=0.13.2",
    "bokeh>=3.8.2",
    
    # Jupyter & Notebooks
    "jupyter>=1.0.0",
    "jupyterlab>=4.5.5",
    "ipykernel>=7.2.0",
    "ipywidgets>=8.0.0",
    "notebook",
    
    # Google Cloud
    "google-cloud-aiplatform>=1.139.0",
    "google-cloud-bigquery>=3.40.1",
    "google-cloud-storage>=3.9.0",
    "google-genai>=1.65.0",
    
    # Utilities
    "loguru>=0.7.3",
    "python-dotenv>=1.2.1",
    "requests>=2.32.5",
    "pyyaml>=6.0.3",
    "tqdm>=4.67.3",
    "pydantic>=2.12.5",
]

[dependency-groups]
dev = [
    "ruff",
    "mypy>=1.0.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[tool.uv]
# uv-specific settings can go here

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
```

#### Step 2.1: Initialize Project

```bash
# Option A: Initialize fresh and add dependencies
uv init

# Option B: Create pyproject.toml manually with the content above
```

#### Step 2.2: Add Dependencies

Since we're creating pyproject.toml manually with all dependencies, we don't need to add them individually. Just run:

```bash
# Sync dependencies to create virtual environment
uv sync

# Sync including dev dependencies
uv sync --group dev
```

---

### Phase 3: Task Migration

Pixi tasks map to uv run commands or scripts.

| Pixi Task | uv Equivalent |
|-----------|---------------|
| `pixi run check` | `uv run python -c "..."` |
| `pixi run test` | `uv run pytest tests/ -v --tb=short` |
| `pixi run test-cov` | `uv run pytest tests/ -v --cov=src --cov-report=term-missing` |
| `pixi run notebook` | `uv run jupyter notebook` |
| `pixi run lab` | `uv run jupyter lab` |
| `pixi run fetch-data` | `uv run python -m src.data_ingestion.fetch_data` |
| `pixi run run` | `uv run python -m src.main` |
| `pixi run -e dev lint` | `uv run --group dev ruff check src/` |
| `pixi run -e dev format` | `uv run --group dev ruff format src/` |
| `pixi run -e dev typecheck` | `uv run --group dev mypy src/` |

#### Option: Define Scripts in pyproject.toml

```toml
[project.scripts]
fetch-data = "src.data_ingestion.fetch_data:main"
run-app = "src.main:main"

[project.entry-points."console_scripts"]
# Alternative entry point style
```

---

### Phase 4: Clean Up

#### Files to Remove

| File | Action |
|------|--------|
| `pixi.toml` | Delete after migration |
| `.pixi/` directory | Delete after migration |
| `requirements.txt` | Delete after migration (per user preference) |

#### Files to Create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Main project configuration |
| `uv.lock` | Lock file (auto-generated) |
| `.python-version` | Python version pin |

---

## Migration Commands Summary

```bash
# 1. Verify uv is installed (already installed via scoop)
uv --version

# 2. Navigate to project
cd C:\Dev\projects\investment_trying

# 3. Pin Python version
uv python install 3.13
uv python pin 3.13

# 4. Create pyproject.toml (manually or via uv init)
# We will create it manually with all dependencies

# 5. Sync/create virtual environment with all dependencies
uv sync

# 6. Sync including dev dependencies
uv sync --group dev

# 7. Verify installation
uv run python -c "import pandas; import numpy; import backtesting; print('All imports successful!')"

# 8. Run tests
uv run pytest tests/ -v

# 9. Clean up old files (after verification)
# del pixi.toml
# rmdir /s .pixi
# del requirements.txt
```

---

## Verification Checklist

- [x] uv installed via scoop
- [ ] Python 3.13 pinned via `.python-version`
- [ ] `pyproject.toml` created with all dependencies
- [ ] `uv.lock` generated
- [ ] Virtual environment created (`.venv/`)
- [ ] All imports work: `uv run python -c "import pandas, numpy, backtesting"`
- [ ] Tests pass: `uv run pytest tests/ -v`
- [ ] Jupyter works: `uv run jupyter lab`
- [ ] Dev tools work: `uv run --group dev ruff check src/`
- [ ] VS Code configured for uv Python interpreter
- [ ] Old files removed: `pixi.toml`, `.pixi/`, `requirements.txt`

---

## Benefits of uv

1. **Speed** - 10-100x faster than pip/pixi for dependency resolution
2. **Simplicity** - Single tool replaces pip, pip-tools, poetry, pyenv
3. **Standard** - Uses `pyproject.toml` (PEP 621 standard)
4. **Lock file** - Deterministic builds with `uv.lock`
5. **Python management** - Built-in Python version management
6. **Compatibility** - Works with existing `requirements.txt` files

---

## Rollback Plan

If issues arise:

1. Keep `environment_backup.yml` as reference
2. Can reinstall pixi and use `pixi.toml` until issues are resolved (keep a backup)
3. uv virtual environment (`.venv/`) can be deleted and recreated
4. `pyproject.toml` can be deleted and start fresh

---

## User Decisions (Confirmed)

1. ✅ **TA-Lib**: Use `pandas-ta` instead (avoids C library dependency)
2. ✅ **Packages**: Include all packages from `environment_backup.yml`
3. ✅ **requirements.txt**: Remove after migration
4. ✅ **VS Code**: Configure VS Code for uv Python interpreter
5. ✅ **uv installation**: Installed via scoop
