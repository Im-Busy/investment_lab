# Conda to Pixi Migration Plan

## Overview

This plan outlines the migration from conda to pixi for the `trading-lab` environment in the investment_trying project.

### Why Pixi?

- **Fast dependency resolution**: Uses uv under the hood for Python packages
- **Conda channel support**: Can still use conda-forge and other conda channels
- **Reproducible environments**: Lock file ensures exact reproducibility
- **Project-local environments**: No more activating environments - just use `pixi run`
- **Multi-platform support**: Easy cross-platform configuration
- **Multiple environments**: Support for different feature sets (dev, jupyter, etc.)

## Current State

- **Environment name**: `trading-lab`
- **Location**: `C:\Dev\projects\investment_trying`
- **Dependencies**: Defined in `requirements.txt`
- **Python version**: 3.13
- **Python packages**: pandas, numpy, backtesting, quantstats, mplfinance, plotly, matplotlib, jupyter, yfinance, loguru, pytest

## Migration Steps

### Phase 1: User Preparation Tasks ✅ COMPLETED

These tasks were completed by you:

- [x] Install pixi via scoop: `scoop install pixi`
- [x] Check Python version: Python 3.13
- [x] Backup conda environment: `conda env export > environment_backup.yml`

### Phase 2: Files Created ✅ COMPLETED

#### 1. pixi.toml Configuration

Created an enhanced `pixi.toml` with:

- **Core dependencies** from conda-forge (pandas, numpy, python)
- **PyPI dependencies** via uv (backtesting, quantstats, etc.)
- **Named tasks** with descriptions for better UX
- **Dev feature** with linting tools (ruff, mypy)
- **Multiple environments** (default, dev)

#### 2. .gitignore Updated

Added pixi-specific entries:
```
# Pixi
.pixi/
pixi.lock
```

### Phase 3: Verification Tasks

After the files are created, run:

#### 1. Initialize Pixi Environment

```powershell
cd C:\Dev\projects\investment_trying
pixi install
```

This will:
- Create a `.pixi` directory with the environment
- Resolve all dependencies using uv (fast!)
- Generate a `pixi.lock` file for reproducibility

#### 2. Test the Environment

```powershell
# Quick import verification
pixi run check

# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov
```

#### 3. Test Jupyter Notebooks

```powershell
# Start Jupyter Notebook
pixi run notebook

# Or start JupyterLab
pixi run lab

# Install as Jupyter kernel (optional)
pixi run install-kernel
```

#### 4. Development Environment (Optional)

```powershell
# Install dev environment with linting tools
pixi install -e dev

# Use dev tools
pixi run -e dev lint
pixi run -e dev format
pixi run -e dev typecheck
```

### Phase 4: Cleanup (Optional)

After verifying everything works:

#### 1. Remove Conda Environment

```powershell
conda deactivate
conda env remove -n trading-lab
```

#### 2. Delete Backup (Optional)

```powershell
del environment_backup.yml
```

## Workflow Changes

### Before (Conda)

```powershell
cd C:\Dev\projects\investment_trying
conda activate trading-lab
python script.py
```

### After (Pixi)

```powershell
cd C:\Dev\projects\investment_trying
pixi run python script.py
```

Or for interactive work:
```powershell
pixi shell
python script.py
```

## Available Tasks

| Task | Command | Description |
|------|---------|-------------|
| `check` | `pixi run check` | Verify all core imports work |
| `test` | `pixi run test` | Run all tests |
| `test-cov` | `pixi run test-cov` | Run tests with coverage |
| `notebook` | `pixi run notebook` | Start Jupyter Notebook |
| `lab` | `pixi run lab` | Start JupyterLab |
| `install-kernel` | `pixi run install-kernel` | Install as Jupyter kernel |
| `fetch-data` | `pixi run fetch-data` | Fetch market data |
| `run` | `pixi run run` | Run main application |

### Dev Environment Tasks

| Task | Command | Description |
|------|---------|-------------|
| `lint` | `pixi run -e dev lint` | Lint source code |
| `format` | `pixi run -e dev format` | Format source code |
| `typecheck` | `pixi run -e dev typecheck` | Type check source code |

## Benefits After Migration

1. **Faster installs**: uv is significantly faster than pip/conda
2. **Reproducible**: Lock file ensures same versions everywhere
3. **No activation needed**: Use `pixi run` directly
4. **Project isolation**: Environment is in project directory
5. **Mixed sources**: Use both conda-forge and PyPI packages
6. **Multiple environments**: Separate dev/prod configurations
7. **Named tasks**: Easy-to-remember commands with descriptions

## Potential Issues and Solutions

### Issue: Package not found in conda-forge

**Solution**: Use `[pypi-dependencies]` section for pip-only packages (already configured)

### Issue: Jupyter kernel not showing

**Solution**: Register the kernel manually:
```powershell
pixi run install-kernel
```

### Issue: VSCode not detecting environment

**Solution**: Select the interpreter at `.pixi/envs/default/python.exe`

### Issue: Need to add a new dependency

**Solution**: 
```powershell
# Add conda package
pixi add pandas

# Add PyPI package
pixi add --pypi some-package
```

## Files Summary

| File | Action | Status |
|------|--------|--------|
| `pixi.toml` | Created | ✅ Done |
| `.gitignore` | Updated | ✅ Done |
| `environment_backup.yml` | Created by user | ✅ Done |
| `.pixi/` | Auto-generated | ⏳ Run `pixi install` |
| `pixi.lock` | Auto-generated | ⏳ Run `pixi install` |

## Next Steps

1. Run `pixi install` to create the environment
2. Run `pixi run check` to verify imports
3. Run `pixi run test` to verify tests pass
4. Run `pixi run notebook` to test Jupyter
5. Optionally cleanup old conda environment

---

**Migration Status**: Ready for testing! Run `pixi install` to proceed.
