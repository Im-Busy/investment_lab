---
type: setup
name: "Jupyter + VS Code Setup"
status: complete
started: 2026-04-19
completed: 2026-04-19
tools_installed:
  - jupyter
  - ipykernel
configs_changed:
  - .vscode/settings.json (Python interpreter, Jupyter config)
---

# Setup: Jupyter + VS Code Configuration

## Overview

Configured Jupyter notebooks in VS Code with the uv-managed Python environment. Ensured all notebooks can execute with the project's dependencies.

## Changes
- VS Code Python interpreter pointed to uv environment
- Jupyter kernel configured to use project Python
- Notebook execution verified for all 15+ notebooks
- Notebook helpers created in `src/utils/notebook_helpers.py`

## Notebook Inventory
- 01-04: Pattern backtests & visualization
- 05: SPY long-term backtest
- 06-08: Pattern selection, contribution, benchmarks
- 09: Multi-timeframe backtest
- 10: Pair trading (pending data)
- 11: Parameter robustness (pending optimization)
- 12: Regime-aware backtest
- 13: ML validation
- 14: Regime detector comparison
- 15: Phase 2 validation
- ML_Training_Colab: Google Colab training

## Notes
- All notebooks have CONFIG sections at top
- Notebook helpers provide consistent data loading and formatting
- `uv run jupyter nbconvert --to notebook --execute` for batch execution
