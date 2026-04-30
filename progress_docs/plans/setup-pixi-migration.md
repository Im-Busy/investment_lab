---
type: setup
name: "Pixi to UV Migration"
status: complete
started: 2026-04-20
completed: 2026-04-20
tools_installed:
  - uv (replaces pixi, installed via scoop)
configs_changed:
  - pyproject.toml (pixi -> uv format)
  - Removed pixi.toml
  - Removed requirements.txt
  - Kept environment_backup.yml as reference
---

# Setup: Pixi to UV Migration

## Overview

Migrated from pixi to uv for Python package and environment management. uv is an extremely fast Python package manager written in Rust.

## Changes

| Before | After |
|--------|-------|
| `pixi.toml` | `pyproject.toml` (uv format) |
| `requirements.txt` | Removed |
| `environment_backup.yml` | Kept as reference |
| pixi commands | uv commands |

## Key Decisions
- Use `pandas-ta` instead of TA-Lib (avoids C library dependency)
- Include all packages from `environment_backup.yml`
- Configure VS Code for uv Python interpreter
- uv installed via scoop

## New Convention
- `uv run <command>` for all Python execution
- `uv add <package>` for dependencies
- `uv sync` for environment sync
- NEVER use `python`, `pip`, `python3`, or `pip3` directly
