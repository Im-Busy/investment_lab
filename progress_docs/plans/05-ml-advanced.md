---
type: phase
phase: "05"
name: "ML Advanced"
status: deferred
deferred_reason: "Requires GPU >=16GB VRAM for CNN training and autoencoder hyperparameter sweeps"
deferred_since: 2026-04-30
revisit_when: "WSL2 with GPU passthrough available OR cloud GPU budget approved (Azure NCas T4 v3, Colab Pro+, Lambda Labs)"
---

# Phase 05: ML Advanced

## Overview

GPU-intensive ML tasks deferred from Phase 04. Includes full-scale CNN regime training, autoencoder hyperparameter sweeps, and integration into the live trading pipeline.

## Held Items

| Item | Reason | Blocked By |
|------|--------|------------|
| CNN regime full training (all symbols, all timeframes) | >16GB VRAM needed | Hardware |
| Autoencoder risk factor hyperparameter sweep | GPU + 32GB RAM | Hardware |
| Ensemble model with XGBoost + LightGBM + CatBoost | Training time prohibitive on CPU | Hardware |
| Walk-forward optimization with 5-year rolling windows | Compute bound | Hardware |

## Planned Tasks (when unblocked)

| Task | Dependency | Estimated Duration |
|------|-----------|-------------------|
| C1: Full CNN regime training | GPU available | 1-2 weeks |
| C2: Autoencoder sweep | C1 complete | 1 week |
| C3: Ensemble model training | C1, C2 complete | 2 weeks |
| C4: Walk-forward validation | C3 complete | 1 week |
| C5: Integration into live pipeline | C4 complete | 1 week |

## Success Criteria
- Regime classifier test accuracy >60%
- Overfit gap <0.10
- ML-enhanced Sharpe > baseline Sharpe on OOS data
- Walk-forward validated on 5+ years
