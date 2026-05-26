# GPU Task Queue

> **Purpose:** Central registry of tasks that require GPU or benefit significantly from it.
> Each task includes a self-contained tutorial so a future agent can attempt it without
> researching the paper/module from scratch.
>
> **Protocol (see AGENTS.md §GPU Task Protocol):**
> 1. On session start, check this file for pending tasks.
> 2. If GPU is available (`torch.cuda.is_available()`), pick the highest-priority pending task.
> 3. Follow the tutorial in the task entry. Do NOT re-research.
> 4. If the task completes: update status to ✅, record results.
> 5. If aborted (OOM, too slow, dependency missing): record reason, mark ⏸️, move to next.
> 6. If all tasks aborted or no GPU: continue with CPU-safe work.
>
> **Last updated:** 2026-05-26
> **GPU status:** ❌ Not available (torch 2.9.1+cpu)

---

## Priority Tiers

| Tier | Criteria | Example |
|------|----------|---------|
| **P0** | Gate-blocking — must run to validate a phase | TTS-GAN benchmark (Phase 27A gate) |
| **P1** | High-impact — would unlock new capabilities | TadGAN anomaly detection (Phase 27C) |
| **P2** | Nice-to-have — incremental improvement | Chronos-2 fine-tuning |
| **P3** | Research — exploratory, no immediate production impact | WaveletDiff generation |

---

## P0 Tasks

### 1. TTS-GAN Gate Validation (Phase 27A)

**Status:** ⏸️ Aborted (CPU too slow — ~10 min for fast-mode 30-epoch training)

**What:** Train TTS-GAN on SPY IS data (2016-2021), augment LSTM training set,
measure directional error reduction on OOS (2022 bear). Gate: ≥10% reduction.

**Why:** Phase 27A completion gate. Paper (Podobinski & Chudziak 2024) showed
consistent MSE reduction across 40 time windows on BTC + S&P500. We need to
validate on SPY 2022 bear market specifically.

**Files involved:**
- `scripts/benchmark_gan_augmentation.py` — full benchmark script (already built)
- `src/ml/gan_data_augmentation.py` — TTS-GAN implementation
- `src/ml/gan_convergence.py` — DTW DeD-iMs metric

**Tutorial:**

```bash
# Step 1: Verify GPU available
uv run python -c "import torch; assert torch.cuda.is_available(), 'GPU required'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Step 2: Run full gate validation
uv run scripts/benchmark_gan_augmentation.py \
    --symbol SPY \
    --start 2016-01-01 \
    --end 2021-12-31 \
    --seq-len 90 \
    --gan-epochs 200 \
    --lstm-epochs 100 \
    --n-trials 3

# Step 3: Production model — train & export for all basket instruments
for SYM in SPY QQQ XLK XLE GLD SLV NUE STLD HAL MPC EOG INTC AMD LMT JNJ MRK NEM; do
    uv run scripts/train_tts_gan.py \
        --symbol $SYM --start 2016-01-01 --end 2021-12-31 \
        --epochs 200 --model-path models/gan/tts_gan_${SYM}.pt
done

# Step 4: Use pre-trained GAN in ML pipeline (offline augmentation)
uv run scripts/train_ml_pipeline_v3.py \
    --basket SPY,QQQ,GLD,XLK \
    --gan-augment --gan-model models/gan/tts_gan_SPY.pt
```

**Expected runtime on GPU:** ~15-30 minutes for 200 GAN epochs + LSTM training
**Paper reference:** `useful_resources/papers_md/Financial time series augmentation using transformer based GAN architecture.md`

---

## P1 Tasks

### 2. TadGAN Regime Anomaly Detection (Phase 27C)

**Status:** ⏳ Not started

**What:** Train cycle-consistent GAN (TadGAN) with LSTM generator/critic on SPY
data. Use reconstruction error + critic output as dual anomaly score. Detect
major crisis events (2020 COVID crash, 2022 bear, 2025 tariff vol, 2026 oil shock).

**Files:**
- `src/ml/anomaly_detection.py` — TadGAN model (encoder, decoder, critic_x, critic_z) — **BUILT**
- `scripts/train_tadgan.py` — CLI for training TadGAN — **BUILT**
- `scripts/benchmark_tadgan.py` — Gate validation script — **BUILT**
- Integrated into `src/strategies/rules_first_strategy.py` via `--use-tadgan-gate`
- Integrated into `scripts/backtest_rules_first.py` via `--use-tadgan-gate`/`--tadgan-model`

**Tutorial:**

```bash
# Step 1: Train TadGAN on pre-crisis SPY data
uv run scripts/train_tadgan.py \
    --symbol SPY --start 2010-01-01 --end 2019-12-31 \
    --epochs 200 --device cuda

# Step 2: Validate crisis detection gate
uv run scripts/benchmark_tadgan.py --symbol SPY --epochs 200 --device cuda
# Expected: "Gate: PASS (4/4 detected, X.X ≤ 5 FP/year)"

# Step 3: Wire into backtest as risk gate
uv run scripts/backtest_rules_first.py SPY --start 2020-01-01 --end 2026-06-01 \
    --use-tadgan-gate --tadgan-model models/anomaly/tadgan_SPY.pt \
    --fast

# Step 4: Full production training
uv run scripts/train_tadgan.py \
    --symbol SPY --start 2010-01-01 --end 2024-12-31 \
    --hidden 64 --latent 20 --layers 2 --epochs 500 --device cuda \
    --output models/anomaly/tadgan_SPY_production.pt
```

**Expected runtime on GPU:** ~20-40 minutes
**Paper reference:** `useful_resources/papers_md/TadGAN Time Series Anomaly Detection Using Generative Adversarial Networks.md`

---

## P2 Tasks

### 3. TTS-GAN Full Training & Model Export

**Status:** ⏳ Not started (blocked on P0 gate)

**What:** Train production-quality TTS-GAN models on all 18 basket instruments,
export .pt files, enable `--gan-model` flag in ML pipeline for offline augmentation.

**Why:** Once gate validates that GAN augmentation works, pre-train models for
all production instruments so training doesn't need to re-train GAN each time.

**Tutorial:**

```bash
# Train TTS-GAN for each instrument in production basket
for SYM in SPY QQQ XLK XLE GLD SLV NUE STLD HAL MPC EOG INTC AMD LMT JNJ MRK NEM; do
    uv run scripts/train_tts_gan.py \
        --symbol $SYM \
        --start 2016-01-01 \
        --end 2021-12-31 \
        --epochs 200 \
        --model-path models/gan/tts_gan_${SYM}.pt
done

# Then train CatBoost with pre-trained GAN augmentation
uv run scripts/train_ml_pipeline_v3.py \
    --basket SPY,QQQ,GLD,XLK \
    --gan-augment \
    --gan-model models/gan/tts_gan_SPY.pt \
    --gan-seq-len 90
```

**Expected runtime on GPU:** ~30 min per instrument × 18 instruments = ~9 hours
**Expected runtime on CPU:** ~3-5 hours per instrument = impractical

---

### 4. Chronos-2 Fine-Tuning (Phase 14/E1)

**Status:** ⏳ Not started

**What:** Fine-tune Amazon's Chronos-2 foundation model on SPY data for
zero-shot forecasting. Compare to CatBoost baseline.

**Tutorial:**

```bash
# Fine-tune chronos-2-small on SPY with LoRA
uv run scripts/finetune_chronos.py \
    --symbol SPY --model chronos-2-small --lora \
    --context 512 --horizon 5 --epochs 10 --device cuda

# Fine-tune on basket
uv run scripts/finetune_chronos.py \
    --basket SPY,QQQ,GLD --model bolt-small --epochs 20 --device cuda

# Evaluate fine-tuned vs zero-shot
uv run scripts/finetune_chronos.py \
    --symbol SPY --eval-only \
    --model-path models/chronos/SPY_20260526_120000/chronos_finetuned_chronos-2-small
```

**Expected runtime on GPU:** ~1-2 hours
**Files:** `src/ml/models/chronos.py` (existing), `scripts/finetune_chronos.py` (BUILT)

---

## P3 Tasks

### 5. WaveletDiff — Diffusion Model for Time Series Generation

**Status:** ⏳ Not started

**What:** Implement diffusion model operating on wavelet coefficients (Wang &
Milenkovic, UIUC 2024). 3× better discriminative scores than GAN baselines.

**Tutorial:**

```bash
# Full training on SPY
uv run scripts/train_wavelet_diffusion.py \
    --symbol SPY --epochs 500 --num-steps 1000 --device cuda

# Fast smoke test
uv run scripts/train_wavelet_diffusion.py \
    --symbol SPY --fast --epochs 50 --device cuda

# Generate samples from trained model (add to script via --generate flag)
```

**Expected runtime on GPU:** ~2-4 hours for training
**Files:** `scripts/train_wavelet_diffusion.py` (BUILT)
**Paper reference:** `useful_resources/papers_md/*WaveletDiff*`

---

### 6. SB3 RL Training (PPO/SAC/CQL) with Larger Networks

**Status:** ⏸️ CPU-trained (Phase 21 D5/D12)

**What:** Re-train PPO/SAC/CQL trade execution agents with larger networks
(256→512 hidden, 3→5 layers) that were impractical on CPU.

**Why:** Phase 21 RL models trained on CPU with tiny networks. Larger models
could capture more complex execution patterns.

**Tutorial:**

```bash
# Re-train with larger architectures
uv run scripts/train_rl_advanced.py \
    --symbol SPY \
    --algo ppo \
    --hidden 512 \
    --layers 5 \
    --timesteps 500000 \
    --device cuda

# Compare:
# CPU baseline (hidden=64, layers=2): Sharpe ~0.15
# GPU target (hidden=512, layers=5): Sharpe > 0.30
```

**Expected runtime on GPU:** ~30-60 minutes per algorithm
**Files:** `src/rl/sb3_executors.py`, `src/rl/offline_rl.py`, `scripts/train_rl_advanced.py`

---

### 7. SHAP Analysis on Full Feature Set

**Status:** ⏸️ Partial (CPU-limited to small samples)

**What:** Run full SHAP analysis on CatBoost model with 200+ features and 100K+
samples. Currently limited to subsampling due to CPU memory.

**Why:** Phase 27B SHAP was limited to 2016-2024 subset. Full analysis could
reveal regime-dependent feature importance shifts.

**Tutorial:**

```bash
# GPU-accelerated SHAP via GPUTreeExplainer
uv run python -c "
import shap
import pickle
import torch

model = pickle.load(open('models/pattern_classifier_v3_SPY.pkl', 'rb'))
X = ... # full 200+ feature matrix, 100K+ samples

explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
shap_values = explainer.shap_values(X[:50000])  # GPU-accelerated

# Output: full SHAP ranking, interaction effects, regime decomposition
"
```

**Expected runtime on GPU:** ~10-30 minutes
**Expected runtime on CPU:** ~4-8 hours or OOM

---

## Summary

| # | Task | Tier | Status | Est. GPU Time |
|---|------|------|--------|---------------|
| 1 | TTS-GAN Gate Validation | P0 | ⏸️ Aborted (CPU slow) | 15-30 min |
| 2 | TadGAN Anomaly Detection | P1 | ⏳ Not started | 20-40 min |
| 3 | TTS-GAN Full Training (18 instruments) | P2 | ⏳ Blocked on P0 | ~9 hours |
| 4 | Chronos-2 Fine-Tuning | P2 | ⏳ Not started | 1-2 hours |
| 5 | WaveletDiff Generation | P3 | ⏳ Not started | 2-4 hours |
| 6 | SB3 RL Larger Networks | P3 | ⏸️ CPU-trained | 30-60 min |
| 7 | SHAP Full Feature Analysis | P3 | ⏸️ Partial | 10-30 min |

**Next action when GPU becomes available:** Start with Task 1 (P0), then Task 2 (P1).
