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
> **Last updated:** 2026-05-27 (v2 — cross-referenced all plans + insight registry)
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

### 5. MarS Market Simulation Stress Testing (Phase 28B P28-15)

**Status:** ⏳ Not started — files not yet built

**What:** Build `src/ml/mars_sim.py` — a Microsoft MarS generative foundation model
wrapper for financial market simulation. Generate synthetic bear, bull, and crash
regimes for stress-testing strategies against edge-case market conditions.

**Why:** Strategy robustness = OOS validation + synthetic stress regimes. MarS
can generate plausible crash scenarios not seen in historical data, evaluating
strategy resilience against tail events.

**Files to create:**
- `src/ml/mars_sim.py` — MarS model wrapper (~200 LOC)
- `scripts/train_mars.py` — CLI for training and generating scenarios
- `scripts/benchmark_mars_stress.py` — Stress-test existing strategies

**Tutorial:**

```bash
# Step 1: Implement MarS wrapper + training script
# (src/ml/mars_sim.py — MarS model from microsoft/mars)

# Step 2: Train MarS on SPY historical data
uv run scripts/train_mars.py \
    --symbol SPY --start 2010-01-01 --end 2024-12-31 \
    --epochs 100 --device cuda \
    --output models/sim/mars_SPY.pt

# Step 3: Generate synthetic stress scenarios
uv run scripts/train_mars.py \
    --generate --model models/sim/mars_SPY.pt \
    --n-scenarios 1000 --regime crash \
    --output data/synthetic/mars_crash_SPY.parquet

# Step 4: Stress-test strategies
uv run scripts/benchmark_mars_stress.py \
    --strategy rules_first --model models/sim/mars_SPY.pt \
    --scenarios data/synthetic/mars_crash_SPY.parquet
```

**Expected runtime on GPU:** ~2-4 hours for training, ~10 min for scenario generation
**Source:** awesome-ai-in-finance (microsoft/Mars)

---

### 6. FinRL Deep RL Trading Strategies (Phase 28B P28-16)

**Status:** ⏳ Not started — files not yet built

**What:** Build `src/strategies/drl_strategies.py` — Deep RL trading strategies
using OpenAI Gym environments (FinRL-style). Train PPO, SAC, and TD3 agents on
trading execution with ensemble strategy weighting.

**Why:** DRL can learn complex non-linear execution patterns that rule-based
strategies miss. Ensemble RL strategies can outperform individual agents.

**Files to create:**
- `src/strategies/drl_strategies.py` — DRL ensemble strategy (~500 LOC)
- `scripts/train_finrl.py` — CLI for training DRL agents
- `scripts/benchmark_finrl.py` — Compare DRL vs rules-first baseline

**Tutorial:**

```bash
# Step 1: Implement DRL ensemble strategy
# (src/strategies/drl_strategies.py — PPO/SAC/TD3 wrappers + ensemble)

# Step 2: Train PPO agent on SPY
uv run scripts/train_finrl.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --algo ppo --timesteps 500000 --device cuda \
    --output models/rl/finrl_ppo_SPY.zip

# Step 3: Train ensemble (PPO + SAC + TD3)
uv run scripts/train_finrl.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --algo ensemble --timesteps 1500000 --device cuda \
    --output models/rl/finrl_ensemble_SPY/

# Step 4: Benchmark against rules-first baseline
uv run scripts/benchmark_finrl.py \
    --model models/rl/finrl_ppo_SPY.zip \
    --symbol SPY --start 2022-01-01 --end 2026-06-01
```

**Expected runtime on GPU:** ~2-4 hours for ensemble training (3 agents × 500K steps)
**Source:** awesome-ai-in-finance (AI4Finance-Foundation/FinRL)

---

### 7. Multi-Agent Trading Framework (Phase 28B P28-17)

**Status:** ⏳ Not started — files not yet built (LLM + GPU)

**What:** Build `src/agents/` — a 25-agent Darwinian selection framework
(ATLAS/TradingAgents-style) with Karpathy-style autoresearch, meta-weighting,
and adversarial debate for signal confluence.

**Why:** Multi-agent ensembles with evolutionary selection can adapt to regime
shifts by weighting agents differently across market conditions. This is the
project's most ambitious AI architecture.

**Files to create:**
- `src/agents/` — new directory (~1000 LOC)
  - `src/agents/base_agent.py` — Agent protocol with `predict()`, `confidence()`
  - `src/agents/darwinian_pool.py` — 25-agent pool with selection pressure
  - `src/agents/meta_weighting.py` — Per-regime agent weighting
  - `src/agents/autoresearch.py` — Karpathy-style self-improvement loop
  - `src/agents/adversarial_debate.py` — Agent debate for signal resolution
- `scripts/train_agent_pool.py` — CLI for training and evaluating agent pools
- `scripts/benchmark_multi_agent.py` — Compare vs single-agent baselines

**Tutorial:**

```bash
# Step 1: Implement agent framework
# (src/agents/ — 25-agent Darwinian selection + meta-weighting)

# Step 2: Train agent pool on SPY
uv run scripts/train_agent_pool.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --n-agents 25 --generations 20 --device cuda \
    --output models/agents/pool_SPY/

# Step 3: Evaluate against baselines
uv run scripts/benchmark_multi_agent.py \
    --pool models/agents/pool_SPY/ \
    --symbol SPY --start 2022-01-01 --end 2026-06-01

# Step 4: Full basket training
for SYM in SPY QQQ GLD XLK; do
    uv run scripts/train_agent_pool.py \
        --symbol $SYM --start 2016-01-01 --end 2021-12-31 \
        --n-agents 25 --generations 20 --device cuda \
        --output models/agents/pool_${SYM}/
done
```

**Expected runtime on GPU:** ~4-8 hours for 25-agent × 20-generation training
**Expected runtime on CPU:** Impractical (days per instrument)
**Source:** awesome-ai-in-finance (ATLAS/TradingAgents)

---

### 11. TimeGAN + WGAN-GP Data Augmentation Benchmark Suite

**Status:** ⏳ Not started — files not yet built

**What:** Implement and benchmark TimeGAN (Mushunje/Allen/Peiris 2024 — Columbia) + WGAN-GP
(Pfenninger/Bigler 2025 — ZHAW) as alternative GAN augmentation methods to TTS-GAN.
Compare all three GAN architectures on SPY directional error reduction.

**Why:** Insight registry I26+I27 — TimeGAN captures fat-tails and long-range dependence
that TTS-GAN may miss. WGAN-GP+gradient penalty stabilizes training on volatile data.
Gate: compare TTS-GAN vs TimeGAN vs WGAN-GP on directional error reduction.

**Files to create/modify:**
- `src/ml/timegan.py` — TimeGAN implementation (~300 LOC)
- `src/ml/wgan_gp.py` — WGAN-GP with LSTM generator/discriminator (~250 LOC)
- `scripts/benchmark_gan_comparison.py` — Head-to-head TTS-GAN vs TimeGAN vs WGAN-GP
- Modify `scripts/benchmark_gan_augmentation.py` to support `--gan-type timegan|wgan-gp|tts-gan`

**Tutorial:**

```bash
# Step 1: Train TimeGAN on SPY IS data
uv run scripts/train_timegan.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --epochs 200 --device cuda \
    --output models/gan/timegan_SPY.pt

# Step 2: Train WGAN-GP for comparison
uv run scripts/train_wgan_gp.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --epochs 200 --gp-lambda 10 --device cuda \
    --output models/gan/wgan_gp_SPY.pt

# Step 3: Head-to-head benchmark (all 3 GAN types)
uv run scripts/benchmark_gan_comparison.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --gan-types tts-gan,timegan,wgan-gp \
    --seq-len 90 --n-trials 3 --device cuda
```

**Expected runtime on GPU:** ~30-60 min per GAN type × 3 = ~2-3 hours total
**Paper reference:** `useful_resources/papers_md/*TimeGAN*`, `useful_resources/papers_md/*WGAN-GP*`
**Insight registry:** I26.1, I26.2, I27.1, I27.2

---

### 12. TsLLM Contextual Forecasting (LLM + Time Series)

**Status:** ⏳ Not started — files not yet built (LLM + GPU)

**What:** Implement TsLLM (Parker/Chan/Zhang/Ghobadi 2025 — JHU) patch-based VAE encoder-decoder
for LLM-augmented time series forecasting. Condition forecasts on unstructured news/sentiment
text via interleaved token sequences. Zero-shot regime detection without retraining.

**Why:** Insight registry I29 — TsLLM achieves zero-shot forecasting + anomaly detection
conditioned on news context. Bridges NLP pipeline (FinBERT, SEC filings) with quantitative
forecasting. Complements Phase 28B Multi-Agent Framework (task 7) as a structured LLM approach.

**Files to create:**
- `src/ml/models/tsllm.py` — Patch-based VAE + LLM bridge (~350 LOC)
- `src/nlp/contextual_forecast.py` — News-conditioned forecast pipeline (~200 LOC)
- `scripts/train_tsllm.py` — CLI for training and zero-shot evaluation

**Tutorial:**

```bash
# Step 1: Implement TsLLM wrapper
# (src/ml/models/tsllm.py — VAE encoder-decoder + LLM text integration)

# Step 2: Train patch-based VAE on SPY
uv run scripts/train_tsllm.py \
    --symbol SPY --start 2010-01-01 --end 2024-12-31 \
    --patch-len 16 --latent-dim 128 --epochs 100 --device cuda \
    --output models/llm/tsllm_SPY.pt

# Step 3: Zero-shot forecasting with news context
uv run scripts/train_tsllm.py \
    --model models/llm/tsllm_SPY.pt --eval-only \
    --context "Fed signals rate cut amid recession fears" \
    --horizon 21

# Step 4: Benchmark vs CatBoost baseline
uv run scripts/benchmark_tsllm.py \
    --symbol SPY --start 2022-01-01 --end 2026-06-01 \
    --model models/llm/tsllm_SPY.pt
```

**Expected runtime on GPU:** ~2-4 hours for VAE training, ~10 min for inference
**Paper reference:** `useful_resources/papers_md/*TsLLM*`
**Insight registry:** I29.1, I29.2, I29.3

---

### 13. TSI-GAN / ALGAN / MIM-GAN Anomaly Detection Variants

**Status:** ⏳ Not started — files not yet built

**What:** Implement 3 additional GAN-based anomaly detection architectures from insight registry
papers P30-P33 as alternatives to TadGAN (task 2). Evaluate which architecture has highest
F1 on crisis-event detection (2020 COVID, 2022 bear, 2025 tariff vol, 2026 oil shock).

**Why:** TadGAN is only one of 4 GAN anomaly architectures in the registry. TSI-GAN (I33.1)
uses 2D image encoding for 13% improvement over MERLIN. ALGAN (I30.1) adds attention-adjusted
LSTM states. MIM-GAN (I31.1) uses exponential information measure to avoid mode collapse.

**Files to create:**
- `src/ml/tsi_gan.py` — Time series → 2D image → convolutional GAN (~280 LOC)
- `src/ml/algan.py` — Attention-adjusted LSTM GAN (~250 LOC)
- `src/ml/mim_gan.py` — MIM-based GAN with exponential loss (~220 LOC)
- `scripts/benchmark_anomaly_gan.py` — 4-way comparison (TadGAN vs TSI-GAN vs ALGAN vs MIM-GAN)

**Tutorial:**

```bash
# Step 1: Train TSI-GAN (2D image encoding approach)
uv run scripts/train_tsi_gan.py \
    --symbol SPY --start 2010-01-01 --end 2019-12-31 \
    --epochs 200 --image-size 64 --device cuda \
    --output models/anomaly/tsi_gan_SPY.pt

# Step 2: Train ALGAN (attention-LSTM)
uv run scripts/train_algan.py \
    --symbol SPY --start 2010-01-01 --end 2019-12-31 \
    --epochs 200 --hidden 64 --device cuda \
    --output models/anomaly/algan_SPY.pt

# Step 3: Train MIM-GAN
uv run scripts/train_mim_gan.py \
    --symbol SPY --start 2010-01-01 --end 2019-12-31 \
    --epochs 200 --device cuda \
    --output models/anomaly/mim_gan_SPY.pt

# Step 4: 4-way comparison on crisis detection
uv run scripts/benchmark_anomaly_gan.py \
    --symbol SPY --start 2020-01-01 --end 2026-06-01 \
    --models models/anomaly/tadgan_SPY.pt,models/anomaly/tsi_gan_SPY.pt,models/anomaly/algan_SPY.pt,models/anomaly/mim_gan_SPY.pt
```

**Expected runtime on GPU:** ~20-40 min per model × 3 = ~1.5-2 hours
**Paper reference:** `useful_resources/papers_md/*TSI-GAN*`, `*ALGAN*`, `*MIM-GAN*`
**Insight registry:** I30.1, I31.1, I31.2, I33.1, I33.2

---

## P3 Tasks

### 8. WaveletDiff — Diffusion Model for Time Series Generation

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

### 9. SB3 RL Training (PPO/SAC/CQL) with Larger Networks

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

### 10. SHAP Analysis on Full Feature Set

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

### 14. Wavelet Transformer Models (WDformer / AWEMixer / DB2-TransF)

**Status:** ⏳ Not started — files not yet built

**What:** Implement 3 wavelet-based transformer architectures from insight registry papers
P35-P37. WDformer uses differential attention on wavelet coefficients. AWEMixer uses
Frequency Router + Coherent Gated Fusion. DB2-TransF replaces O(n²) self-attention with
learnable Daubechies wavelets for linear complexity.

**Why:** Insight registry I35-I37 — wavelet-frequency localization beats FFT for
non-stationary financial signals. DB2-TransF achieves comparable/better accuracy at
substantially lower compute vs transformers (O(n) vs O(n²)). Builds on Phase 27B
wavelet features by replacing the CatBoost feature pipeline with end-to-end DL.

**Files to create:**
- `src/ml/models/wdformer.py` — Differential attention + wavelet coefficients (~350 LOC)
- `src/ml/models/awemixer.py` — Frequency Router + Coherent Gated Fusion (~300 LOC)
- `src/ml/models/db2_transf.py` — Learnable Daubechies wavelet attention (~280 LOC)
- `scripts/benchmark_wavelet_transformers.py` — 3-way comparison + vs CatBoost baseline

**Tutorial:**

```bash
# Step 1: Train WDformer on SPY
uv run scripts/train_wdformer.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --epochs 100 --device cuda \
    --output models/wavelet/wdformer_SPY.pt

# Step 2: Train AWEMixer
uv run scripts/train_awemixer.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --epochs 100 --freq-bands 8 --device cuda \
    --output models/wavelet/awemixer_SPY.pt

# Step 3: Train DB2-TransF
uv run scripts/train_db2_transf.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --epochs 100 --wavelet-levels 4 --device cuda \
    --output models/wavelet/db2_transf_SPY.pt

# Step 4: Head-to-head comparison vs CatBoost + wavelet features (Phase 27B)
uv run scripts/benchmark_wavelet_transformers.py \
    --symbol SPY --start 2022-01-01 --end 2026-06-01 \
    --models models/wavelet/wdformer_SPY.pt,models/wavelet/awemixer_SPY.pt,models/wavelet/db2_transf_SPY.pt \
    --baseline models/pattern_classifier_v3_SPY.pkl
```

**Expected runtime on GPU:** ~1-2 hours per model × 3 = ~3-6 hours
**Paper reference:** `useful_resources/papers_md/*WDformer*`, `*AWEMixer*`, `*DB2-TransF*`
**Insight registry:** I35.1, I35.2, I36.1-I36.4, I37.1-I37.3

---

### 15. Non-Linear Factor Models (KAN / MFIN / Tensor Factor)

**Status:** ⏳ Not started — files not yet built

**What:** Implement Kolmogorov-Arnold Networks (KAN) and Matrix Factor Interaction Networks (MFIN)
for non-linear factor discovery, replacing linear PCA/factor models. Tensor factor models
capture multi-way interactions across time × asset × factor dimensions.

**Why:** Insight registry I18.4 — non-linear factor architectures outperform linear models.
Gated on GPU >= 8GB per paper requirement. Complements existing factor purification (Phase 17 R2)
and IR-weighted synthesis (Phase 17 R1) with non-linear alternatives.

**Files to create:**
- `src/ml/kan_factors.py` — KAN-based factor discovery (~250 LOC)
- `src/ml/mfin_factors.py` — MFIN interaction network factors (~200 LOC)
- `scripts/train_nonlinear_factors.py` — CLI for training + IC comparison

**Tutorial:**

```bash
# Step 1: Train KAN factor model
uv run scripts/train_nonlinear_factors.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --algo kan --n-factors 20 --epochs 50 --device cuda \
    --output models/factors/kan_SPY.pt

# Step 2: Train MFIN factor model
uv run scripts/train_nonlinear_factors.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --algo mfin --n-factors 20 --epochs 50 --device cuda \
    --output models/factors/mfin_SPY.pt

# Step 3: Compare IC against linear PCA factor model
uv run scripts/train_nonlinear_factors.py \
    --symbol SPY --compare-linear \
    --models models/factors/kan_SPY.pt,models/factors/mfin_SPY.pt
```

**Expected runtime on GPU:** ~1-2 hours per model × 2 = ~2-4 hours
**Insight registry:** I18.4
**Requirement:** GPU >= 8GB VRAM

---

### 16. TabNet Regime Classification

**Status:** ⏳ Not started

**What:** Train TabNet (attention-based tabular DL) for regime classification on high-dimensional
feature sets (>100 features). Compare against existing CatBoost and InterpretML EBM regime
classifiers from Phase 04 (ML Foundation).

**Why:** Phase 06b RS3 — TabNet uses sequential attention to select which features to use at
each decision step. Only viable with >10K samples and GPU. May capture non-linear regime
interactions that CatBoost misses.

**Files to create:**
- `src/ml/tabnet_regime.py` — TabNet regime classifier (~200 LOC)
- `scripts/train_tabnet_regime.py` — CLI training script

**Tutorial:**

```bash
# Train TabNet regime classifier
uv run scripts/train_tabnet_regime.py \
    --symbol SPY --start 2016-01-01 --end 2021-12-31 \
    --n-steps 5 --attention-dim 64 --epochs 100 --device cuda \
    --output models/regime/tabnet_SPY.pt

# Compare vs CatBoost + EBM
uv run scripts/benchmark_regime_classifiers.py \
    --symbol SPY --start 2022-01-01 --end 2026-06-01 \
    --models models/regime/tabnet_SPY.pt,models/regime/ebm_SPY.pkl
```

**Expected runtime on GPU:** ~1-2 hours
**Source:** Phase 6b Enhancement RS3
**Insight registry:** RS3

---

### 17. Phase 05 + Phase 16 Phase T Deep Learning Models

**Status:** ⏳ Not started — permanently gated (requires GPU)

**What:** Implement the full Phase 05 (ML Advanced) and Phase 16 Phase T (Deep Learning)
model catalog on GPU. Covers: LSTM/GRU forecasting, Transformer time series, 1D CNN patterns,
TFT (Temporal Fusion Transformer), GAN market data generation, autoencoder risk factors,
CNN regime classification, XGB+LGBM+CatBoost ensemble on GPU.

**Why:** Phases 05 and 16 Phase T are permanently gated on GPU availability. These are the
project's only remaining unimplemented major phases. All code designs and specifications
exist in the phase plans — only training/inference needs hardware.

**Files to create:**
- `src/ml/models/lstm_forecaster.py` — LSTM/GRU time series predictor (~200 LOC)
- `src/ml/models/transformer_forecaster.py` — Transformer time series (~300 LOC)
- `src/ml/models/cnn_patterns.py` — 1D conv on OHLCV patterns (~200 LOC)
- `src/ml/models/tft_forecaster.py` — Temporal Fusion Transformer (~400 LOC)
- `src/ml/models/autoencoder_risk.py` — Autoencoder risk factor extraction (~250 LOC)
- `src/ml/models/cnn_regime.py` — CNN regime classifier (~200 LOC)
- `scripts/benchmark_dl_catalog.py` — Compare all DL models vs CatBoost baseline

**Tutorial:**

```bash
# Train each DL model (sequential to avoid OOM)
for MODEL in lstm transformer cnn tft autoencoder cnn_regime; do
    uv run scripts/train_dl_model.py \
        --model $MODEL --symbol SPY \
        --start 2016-01-01 --end 2021-12-31 \
        --epochs 100 --device cuda \
        --output models/dl/${MODEL}_SPY.pt
done

# Benchmark all vs CatBoost baseline
uv run scripts/benchmark_dl_catalog.py \
    --symbol SPY --start 2022-01-01 --end 2026-06-01 \
    --baseline models/pattern_classifier_v3_SPY.pkl
# Gate: at least one DL model beats CatBoost AUC by ≥ 0.02
```

**Expected runtime on GPU:** ~30-90 min per model × 6 models = ~3-9 hours
**Source:** Phase 05 (ML Advanced) + Phase 16 Phase T (Deep Learning)
**Note:** Permanently gated on GPU. All models should be compared against existing
CatBoost baseline as gate condition (DL AUC must exceed CatBoost by ≥ 0.02).

---

### 18. VAE Latent Embeddings — Full Training

**Status:** ⏳ Not started — CPU viable but GPU significantly faster

**What:** Train variational autoencoder on full SPY price history to discover latent market
structure embeddings. Currently FS15-lite (Phase 06b) is designed for CPU with small networks.
Re-train with larger architectures (256 hidden, 64 latent) impractical on CPU.

**Why:** Phase 06b FS15 — VAE discovers market structure embeddings as features. CPU-limited
small network may miss non-linear structure. GPU enables deeper architectures.

**Files:**
- `src/ml/vae_embeddings.py` — VAE latent feature extractor (already BUILT for CPU)
- `scripts/train_vae_latent.py` — CLI (already BUILT for CPU)

**Tutorial:**

```bash
# Train large VAE on GPU
uv run scripts/train_vae_latent.py \
    --symbol SPY --start 2016-01-01 --end 2024-12-31 \
    --hidden 256 --latent 64 --epochs 200 --device cuda \
    --output models/vae/vae_embeddings_SPY.pt

# Compare CPU-lite vs GPU-large embedding quality
uv run scripts/benchmark_vae_embeddings.py \
    --cpu-model models/vae/vae_lite_SPY.pt \
    --gpu-model models/vae/vae_embeddings_SPY.pt
```

**Expected runtime on GPU:** ~1-2 hours
**Expected runtime on CPU:** ~8-12 hours (with smaller network)
**Source:** Phase 06b FS15, FS15-lite

---

## Summary

| # | Task | Tier | Status | Est. GPU Time |
|---|------|------|--------|---------------|
| 1 | TTS-GAN Gate Validation | P0 | ⏸️ Aborted (CPU slow) | 15-30 min |
| 2 | TadGAN Anomaly Detection | P1 | ⏳ Not started | 20-40 min |
| 3 | TTS-GAN Full Training (18 instruments) | P2 | ⏳ Blocked on P0 | ~9 hours |
| 4 | Chronos-2 Fine-Tuning | P2 | ⏳ Not started | 1-2 hours |
| 5 | MarS Market Simulation (P28-15) | P2 | ⏳ Not started — files not built | 2-4 hours |
| 6 | FinRL DRL Strategies (P28-16) | P2 | ⏳ Not started — files not built | 2-4 hours |
| 7 | Multi-Agent Framework (P28-17) | P2 | ⏳ Not started — files not built | 4-8 hours |
| 8 | WaveletDiff Generation | P3 | ⏳ Not started | 2-4 hours |
| 9 | SB3 RL Larger Networks | P3 | ⏸️ CPU-trained | 30-60 min |
| 10 | SHAP Full Feature Analysis | P3 | ⏸️ Partial | 10-30 min |
| 11 | TimeGAN + WGAN-GP Benchmark Suite | P2 | ⏳ Not started — files not built | 2-3 hours |
| 12 | TsLLM Contextual Forecasting (LLM+GPU) | P2 | ⏳ Not started — files not built | 2-4 hours |
| 13 | TSI-GAN / ALGAN / MIM-GAN Anomaly Variants | P2 | ⏳ Not started — files not built | 1.5-2 hours |
| 14 | Wavelet Transformers (WDformer/AWEMixer/DB2-TransF) | P3 | ⏳ Not started — files not built | 3-6 hours |
| 15 | Non-Linear Factor KAN/MFIN Models | P3 | ⏳ Not started — files not built | 2-4 hours |
| 16 | TabNet Regime Classification | P3 | ⏳ Not started | 1-2 hours |
| 17 | Phase 05 + Phase 16 Phase T DL Models | P3 | ⏳ Not started — permanently gated | 3-9 hours |
| 18 | VAE Latent Embeddings Full Training | P3 | ⏳ Not started | 1-2 hours |

**Total estimated GPU time for all pending tasks:** ~56-85 hours (full queue)

**Next action when GPU becomes available:** Start with Task 1 (P0), then Task 2 (P1).
