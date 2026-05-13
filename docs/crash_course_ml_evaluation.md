# Machine Learning Evaluation Crash Course — For Trading

> A plain-English guide to AUC, overfit gap, IC, feature importance, and cross-validation.
> Written for someone who wants to understand ML model reports, not write ML papers.

---

## 1. What We're Actually Trying to Do

We give the computer a job: "Look at this stock's price history, and tell me — will it go up or down in the next 5 days?"

The computer (CatBoost model) produces two things:
1. A **prediction** (UP or DOWN)
2. A **confidence score** (a number between 0 and 1, where 0.9 means "I'm 90% sure it'll go up")

Now we need to judge: how good is the computer at this job?

---

## 2. AUC — The Single Number That Tells You Everything

### What it stands for
Area Under the (ROC) Curve.

### What it actually means (forget the math for a moment)

**AUC answers this question:** If I randomly pick one day where the stock actually went UP, and one day where it actually went DOWN, does my model give a higher confidence score to the UP day?

- **AUC = 1.0 (Perfect):** Every UP day gets a higher score than every DOWN day. The model is never wrong about ranking.
- **AUC = 0.75 (Good):** 75% of the time, the model ranks an UP day higher than a DOWN day.
- **AUC = 0.50 (Useless):** Same as flipping a coin. The model has no idea which days are good.
- **AUC < 0.50 (Worse than random):** The model is actively confused — it's better to flip the predictions.

### Why we use AUC instead of "accuracy"

Imagine a stock that goes up only 10% of days. A model that always says "DOWN" will be 90% accurate — but it's completely useless for trading.

**AUC is immune to this problem.** It doesn't care about class balance. It only cares about whether the model can separate the good days from the bad days.

### Real-world interpretation for our experiment

| AUC | What it means for trading |
|-----|--------------------------|
| 0.60+ | Model has real signal — can potentially make money |
| 0.55-0.60 | Marginal signal — might work with good risk management |
| 0.50-0.55 | Very weak signal — needs improvement |
| Below 0.50 | Model is actively harmful |

**Source:** [Google ML Crash Course — ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## 3. Overfit Gap — Why a Perfect Score Is a Red Flag

### The core problem: Memorization vs. Learning

Imagine you study for a test by memorizing the practice exam answers word-for-word. You get 100% on the practice test. Then the real test comes with different questions — and you fail.

That's overfitting. The model memorized the training data instead of learning general patterns.

### Train AUC vs. Test AUC

We split the data into two parts:
- **Training set (70%):** The model studies this. This is the "practice exam."
- **Test set (30%):** The model has never seen this. This is the "real exam."

**Train AUC** = performance on the practice exam
**Test AUC** = performance on the real exam

### The Overfit Gap

```
Overfit Gap = Train AUC - Test AUC
```

- **Gap < 0.05:** The model generalizes well — it learned real patterns.
- **Gap 0.05-0.15:** Acceptable — some memorization but not dangerous.
- **Gap > 0.15:** Danger zone — the model is memorizing too much.
- **Gap > 0.30:** The model is mostly memorizing — test predictions are unreliable.

### The V1 Disaster (Why we built V2)

Our V1 model had:
- Train AUC: **0.999** (basically perfect on training data)
- Test AUC: **0.48-0.56** (barely better than random)
- Overfit Gap: **0.44-0.52** (massive memorization)

The model was just memorizing which specific dates went up, not learning why stocks go up. This is the #1 mistake in financial ML.

**Source:** [Google ML Crash Course — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)

---

## 4. Information Coefficient (IC) — The Feature Quality Filter

### The problem: too many features

We generate 100-150 features per stock (RSI, moving averages, volatility, etc.). Most of them are noise. If we feed all of them to the model, it will find random patterns in the noise and overfit.

### What IC measures

IC asks: "Does this feature have any relationship with future returns?"

```
IC = correlation(feature_value_today, return_over_next_5_days)
```

- **IC > 0.05:** Strong relationship — this feature probably matters.
- **IC 0.02-0.05:** Weak but real relationship — might help.
- **IC < 0.02:** Effectively random — drop this feature.
- **IC < 0:** Inverse relationship — higher feature values predict LOWER returns.

### Rank IC (the version we actually use)

Standard IC is thrown off by outliers. One crazy day can make a useless feature look important. **Rank IC** solves this by ranking the values first:

Instead of comparing raw numbers, it asks: "On days when this feature ranks in the top 10%, do returns tend to be higher? On days in the bottom 10%, do returns tend to be lower?"

**We filter out any feature with Rank IC < 0.02.** This is how we go from 158 features to 64-107 features.

**Source:** [ML for Trading — The Information Coefficient](https://ml4trading.io/primer/the-information-coefficient/)

---

## 5. Feature Importance — What the Model Actually Uses

### After training, the model tells us what mattered

CatBoost (our model) calculates feature importance by measuring:
> "How much worse would my predictions get if I randomly shuffle this feature?"

If shuffling a feature (destroying its signal) makes predictions much worse, that feature is important. If shuffling does nothing, the model wasn't using it.

### Reading the top 10

From our SPY + cross-asset experiment:

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | beta_QQQ_60d | 4.14 | Cross-asset (beta) |
| 2 | avg_cross_corr_20d | 3.60 | Cross-asset (correlation) |
| 3 | price_to_ma_100 | 3.35 | Instrument (trend) |
| 4 | price_to_ma_10 | 3.22 | Instrument (trend) |
| 5 | std_return_50 | 3.05 | Instrument (volatility) |

**Key insight:** Cross-asset features (beta, correlation) appear at ranks 1 and 2 — the model learned that market context matters more than the stock's own indicators.

---

## 6. Cross-Validation — Testing Without Cheating

### Why a simple train/test split isn't enough for time series

In regular ML (cat pictures, spam detection), you can randomly split data. In finance, you CANNOT — because time leaks.

If you randomly split, the model might train on December 2023 data and test on November 2023. It has already "seen the future" (December influences how you interpret November if dates are mixed).

### Purged K-Fold Cross-Validation

We split the timeline into 5 segments (folds), like reading a book:

```
[Fold 1: 2015-2017] [Fold 2: 2018-2019] [Fold 3: 2020-2021] [Fold 4: 2022-2023] [Fold 5: 2024]
```

Each fold tests on future data only — never past data. Between folds, we add an **embargo gap** (remove the border days) so the model can't peek across folds through overlapping labels.

### Nested CV (why our pipeline has inner AND outer loops)

- **Outer loop (5 folds):** Measures honest performance. "How good is this approach?"
- **Inner loop (3 folds):** Picks the best hyperparameters without leaking information to the outer loop.

This is the gold standard for time-series ML validation.

**Source:** [Sasse et al. 2025 — Overview of leakage scenarios in supervised ML](https://arxiv.org/abs/2505.11065) and [Vabalas et al. 2019 — ML validation with limited sample size](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365)

---

## 7. Triple-Barrier Labeling — Defining "Up" Honestly

### The problem with simple labels

"Stock went up in 5 days" sounds simple, but is misleading:
- Stock goes up 1%, then crashes 20% → labeled "UP" (you lost money)
- Stock goes down, then recovers to +2% → labeled "UP" (you got stopped out)

### Triple-barrier method

Three outcomes instead of "up or down":
1. **Take Profit (+1):** Price hits an upper barrier first → good trade
2. **Stop Loss (-1):** Price hits a lower barrier first → bad trade
3. **Time Limit (0):** Neither barrier hit within the horizon → neutral, don't trade

This creates cleaner labels — you're labeling based on actual trade outcomes, not just "did the close price go up."

---

## 8. Putting It All Together — Reading an Experiment Report

Here's a real result from our cross-asset experiment (JOE with cross-asset features):

```
Train AUC:  0.784
Test AUC:   0.620
Overfit Gap: 0.164
CV AUC:     0.533 ± 0.025
Features:   82 (after IC filter from 158)
```

**What this tells us:**

| Metric | Value | Judgment |
|--------|-------|----------|
| Test AUC | 0.620 | **Real signal.** Model predicts direction better than random. |
| Overfit Gap | 0.164 | **Borderline.** Slightly above the 0.15 threshold. Some memorization, but not catastrophic. |
| CV AUC ± | 0.533 ± 0.025 | **Stable.** Performance doesn't swing wildly between time periods. |
| Features | 82/158 | **Reasonable.** IC filter removed ~half the features as noise. |

**Compare to the baseline without cross-asset features:**

```
Test AUC:   0.471  (below random — actively harmful!)
Overfit Gap: 0.378  (massive memorization)
```

Cross-asset features transformed JOE from worse-than-random to a meaningful predictor. This is the power of market context.

---

## 9. Quick Reference — Cheat Sheet

| Term | One-sentence definition | Good value |
|------|------------------------|------------|
| **AUC** | Probability model ranks an UP day above a DOWN day | > 0.55 (trading), > 0.70 (general ML) |
| **Train AUC** | AUC on data the model studied | 0.60-0.80 |
| **Test AUC** | AUC on data the model never saw | > 0.55 |
| **Overfit Gap** | Train AUC minus Test AUC | < 0.15 |
| **IC** | Correlation between a feature and future returns | > 0.02 (keep), < 0.02 (drop) |
| **Rank IC** | IC using ranked values (robust to outliers) | > 0.02 |
| **Feature Importance** | How much worse predictions get if you shuffle this feature | Higher = more useful |
| **CV (Cross-Validation)** | Testing on multiple time periods to avoid lucky splits | Lower std = more stable |
| **CV Std** | How much AUC varies across time periods | < 0.05 |
| **Triple-Barrier** | Labels based on actual trade outcomes, not just close price | Use always for trading ML |

---

## 10. Key Papers (For Deeper Reading)

1. **ROC and AUC:** [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) — Best beginner resource, interactive diagrams.

2. **Overfitting:** [Google ML Crash Course — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) — Clear explanation with visual loss curves.

3. **Information Coefficient:** [ML for Trading — Stefan Jansen](https://ml4trading.io/primer/the-information-coefficient/) — The definitive guide to IC in financial ML.

4. **Leakage in Financial ML:** [Sasse et al. 2025 — Overview of leakage scenarios](https://arxiv.org/abs/2505.11065) — Academic survey of all the ways ML can cheat in finance.

5. **Validation with Small Samples:** [Vabalas et al. 2019](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365) — Why nested CV matters when you don't have millions of data points.

---

*Last updated: 2026-05-09*
