# Direction N: NLP, Quant Schools & Strategy Expansion — Agent Execution Plan

> **Source:** NLP in Quant 101 course + Chinese 5 Quant Schools framework
> **Scope:** 81 ideas → 9 phases → prioritized by impact/complexity ratio
> **Date:** 2026-05-15

---

## Priority Logic

```
LOW complexity + HIGH impact  →  P0 (Phase N, P)
MEDIUM complexity + HIGH impact  →  P1 (Phase O, Q)
MEDIUM complexity + MEDIUM impact  →  P2 (Phase R, S, V)
HIGH complexity OR gated on hardware/data  →  P3 (Phase T, U)
```

---

## Phase N: NLP Foundation — Dictionary-Based Financial Sentiment ✅ COMPLETE (2026-05-15)

**Priority:** P0 — Highest value-per-line-of-code. Zero API cost.
**Depends On:** Nothing. Standalone.
**Source:** NLP 101 Modules 2-3
**Status:** N1-N3 complete. N4 (degree adverbs) deferred.

### Pre-Flight Checklist

```
□ uv sync — ensure nltk or spaCy available (check pyproject.toml)
□ uv run ruff check src/ — clean baseline
□ Read: useful_resources/papers_md/ (4 sentiment papers already summarized)
```

---

### AGENT TASK N1: Install dependencies + load L&M dictionary

**New files:** `src/signals/sentiment/dictionary.py`, `src/signals/sentiment/__init__.py`

1. Download Loughran-McDonald Master Dictionary (Notre Dame SAR):
   - URL: `https://sraf.nd.edu/loughranmcdonald-master-dictionary/`
   - Download CSV to `data/sentiment/LM_Master_Dictionary.csv`
   - Columns: Word, Positive, Negative, Uncertainty, Litigious, etc.
   - ~80,000 words with financial-domain tags

2. Create `src/signals/sentiment/__init__.py` with module exports.

3. Create `src/signals/sentiment/dictionary.py`:

```python
"""Loughran-McDonald financial sentiment dictionary loader and scorer.

The LM dictionary categorizes ~80,000 words by financial-domain sentiment:
- Positive: e.g., "profit", "growth", "achieve", "opportunity"
- Negative: e.g., "loss", "decline", "default", "terminate"
- Uncertainty: e.g., "may", "could", "approximately", "dependent"
- Litigious: e.g., "lawsuit", "plaintiff", "allegation", "breach"

Standard formula: Score = (N_pos - N_neg) / N_total
"""

from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


LM_DEFAULT_PATH = Path("data/sentiment/LM_Master_Dictionary.csv")


class LMDictionary:
    """Loughran-McDonald financial sentiment dictionary."""

    def __init__(self, csv_path: Path = LM_DEFAULT_PATH) -> None:
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None
        self._pos_words: set[str] = set()
        self._neg_words: set[str] = set()
        self._unc_words: set[str] = set()
        self._lit_words: set[str] = set()

    def _load(self) -> None:
        if self._df is not None:
            return
        self._df = pd.read_csv(self.csv_path)
        # LM dictionary marks positive words with Positive > 0
        self._pos_words = set(
            self._df[self._df.get("Positive", 0) > 0]["Word"].str.lower()
        )
        self._neg_words = set(
            self._df[self._df.get("Negative", 0) > 0]["Word"].str.lower()
        )
        self._unc_words = set(
            self._df[self._df.get("Uncertainty", 0) > 0]["Word"].str.lower()
        )
        self._lit_words = set(
            self._df[self._df.get("Litigious", 0) > 0]["Word"].str.lower()
        )
        logger.info(
            "LM Dictionary loaded: %d pos, %d neg, %d unc, %d lit words",
            len(self._pos_words),
            len(self._neg_words),
            len(self._unc_words),
            len(self._lit_words),
        )

    @property
    def positive_words(self) -> set[str]:
        self._load()
        return self._pos_words

    @property
    def negative_words(self) -> set[str]:
        self._load()
        return self._neg_words

    @property
    def uncertainty_words(self) -> set[str]:
        self._load()
        return self._unc_words

    @property
    def litigious_words(self) -> set[str]:
        self._load()
        return self._lit_words


class LMSentimentScorer:
    """Score financial text using the LM dictionary."""

    def __init__(self, dictionary: LMDictionary | None = None) -> None:
        self.dictionary = dictionary or LMDictionary()

    def score_text(self, text: str) -> dict[str, float]:
        """Score a single text document.

        Returns dict with keys: sentiment, uncertainty_ratio, litigious_ratio,
        positive_count, negative_count, uncertainty_count, litigious_count, total_words.
        """
        words = text.lower().split()
        total = len(words)
        if total == 0:
            return {
                "sentiment": 0.0,
                "uncertainty_ratio": 0.0,
                "litigious_ratio": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "uncertainty_count": 0,
                "litigious_count": 0,
                "total_words": 0,
            }

        pos = sum(1 for w in words if w in self.dictionary.positive_words)
        neg = sum(1 for w in words if w in self.dictionary.negative_words)
        unc = sum(1 for w in words if w in self.dictionary.uncertainty_words)
        lit = sum(1 for w in words if w in self.dictionary.litigious_words)

        sentiment = (pos - neg) / total
        uncertainty_ratio = unc / (pos + neg + 1)  # +1 avoid div/0
        litigious_ratio = lit / total

        return {
            "sentiment": round(sentiment, 4),
            "uncertainty_ratio": round(uncertainty_ratio, 4),
            "litigious_ratio": round(litigious_ratio, 4),
            "positive_count": pos,
            "negative_count": neg,
            "uncertainty_count": unc,
            "litigious_count": lit,
            "total_words": total,
        }

    def score_batch(
        self, texts: list[str], dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Score multiple texts, returning a time-indexed DataFrame."""
        results = [self.score_text(t) for t in texts]
        return pd.DataFrame(results, index=dates)


class LMTradingSignalModifier:
    """Convert LM sentiment scores into trading signal adjustments.

    Replaces the trivial linear multiplier in SentimentSignalModifier
    with LM dictionary-based scoring.

    Formula: adjusted_signal = base_signal * (1 + weight * sentiment)
    with additional penalties for high uncertainty (masking) and litigation risk.
    """

    def __init__(
        self,
        scorer: LMSentimentScorer | None = None,
        sentiment_weight: float = 0.15,
        uncertainty_penalty: float = 0.05,
        litigious_penalty: float = 0.10,
        uncertainty_threshold: float = 0.30,
        litigious_threshold: float = 0.05,
    ) -> None:
        self.scorer = scorer or LMSentimentScorer()
        self.sentiment_weight = sentiment_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.litigious_penalty = litigious_penalty
        self.uncertainty_threshold = uncertainty_threshold
        self.litigious_threshold = litigious_threshold

    def adjust_signal(self, base_signal: float, text: str) -> float:
        """Adjust a single trading signal using LM sentiment on associated text.

        Args:
            base_signal: ML probability in [0, 1]
            text: Associated news/filing text

        Returns:
            Adjusted probability in [0, 1]
        """
        scores = self.scorer.score_text(text)

        # Base sentiment adjustment
        adjustment = 1.0 + self.sentiment_weight * scores["sentiment"]

        # Uncertainty penalty — high uncertainty = management masking
        if scores["uncertainty_ratio"] > self.uncertainty_threshold:
            adjustment -= self.uncertainty_penalty

        # Litigious penalty — legal risk
        if scores["litigious_ratio"] > self.litigious_threshold:
            adjustment -= self.litigious_penalty

        adjusted = base_signal * max(adjustment, 0.5)
        return float(np.clip(adjusted, 0, 1))
```

### AGENT TASK N2: Integrate LM scorer into MLStrategy + backtest CLI

**Modified files:** `src/strategies/ml_strategy.py`, `scripts/run_ml_backtest.py`

1. In `ml_strategy.py`, add params: `use_lm_sentiment: bool = False`, `lm_sentiment_weight: float = 0.15`
2. In `init()`, instantiate `LMTradingSignalModifier` if `use_lm_sentiment=True`
3. In `next()`, after computing ML probability, adjust with LM sentiment (requires text source — for v1, use synthetic headline matching to establish the integration)
4. Add `--use-lm-sentiment` and `--lm-sentiment-weight` flags to `run_ml_backtest.py`

### AGENT TASK N3: Smoke test + validate

```bash
# 1. Download LM dictionary (manual step)
# Place at: data/sentiment/LM_Master_Dictionary.csv

# 2. Smoke test dictionary loader
uv run python -c "
from src.signals.sentiment.dictionary import LMDictionary, LMSentimentScorer
d = LMDictionary()
s = LMSentimentScorer(d)
# Test financial text
result = s.score_text('The company reported strong profit growth and expanding margins')
assert result['sentiment'] > 0, f'Should be positive, got {result[\"sentiment\"]}'
print(f'Positive: sentiment={result[\"sentiment\"]:.3f}')
# Test negative text
result2 = s.score_text('The company faces litigation risk and potential default on debt obligations')
assert result2['sentiment'] < 0, f'Should be negative, got {result2[\"sentiment\"]}'
assert result2['litigious_ratio'] > 0, f'Should detect litigation terms'
print(f'Negative: sentiment={result2[\"sentiment\"]:.3f}, litigious={result2[\"litigious_ratio\"]:.3f}')
# Test uncertainty masking
result3 = s.score_text('Revenue may grow approximately 5 percent but could decline if conditions deteriorate')
print(f'Uncertainty: sentiment={result3[\"sentiment\"]:.3f}, unc_ratio={result3[\"uncertainty_ratio\"]:.3f}')
assert result3['uncertainty_ratio'] > 0.2, 'Should detect high uncertainty'
"

# 3. Integration test with ml_strategy
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop --use-lm-sentiment --lm-sentiment-weight 0.15

# 4. Compare vs baseline
uv run scripts/run_ml_backtest.py SPY --start 2016-05-12 --trail-stop
```

### AGENT TASK N4: (Optional) Degree adverb weighting

**New file:** `src/signals/sentiment/adverb_weighting.py`

Implement keyword weighting by intensity:
```
"surging" > "growing" > "increasing" > "edging up"
"plunging" > "declining" > "decreasing" > "edging down"
```

Map to multipliers: surging/plunging = 2.0×, growing/declining = 1.5×, increasing/decreasing = 1.0×, edging = 0.5×

### Success Criteria for Phase N

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| LM dictionary loads from CSV | ≥ 2000 positive, ≥ 2000 negative words | dict test |
| Sentiment scoring direction correct | Positive text → score > 0; Negative → score < 0 | smoke test |
| Uncertainty detection works | High-uncertainty text → ratio > 0.2 | smoke test |
| Litigious detection works | Legal text → litigious > 0 | smoke test |
| MLStrategy integration runs | Backtest completes without error | backtest |
| Ruff clean | 0 errors | `uv run ruff check src/signals/sentiment/` |
| Existing tests pass | 0 failures | `uv run pytest tests/ -x -q` |

### Decision Gate after Phase N

```
IF LM dictionary scores produce valid sentiment AND integration runs:
  → N PASSES. Real financial-domain sentiment is now available.
  → Note: Text source still needed (news API or SEC filings) for live data.
  → Proceed to Phase P (Mean Reversion) or Phase O (social media validation).
ELSE:
  → DEBUG: check LM dictionary CSV format. Verify word matching (lowercase both sides).
  → Check nltk tokenization for word splitting.
```

---

## Phase P: Dedicated Mean Reversion System ✅ COMPLETE (2026-05-15)

**Priority:** P0 — Indicators already exist. Just needs routing + stops. ~400 lines.
**Depends On:** Nothing. Uses existing pattern detectors + indicators.
**Source:** Mean Reversion School
**Status:** P1-P4 complete. RegimeRouter (209.4% ret, Sharpe 0.68) validates routing.

### Pre-Flight

```
□ Read src/patterns/momentum/ — all existing mean-reversion indicators
□ Read src/strategies/rules_first_strategy.py — understand PatternBoostFilter pattern
□ Understand Phase 03 regime table (Trending/Ranging/Volatile/Transition)
```

---

### AGENT TASK P1: Create MeanReversionStrategy

**New file:** `src/strategies/mean_reversion_strategy.py`

```python
"""Dedicated mean-reversion strategy (backtesting.py Strategy).

Separates mean-reversion from trend-following — different stop logic,
different entry criteria, different regime routing.

Course insight: "King of ranging markets." Trend following loses here,
mean reversion prints money. But mean reversion gets steamrolled in trends —
so regime routing is CRITICAL.

Indicators (already implemented in src/patterns/momentum/):
- RSI divergence/oversold/overbought
- Williams %R
- Stoch RSI
- CCI divergence
- MFI (Money Flow Index)
- Bollinger Band touch/rejection
"""

from backtesting import Strategy

from src.patterns.momentum.rsi import RSIOversold, RSIOverbought
from src.patterns.momentum.williams_r import WilliamsROverbought, WilliamsROversold
from src.patterns.momentum.stoch_rsi import StochRSI
from src.patterns.momentum.cci import CCIDivergence
from src.patterns.momentum.mfi import MFIOversold, MFIOverbought
from src.patterns.volatility.bollinger import BollingerBandTouch


class MeanReversionStrategy(Strategy):
    """Pure mean-reversion strategy for ranging/sideways markets.

    Key differences from trend-following:
    - Fixed take-profit at mean, NOT trailing stop
    - Wider initial stop (reversion can overshoot before snapping back)
    - Time-based exit (if hasn't reverted in N bars, exit)
    - Requires regime gate: MUST be in Ranging regime (ADX < 20)
    - Fades extremes: buys oversold, shorts/sells overbought

    Not for trending markets — use RulesFirstStrategy there.
    """

    # Entry parameters
    rsi_oversold = 30
    rsi_overbought = 70
    bollinger_std = 2.0
    bollinger_period = 20

    # Exit parameters (mean-reversion specific)
    tp_atr_mult = 2.0        # Wider TP — wait for full reversion
    sl_atr_mult = 3.0        # Wider SL — reversion can overshoot
    max_hold_bars = 10       # Time stop — exit if not reverted in 10 bars

    # Regime gate
    require_ranging = True    # Only trade when ADX < 20
    adx_period = 14
    adx_threshold = 20

    # Sizing
    risk_per_trade = 0.01    # 1% risk per trade

    def init(self):
        # ... init indicator arrays ...
        pass

    def next(self):
        # ... check regime gate, detect signals, manage positions ...
        pass
```

### AGENT TASK P2: Implement regime-appropriate routing

**New file:** `src/strategies/regime_router_strategy.py`

A meta-strategy that routes between:
- **Trending (ADX > 25):** Route to `RulesFirstStrategy` signals
- **Ranging (ADX < 20):** Route to `MeanReversionStrategy` signals
- **Transition (ADX 20-25):** Hold current positions, no new entries

Track P&L separately per strategy. Report: which strategy generates the edge?

### AGENT TASK P3: Implement mean-reversion-specific stop management

| Type | Trend Following | Mean Reversion | Why |
|------|----------------|---------------|-----|
| Take-profit | ATR trail (dynamic) | Fixed TP at mean/ATR target | Reversion terminates at mean — trail would exit too early |
| Stop-loss | ATR trail (tight) | Wider ATR multiple (3× vs 1.5×) | Reversion often overshoots before snapping |
| Time stop | None | Exit after N bars if not reverted | Reversion is time-decaying — if it hasn't happened by N, it won't |
| Volatility filter | Optional | Required (low vol = mean reversion works) | High vol = trending = mean reversion gets wrecked |

### AGENT TASK P4: Backtest + compare

```bash
# Backtest mean-reversion standalone on SPY 2016-2024 (ranging periods only)
uv run scripts/backtest_mean_reversion.py SPY --start 2016-01-01 --end 2024-12-31

# Backtest regime-routed system (trend + reversion)
uv run scripts/backtest_regime_router.py SPY --start 2016-01-01 --end 2024-12-31

# Compare: trend-only vs reversion-only vs routed
# Expected: routing should improve Sharpe by capturing both regimes
```

### Success Criteria for Phase P

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| MeanReversionStrategy runs | Backtest completes | IS 2016-2024 |
| Regime routing fires both strategies | >0 trades in each engine | Check trade log |
| Mean-reversion profitable in ranging periods | Sharpe > 0 in ADX<20 bars | Regime-conditioned |
| Mean-reversion stops prevent steamrolling | MaxDD lower than unmanaged | Stop logic test |
| Ruff clean | 0 errors | Lint check |

### Decision Gate after Phase P

```
IF mean-reversion produces positive edge in ranging regimes AND routing works:
  → P PASSES. Project now handles both Trending AND Ranging markets.
  → Proceed to Phase O (social media validation).
ELSE:
  → DEBUG: check regime gate threshold. ADX < 20 may be too restrictive.
  → Try ADX < 25. Check indicator parameters per-asset.
```

---

## Phase O: Social Media Sentiment — Validation-First Approach

**Priority:** P1 — MUST validate sentiment predicts price BEFORE building ingestion.
**Depends On:** Phase N (needs LM dictionary for baseline sentiment). Phase P is independent.
**Source:** NLP 101 Modules 6-7
**Why validate first:** The course warns: social media is "high frequency, high noise, colloquial." Building scraping infrastructure BEFORE proving sentiment predicts price is a waste. Validate with existing labeled data first.

### Pre-Flight

```
□ Phase N complete — LM dictionary available
□ Check if any labeled financial Twitter datasets are available (StockNet, FiQA, etc.)
□ Read: useful_resources/papers_md/ (sentiment papers)
```

---

### AGENT TASK O1: Sentiment Lead/Lag Analysis

**New file:** `scripts/analyze_sentiment_lead_lag.py`

```python
"""Validate whether sentiment leads or lags price.

Crucial gate before ANY social media infrastructure investment.

Formula from course: ρ(τ) = Corr(S_t, P_{t+τ})
- If ρ is significant for τ > 0 → sentiment LEADS price → worth pursuing
- If ρ peaks at τ = 0 → sentiment REFLECTS price → no alpha
- If ρ peaks at τ < 0 → price leads sentiment → sentiment is reactive noise

Test with:
1. Financial news headlines (Bloomberg/Reuters archives — available historically)
2. Earnings call transcripts (SEC EDGAR)
3. Pre-labeled Twitter datasets (StockNet, etc.)
"""

# Sweep τ from -10 to +20 days
# Compute Pearson and Spearman correlation at each lag
# Report: max_corr, optimal_lag, significance, direction
```

### AGENT TASK O2: Social Media Noise Processing

**New files:** `src/signals/sentiment/noise_filter.py`

Implement three filters (can be tested on labeled datasets):

1. **Sarcasm detector:** Transformer-based binary classifier fine-tuned on financial sarcasm examples. "Great, another earnings miss 🙃" → negative.

2. **Emoji-to-sentiment mapping:**
   ```python
   FINANCIAL_EMOJI_MAP: dict[str, float] = {
       "🚀": 0.8, "📈": 0.6, "💵": 0.5, "💎": 0.5, "🙌": 0.4,
       "📉": -0.6, "🐻": -0.5, "💀": -0.7, "🤡": -0.3, "🔻": -0.4,
       "🤔": 0.0, "👀": 0.1, "📊": 0.0, "⚡": 0.2,
   }
   ```

3. **Bot detection heuristics:**
   - Account age < 30 days + high post frequency → flag
   - Identical text posted by multiple accounts within same hour → coordinated
   - Username pattern: random alphanumeric, no profile picture, low followers → likely bot

### AGENT TASK O3: (Conditional) Social Media Data Ingestion

**Gated on O1 passing** (sentiment must lead price):

**New files:** `src/data_ingestion/twitter_scraper.py`, `src/data_ingestion/reddit_scraper.py`

| Source | API | Notes |
|--------|-----|-------|
| Twitter/X | X API v2 (free tier: 1500 tweets/month) | Filter by $cashtag |
| Reddit | PRAW (free) | r/wallstreetbets, r/stocks, r/investing |
| StockTwits | StockTwits API (free) | Messages pre-tagged with ticker + sentiment |

### Success Criteria for Phase O

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Lead/lag analysis runs | τ sweep -10 to +20 complete | Script output |
| Correlation direction identified | Max ρ at identifiable lag | Report |
| GATE: sentiment leads price | Max ρ at τ > 1 with p < 0.05 | Statistical test |
| Noise filters tested | Sarcasm + emoji + bot detection functional | Unit tests |

### Decision Gate after Phase O

```
IF sentiment LEAD/LAG analysis shows sentiment PREDICTS price (τ > 0, p < 0.05):
  → O PASSES. Sentiment is a real alpha source. Proceed to Phase Q.
  → Social media ingestion (O3) is now justified.
ELSE:
  → O BLOCKED. Sentiment does NOT predict price. Do not build social media infra.
  → Sentiment is reactive/reflective, not predictive. Move to Phase Q directly.
  → Revisit if new labeled datasets become available.
```

---

## Phase Q: Multi-Factor Fundamental Factors

**Priority:** P1 — New feature dimension for CatBoost.
**Depends On:** Nothing. Can run in parallel with N/P/O.
**Source:** Multi-Factor School
**Why here:** The project's 88 features are all technical/price-derived. Adding fundamental factors (Value, Quality, Size, etc.) gives CatBoost a completely new dimension to learn from. Requires fundamental data source.

### Pre-Flight

```
□ Check: does yfinance expose fundamental data? (P/E, P/B, market cap, ROE)
□ Alternative: Polygon.io Fundamentals API (paid but comprehensive)
□ Alternative: Financial Modeling Prep API (free tier: 250 requests/day)
□ Check: pyproject.toml for data source libraries
```

---

### AGENT TASK Q1: Fundamental Factor Extractor

**New file:** `src/ml/fundamental_features.py`

Extract these factors per ticker (use yfinance `ticker.info` for v1):

| Factor | Field | Category | Direction |
|--------|-------|----------|-----------|
| P/E ratio | trailingPE | Value | Lower = cheaper |
| P/B ratio | priceToBook | Value | Lower = cheaper |
| P/S ratio | priceToSales | Value | Lower = cheaper |
| EV/EBITDA | enterpriseToEbitda | Value | Lower = cheaper |
| ROE | returnOnEquity | Quality | Higher = better |
| ROA | returnOnAssets | Quality | Higher = better |
| Profit margin | profitMargins | Quality | Higher = better |
| Debt/Equity | debtToEquity | Quality | Lower = safer |
| Market cap | marketCap | Size | Smaller may outperform |
| Revenue growth | revenueGrowth | Growth | Higher = better |
| Earnings growth | earningsGrowth | Growth | Higher = better |
| Dividend yield | dividendYield | Income | Higher = better |
| Beta | beta | Risk | Lower = defensive |
| Short % float | shortPercentOfFloat | Sentiment | High = bearish |

**Implementation:**
```python
class FundamentalFeatureExtractor:
    """Extract fundamental factors from yfinance or Polygon.io."""

    def __init__(self, source: str = "yfinance"):
        self.source = source

    def extract(self, tickers: list[str]) -> pd.DataFrame:
        """Return DataFrame of fundamental factors per ticker.
        Index: ticker, Columns: pe_ratio, pb_ratio, roe, ...
        """
        ...

    def z_score_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectionally z-score all factors."""
        return (df - df.mean()) / df.std()

    def composite_score(self, df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
        """Weighted sum of z-scored factors → composite alpha score per ticker."""
        ...
```

### AGENT TASK Q2: Integrate Fundamentals into Feature Pipeline

**Modified files:** `src/ml/feature_engineering.py`, `src/ml/cross_asset_features.py`

Add fundamental factor columns to the feature matrix. Join on ticker + date. Forward-fill quarterly fundamentals (they update slowly).

### AGENT TASK Q3: Multi-Factor Scoring Backtest

**New file:** `scripts/backtest_multi_factor.py`

Long top-quintile stocks, equal-weight. Monthly rebalance. Compare vs SPY B&H.

### Success Criteria for Phase Q

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Fundamental factors extracted | ≥ 10 factors per ticker | Smoke test on SPY |
| Feature matrix includes fundamentals | New columns in feature df | Shape check |
| Multi-factor backtest runs | ≥ 5 stocks, monthly rebalance | Backtest completes |
| Factor IC computed | Information Coefficient per factor | Report |

### Decision Gate after Phase Q

```
IF fundamental factors add new predictive information (IC > 0.02 for any factor):
  → Q PASSES. Fundamentals add real alpha. CatBoost can use them.
  → Proceed to Phase R.
ELSE:
  → Technical factors may already capture what fundamentals provide.
  → Still include fundamentals — they may help in different regimes.
```

---

## Phase R: Advanced NLP + LLM Deployment

**Priority:** P2 — Builds on Phase N. Requires GPU for FinBERT (or CPU inference, slower).
**Depends On:** Phase N (LM dictionary foundation)
**Source:** NLP 101 Module 8 + AI Intelligence School
**Why here:** FinBERT + LLM analysis is the "endgame" for NLP. But it's P2 because Phase N (cheap dictionary) and Phase P (mean reversion) deliver more value per line of code.

### Pre-Flight

```
□ Phase N complete — LM dictionary available, text preprocessing pipeline ready
□ GPU preferred for FinBERT inference (CPU inference: ~0.5s/sentence, acceptable for daily data)
□ uv add transformers torch sentence-transformers
```

---

### AGENT TASK R1: FinBERT Deployment

**New files:** `src/signals/sentiment/finbert.py`

```python
"""FinBERT financial sentiment model (ProsusAI/finbert).

ProsusAI/finbert: BERT fine-tuned on financial text (TRC2-financial + FiQA).
Outputs: positive, negative, neutral probabilities per sentence.

Usage:
    model = FinBERTSentiment(model_name="ProsusAI/finbert")
    scores = model.predict(["Revenue grew 20% YoY", "Company faces headwinds"])
    # Returns: [[0.85, 0.05, 0.10], [0.02, 0.92, 0.06]]  # [pos, neg, neutral]
```

Three providers implementing the existing SentimentProvider Protocol:
1. **FinBERTSentiment** — local FinBERT inference (CPU-viable for daily data)
2. **LMDictionarySentiment** — LM dictionary (from Phase N, zero-cost fallback)
3. **Comparison harness** — compare LM vs FinBERT on same text, measure agreement

### AGENT TASK R2: SEC Filing Mining

**New files:** `src/data_ingestion/sec_filing_scraper.py`, `src/signals/sentiment/filing_analyzer.py`

1. **SEC EDGAR scraper:** Download 10-K/10-Q filings for any ticker via SEC EDGAR API. Parse HTML → extract MD&A section.

2. **Filing analyzer:**
   - Year-over-year text similarity (cosine similarity between MD&A sections)
   - Tone change detection (sentiment shift from LM or FinBERT)
   - Uncertainty ratio trends (is management getting more vague?)
   - Emerging keyword detection (TF-IDF: words new this year vs. corpus)

### AGENT TASK R3: Multi-Source Sentiment Fusion

**New file:** `src/signals/sentiment/multi_source_fusion.py`

Combine multiple sentiment sources into one signal:

| Source | Weight (init) | How |
|--------|--------------|-----|
| LM dictionary (news) | 0.30 | Phase N |
| FinBERT (news) | 0.30 | Phase R |
| SEC filing tone | 0.20 | Phase R |
| Social media sentiment | 0.20 | Phase O (if gate passed) |

Weights rebalanced by rolling predictive power (IC-based). Only trade when ≥ 2 sources agree on direction.

### AGENT TASK R4: NLP+Financial Fusion Model

**Modified file:** `src/ml/pattern_classifier.py`

Train CatBoost with BOTH traditional price features (88) AND NLP-derived features:
- LM sentiment score (from news/filings)
- Uncertainty ratio
- Litigious ratio
- Text drift score (cosine similarity year-over-year)
- FinBERT positive probability
- Sentiment source agreement score (0 = all disagree, 1 = all agree)

Compare vs price-only model. Expected: NLP features should add incremental predictive power.

### Success Criteria for Phase R

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| FinBERT loads and predicts | >0 inferences/sec on CPU | Smoke test |
| SEC scraper downloads filing | 10-K text extracted | Test on SPY |
| Text similarity between years computed | 0 < sim < 1 | Cosine check |
| NLP+Financial model trained | CV AUC reported | Pipeline output |

### Decision Gate after Phase R

```
IF NLP features improve CatBoost AUC by ≥ 0.01:
  → R PASSES. NLP adds real predictive power.
  → Sentiment module is now production-grade.
ELSE IF AUC unchanged::
  → NLP features are redundant with price features.
  → Still include for regime-dependent value (sentiment may help in crashes).
```

---

## Phase S: Statistical Arbitrage — Pairs & Spread Trading

**Priority:** P2 — New alpha source, orthogonal to trend-following.
**Depends On:** Phase P (mean-reversion engine — pairs trading is a specialization)
**Source:** Statistical Arbitrage School
**Why here:** Value per line of code is lower than P or N. Requires dedicated infrastructure (cointegration testing, spread calculation, dual-leg execution). But the alpha is genuinely independent from trend-following.

### Pre-Flight

```
□ Phase P complete — mean-reversion framework exists
□ uv add statsmodels (if not already) — for cointegration tests
□ Read: existing CrossAssetFeatureExtractor (src/ml/cross_asset_features.py) — correlation features already computed
```

---

### AGENT TASK S1: Pairs Trading Engine

**New file:** `src/strategies/pairs_trading_strategy.py`

Pipeline:
1. **Correlation screen:** Compute 252d rolling correlation for all ticker pairs in universe
2. **Cointegration test:** Engle-Granger cointegration test (statsmodels). Only trade cointegrated pairs.
3. **Hedge ratio:** Kalman filter (pykalman) for dynamic hedge ratio. More robust than static OLS.
4. **Spread:** `spread = log(P_A) - hedge_ratio * log(P_B)`
5. **Entry:** Z-score of spread > 2.0 (short the spread) or < -2.0 (long the spread)
6. **Exit:** Z-score crosses 0 (or stop-loss at 3.0)

### AGENT TASK S2: Pre-Built Sector Pairs Universe

Hardcoded high-corr pairs within sectors:

| Sector | Pair | Rationale |
|--------|------|-----------|
| Consumer | KO-PEP | Coke vs Pepsi — textbook pair |
| Financial | JPM-GS | Major banks |
| Energy | CVX-XOM | Supermajors |
| Tech | QQQ-SPY | Tech-heavy vs broad market |
| Retail | WMT-TGT | Big-box retail |
| Defense | LMT-NOC | Defense contractors |

Add more pairs via automated correlation screen.

### AGENT TASK S3: Backtest Pairs Strategy

```bash
uv run scripts/backtest_pairs.py KO-PEP --start 2016-01-01 --end 2024-12-31
uv run scripts/backtest_pairs.py --auto-pairs --min-correlation 0.7 --n-pairs 10
```

### Success Criteria for Phase S

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Cointegration test works | Engle-Granger p-value output | Test on KO-PEP |
| Spread z-score computed | Finite values | Smoke test |
| Pairs backtest completes | ≥ 10 trades | KO-PEP 2016-2024 |
| Pairs Sharpe independent of trend | Correlation < 0.3 with RulesFirst | Cross-check |

### Decision Gate after Phase S

```
IF pairs trading produces positive Sharpe with < 0.3 correlation to trend-following:
  → S PASSES. Pairs trading is a genuine diversifier.
ELSE:
  → Pairs may not have enough dispersion in chosen universe.
  → Expand to more pairs or try ETF/index pairs.
```

---

## Phase V: Strategy Architecture v2 — Short-Side, Multi-Asset, Position Sizing

**Priority:** P2 — Structural improvements, not new alpha sources.
**Depends On:** Phase P (mean reversion) + Phase S (pairs). Partially independent.
**Source:** Cross-school (Trend Following, Multi-Factor)
**Why here:** Enables short-selling (Crisis Alpha), crypto expansion, correlation-aware portfolio allocation. Needed for production-grade multi-asset deployment.

### Pre-Flight

```
□ Phase P complete — two strategy engines exist (trend + reversion)
□ Check: does backtesting.py support short positions? (Yes — negative size)
□ T10c-1: CCXT already planned (deferred) — crypto API for Phase V
```

---

### AGENT TASK V1: Short-Side Pattern Activation

**Modified file:** `src/strategies/rules_first_strategy.py`

Enable bearish patterns to trigger short entries:
- Head & Shoulders → short
- Double Top → short
- Bearish engulfing → short
- Bearish harmonic patterns → short

For backtesting.py: `self.sell(size=size)` instead of `self.buy()`, or use negative position sizes.

Implementation: `use_short: bool = True` param. Short via inverse ETFs if direct short not supported.

### AGENT TASK V2: Strategy-Specific Position Sizing

**New file:** `src/risk/strategy_aware_sizing.py`

| Strategy Type | Kelly Fraction | Rationale |
|--------------|---------------|-----------|
| Trend Following | 0.5 (Half-Kelly) | Reliable in trending markets, lower uncertainty |
| Mean Reversion | 0.25 (Quarter-Kelly) | More fragile — can get steamrolled |
| Pairs Trading | 0.5 (Half-Kelly) | Market-neutral, highest confidence |
| ML Signal | 0.25 (Quarter-Kelly) | Model confidence may be overestimated |

Kelly formula: `f* = (p * b - q) / b` where p = win_rate, b = avg_win/avg_loss, q = 1-p.

### AGENT TASK V3: Multi-Asset Correlation-Aware Portfolio

**New file:** `src/portfolio/multi_asset_allocator.py`

When running strategies on N instruments simultaneously:
1. Compute N×N correlation matrix from rolling 60d returns
2. When two instruments have correlation > 0.7 → treat as cluster
3. Cluster-level allocation = 1/num_clusters (equal risk to each cluster)
4. Within cluster: equal-weight or signal-strength-weight

Prevents over-concentration in correlated bets (e.g., SPY + QQQ + XLK = triple tech bet).

### AGENT TASK V4: Crypto Trading Expansion

**Modified files:** `scripts/run_ml_backtest.py`, `scripts/backtest_rules_first.py`

Add `--asset-class crypto` flag. Use CCXT (T10c-1) for data. Format: `BTC_USD`.

Key differences from equities:
- 24/7 trading (no market hours constraint)
- Higher volatility (wider stops, larger ATR multiples)
- Different ticker universe (BTC, ETH, SOL, AVAX, etc.)

### Success Criteria for Phase V

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| Short entries fire on bearish patterns | Short trades in trade log | Bear market backtest |
| Strategy-aware sizing allocates correctly | Different sizes per strategy type | Sizing debug output |
| Multi-asset portfolio respects corr limits | No correlated cluster > 30% | Correlation check |
| Crypto backtest runs | Trades on BTC_USD | Backtest |

### Decision Gate after Phase V

```
IF short-side + multi-asset + crypto all work end-to-end:
  → V PASSES. Production-grade multi-strategy, multi-asset system.
ELSE:
  → Fix individual component. Each is independently testable.
```

---

## Phase T: Deep Learning Models

**Priority:** P3 — Gated on GPU availability. Same gate as Phase 05 (ML Advanced).
**Depends On:** GPU hardware (≥16GB VRAM preferred, ≥8GB minimum)
**Source:** AI Intelligence School
**Why last:** Deep learning requires GPU. Classical ML (CatBoost) + rules already achieve OOS Sharpe +0.76. Deep learning may or may not improve on this — it's an experiment, not a gap-closer.

### Pre-Flight

```
□ GPU available? (nvidia-smi or WSL2 with GPU passthrough)
□ If NO GPU: SKIP Phase T. Return when hardware available.
□ uv add torch torchvision torchaudio pytorch-lightning
□ Expand Phase 05 (ML Advanced) with these additional models
```

### AGENT TASK T1: LSTM/GRU Price Sequence Predictor

**New file:** `src/ml/models/lstm_predictor.py`

Input: (n_bars, n_features) sequence → LSTM/GRU → predict next-bar direction or N-bar return.
Compare AUC vs CatBoost. Only promote if LSTM AUC > CatBoost AUC by ≥ 0.02.

### AGENT TASK T2: Transformer for Multivariate Time Series

**New file:** `src/ml/models/transformer_predictor.py`

Self-attention over price history + features. Capture long-range dependencies.
Compare vs LSTM and CatBoost.

### AGENT TASK T3: CNN for Chart Pattern Recognition

**New file:** `src/ml/models/cnn_patterns.py`

Treat OHLCV as image-like input. CNN learns visual patterns directly.
Compare learned patterns vs 34 hand-crafted detectors. Do they find the same things?

### AGENT TASK T4: Temporal Fusion Transformer (TFT)

**New file:** `src/ml/models/tft_forecaster.py`

Google's interpretable multi-horizon forecaster. Variable selection + attention + quantile outputs.
Predict 1/5/21-day returns with prediction intervals.

### AGENT TASK T5: GAN for Synthetic Market Data

**New file:** `src/ml/models/gan_market.py`

Generate synthetic OHLCV data for stress-testing strategies:
- Train on crash periods → generate "extreme but realistic" scenarios
- Test strategy robustness against regime extremes

### Success Criteria for Phase T

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| LSTM predicts next-bar direction | AUC > 0.50 | Backtest on SPY |
| Transformer matches LSTM performance | AUC within 0.01 | Comparison |
| CNN pattern detection works | Visual inspection of learned filters | Plot |
| GAN generates realistic OHLCV | Visual inspection, statistical checks | KS test |

### Decision Gate after Phase T

```
IF LSTM/Transformer AUC > CatBoost AUC + 0.02:
  → T PASSES (partial). Deep learning adds value. Consider ensemble.
ELSE:
  → Deep learning does NOT beat classical ML on this data.
  → CatBoost remains the ML backbone. GPU not needed — skip remaining T tasks.
```

---

## Phase U: Alternative Data Sources

**Priority:** P3 — Requires paid data subscriptions. Zero existing integration.
**Depends On:** Data provider subscriptions (satellite, credit card, supply chain)
**Source:** Multi-Factor School Stage 2 (data evolution)
**Why last:** These are experiments, not core infrastructure. High cost, uncertain payoff.

### AGENT TASK U1-U5: (Deferred until data access confirmed)

| # | Data Source | Provider | Cost | Potential Signal |
|---|------------|----------|------|-----------------|
| U1 | Satellite imagery (retail parking) | Orbital Insight, RS Metrics | $10K+/yr | Predict retail revenue before earnings |
| U2 | Credit card transaction data | Second Measure, Earnest | $10K+/yr | Real-time consumer spending proxy |
| U3 | Supply chain / shipping | MarineTraffic, Freightos | $1K+/yr | Economic activity leading indicator |
| U4 | Job posting data | Revelio Labs, LinkUp | $5K+/yr | Company growth/hiring signal |
| U5 | Google Trends | Free API | $0 | Retail attention / search volume indicator |

### Success Criteria for Phase U

| Criterion | Threshold | Verification |
|-----------|-----------|-------------|
| At least 1 data source integrated | Data flows into feature pipeline | Backtest |
| Alternative data improves AUC | ΔAUC > 0.01 | Comparison |
| Cost-benefit justified | Return improvement > data cost | P&L analysis |

---

## Master Priority Order

```
Phase N  (P0) — NLP Foundation: LM Dictionary Scorer ................ ████████  ~300 lines
Phase P  (P0) — Mean Reversion System ............................... ████████  ~400 lines
Phase O  (P1) — Social Media Validation (gate on lead/lag) .......... ██████    validation-first
Phase Q  (P1) — Multi-Factor Fundamental Factors .................... ██████    new features
Phase R  (P2) — Advanced NLP: FinBERT + SEC Filings ................. ████      builds on N
Phase V  (P2) — Strategy Architecture v2 (short/crypto/portfolio) ... ████      structural
Phase S  (P2) — Statistical Arbitrage: Pairs Trading ................ ████      orthogonal alpha
Phase T  (P3) — Deep Learning (GPU gate) ............................ ██        gated on hardware
Phase U  (P3) — Alternative Data (paid gate) ........................ █         gated on budget
```

---

## Cross-Cutting Rules (same as all agents)

1. **uv for everything:** `uv run`, `uv add`, NEVER bare `python` or `pip`
2. **Type hints:** All function signatures must have type hints
3. **Docstrings:** Google-style, 1-3 lines on public functions
4. **Ruff clean:** Run `uv run ruff check src/` before marking any task complete
5. **Update docs:** After creating any new CLI script, update `docs/COMMAND_CHEATSHEET.md`
6. **Update MEMORY.md:** After each phase, update completed tasks
7. **Update BESTS.md:** After any backtest producing results, update the leaderboard
8. **Log to current.md:** After each significant action, append timestamped row
9. **Never commit:** Unless explicitly asked
10. **Check existing patterns:** Before creating new files, look at existing similar files
