# Unplanned Ideas: NLP Course + Quant Schools

> **Source:** NLP in Quant 101 course curriculum + Chinese 5 Quant Schools framework
> **Scoped:** Only ideas that (a) could help this project AND (b) are NOT currently implemented AND (c) are NOT tracked in any existing plan
> **Date:** 2026-05-15

**Cross-reference verified against:** `full.md` (all phases), `direction-c-research-signals.md`, `enhance-tool-evaluation.md`, `enhance-knowledge-graph.md`, `fix-overfitting.md`, `post_retrain_next_steps.md`

---

## Category 1: NLP Pipeline Infrastructure (NLP 101 Modules 2–3)

> The project has `SentimentSignalModifier` (a linear probability multiplier) but zero upstream NLP.

### 1.1 Text Preprocessing Pipeline

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N1 | Tokenization & segmentation | NLP M2 | Break financial text into words/tokens. NLTK or spaCy-based sentence→word pipeline. |
| N2 | Stop-word removal for finance | NLP M2 | Remove useless words ("the", "is", "and") but preserve financial-significant stop words ("above", "below" → directional cues). |
| N3 | Financial dictionary matching engine | NLP M2 | Pluggable dictionary interface: load word list → tag tokens → count matches. Foundation for all scoring below. |

### 1.2 Loughran & McDonald Financial Dictionary (Core Gap)

> **This is the single highest-value unimplemented idea.** Standard in academic finance NLP. Zero API cost. Replaces all synthetic/test sentiment with real domain-specific signal.

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N4 | L&M dictionary loader | NLP M3 | Load 2000+ financial-domain words across 4 dimensions: **Positive** (e.g., "profit", "growth"), **Negative** (e.g., "loss", "decline"), **Uncertainty** (e.g., "may", "could", "approximately"), **Litigious** (e.g., "lawsuit", "claimant", "plaintiff"). Source: Notre Dame SAR database. |
| N5 | L&M sentiment scorer | NLP M3 | Implement the course formula: `Score = (N_pos - N_neg) / N_total`. Produce per-document sentiment in [-1, 1]. Extend to 4-dimension vector output: [pos_count, neg_count, unc_count, lit_count]. |
| N6 | L&M uncertainty detector | NLP M3 | High uncertainty + low positive = management masking bad news. Signal: uncertainty_ratio triggers negative alpha even when net sentiment is neutral. |
| N7 | L&M litigious risk flag | NLP M3 | Sudden spike in litigious terms → legal risk → negative alpha on affected stocks. Different signal from general negative sentiment. |
| N8 | Keyword weighting (degree adverbs) | NLP M3 | Weight verbs by degree: "surged" > "increased" > "edged up". Build financial adverb-intensity mapping table. |

### 1.3 Financial Text Vectorization (Beyond Dictionary Counting)

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N9 | TF-IDF feature extraction | NLP M2 | Term Frequency-Inverse Document Frequency for financial text. Identifies words frequent in THIS news but rare in the corpus → emerging signals. |
| N10 | Word2Vec / FastText embeddings for finance | NLP M2 | Pre-trained or custom-trained word embeddings on financial corpus. Captures semantic relationships (e.g., "EBITDA" ≈ "profit" in vector space). |
| N11 | Transformer embedding (sentence-level) | NLP M2 | Use sentence-transformers to convert full news paragraphs into dense vectors. Compare cosine similarity across time for narrative drift detection. |

---

## Category 2: News & Report Analysis (NLP 101 Modules 3–5 + AI Intelligence School)

### 2.1 Real-Time News Ingestion

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N12 | News API integration | NLP M3 | Polygon.io News API, Benzinga, Alpaca News, or NewsAPI for real-time financial headline scraping. |
| N13 | News-to-symbol entity mapping | NLP M3 | Given a news headline, determine which tickers it affects. Disambiguate "Apple" → $AAPL vs fruit company. |
| N14 | Abnormal Return (AR) event study | NLP M3 | On news release, compute: `AR = Actual Return - Expected Return` (from market model CAPM β). Flag statistically significant AR as tradeable event. |

### 2.2 SEC Filing / Financial Report Mining

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N15 | SEC EDGAR 10-K/10-Q scraper | NLP M4 | Automated download of annual/quarterly reports via SEC EDGAR API for any ticker. Store raw HTML/text. |
| N16 | MD&A section extractor | NLP M4 | Isolate "Management Discussion and Analysis" section — contains management's narrative about performance, risks, and outlook. |
| N17 | Year-over-year text similarity comparison | NLP M5 | Cosine similarity between this year's MD&A vs last year's. Low similarity = narrative change = potential strategy shift or hidden issue. |
| N18 | Tone certainty classifier | NLP M4 | Classify sentences as "certain" vs "vague." High vagueness in earnings discussion = management hiding bad news. Use L&M uncertainty dimension + syntactic patterns. |
| N19 | Emerging keyword detection via TF-IDF | NLP M5 | Words that appear frequently in this filing but rarely historically → new technology adoption, new risk factors, new business lines before analysts notice. |

### 2.3 Multi-Factor Alpha Fusion (Text + Financial Metrics)

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N20 | NLP+financial feature fusion model | NLP M5 | Train CatBoost with both traditional price features AND NLP-derived features (sentiment score, uncertainty ratio, text drift score). Single unified model predicting alpha. |
| N21 | Earnings call transcript analysis | AI School | Analyze quarterly earnings call transcripts: CEO tone, Q&A sentiment, analyst question patterns. Feed into NLP pipeline. |

---

## Category 3: Social Media Analysis (NLP 101 Modules 6–7)

> **Zero social media infrastructure exists.** 22 sentiment papers in the knowledge graph but no module.

### 3.1 Social Media Data Ingestion

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N22 | Twitter/X API sentiment stream | NLP M6 | Real-time tweet collection filtered by ticker/$cashtag. Historical search for backtesting. |
| N23 | Reddit (r/wallstreetbets, r/stocks) scraper | NLP M6 | Pull posts/comments with ticker mentions. High noise but captures retail sentiment herding. |
| N24 | StockTwits sentiment feed | NLP M6 | Dedicated financial social network. Messages tagged with ticker and sentiment (bullish/bearish). Structured data easier to process than Twitter. |

### 3.2 Social Media Noise Processing

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N25 | Sarcasm / irony detector | NLP M7 | "Great, another earnings miss 😂" → actually negative. Use transformer-based sarcasm classifier fine-tuned on financial text. |
| N26 | Emoji-to-sentiment mapping | NLP M7 | Map emojis to sentiment scores: 🚀→+0.8, 📉→-0.7, 💎🙌→+0.5 (holding), 🤡→-0.4 (mocking). Financial-emoji dictionary. |
| N27 | Bot / "water army" detection | NLP M7 | Identify coordinated pump-and-dump accounts: (a) account creation date clustering, (b) identical message patterns, (c) sudden coordinated activity spikes. Filter signals from flagged accounts. |
| N28 | Behavioral fingerprinting | NLP M7 | Per-account metrics: post frequency distribution, mean sentiment, follower/following ratio, engagement patterns. Flag statistical outliers as likely bots. |

### 3.3 KOL (Key Opinion Leader) Propagation

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N29 | Influencer identification | NLP M6 | By sector/ticker: who moves the conversation? Track top-N accounts by engagement-weighted follower count. |
| N30 | Sentiment propagation network | NLP M6 | When Influencer A turns bearish, measure how sentiment spreads to B, C, D over time. Build directed graph of sentiment contagion. |
| N31 | Public opinion crisis early warning | NLP M6 | Real-time anomaly detection on social media volume + sentiment. Volume spike + sentiment crash = breaking scandal → trade BEFORE traditional news picks it up. |

### 3.4 Sentiment Time Series Analysis

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N32 | Sentiment lead/lag correlation | NLP M7 | `ρ(τ) = Corr(S_t, P_{t+τ})` — does sentiment at time t predict price at t+τ? Sweep τ from 1 to 20 days. Only trade if ρ is significant. |
| N33 | Sentiment volatility prediction | NLP M7 | Social media sentiment dispersion (std of scores) predicts next-day realized volatility. Feed into dynamic position sizing. |
| N34 | Sentiment regime shift detection | NLP M7 | When mean sentiment shifts >2σ from 60d rolling → regime change. Aggregate across tickers for market-wide sentiment regime indicator. |

---

## Category 4: Multi-Factor / Fundamental Factor Investing (Multi-Factor School)

> The project has technical factors only (price-derived). Zero fundamental factors.

### 4.1 Fundamental Factor Extraction

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N35 | Value factors | Multi-Factor | P/E, P/B, P/S, EV/EBITDA, FCF yield, dividend yield. Requires fundamental data (Yahoo Finance fundamentals or Polygon.io). |
| N36 | Quality factors | Multi-Factor | ROE, ROA, profit margins, debt-to-equity, accruals, earnings quality. Low-debt high-ROE companies tend to outperform. |
| N37 | Size factor | Multi-Factor | Market capitalization. Small-cap premium historically significant. Requires market cap data per ticker. |
| N38 | Low volatility factor | Multi-Factor | 60d/252d realized volatility percentile within sector. Low-vol stocks tend to have higher risk-adjusted returns. |
| N39 | Growth factors | Multi-Factor | Revenue growth YoY, earnings growth YoY, EPS estimate revisions. |

### 4.2 Factor-Based Portfolio Construction

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N40 | Multi-factor stock scoring | Multi-Factor | Z-score each factor, combine with equal or IC-weighted sum. Rank all stocks. Long top quintile, short bottom quintile (or long-only in backtesting.py). |
| N41 | Market-neutral factor portfolio | Multi-Factor | Long top-scored stocks, short bottom-scored stocks → beta-neutral. Profits from factor spread alone, independent of market direction. Requires short capability. |
| N42 | Factor timing / dynamic rotation | Multi-Factor | Which factor works NOW? Momentum factor dominates in trending regimes; Value in recovery; Low Vol in crashes. Rotate weights based on regime. |
| N43 | Industry-neutral factor scores | Multi-Factor | Score within industry, not cross-industry. Prevents sector bets masquerading as factor signals (e.g., all tech scores high on momentum). |

---

## Category 5: Dedicated Mean Reversion System (Mean Reversion School)

> Indicators exist but are mixed into the trend-following system. No dedicated mean-reversion subsystem.

### 5.1 Mean Reversion Strategy Engine

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N44 | Bias rate trigger | Mean Reversion | `Bias = (Close - MA(n)) / MA(n) * 100`. Extreme bias → mean-reversion entry. Different thresholds for different assets/vol regimes. |
| N45 | Bollinger Band mean-reversion with confidence | Mean Reversion | Not just "touched band" → requires confirmation: did price REJECT the band (intraday reversal)? Did volume confirm? Multiple timeframes agree? |
| N46 | RSI extreme zone with volatility filter | Mean Reversion | RSI > 70 + ATR contracting = fading opportunity. RSI > 70 + ATR expanding = trend continuation — do NOT fade. Volatility context distinguishes reversion from momentum. |
| N47 | Mean-reversion-specific stop management | Mean Reversion | Trend-following stops (ATR trail) WRECK mean-reversion. Need: wider initial stop, time-based exit (if hasn't reverted in N days, exit regardless), no trailing — fixed target at mean. |
| N48 | Regime-appropriate strategy routing | Mean Reversion | ADX < 20 → route to Mean Reversion engine. ADX > 25 → route to Trend Following engine. ADX 20-25 → hold. Separate P&L tracking per engine. |

### 5.2 Pair Spread Mean Reversion

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N49 | Cointegration-based pair selection | Mean Reversion | Engle-Granger or Johansen cointegration test on pairs of correlated tickers. Only trade pairs that are cointegrated (stationary spread). |
| N50 | Spread z-score entry/exit | Mean Reversion | `z = (spread - mean_spread) / std_spread`. Entry at z > 2.0, exit at z < 0.5. Market-neutral by construction. |

---

## Category 6: Statistical Arbitrage (Statistical Arbitrage School)

> Zero stat-arb infrastructure. Cross-asset features exist but are used for directional prediction, not spread trading.

### 6.1 Pairs Trading

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N51 | Pairs trading engine | Stat Arb | Full pipeline: correlation screen → cointegration test → Kalman filter hedge ratio (dynamic) → spread calculation → entry/exit signals → P&L tracking. |
| N52 | Sector pairs universe | Stat Arb | Pre-screen within sectors: KO-PEP (consumer), JPM-GS (financials), CVX-XOM (energy), QQQ-SPY (broad market). Higher cointegration probability intra-sector. |
| N53 | Kalman filter dynamic hedge ratio | Stat Arb | Higher-order extension of N50. Hedge ratio adapts as the relationship shifts. More robust than static OLS hedge. |

### 6.2 ETF Arbitrage

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N54 | ETF-NAV deviation arbitrage | Stat Arb | ETF price vs sum of constituent prices (NAV). When deviation > threshold, buy cheap, sell expensive. Risk-free in theory; requires execution speed. |

### 6.3 Futures/Calendar Spreads

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N55 | Futures calendar spread | Stat Arb | Long near-month, short far-month (or reverse). Bet on term structure convergence. Requires futures data (not currently in project scope). |
| N56 | Cross-asset spread arbitrage | Stat Arb | Gold vs Gold Miners (GLD vs GDX), Oil vs Energy sector (USO vs XLE), Bonds vs Equities (TLT vs SPY divergences). |

---

## Category 7: Advanced AI / Deep Learning (AI Intelligence School)

> Classical ML (CatBoost, RL DQN, ensemble) is well-covered. Everything below requires GPU or is deep learning.

### 7.1 Deep Learning Models

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N57 | LSTM / GRU for price sequence prediction | AI School | Sequential price data → LSTM predicts next-N-day return. Captures temporal dependencies CatBoost misses. Requires GPU. |
| N58 | Transformer for multivariate time series | AI School | Attention-based architecture for OHLCV + features. Self-attention captures long-range dependencies better than LSTM. |
| N59 | CNN for candlestick pattern recognition | AI School | Treat price charts as images. CNN learns visual patterns beyond 34 hand-crafted detectors. |
| N60 | TFT (Temporal Fusion Transformer) | AI School | Google's model for interpretable multi-horizon time series. Variable selection + attention + quantile outputs. |
| N61 | GAN for synthetic market data | AI School | Generate realistic OHLCV data for stress-testing strategies in extreme regimes. Train on crash periods, generate "what-if" scenarios. |

### 7.2 LLM Deployment

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N62 | FinBERT deployment | NLP M8 | Deploy `ProsusAI/finbert` locally. Financial-domain BERT fine-tuned on earnings calls + SEC filings. Produces sentiment with financial context awareness. |
| N63 | LLM-based earnings call Q&A analysis | NLP M8 | Feed earnings call transcripts to LLM. Extract: (1) analyst skepticism level, (2) management defensiveness, (3) guidance language shifts. |
| N64 | LLM news summarization → signal extraction | NLP M8 | Multiple news articles about same ticker → LLM synthesizes key themes → extracted structured signal (not just sentiment, but REASONS). |
| N65 | LLM causal reasoning chain | AI School | "Fed raised rates → tech valuations compress → growth stocks underperform" — LLM generates causal chain, each link testable as a trading hypothesis. |

### 7.3 Multi-Source Fusion

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N66 | Cross-modal signal fusion | NLP M10 | Combine: News sentiment + Social media volume + SEC filing tone + Price momentum → unified alpha score. Weighted by each source's historical predictive power. |
| N67 | Attention-based news weighting | NLP M8 | Multiple news items about same ticker → attention mechanism weights by relevance (not equal weight). Revenue guidance > product launch > executive hire. |
| N68 | Foundation model ensemble (NLP + Time Series) | AI School | Combine Chronos-2 (time series forecast) + FinBERT (sentiment) into one signal. Only trade when both agree on direction. |

---

## Category 8: Alternative Data Sources (Multi-Factor School Stage 2)

> The data evolution framework identifies alternative data as the bridge between traditional quant and NLP.

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N69 | Satellite imagery (retail parking lots) | Multi-Factor Stage 2 | Count cars in Walmart/Target parking lots → predict quarterly revenue before earnings. Requires satellite API. |
| N70 | Credit card transaction data | Multi-Factor Stage 2 | Aggregated/anonymized consumer spending data → real-time revenue proxy. |
| N71 | Supply chain / shipping data | Multi-Factor Stage 2 | Port traffic, container shipping rates, supplier delivery times → economic activity indicators. |
| N72 | Job posting data | Multi-Factor Stage 2 | Company job posting volume and role types → growth/hiring signal. |
| N73 | Google Trends / search volume | Multi-Factor Stage 2 | Search volume for ticker or product → retail attention indicator. |

---

## Category 9: Strategy Architecture Improvements (Cross-School)

> Structural improvements to how strategies are organized and combined.

### 9.1 Regime-Based Strategy Routing

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N74 | Dedicated strategy-per-regime allocation | Trend + MR | Trending → 100% trend-following weight. Ranging → 50% mean-reversion + 50% stat-arb. Volatile → 70% trend + 30% cash. Transition → hold. |
| N75 | Strategy correlation matrix | All schools | Compute rolling correlation between strategies. When trend + mean-reversion both profitable → normal. When both losing simultaneously → regime shift → reduce exposure. |

### 9.2 Short-Side & Directional Flexibility

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N76 | Short-selling capability | Trend Following | Enable short entries on bearish patterns (inverse ETFs if backtesting.py doesn't support shorts directly). Crisis Alpha requires short-side during crashes. |
| N77 | Crypto 24/7 trading | Trend Following | Deploy same pattern detectors on crypto pairs. No market hours constraints, different volatility profile. Requires CCXT integration (T10c-1 planned). |

### 9.3 Position Sizing by Strategy Type

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N78 | Strategy-type-specific position sizing | All schools | Trend-following: Kelly based on win rate + avg win/avg loss. Mean-reversion: half-Kelly (more fragile). Stat-arb: full Kelly (highest confidence). |
| N79 | Volatility-targeted position sizing | Multi-Factor | Scale position size inversely to forecast volatility. Target portfolio vol of 15% annualized. Standard in institutional quant. |

### 9.4 Multi-Asset Portfolio

| # | Idea | Source | Description |
|---|------|--------|-------------|
| N80 | Correlation-aware multi-asset allocation | All schools | When running strategies on N instruments simultaneously, size positions based on correlation matrix. Two highly correlated signals = split allocation, don't double-bet. |
| N81 | Cross-asset hedge layer | All schools | When equity trend-following is heavily long, check bond/crypto correlation. If equities + bonds diverge, add bond hedge to reduce portfolio beta. |

---

## Summary: Highest-Impact Items (Top 10)

| Rank | ID | Idea | Category | Complexity | Impact | Why |
|------|----|------|----------|-----------|--------|-----|
| 1 | N4-N5 | L&M Financial Dictionary scorer | NLP Pipeline | LOW | HIGH | Zero API cost. Immediately replaces synthetic sentiment. Academic standard. ~200 lines. |
| 2 | N12-N13 | News API + entity mapping | News Analysis | MEDIUM | HIGH | Real alpha from news events. Several free tiers available. Enables everything in Category 2. |
| 3 | N44-N48 | Dedicated Mean Reversion engine | Mean Reversion | MEDIUM | HIGH | Indicators already exist. Just needs routing + stop logic. Covers the "ranging market" blind spot. |
| 4 | N62 | FinBERT deployment | AI/LLM | MEDIUM | HIGH | HuggingFace model, pre-trained, drop-in. Financial-domain BERT > general sentiment. |
| 5 | N32 | Sentiment lead/lag analysis | Social Media | MEDIUM | HIGH | Critical validation: does sentiment PREDICT or just REFLECT price? Gate before any social media investment. |
| 6 | N35-N39 | Fundamental factors (Value, Quality, etc.) | Multi-Factor | MEDIUM | HIGH | New feature dimension. CatBoost already handles high-dim input. |
| 7 | N40-N41 | Multi-factor scoring + market-neutral | Multi-Factor | LARGE | HIGH | Institutional quant standard. Unlocks multi-ticker portfolio. Requires short capability. |
| 8 | N20 | NLP+financial feature fusion model | News Analysis | MEDIUM | HIGH | Combine what already works (price features) with NLP. Unified model. |
| 9 | N9-N10 | TF-IDF + Word2Vec embeddings | NLP Pipeline | MEDIUM | MEDIUM | Unlocks narrative drift detection and emerging keyword identification. Foundation for all advanced NLP. |
| 10 | N51-N53 | Pairs trading engine | Stat Arb | LARGE | MEDIUM | New alpha source orthogonal to trend-following. Market-neutral by construction. Requires cointegration infrastructure. |

---

## Cross-Reference: What These Map To

| Quant School | Related Ideas | Current Project Coverage |
|-------------|--------------|------------------------|
| **Trend Following** | N76, N77 (short-side, crypto) | **85%** — Home territory |
| **Mean Reversion** | N44-N50 (dedicated engine) | **40%** — Indicators exist, no system |
| **Multi-Factor** | N35-N43, N79, N80 | **30%** — Technical factors only |
| **AI Intelligence** | N57-N68 (deep learning + LLM) | **35%** — Classical ML only |
| **Statistical Arb** | N51-N56 | **10%** — Cross-asset awareness only |
| **NLP Pipeline** | N1-N34 (all NLP) | **5%** — Placeholder modifier only |

---

## Important Notes

1. **GPU constraint:** Items N57-N61 (deep learning) are gated on GPU availability, same as Phase 05.
2. **Short-selling constraint:** Items N41, N76 require short capability. `backtesting.py` supports short positions with negative size but rules-first strategy currently only goes long.
3. **Data cost:** Items N69-N73 (alternative data) typically require paid data subscriptions. N12 (news API) has free tiers for development.
4. **These are NOT plans.** They are a catalog of opportunities. Any item promoted to implementation should get its own plan file with tasks, file paths, and success criteria following the existing plan format in `progress_docs/plans/`.
