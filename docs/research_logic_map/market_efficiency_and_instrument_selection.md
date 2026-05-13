# Market Efficiency & Instrument Selection: Where to Find Alpha

**Created:** 2026-05-08
**Status:** Research Note (discussion phase)
**Related Papers:** Bartram & Grinblatt (2019), Damodaran (Market Efficiency), McLean & Pontiff (2016), Falck et al. (2021), DeMiguel et al. (2024), Dickerson et al. (2024)
**Tags:** `Market.Efficiency` `Alpha.Decay` `Instrument.Selection` `ML.Training` `Portfolio`

---

## Core Thesis

> The more a financial instrument is traded and analyzed, the harder it is to consistently beat the market on it. Alpha is inversely proportional to arbitrage attention.

This is formalized in Damodaran's three propositions of market efficiency:

- **Proposition 1**: The probability of finding an inefficiency in an asset market **decreases** as the ease of trading on the asset increases.
- **Proposition 2**: The probability of finding an inefficiency **increases** as the transaction and information cost of exploiting it increases.
- **Proposition 3**: The speed at which an inefficiency is resolved is directly related to how easily the strategy can be replicated.

---

## The Alpha Decay Mechanism

When a predictive signal (anomaly, factor, pattern) is discovered:

1. **Pre-discovery**: Few know about it. Alpha is high.
2. **Publication/dissemination**: More arbitrageurs learn of it. Capital flows in.
3. **Competition phase**: Multiple parties trade on the same signal. Alpha compresses.
4. **Decay equilibrium**: Alpha asymptotically approaches zero (or transaction cost threshold).

**Quantified decay rates (from literature):**
- Falck et al. (2021): Published anomalies lose ~5 percentage points of Sharpe **per year** post-publication.
- **Year of publication alone** explains 30% of the variance in Sharpe decay across factors.
- McLean & Pontiff (2016): Post-publication anomaly returns decline by ~35% on average.
- Alpha decay is generally **non-stationary** — asset pricing tests that impose stationarity produce biased inference.

---

## The Efficiency Hierarchy: Which Instruments Are "Beatable"

| Factor | More Efficient (Harder to Beat) | Less Efficient (Easier to Beat) |
|---|---|---|
| **Liquidity** | SPY, mega-cap tech, EUR/USD | Small/mid-caps, niche commodities |
| **Analyst coverage** | AAPL, MSFT (50+ analysts) | Stocks with ≤3 analysts |
| **Geography** | US large-caps | Emerging markets, Asia-Pacific |
| **Asset class** | US equities, major FX | Corporate bonds, CEFs, options, niche futures |
| **Market cap** | $100B+ mega-caps | $500M-$5B small-caps |
| **Institutional ownership** | Heavily owned (>80%) | Low institutional ownership (<30%) |
| **Information availability** | High-frequency, real-time, widely disseminated | Delayed, fragmented, over-the-counter |
| **Derivative complexity** | Plain vanilla ETFs | Volatility products, structured notes, CEFs |

### Key Research Findings

1. **Bartram & Grinblatt (2019) — "Global Market Inefficiencies"**
   - Studied 25,000+ firms across 36 countries, 1993-2016
   - Alpha is **40-70 bps/month higher** in emerging markets vs. developed markets
   - A country's pre-transaction-cost alpha is **positively correlated with its trading costs**
   - Trading costs deter arbitrageurs → alpha persists

2. **DeMiguel et al. (2024) — ML on Trades & Holdings**
   - ML return predictability is **stronger for smaller/illiquid stocks**
   - Stronger for stocks with **lower analyst coverage**
   - Stronger for stocks with **higher idiosyncratic volatility**
   - "Nonlinear interactions between trades and holdings reveals valuable information for price discovery"

3. **Dickerson et al. (2024) — "Factor Investing with Delays"**
   - In infrequently traded corporate bonds, ML strategies beat the market **before costs**
   - Alpha **disappears** after accounting for transaction delays (the double-edged sword of illiquidity)
   - Key lesson: signal exists, but execution capacity is the binding constraint

4. **Bryzgalova, Pavlova & Sikorskaya (2025) — Option Arbitrage**
   - Only 57% of profitable option arbitrage opportunities attract ANY arbitrageur
   - ~50% of those that do are exploited by **only one** arbitrageur
   - Even in highly visible markets, arbitrage is far from complete

5. **Lassance & Martin-Utrera (2024) — "Does the Factor Zoo Pay Off?"**
   - Optimal arbitrage portfolio still yields efficiency gains after controlling for costs + constraints
   - **But**: proliferation of new anomalies since the 1980s has **not** translated into additional mean-variance benefits
   - Low-beta assets are a key driver of remaining exploitable mispricing

---

## Screening Criteria: How to Find Beatable Instruments

### Quantitative Filter

```python
# Market inefficiency score (higher = more beatable)
inefficiency_score = (
    -0.40 * zscore(log(dollar_volume))          # Lower volume → more inefficiency
    -0.30 * zscore(analyst_coverage)             # Fewer analysts → more inefficiency
    -0.20 * zscore(institutional_ownership_pct)  # Lower ownership → more inefficiency
    +0.10 * zscore(amihud_illiquidity)           # Higher illiquidity → more inefficiency
)
```

### Five Specific Screening Methods

1. **Liquidity Friction**: Amihud illiquidity = |daily return| / dollar volume. Higher → stickier mispricing.
2. **Analyst Coverage Deficit**: Stocks with ≤3 analysts. Hong, Lim & Stein (2000) + confirmed by DeMiguel (2024).
3. **Institutional Ownership Gap**: Low institutional ownership → less arbitrage capital watching → slower price correction.
4. **Information Asymmetry Proxies**: Wide bid-ask spread, high idiosyncratic volatility, high Kyle's lambda (price impact).
5. **Cross-Asset Fragmentation**: Markets where information is fragmented (corporate bonds trade OTC, CEFs have persistent NAV discounts).

### The Sweet Spot

Instruments that are:
- **Liquid enough to trade** (can enter/exit without destroying alpha)
- **Inefficient enough** that alpha persists (not fully arbitraged)
- **Information-sparse enough** that ML can find nonlinear patterns others miss

Best candidates: mid-cap equities ($2B-$10B market cap), sector ETFs with high dispersion, moderately traded commodity futures spreads, closed-end funds, emerging market ETFs.

---

## Mathematical Methods to Detect Inefficient Instruments

### Method 1: Alpha Decay Rate Estimation

For each instrument, measure how quickly a predictive signal loses power:

1. Train model on t-24m to t-12m (in-sample)
2. Measure OOS performance on t-12m to t-6m (recent OOS)
3. Compare to performance on t-6m to t (current)
4. Instruments with **slower decay** = less arbitraged = better candidates

### Method 2: Cross-Sectional Signal Dispersion

In efficient markets, all stocks within a sector should have similar predictability. If some stocks show much higher predictability than peers (controlling for risk), those are the inefficiency candidates.

### Method 3: Granger Causality from Macro to Micro

Test whether macro variables (rates, VIX, commodities) predict individual asset returns. Assets where macro **does** predict returns → information not yet fully incorporated → less efficient.

### Method 4: Market Efficiency Spectrum Ranking

Rank your entire universe by efficiency proxies. Target the inefficient tail (bottom 20-30%) for alpha strategies, use the efficient head (top 20%) for beta/hedging.

---

## ML Training Strategy: Niche vs. Mainstream Instruments

| Training Approach | What It Does | Best Use Case |
|---|---|---|
| **Train on liquid only** (SPY, QQQ, AAPL) | Learns features that work in highly competitive, noise-dominated environments | Robustness, generalizable feature engineering, regime detection |
| **Train on illiquid only** (small-caps, niche) | Higher signal-to-noise ratio; captures structural inefficiencies | Alpha discovery, pattern identification |
| **Cross-asset transfer learning** | Train on niche → fine-tune on liquid; or use cross-asset features | Best of both: alpha patterns from niche, execution capacity from liquid |
| **Universal multi-asset training** | Train on thousands of diverse assets with different liquidity regimes | TradeFM (2025) showed this generalizes best; learns liquidity-invariant representations |

### Recommendation

1. **Don't abandon liquid instruments** — they're essential for execution capacity and regime detection
2. **Expand to the inefficient frontier** — add mid-caps, emerging markets, niche sector ETFs
3. **Use cross-asset features** — condition equity predictions on rates (TLT), commodities (GLD, USO), and volatility (VIX)
4. **Train on diversity** — TradeFM (2025) demonstrated that training on thousands of assets across sectors and liquidity regimes produces models that generalize better than single-asset models

---

## Assessment: Our Current Instrument Universe

### What We Have (Active Trading Universe)

| Instrument | Efficiency Level | Alpha Difficulty |
|---|---|---|
| **SPY** | World's most efficient ETF | Extremely hard |
| **QQQ** | Ultra-liquid, heavily arbitraged | Very hard |
| **AAPL, MSFT, NVDA, GOOGL, AMZN, META** | Most analyzed stocks on Earth | Extremely hard |
| **BTC-USD, ETH-USD** | Most traded cryptocurrencies | Hard (24/7 arbitrage) |
| **EUR/USD** | Most liquid FX pair | Extremely hard |
| **GLD, TLT, IEF** | Heavily traded macro ETFs | Moderately hard |
| **Sector ETFs (XLF, XLE, etc.)** | Diversified, institutional flows | Moderate |
| **IWM (Russell 2000)** | Our least efficient instrument | Moderate-Hard |

**Verdict**: We are exclusively training on the most efficient, most arbitraged instruments in existence. This is "hard mode" — it doesn't mean ML can't work (cross-asset feature engineering and regime detection can still find edge), but we're competing against the most sophisticated players on the most crowded field.

### What We Should Add

| Category | Suggested Instruments | Rationale |
|---|---|---|
| **Mid-cap low-coverage stocks** | Russell 2000 components with <5 analysts | DeMiguel (2024) confirms stronger ML predictability |
| **Closed-end funds** | Various CEFs | Persistent NAV discounts that mean-revert — a proven, persistent inefficiency |
| **Emerging market ETFs** | EEM, VWO, FM | 40-70 bps/month alpha premium documented (Bartram 2019) |
| **Niche sector ETFs** | XBI (biotech), SMH (semiconductors), URA (uranium), XRT (retail) | More idiosyncratic, less institutional crowding |
| **Commodity spread instruments** | Crack spreads, calendar spreads | Less crowded than outright futures |
| **Volatility products** | VIX futures, VXX | Complex payoff structures deter simple arbitrage |
| **Small-cap value** | AVUV, VBR | Factor zoo research shows value + small-cap still has residual alpha |
| **International small-caps** | SCHC, VSS | Less analyst coverage, less institutional ownership |

---

## Manual Instrument Selection: The Attribute Checklist

Use these criteria when manually filtering candidates. Apply hard filters first, then rank survivors on the scorecard.

---

### Hard Filters (Must Pass All)

| # | Attribute | Reject If | Rationale |
|---|-----------|-----------|-----------|
| F1 | **Average Daily Dollar Volume** | < $10M | Can't enter/exit without moving price. Below $10M means slippage will eat alpha. |
| F2 | **Market Cap** | < $300M | Micro-caps have survivorship bias, data quality issues, and extreme illiquidity. |
| F3 | **Price** | < $5 (stocks only) | Penny stocks have structural issues (low institutional ownership is extreme, not informational — it's toxicity). |
| F4 | **Price** | < $10 (ETFs/crypto/futures) | Same logic; too cheap = too much noise. |
| F5 | **Data History** | < 3 years of daily data | ML needs enough observations. Less than ~750 bars is insufficient for training. |
| F6 | **Listing** | OTC / pink sheets | Reporting requirements are lax; data quality is unreliable. Stick to NYSE, NASDAQ, major exchanges. |
| F7 | **ETF AUM** | < $100M (ETFs only) | Small ETFs risk closure/liquidation. Also have wide spreads. |
| F8 | **Leveraged / Inverse Products** | Yes, unless specifically desired | These decay over time due to volatility drag. Their price series is not stationary and confuses ML. |

---

### Desirability Scorecard (Rank Survivors)

Score each surviving instrument. Higher total score = better candidate.

#### Liquidity (lower = more inefficiency, but need minimum viability)

| Score | ADDV (daily $) | Interpretation |
|-------|----------------|----------------|
| **+3** | $10M – $50M | Thinly traded — alpha persists, but position sizing must be careful |
| **+2** | $50M – $200M | Moderate liquidity — good balance of tradability and inefficiency |
| **+1** | $200M – $1B | Amplitude liquid — starting to get competitive |
| **0** | > $1B | Very liquid — alpha decays fast; efficient market territory |

#### Market Cap (lower = less analyst attention)

| Score | Market Cap | Interpretation |
|-------|------------|----------------|
| **+3** | $500M – $5B | Small-cap sweet spot. Low coverage, high ML edge (DeMiguel 2024). |
| **+2** | $5B – $20B | Mid-cap. Moderate coverage. Still inefficiency. |
| **+1** | $20B – $100B | Large-cap. Significant coverage, but some residual alpha. |
| **0** | > $100B | Mega-cap. Extremely efficient. SPY/AAPL territory. |
| **-1** | < $500M | Growing but risky — borderline data quality. |

#### Analyst Coverage (critical alpha predictor — **stocks only**)

DeMiguel et al. (2024) explicitly found that ML return predictability is **stronger** for stocks with lower analyst coverage. This is a direct measure of how much "smart attention" an instrument receives. Fewer analysts → less information competition → more room for ML to find unpriced patterns.

**Important**: This dimension applies to individual stocks. ETFs, crypto, forex, and futures do not have analyst coverage in the traditional sense — skip this dimension for those and use institutional ownership + sector bonuses instead.

| Score | # Analysts | Interpretation |
|-------|------------|----------------|
| **+4** | 0 | Zero coverage. Maximum neglect. Nobody is modeling this stock. Every pattern you find is yours alone. **But**: only score here if the instrument passes ALL hard filters — a stock with zero analysts AND low volume is a red flag, not a signal. Zero-analyst stocks that clear the ADDV + market cap + data history gates are the rarest and best candidates. |
| **+3** | 1 – 5 | Very low coverage. DeMiguel (2024) confirms this is where ML shines hardest. |
| **+2** | 6 – 10 | Moderate coverage. Information gaps exist. |
| **+1** | 11 – 15 | Above-average coverage. Less edge, but not zero. |
| **0** | 16 – 25 | Well-covered. Most mispricing is arbed away. |
| **-1** | > 25 | Hyper-covered (AAPL, MSFT, GOOGL territory). Every quarterly decimal, every product launch, every supply chain rumor is modeled by dozens of analysts. ML competing here is like playing poker against 25 opponents who all see your cards. |

**Edge case: 0 analysts + weak fundamentals.** If a stock has 0 analysts AND fails any hard filter (ADDV, market cap, price), do not score it. It's not "undiscovered" — it's uninvestable. The 0-analyst bonus only applies to stocks that clear all hard filters.

#### Institutional Ownership (lower = less arbitrage capital)

| Score | Institutional % | Interpretation |
|-------|-----------------|----------------|
| **+3** | 15% – 40% | Low ownership. Few arbitrageurs watching. |
| **+2** | 40% – 60% | Moderate. |
| **+1** | 60% – 80% | High. Lots of smart money competing. |
| **0** | > 80% | Dominated by institutions. Very efficient. |

#### Bid-Ask Spread (wider = more asymmetry, but too wide kills execution)

| Score | Spread % | Interpretation |
|-------|----------|----------------|
| **+3** | 0.15% – 0.30% | Moderate spread. Inefficiency signal without cost killing you. |
| **+2** | 0.30% – 0.50% | Wider. Good for alpha, need to account in cost model. |
| **+1** | 0.05% – 0.15% | Tight. Liquid but also efficient. |
| **0** | < 0.05% | Ultra-tight. Too efficient. |
| **-1** | > 0.50% | Too wide. Transaction costs will dominate any alpha. |

#### Sector / Category Bonus

Different sectors attract different amounts of arbitrage attention. This bonus adjusts for structural inefficiency baked into the instrument's category — independent of the liquidity/size scores above.

| Bonus | Condition | Rationale |
|-------|-----------|-----------|
| **+2** | Biotech (XBI, IBB), clean energy (ICLN, TAN), uranium (URA), cannabis (MSOS), emerging fintech (FINX, ARKF) | These sectors are defined by **binary event risk** (FDA approvals, regulatory rulings, clinical trial results). The outcomes are fundamentally unpredictable from price data alone, which means (a) price often fails to fully incorporate probabilistic outcomes, and (b) institutional models that rely on fundamentals struggle to price them. ML can capture the pre-event drift and post-event reaction patterns that simple models miss. Additionally, these sectors have high **dispersion** — constituent stocks move independently rather than as a bloc — which means there are always relative mispricings to exploit. |
| **+1** | Materials (XLB), energy (XLE), industrials (XLI), real estate (XLRE) | Cyclical sectors where mispricing is **regime-dependent**. Commodity cycles, rate cycles, and economic cycles create predictable rotation patterns that repeat across decades. The patterns are durable because they're driven by macro fundamentals that don't change quickly, but they're hard to arbitrage because timing the cycle is a multi-quarter bet most funds can't hold. ML trained on cross-asset features (commodity prices, rates, yield curve) can capture these rotations. |
| **+1** | Emerging market (EEM, VWO, FM, any country-specific ETF) | Bartram & Grinblatt (2019): alpha is **40-70 bps/month higher** in emerging vs. developed markets. Lower institutional ownership, fewer analysts per stock, higher trading costs that deter foreign arbitrageurs, less efficient information dissemination (local news takes longer to reach global markets). Caveat: ensure the ETF itself has sufficient ADDV — some country ETFs are surprisingly illiquid despite tracking liquid underlying stocks. |
| **0** | Technology (XLK), financials (XLF), consumer staples (XLP), healthcare (XLV) | Heavily covered sectors. Every major fund has dedicated sector teams. Alpha exists but requires superior information — not something ML on price data alone can reliably provide. |
| **-1** | Broad market index ETF (SPY, QQQ, IWM, DIA) | These are the **most efficient instruments on Earth**. They are the primary vehicle for global macro allocation, the benchmark for every fund, and the subject of more academic research than any other security. If you find an edge on SPY via pure price patterns, assume it's overfitting until proven otherwise across multiple OOS periods. That said, cross-asset features (H14) can still work here because they inject information that isn't in SPY's own price history. |

**How to score an ETF that sits between categories**: Use the primary sector. For example, SMH (semiconductor ETF) is Technology (+0), despite being concentrated — it's still tech. SOXX similarly. XME (metals & mining) is Materials (+1). XOP (oil & gas exploration) is Energy (+1).

#### Volatility Profile

Volatility drives the magnitude of potential mispricing. An instrument that moves 2% per day has 10x the daily opportunity of one that moves 0.2% — but also 10x the noise. The sweet spot is high enough that moves are meaningful relative to transaction costs, but not so high that noise dominates signal.

| Score | Annualized Vol | Interpretation |
|-------|----------------|----------------|
| **+2** | 35% – 60% | High dispersion. Large daily moves create **pricing gaps** that fast arbitrageurs can't fully close intraday. The magnitude of potential alpha relative to fixed transaction costs (spread, commissions) is favorable. Typical of biotech, crypto, small-cap growth, emerging markets. |
| **+1** | 25% – 35% | Above-average. Enough price movement that a modest edge compounds. Typical of mid-cap stocks, sector ETFs, commodities. |
| **0** | 15% – 25% | Average. SPY territory. The edge needs to be proportionally larger to overcome costs. |
| **-1** | < 15% | Low vol. Utility stocks, consumer staples, bonds, TLT. **Not a hard reject** — just means the strategy must be lower-turnover and higher-conviction, because each trade's profit margin is thin relative to costs. Pair trading and mean-reversion strategies actually prefer lower vol. |

**Important**: Volatility is not a standalone score. A +2 vol score on a broad market ETF (-1 sector) still nets to near-zero. The dimensions compound: high vol (+2) + ignored sector (+2) + low analyst coverage (+4) + thin liquidity (+3) = an instrument where ML has a genuine structural advantage over the market.

---

### Scorecard Interpretation

Maximum possible score: 20+ (varies by whether analyst coverage applies).

| Total Score | Verdict | Action |
|-------------|---------|--------|
| **14 – 20** | Excellent candidate | Priority for data acquisition and ML training |
| **10 – 13** | Good candidate | Add to active universe |
| **6 – 9** | Acceptable | Add for diversification, not primary alpha source |
| **3 – 5** | Marginal | Beta/hedging only |
| **< 3** | Avoid | Too efficient or structurally compromised |

---

### Worked Examples

**SPY (current dominant instrument):**
| Filter | Value | Score |
|--------|-------|-------|
| ADDV | ~$30B | 0 |
| Market Cap | N/A (ETF) | N/A (use AUM ≈ $600B) |
| Analyst Coverage | N/A (ETF) | N/A |
| Institutional Ownership | ~80% | +1 |
| Spread | ~0.01% | 0 |
| Sector | Broad market | -1 |
| Volatility | ~18% | 0 |
| **Total** | | **0 / N/A** |

SPY scores 0 — it's the definition of "too efficient." ML might work via cross-asset features (H14), but pure price-based alpha is unlikely to persist.

**IWM (Russell 2000, already in our universe):**
| Filter | Value | Score |
|--------|-------|-------|
| ADDV | ~$5B | 0 |
| AUM | ~$70B | — |
| Spread | ~0.02% | +1 |
| Sector | Broad market | -1 |
| Volatility | ~22% | 0 |
| Underlying avg market cap | ~$3B | +3 (by proxy) |
| Underlying avg analyst coverage | ~8 | +2 (by proxy) |
| **Total (by proxy)** | | **~5** |

IWM is acceptable but its broad-market nature holds it back. Better: a small-cap value ETF like AVUV would score higher.

**XBI (Biotech ETF — suggested addition):**
| Filter | Value | Score |
|--------|-------|-------|
| ADDV | ~$800M | +1 |
| AUM | ~$7B | — |
| Spread | ~0.05% | +1 |
| Sector | Biotech (high idiosyncratic) | +2 |
| Volatility | ~35% | +2 |
| Underlying avg market cap | ~$8B | +2 (by proxy) |
| Underlying avg analyst coverage | ~10 | +2 (by proxy) |
| **Total** | | **~10** |

XBI is a strong candidate — high volatility, high dispersion, niche sector.

**SMH (Semiconductor ETF — suggested addition):**
| Filter | Value | Score |
|--------|-------|-------|
| ADDV | ~$1.5B | +1 |
| AUM | ~$25B | — |
| Spread | ~0.03% | +1 |
| Sector | Technology (but concentrated) | 0 |
| Volatility | ~38% | +2 |
| Underlying concentration | Top 3 = 40% (NVDA, TSM, AVGO) | -1 (concentration reduces diversification benefit) |
| **Total** | | **~3** |

SMH is marginal — it's tech-concentrated and essentially a leveraged bet on NVDA/TSM.

---

### Data Sources for Attributes

| Attribute | Source | Method |
|-----------|--------|--------|
| ADDV, Market Cap, Price, Spread | yfinance | `.info` dict or `ticker.history()` |
| Analyst Coverage | yfinance | `.info['numberOfAnalystOpinions']` |
| Institutional Ownership % | yfinance | `.info['heldPercentInstitutions']` |
| Sector / Industry | yfinance | `.info['sector']`, `.info['industry']` |
| ETF AUM | yfinance | `.info['totalAssets']` |
| Bid-Ask Spread | yfinance | `(ask - bid) / mid` from recent quotes (or estimate from `.info`) |
| Historical Volatility | Calculate | `df['Close'].pct_change().std() * sqrt(252)` |
| Free Float | yfinance | `.info['floatShares']` |

---

## Key Takeaways

1. **Our core thesis is correct**: More traded = harder to beat. This is mathematically formalized and empirically validated across multiple studies.
2. **Alpha decay is the dominant dynamic**: Every year a signal is known, it loses ~5% of its Sharpe. This applies to our chart patterns too.
3. **Our current universe is too narrow and too efficient**: We're only playing on the hardest difficulty. We need to diversify to less efficient instruments.
4. **The sweet spot is the "neglected middle"**: Liquid enough to trade, inefficient enough to have persistent alpha, complex enough that simple arbitrage can't fully price it.
5. **Cross-asset features are force multipliers**: Even on efficient instruments like SPY, conditioning on macro/commodity/crypto cross-asset signals can find edge that pure price-based features miss.

---

## Sources

- **Bartram & Grinblatt** (2019). "Global Market Inefficiencies." SSRN-3518570.
- **Damodaran, A.** "Market Efficiency — Definition, Tests, and Evidence." NYU Stern.
- **DeMiguel, Sang, Zhang** (2024). "Do Trades and Holdings of Market Participants Contain Information About Stocks?" SSRN-5071465.
- **Dickerson, Robotti, Nozawa** (2024). "Factor Investing with Delays." SSRN-5074221.
- **Falck, Rej, Thesmar** (2021). "When Systematic Strategies Decay." SSRN-3845928.
- **McLean & Pontiff** (2016). "Does Academic Research Destroy Stock Return Predictability?" Journal of Finance.
- **Bai** (2023). "Can the changes in fundamentals explain the attenuation of anomalies?" J. Financial Economics.
- **Lassance & Martin-Utrera** (2024). "Does the Factor Zoo Pay Off?" SSRN-4760599.
- **Bryzgalova, Pavlova, Sikorskaya** (2025). "Arbitrage Capital and Limits to Arbitrage."
- **TradeFM** (2025). arXiv:2602.23784 — Multi-asset training with scale-invariant features.
- **Di Mascio, Lines, Naik** (2017). "Alpha Decay and Institutional Trading." SSRN-2580551.

---

*This document is a living research note. Update as new instruments are added, new backtests are run, or new efficiency metrics are computed.*
