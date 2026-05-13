# Ticker Test Log

> All tickers tested for basket inclusion, with criteria thresholds and results.

**Model**: `models/pattern_classifier_v3_SPY_20260511_145927.pkl` (43 features, 12-all basket trained)
**Walk-forward params**: initial_train=3×252d, step=6×21d, horizon=5d
**IC threshold**: mean rank IC > 0.03 = PASS
**Last updated**: 2026-05-11

---

## Criteria Thresholds Used

| Criterion | v1 (strict) | v2 (relaxed) |
|-----------|-------------|---------------|
| Market cap | > $500M | > $200M |
| Daily notional volume | > $10M | > $5M |
| Institutional ownership | > 40% | > 25% |
| Data history | > 8 years | > 5 years |
| Annualized vol | 15-50% | 12-55% |
| Sector decorrelation | r < 0.50 vs SPY/QQQ/XLK | r < 0.70 |
| Exchange | NYSE/NASDAQ | NYSE/NASDAQ |
| Bid-ask spread | < 0.15% | < 0.25% |

---

## Test Log

### Batch 1 — User Suggested (v1 criteria)

| Date | Ticker | Name | Sector | MC | Vol $M/d | Inst% | History | AnnVol% | IC | Verdict | Notes |
|------|--------|------|--------|----|---------|-------|---------|---------|----|---------|-------|
| 2026-05-11 | REPL | Replimune Group | Healthcare | $336M | $27M | 99% | 8yr | ~40% | — | REJECT | MC < $500M, beta 0.12 |
| 2026-05-11 | CAR | Avis Budget Group | Industrials | $5.1B | $230M | 143% | 16yr | ~50% | +0.005 | REJECT | IC < 0.03 |

### Batch 2 — Blue-Chip Non-Tech S&P 500 (v1 criteria)

| Date | Ticker | Name | Sector | MC | Vol $M/d | Inst% | History | AnnVol% | IC | Verdict | Notes |
|------|--------|------|--------|----|---------|-------|---------|---------|----|---------|-------|
| 2026-05-11 | LMT | Lockheed Martin | Defense | $117B | $810M | 77% | 16yr | 21.5% | +0.009 | REJECT | |
| 2026-05-11 | COST | Costco | Consumer Def | $448B | $1859M | 75% | 16yr | 20.3% | +0.001 | REJECT | |
| 2026-05-11 | HD | Home Depot | Consumer Cyc | $316B | $1293M | 77% | 16yr | 23.3% | +0.024 | REJECT | Borderline |
| 2026-05-11 | FCX | Freeport-McMoRan | Materials | $89B | $1074M | 91% | 16yr | 49.5% | -0.045 | REJECT | Negative |
| 2026-05-11 | CAT | Caterpillar | Industrials | $413B | $2342M | 76% | 16yr | 29.5% | +0.004 | REJECT | |
| 2026-05-11 | JPM | JPMorgan Chase | Financials | $810B | $2904M | 78% | 16yr | 27.5% | -0.003 | REJECT | Negative |

### Batch 3 — Also Checked (Fundamental Only, No IC Test)

| Date | Ticker | Name | Sector | MC | Vol $M/d | Inst% | History | AnnVol% | IC | Verdict | Notes |
|------|--------|------|--------|----|---------|-------|---------|---------|----|---------|-------|
| 2026-05-11 | XOM | Exxon Mobil | Energy | $599B | $3142M | 71% | 16yr | 25.0% | — | — | Not tested (already have XLE) |
| 2026-05-11 | JNJ | Johnson & Johnson | Healthcare | $533B | $1792M | 76% | 16yr | 16.9% | — | — | Not tested (already have XLV) |
| 2026-05-11 | ABBV | AbbVie | Healthcare | $357B | $1396M | 80% | 13yr | 26.3% | — | — | Not tested (already have XLV) |
| 2026-05-11 | CVX | Chevron | Energy | $362B | $2223M | 70% | 16yr | 26.6% | — | — | Not tested (already have XLE) |
| 2026-05-11 | DE | Deere & Co | Industrials | $155B | $825M | 84% | 16yr | 28.3% | — | — | Not tested (similar to CAT) |
| 2026-05-11 | NEM | Newmont | Gold/Materials | $124B | $1116M | 86% | 16yr | 36.2% | — | — | Not tested (overlaps GLD) |
| 2026-05-11 | NOC | Northrop Grumman | Defense | $78B | $458M | 85% | 16yr | 23.4% | — | — | Not tested (similar to LMT) |
| 2026-05-11 | BSX | Boston Scientific | Medical Devices | $80B | $866M | 96% | 16yr | 28.7% | — | — | Not tested (already have XLV) |
| 2026-05-11 | MRK | Merck | Healthcare | $275B | $1090M | 84% | 16yr | 21.5% | — | — | Not tested (already have XLV) |

---

## Summary

| Batch | Tested | Passed | Pass Rate | Best IC |
|-------|--------|--------|-----------|---------|
| User suggested | 2 | 0 | 0% | CAR +0.005 |
| Blue-chip non-tech | 6 | 0 | 0% | HD +0.024 |
| Utilities | 9 | 4 | 44% | PEG +0.052 |
| Real Estate | 8 | 3 | 38% | AVB +0.050 |
| Transportation | 9 | 3 | 33% | NSC +0.047 |
| Metals/Mining | 6 | 1 | 17% | NEM +0.052 |
| Consumer Staples | 9 | 3 | 33% | KO +0.054 |
| Energy Singles | 10 | 2 | 20% | EOG +0.044 |
| Healthcare Singles | 10 | 3 | 30% | VRTX +0.071 |
| Midcap Niche | 6 | 3 | 50% | CRVL +0.066 |
| **Total** | **75** | **22** | **29%** | VRTX +0.071 |

**Key finding**: The model does NOT work on blue-chip individual stocks but DOES work on sector-specific names in under-represented sectors. The strongest passes come from:
1. **Healthcare biotech** (VRTX +0.071, BMY +0.054, REGN +0.036)
2. **Niche midcaps** (CRVL +0.066, HIFS +0.036, JOE +0.032)
3. **Utilities** (PEG +0.052, D +0.049, SO +0.047, SRE +0.045)
4. **Consumer staples** (KO +0.054, WMT +0.040, PEP +0.031)
5. **Real estate REITs** (AVB +0.050, PSA +0.033, SPG +0.030)
6. **Transportation railroads** (NSC +0.047, UNP +0.039, CSX +0.031)
7. **Energy E&P** (EOG +0.044, OXY +0.038)
8. **Gold mining** (NEM +0.052)
| 2026-05-11 | XLU | State Street Utilities Select S | N/A | $0.0B | $1141M | 0% | 16yr | 17.5% | � | FAIL (fundamental) | MC $0M < $200M; Inst 0% |
| 2026-05-11 | XLRE | State Street Real Estate Select | N/A | $0.0B | $358M | 0% | 11yr | 20.4% | � | FAIL (fundamental) | MC $0M < $200M; Inst 0% |
| 2026-05-11 | VNQ | Vanguard Real Estate ETF | N/A | $0.0B | $364M | 0% | 16yr | 20.4% | � | FAIL (fundamental) | MC $0M < $200M; Inst 0% |
| 2026-05-11 | XTN | State Street SPDR S&P Transport | N/A | $0.0B | $7M | 0% | 15yr | 24.7% | � | FAIL (fundamental) | MC $0M < $200M; Inst 0% |
| 2026-05-11 | CLF | Cleveland-Cliffs Inc. | Basic Materials | $6.3B | $202M | 87% | 16yr | 66.2% | � | FAIL (fundamental) | Vol 66.2% |
| 2026-05-11 | X | X | N/A | $0.0B | $0M | 0% | 0yr | 0.0% | � | FAIL (fundamental) | Only 0 bars |
| 2026-05-11 | SCCO | Southern Copper Corporation | Basic Materials | $153.1B | $303M | 10% | 16yr | 36.1% | � | FAIL (fundamental) | Inst 10% |
| 2026-05-11 | XLP | State Street Consumer Staples S | N/A | $0.0B | $1435M | 0% | 16yr | 13.7% | � | FAIL (fundamental) | MC $0M < $200M; Inst 0% |
| 2026-05-11 | PEG | Public Service Enterprise Group | Utilities | $38.4B | $209M | 84% | 16yr | 20.8% | +0.052 | PASS |  |
| 2026-05-11 | D | Dominion Energy, Inc. | Utilities | $54.4B | $311M | 89% | 16yr | 20.9% | +0.049 | PASS |  |
| 2026-05-11 | SO | Southern Company (The) | Utilities | $103.5B | $534M | 74% | 16yr | 19.1% | +0.047 | PASS |  |
| 2026-05-11 | SRE | DBA Sempra | Utilities | $59.8B | $319M | 101% | 16yr | 22.2% | +0.045 | PASS |  |
| 2026-05-11 | AEP | American Electric Power Company | Utilities | $70.8B | $433M | 88% | 16yr | 19.4% | +0.027 | FAIL |  |
| 2026-05-11 | ED | Consolidated Edison, Inc. | Utilities | $39.2B | $235M | 78% | 16yr | 18.8% | +0.015 | FAIL |  |
| 2026-05-11 | EXC | Exelon Corporation | Utilities | $44.9B | $394M | 98% | 16yr | 22.5% | +0.011 | FAIL |  |
| 2026-05-11 | DUK | Duke Energy Corporation (Holdin | Utilities | $96.8B | $507M | 73% | 16yr | 18.5% | +0.010 | FAIL |  |
| 2026-05-11 | NEE | NextEra Energy, Inc. | Utilities | $194.1B | $887M | 88% | 16yr | 22.4% | +0.009 | FAIL |  |
| 2026-05-11 | KODK | Eastman Kodak Company | Industrials | $1.1B | $14M | 38% | 13yr | 132.9% | � | FAIL (fundamental) | Vol 132.9% |
| 2026-05-11 | CAR | Avis Budget Group, Inc. | Industrials | $5.1B | $420M | 143% | 16yr | 68.9% | � | FAIL (fundamental) | Vol 68.9% |
| 2026-05-11 | REPL | Replimune Group, Inc. | Healthcare | $0.3B | $21M | 99% | 8yr | 103.4% | � | FAIL (fundamental) | Vol 103.4% |
| 2026-05-11 | NEM | Newmont Corporation | Basic Materials | $124.4B | $1116M | 86% | 16yr | 36.2% | +0.052 | PASS |  |
| 2026-05-11 | AEM | Agnico Eagle Mines Limited | Basic Materials | $96.6B | $502M | 74% | 16yr | 40.7% | +0.026 | FAIL |  |
| 2026-05-11 | GOLD | Gold.com, Inc. | Financial Services | $1.3B | $29M | 59% | 12yr | 45.0% | +0.010 | FAIL |  |
| 2026-05-11 | STLD | Steel Dynamics, Inc. | Basic Materials | $33.9B | $283M | 93% | 16yr | 37.9% | -0.013 | FAIL |  |
| 2026-05-11 | AA | Alcoa Corporation | Basic Materials | $16.7B | $369M | 87% | 16yr | 48.3% | -0.014 | FAIL |  |
| 2026-05-11 | FCX | Freeport-McMoRan, Inc. | Basic Materials | $88.6B | $1074M | 91% | 16yr | 49.5% | -0.045 | FAIL |  |
| 2026-05-11 | NUE | Nucor Corporation | Basic Materials | $51.8B | $352M | 84% | 16yr | 32.5% | -0.049 | FAIL |  |
| 2026-05-11 | AVB | AvalonBay Communities, Inc. | Real Estate | $25.6B | $189M | 94% | 16yr | 23.8% | +0.050 | PASS |  |
| 2026-05-11 | PSA | Public Storage | Real Estate | $54.6B | $374M | 91% | 16yr | 22.4% | +0.033 | PASS |  |
| 2026-05-11 | SPG | Simon Property Group, Inc. | Real Estate | $76.9B | $311M | 100% | 16yr | 32.0% | +0.030 | PASS |  |
| 2026-05-11 | AMT | American Tower Corporation (REI | Real Estate | $82.2B | $553M | 89% | 16yr | 24.3% | +0.028 | FAIL |  |
| 2026-05-11 | EQR | Equity Residential | Real Estate | $25.3B | $172M | 106% | 16yr | 24.3% | +0.001 | FAIL |  |
| 2026-05-11 | O | Realty Income Corporation | Real Estate | $57.7B | $382M | 89% | 16yr | 23.9% | -0.011 | FAIL |  |
| 2026-05-11 | PLD | Prologis, Inc. | Real Estate | $134.3B | $504M | 103% | 16yr | 27.2% | -0.013 | FAIL |  |
| 2026-05-11 | WELL | Welltower Inc. | Real Estate | $151.5B | $656M | 105% | 16yr | 28.3% | -0.030 | FAIL |  |
| 2026-05-11 | NSC | Norfolk Southern Corporation | Industrials | $70.1B | $409M | 80% | 16yr | 26.8% | +0.047 | PASS |  |
| 2026-05-11 | UNP | Union Pacific Corporation | Industrials | $157.1B | $807M | 91% | 16yr | 24.7% | +0.039 | PASS |  |
| 2026-05-11 | CSX | CSX Corporation | Industrials | $83.3B | $565M | 82% | 16yr | 27.4% | +0.031 | PASS |  |
| 2026-05-11 | UAL | United Airlines Holdings, Inc. | Industrials | $32.3B | $792M | 95% | 16yr | 48.9% | +0.004 | FAIL |  |
| 2026-05-11 | FDX | FedEx Corporation | Industrials | $90.3B | $696M | 83% | 16yr | 30.1% | +0.000 | FAIL |  |
| 2026-05-11 | UPS | United Parcel Service, Inc. | Industrials | $85.7B | $636M | 72% | 16yr | 24.2% | -0.018 | FAIL |  |
| 2026-05-11 | DAL | Delta Air Lines, Inc. | Industrials | $48.2B | $864M | 90% | 16yr | 41.1% | -0.029 | FAIL |  |
| 2026-05-11 | AAL | American Airlines Group, Inc. | Industrials | $8.8B | $892M | 77% | 16yr | 51.7% | -0.052 | FAIL |  |
| 2026-05-11 | LUV | Southwest Airlines Company | Industrials | $20.2B | $369M | 98% | 16yr | 34.4% | -0.052 | FAIL |  |
| 2026-05-11 | KO | Coca-Cola Company (The) | Consumer Defensive | $337.4B | $1234M | 69% | 16yr | 17.1% | +0.054 | PASS |  |
| 2026-05-11 | WMT | Walmart Inc. | Consumer Defensive | $1039.7B | $2627M | 40% | 16yr | 19.7% | +0.040 | PASS |  |
| 2026-05-11 | PEP | Pepsico, Inc. | Consumer Defensive | $211.4B | $982M | 82% | 16yr | 17.7% | +0.031 | PASS |  |
| 2026-05-11 | KMB | Kimberly-Clark Corporation | Consumer Defensive | $32.6B | $481M | 95% | 16yr | 18.8% | +0.026 | FAIL |  |
| 2026-05-11 | PM | Philip Morris International Inc | Consumer Defensive | $266.5B | $833M | 85% | 16yr | 21.9% | +0.026 | FAIL |  |
| 2026-05-11 | CL | Colgate-Palmolive Company | Consumer Defensive | $70.1B | $532M | 90% | 16yr | 18.3% | +0.011 | FAIL |  |
| 2026-05-11 | PG | Procter & Gamble Company (The) | Consumer Defensive | $341.0B | $1501M | 73% | 16yr | 17.3% | +0.010 | FAIL |  |
| 2026-05-11 | COST | Costco Wholesale Corporation | Consumer Defensive | $447.6B | $1859M | 75% | 16yr | 20.3% | +0.001 | FAIL |  |
| 2026-05-11 | MO | Altria Group, Inc. | Consumer Defensive | $113.8B | $644M | 66% | 16yr | 20.4% | -0.003 | FAIL |  |
| 2026-05-11 | CRVL | CorVel Corp. | Financial Services | $3.0B | $14M | 57% | 16yr | 33.3% | +0.066 | PASS |  |
| 2026-05-11 | HIFS | Hingham Institution for Savings | Financial Services | $0.6B | $14M | 67% | 16yr | 31.7% | +0.036 | PASS |  |
| 2026-05-11 | JOE | St. Joe Company (The) | Real Estate | $3.8B | $17M | 95% | 16yr | 33.8% | +0.032 | PASS |  |
| 2026-05-11 | HD | Home Depot, Inc. (The) | Consumer Cyclical | $316.2B | $1293M | 77% | 16yr | 23.3% | +0.024 | FAIL |  |
| 2026-05-11 | LMT | Lockheed Martin Corporation | Industrials | $116.8B | $810M | 77% | 16yr | 21.5% | +0.008 | FAIL |  |
| 2026-05-11 | CAT | Caterpillar, Inc. | Industrials | $413.4B | $2342M | 76% | 16yr | 29.5% | +0.004 | FAIL |  |
| 2026-05-11 | VRTX | Vertex Pharmaceuticals Incorpor | Healthcare | $109.1B | $569M | 99% | 16yr | 42.6% | +0.071 | PASS |  |
| 2026-05-11 | BMY | Bristol-Myers Squibb Company | Healthcare | $114.7B | $677M | 86% | 16yr | 23.7% | +0.054 | PASS |  |
| 2026-05-11 | REGN | Regeneron Pharmaceuticals, Inc. | Healthcare | $74.9B | $484M | 93% | 16yr | 38.0% | +0.036 | PASS |  |
| 2026-05-11 | ABBV | AbbVie Inc. | Healthcare | $356.5B | $1396M | 80% | 13yr | 26.3% | +0.029 | FAIL |  |
| 2026-05-11 | GILD | Gilead Sciences, Inc. | Healthcare | $163.1B | $825M | 95% | 16yr | 27.5% | +0.027 | FAIL |  |
| 2026-05-11 | MRK | Merck & Company, Inc. | Healthcare | $275.1B | $1090M | 84% | 16yr | 21.5% | +0.020 | FAIL |  |
| 2026-05-11 | AMGN | Amgen Inc. | Healthcare | $179.1B | $881M | 88% | 16yr | 24.5% | +0.013 | FAIL |  |
| 2026-05-11 | BIIB | Biogen Inc. | Healthcare | $28.6B | $214M | 102% | 16yr | 38.2% | +0.012 | FAIL |  |
| 2026-05-11 | PFE | Pfizer, Inc. | Healthcare | $146.4B | $985M | 72% | 16yr | 22.1% | +0.010 | FAIL |  |
| 2026-05-11 | JNJ | Johnson & Johnson | Healthcare | $532.8B | $1792M | 76% | 16yr | 16.9% | -0.007 | FAIL |  |
| 2026-05-11 | EOG | EOG Resources, Inc. | Energy | $69.3B | $667M | 99% | 16yr | 37.1% | +0.044 | PASS |  |
| 2026-05-11 | OXY | Occidental Petroleum Corporatio | Energy | $52.7B | $923M | 55% | 16yr | 41.8% | +0.038 | PASS |  |
| 2026-05-11 | CVX | Chevron Corporation | Energy | $361.7B | $2223M | 70% | 16yr | 26.6% | +0.017 | FAIL |  |
| 2026-05-11 | XOM | Exxon Mobil Corporation | Energy | $599.2B | $3142M | 71% | 16yr | 25.0% | +0.016 | FAIL |  |
| 2026-05-11 | HAL | Halliburton Company | Energy | $33.3B | $595M | 90% | 16yr | 42.5% | +0.004 | FAIL |  |
| 2026-05-11 | MPC | Marathon Petroleum Corporation | Energy | $71.5B | $625M | 83% | 15yr | 39.3% | +0.003 | FAIL |  |
| 2026-05-11 | SLB | SLB Limited | Energy | $79.6B | $924M | 95% | 16yr | 36.4% | -0.001 | FAIL |  |
| 2026-05-11 | COP | ConocoPhillips | Energy | $138.7B | $1128M | 88% | 16yr | 33.7% | -0.007 | FAIL |  |
| 2026-05-11 | PSX | Phillips 66 | Energy | $68.8B | $550M | 84% | 14yr | 33.6% | -0.020 | FAIL |  |
| 2026-05-11 | VLO | Valero Energy Corporation | Energy | $71.6B | $889M | 94% | 16yr | 38.8% | -0.042 | FAIL |  |
