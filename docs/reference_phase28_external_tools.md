# Phase 28 Reference: External Tool Evaluations

> Source: awesome-ai-in-finance, stock-sdk, OpenStock. Reference-only — no code implementation needed.

## P28-23: OpenStock Full-Stack Architecture Reference

**OpenStock** uses a modern full-stack architecture that could serve as a blueprint for a future web dashboard:

| Layer | Technology | Project Equivalent |
|-------|-----------|-------------------|
| **Framework** | Next.js 15 (App Router) | N/A (CLI-only) |
| **UI** | shadcn/ui + Radix | N/A |
| **Charts** | TradingView Lightweight Charts | matplotlib (backend) |
| **Database** | MongoDB + Mongoose | CSV/JSON logs |
| **Auth** | NextAuth.js + Better Auth | N/A (local) |
| **Email** | Resend + React Email | N/A |
| **Scheduling** | Inngest (serverless cron) | Windows Task Scheduler |
| **Deployment** | Docker Compose | `docker-compose.prod.yml` (P28-24) |

**Key patterns worth adopting if building a web UI:**
1. **TradingView charts** embedded in React components — real-time OHLCV with indicator overlays
2. **MongoDB aggregation pipelines** for portfolio analytics (group by sector, time-series bucketing)
3. **Inngest-style cron** for daily report generation (P28-22 implements the CLI equivalent)
4. **Personalized onboarding flow** collecting country/goals/risk-tolerance → strategy config (P28-25)

---

## P28-26: WFGY LLM Agent Stress-Test Framework

**WFGY (Wang-Feng-Guo-Yang)** proposes a 16-mode failure map for LLM-based trading agents:

| Mode | Failure Type | Detection Method |
|------|-------------|-----------------|
| 1 | Over-trading in low signal | Trade count vs volatility |
| 2 | Under-trading in high signal | Missed opportunity count |
| 3 | Position size drift | Size vs Kelly fraction |
| 4 | Stale data dependency | Timestamp freshness check |
| 5 | Prompt injection | Input sanitization |
| 6 | Hallucinated signals | Signal vs known pattern overlap |
| 7 | Recency bias | Trade distribution by date |
| 8 | Regime blindness | Return by market regime |
| 9 | Overtfitting to prompt | Prompt ablation test |
| 10 | Context window truncation | Full context vs last-N comparison |
| 11 | Tool misuse | Tool call parameter validation |
| 12 | Latency cascade | Sequential tool call timing |
| 13 | Position concentration | HHI of holdings |
| 14 | Fee blindness | Net vs gross return gap |
| 15 | Benchmark anchoring | Deviation from benchmark allocation |
| 16 | Model update shock | Pre/post model version comparison |

**Application:** If this project adds LLM agents (P28-17, deferred), these 16 modes form the acceptance test suite. Each mode is a check that must PASS before deployment.

---

## P28-27: tf-quant-finance Evaluation

**Google's TensorFlow Quant Finance** (`tf-quant-finance`) provides:

| Module | Use Case | Project Relevance |
|--------|----------|-------------------|
| `math.pde` | PDE solvers for options pricing | Covered by `src/ml/options_pricing.py` (BS/Binomial/MC/Heston) |
| `models.hull_white` | Interest rate models | Covered by `src/ml/fixed_income_models.py` (Vasicek/CIR) |
| `monte_carlo` | GPU-accelerated MC | CPU-only currently — could speed up MC pricing if GPU available |
| `math.random` | Sobol/Halton quasi-random | Better convergence than numpy random for MC pricing |

**Verdict:** Not worth adopting. Project already has scipy-based equivalents for all relevant modules. The GPU acceleration benefit is gated on GPU availability, and the dependency weight (TensorFlow ~500MB) is disproportionate. Documented for awareness.

---

## P28-19: Indicator Computation Reference

Stock-sdk (Node.js) implements these indicators. Comparison against project Python implementations:

| Indicator | stock-sdk (JS) | Project (Python) | Status |
|-----------|---------------|-------------------|--------|
| **MA** | `getMA(data, dayCount)` | `pd.Series.rolling(dayCount).mean()` | ✅ Identical |
| **MACD** | `getMACD(data, fast, slow, signal)` | `src/indicators/pinescript_helpers.py:macd()` | ✅ Identical |
| **BOLL** | `getBOLL(data, dayCount, k=2)` | `pd.rolling(20).mean()` + `2*std` | ✅ Identical |
| **KDJ** | `getKDJ(data, n=9, m1=3, m2=3)` | Not implemented | ⚠️ Missing — Stochastic K/D/J equivalent |
| **RSI** | `getRSI(data, dayCount=14)` | `src/indicators/pinescript_helpers.py:rsi()` | ✅ Identical |
| **WR** | `getWR(data, dayCount=14)` | `src/patterns/technical/williams_r.py` | ✅ Identical |
| **BIAS** | `getBIAS(data, dayCount=6)` | Not implemented | ⚠️ Missing — BIAS = (Close-MA)/MA |
| **CCI** | `getCCI(data, dayCount=14)` | `src/patterns/technical/cci.py` | ✅ Identical |
| **ATR** | `getATR(data, dayCount=14)` | `src/indicators/pinescript_helpers.py:atr()` | ✅ Identical |

**Gaps identified:** KDJ (Stochastic oscillator), BIAS (deviation from MA). These are trivial to implement via existing SMA/rolling infrastructure — no dedicated file needed.
