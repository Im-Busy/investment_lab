# Project Decision Loop: DEEPEN / BROADEN / PIVOT / CONCLUDE

This project operates in a continuous test→diagnose→fix→validate loop, not a linear plan. After each phase, the agent decides which direction to take based on evidence, not wishful thinking.

## How to Decide

After every completed phase or significant experiment, evaluate the evidence against ALL four directions before choosing. Apply the **hardest gate first** — if CONCLUDE triggers, it wins. Otherwise, prefer DEEPEN over BROADEN over PIVOT.

## DEEPEN — A Result Raises Follow-Up Questions

**Trigger:** The current approach is showing promise, but specific weaknesses or open questions remain.

**Trading-specific scenarios:**
- Backtest returns are good (>50%) but drawdown is too high (>25%) → deepen risk management (trailing stop, vol gate, position sizing)
- Model is profitable in Trending/Bull/Bear but weak in Transition → deepen regime gating
- Feature importance is unstable (SHAP≠MDI) → deepen understanding of what drives predictions
- Calibration is off (ECE > 0.10) → deepen probability calibration
- A few tickers dominate while others fail → deepen ticker screening or sector modeling
- Win rate is below 40% but PF is above 1.5 → deepen exit optimization

**Action:** Generate sub-tasks that drill into the specific weakness. Do NOT change the overall approach.

**Example:** Bear market WR=0% → deepen: test with shorter holding periods during VIX>30, add volatility stop, or gate out bear regimes entirely.

## BROADEN — Current Results Are Solid, Adjacent Questions Remain

**Trigger:** The current approach has passed its gates and produces acceptable results, but the same methodology could apply to broader contexts.

**Trading-specific scenarios:**
- SPY model works at acceptable Sharpe (>0.5) → broaden to basket (QQQ, IWM, DIA, sector ETFs)
- Daily timeframe works → broaden to weekly or intraday
- Equity model works → broaden to crypto or futures
- Entry signals are solid → broaden to exit optimization or pyramiding
- Single model works → broaden to ensemble or per-sector models
- Walk-forward IC is positive → broaden to live paper trading

**Action:** Apply the SAME methodology to new assets/timeframes/signals. Do NOT change the methodology — just expand its footprint.

**Gate before broadening:** The base configuration must pass its quality gates first. Never broaden a broken model.

**Example:** SPY model Sharpe 0.73 on 2016-2024 → broaden: train basket model on SPY+QQQ+IWM+DIA+XLK+XLF+XLE.

## PIVOT — Results Invalidate Key Assumptions

**Trigger:** Evidence contradicts a core assumption of the current approach. The approach isn't working and minor fixes won't save it.

**Trading-specific scenarios:**
- Pattern-based signals have no edge → pivot to pure ML signal generation
- ML model consistently overfits (OOS Sharpe < 0, train/OOS gap > 0.5) → pivot to simpler models or rule-based
- Feature engineering isn't helping (IC < 0.02 after all fixes) → pivot to raw price/volume as direct inputs
- Daily frequency has no edge → pivot to lower frequency (weekly swing) or higher (intraday scalping)
- Asset class shows no predictability → pivot to a different asset class
- Cross-asset features hurt rather than help → pivot to single-asset only
- Backtest says strategy works but live paper trading fails → pivot to execution research (slippage, market impact)
- Walk-forward IC drops below 0.03 across ALL tickers → pivot to a fundamentally different signal source

**Action:** Return to the problem statement. Search literature for alternative approaches. Re-bootstrap with a different hypothesis. Discard the current approach — do not keep tweaking.

**Crucial distinction:** A single failed experiment is NOT a pivot trigger. Pivot only when (a) multiple independent experiments converge on failure, or (b) a fundamental assumption is proven false.

**Example:** After B9-B14 fixes, model still has no edge in 2025-2026 OOS (Sharpe -0.27 vs train +0.73) → this was NOT a pivot trigger because it was retrained and improved. But if retrained model still shows Sharpe < 0 in OOS → PIVOT to entirely different approach.

## CONCLUDE — Sufficient Evidence for a Decision

**Trigger:** The question has been answered definitively. There is nothing more to learn from further experimentation on this approach.

**Trading-specific scenarios:**

### CONCLUDE: DEPLOY
- Sharpe > 0.5 OOS, PF > 1.3, Win Rate > 35%, 30+ trades, DD < 25%
- Walk-forward IC > 0.03 across multiple time periods
- Model health checks pass (ECE < 0.10, no feature flips, prediction stability ok)
- Backtest → live paper trade correlation > 0.7
- **Action:** Deploy with small capital, set up monitoring, move to live operations

### CONCLUDE: ARCHIVE
- Sharpe < 0 OOS after retraining with all available fixes
- Walk-forward IC < 0.01 across ALL tested tickers
- No configuration achieves PF > 1.0 in OOS
- Every attempt to fix overfitting has failed
- **Action:** Document findings in `progress_docs/plans/`, move on. The negative result is valuable — it tells you what doesn't work.

### CONCLUDE: ACCEPT LIMITS
- Model works but only in specific conditions (e.g., only during bull markets, only on tech tickers)
- Maximum achievable Sharpe is capped at ~0.5-0.7 with current methodology
- Diminishing returns on further optimization
- **Action:** Document the known envelope of performance. Use as a component in a larger system. Move to BROADEN (apply to other assets within the envelope).

## Decision Heuristics

| Metric | DEEPEN | BROADEN | PIVOT | CONCLUDE (DEPLOY) |
|--------|--------|---------|-------|-------------------|
| OOS Sharpe | 0.2–0.5 | >0.5 | <0 (after fixes) | >0.5 stable |
| Train/OOS Gap | 0.05–0.15 | <0.05 | >0.2 | <0.05 |
| Win Rate | 35–45% | >40% | <25% | >35% |
| Calibration ECE | 0.10–0.20 | <0.10 | >0.25 (after fix) | <0.10 |
| Walk-Forward IC | 0.02–0.05 | >0.05 | <0.01 | >0.03 stable |
| Feature Flips | 1–3 | 0 | 5+ | 0 |

## Anti-Patterns

1. **DEEPEN forever** — re-optimizing the same thing when nothing is improving. If 3 consecutive DEEPEN cycles produce no improvement → PIVOT.
2. **BROADEN prematurely** — applying a broken methodology to new assets. Fix first, then expand.
3. **PIVOT too early** — one bad backtest does not justify throwing away months of work. Require converging evidence.
4. **Never CONCLUDE** — accumulating experiments without ever deciding. The goal is deployment OR documented failure, not infinite experimentation.
5. **Confusing noise for signal** — Sharpe improvement <0.1 is likely noise. Don't chase it.

## Loop Protocol

Each session follows this protocol:

1. **Read MEMORY.md** — know the current objective and state
2. **Read `progress_docs/plans/full.md`** — know the master plan
3. **Run diagnostics** — `/model-diagnose` to get current health
4. **Execute** — work on the highest-priority open task
5. **Validate** — run backtests, check metrics against known baselines
6. **Update BESTS.md** — if any backtest produces new bests
7. **Decide direction** — DEEPEN / BROADEN / PIVOT / CONCLUDE
8. **Update MEMORY.md** — with new state, completed tasks, and next steps
9. **Create handover** — if session is ending, update `progress_docs/handovers/` and `progress_docs/current.md`
