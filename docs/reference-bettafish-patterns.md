# BettaFish ForumEngine — Multi-Agent Debate Patterns

> Source: [BettaFish](https://github.com/666ghj/BettaFish) (`C:\Dev\useful_repos\BettaFish`)
> Studied: 2026-05-16 (R19f)
> Status: Study only — implementation deferred until multi-agent signal fusion is prioritized.

## Architecture Overview

BettaFish is a multi-agent public opinion analysis system. Its ForumEngine orchestrates debate between 3 specialized LLM agents, moderated by a 4th "ForumHost" (Qwen3). The pattern is: **specialized agents produce independent analyses → moderator synthesizes, identifies disagreements, and guides next round**.

```
┌─────────────────────────────────────────────────────────────┐
│                        ForumHost                            │
│  (Qwen3 — Moderator/Chairperson)                            │
│  • Reads agent speeches from log files                      │
│  • Triggers every 5 agent speeches                          │
│  • Structured output: timeline, consensus/divergence,       │
│    deep analysis, next questions                            │
└──────────┬──────────┬──────────┬────────────────────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼──────┐
    │ INSIGHT │ │ MEDIA   │ │ QUERY    │
    │ (Kimi)  │ │ (Gemini)│ │(DeepSeek)│
    │ DB mine │ │Multi-   │ │Web search│
    │ history │ │modal    │ │real-time │
    └─────────┘ └─────────┘ └──────────┘
```

## Core Components

### 1. ForumHost (`ForumEngine/llm_host.py`, 262 lines)

The moderator LLM agent. Key design decisions:

| Aspect | Implementation |
|--------|---------------|
| **Model** | Qwen3 via SiliconFlow (configurable) |
| **Trigger** | Every 5 agent speeches to `forum.log` |
| **Input** | Parsed log lines: `[HH:MM:SS] [AGENT] content` |
| **Output** | Structured 4-section synthesis (1000-char limit) |
| **Deduplication** | Tracks `previous_summaries` to avoid repeats |
| **Retry** | `@with_graceful_retry` decorator with exponential backoff |

**System prompt structure** (6 responsibilities):
1. Event timeline construction — identify key events, people, time nodes
2. Discussion guidance — probe deep causes
3. Error correction — flag factual contradictions across agents
4. Viewpoint integration — consensus vs divergence
5. Trend prediction — risk points, evolution direction
6. Analysis advancement — new angles, next questions

**User prompt structure** (4 output sections):
1. Event timeline & causal chain
2. Viewpoint integration — cross-agent comparison, information value analysis
3. Deep analysis & trend prediction — root causes, risk indicators
4. Question guidance — 2-3 follow-up questions for agents

### 2. LogMonitor (`ForumEngine/monitor.py`, 859 lines)

File-based asynchronous monitor that watches 3 agent log files:

| Log | Agent | Source |
|-----|-------|--------|
| `insight.log` | INSIGHT | Proprietary DB queries, historical pattern matching |
| `media.log` | MEDIA | Multimodal content (images, video, audio) |
| `query.log` | QUERY | Web search, real-time information |

**Key mechanisms:**
- **Target node detection**: Regex patterns matching `SummaryNode` outputs (FirstSummaryNode, ReflectionSummaryNode)
- **Error filtering**: Excludes ERROR-level logs, JSON parse failures, tracebacks
- **Valuable content gating**: Filters out short status messages ("正在生成", "处理完成")
- **Thread-safe writes**: `write_lock` mutex on `forum.log`
- **Agent speech buffering**: Accumulates 5 speeches before triggering host
- **JSON capture**: Multi-line JSON state machine for structured agent outputs

### 3. Agent Configuration (`config.py`, 131 lines)

Each agent uses a **different LLM provider** — this is deliberate to prevent model homogeneity:

| Agent | Model | Provider | Role |
|-------|-------|----------|------|
| INSIGHT | Kimi K2 | Moonshot | Historical DB mining |
| MEDIA | Gemini 2.5 Pro | AIHubMix | Multimodal analysis |
| QUERY | DeepSeek Chat | DeepSeek | Real-time search |
| REPORT | Gemini 2.5 Pro | AIHubMix | Report generation |
| HOST | Qwen Plus | Alibaba/SiliconFlow | Debate moderation |

## Key Design Patterns

### Pattern 1: Role Specialization with Diverse Models

**BettaFish approach:** Give each agent a distinct model family + distinct tool set. This prevents the "echo chamber" problem where same-model agents converge on identical answers.

**Trading application:** For signal fusion, run 3 different "analyst" agents:
- **Trend Analyst** (CatBoost): Price geometry, pattern detection
- **Sentiment Analyst** (FinBERT/LM): News, filings, social media
- **Regime Analyst** (HMM/Simple): Market state, volatility regime
- Each runs independently, then a **Signal Moderator** synthesizes.

### Pattern 2: Structured Debate Output Format

**BettaFish approach:** Host output always follows 4-section structure:
1. Event timeline
2. Viewpoint integration (consensus + divergence)
3. Deep analysis + trends
4. Follow-up questions

**Trading application:** A signal fusion moderator would output:
1. **Signal summary** — what each analyst says (buy/sell/neutral, confidence)
2. **Consensus/divergence** — where analysts agree vs disagree
3. **Conflict resolution** — which analyst to trust based on regime/context
4. **Trade recommendation** — aggregated signal with confidence score + caveats

### Pattern 3: Asynchronous Log-Based Agent Communication

**BettaFish approach:** Agents don't call each other directly. They write to log files, and the monitor reads them. This decouples agent execution and allows independent parallel runs.

**Trading application:** Pattern detectors (40+) produce signals independently. A "signal aggregator" reads all outputs and applies:
- Voting (majority/minority detection)
- Regime-based weighting (trend patterns > MR patterns in trending markets)
- Confluence scoring (agreement across categories = higher confidence)

### Pattern 4: Threshold-Triggered Moderation

**BettaFish approach:** Host triggers after every 5 agent speeches (configurable via `host_speech_threshold`). This prevents the host from interrupting too frequently.

**Trading application:** Rebalance/re-evaluate signal fusion only when:
- N new pattern signals accumulate (e.g., 5 new signals)
- Significant market event (gap, volatility spike)
- Time-based (end of day, weekly rebalance)

### Pattern 5: Error Gating & Quality Filtering

**BettaFish approach:** Multiple exclusion layers before content reaches the host:
1. ERROR-level logs excluded
2. Error keywords filtered ("JSON解析失败", "Traceback")
3. Short status messages excluded ("正在生成", "处理完成")
4. Only `SummaryNode` outputs qualify as "agent speeches"

**Trading application:** Before aggregating signals, filter:
- Patterns with below-threshold reliability (`min_reliability < threshold`)
- Contradictory signals from same category (e.g., both bullish and bearish triangle)
- Stale signals (generated > N bars ago, not re-confirmed)
- Low-volume confirmation patterns

## Implementation Potential for Trading Signal Fusion

### Quick Win (~80 LOC)
Adapt the structured debate output format for `SignalAggregator` in `src/portfolio/signal_aggregator.py`:
```python
@dataclass
class SignalDebateResult:
    signal_summary: dict[str, float]      # agent → signal score
    consensus: list[str]                   # where agents agree
    divergence: list[str]                  # where agents disagree
    resolution: str                        # how conflict was resolved
    recommendation: float                  # final aggregated signal [-1, 1]
    confidence: float                      # 0-1 based on agreement level
    caveats: list[str]                     # risk factors, regime notes
```

### Medium Effort (~200 LOC)
Build a `SignalForumHost` class mirroring `ForumHost`:
- Takes outputs from 3-4 signal sources (rules, ML, sentiment, regime)
- Uses an LLM (or rule-based) to produce structured fusion report
- Triggers on significant divergence (disagreement > threshold)

### Full Integration (~500 LOC)
Full multi-agent signal fusion with:
- Async log-based agent communication (Pattern 3)
- Threshold-triggered re-evaluation (Pattern 4)
- Quality gating pipeline (Pattern 5)
- Requires LLM API key for moderator agent

## Files Referenced

| File | Lines | Purpose |
|------|-------|---------|
| `BettaFish/ForumEngine/__init__.py` | 7 | Package init, exports LogMonitor |
| `BettaFish/ForumEngine/llm_host.py` | 262 | ForumHost class, prompt engineering, Qwen API |
| `BettaFish/ForumEngine/monitor.py` | 859 | LogMonitor, file watching, speech buffering, JSON capture |
| `BettaFish/config.py` | 131 | Agent model configs, API keys, search settings |
| `BettaFish/app.py` | — | Flask orchestration, engine lifecycle |

## Decision

**Deferred.** The ForumEngine debate pattern is a strong conceptual match for signal confluence resolution, but implementation requires either:
1. An LLM API key for the moderator (adds cost + latency), OR
2. A rule-based approximation (loses the "intelligence" of the moderator)

The structured output format (Pattern 2) can be adopted immediately in `SignalAggregator`. Full multi-agent debate deferred until multi-agent signal fusion becomes a priority.
