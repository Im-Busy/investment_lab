#!/usr/bin/env -S uv run python
"""
Knowledge Graph: Cross-reference research papers with project modules.

Usage:
  uv run useful_resources/_knowledge_analysis.py          # Full analysis
  uv run useful_resources/_knowledge_analysis.py --scan   # Only detect new papers
"""
from pathlib import Path
from collections import Counter, defaultdict
import re, sys, io, argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).parent.parent
PAPERS = ROOT / "useful_resources" / "papers"
PAPERS_MD = ROOT / "useful_resources" / "papers_md"
SRC = ROOT / "src"
INSIGHTS_MD = PAPERS_MD / "KNOWLEDGE_GRAPH_INSIGHTS.md"

TOPIC_KEYWORDS = {
    "Overfitting & Data Leakage": [
        "overfitting", "overfit", "over-fit", "over-fitting", "data leak", "data snoop",
        "look-ahead bias", "lookahead bias", "out-of-sample", "out of sample",
        "cross-validation", "cross validation", "purged", "embargo", "backtest overfit",
        "defensive overfitting", "latent recovery", "generalization"
    ],
    "Reinforcement Learning": [
        "reinforcement learning", "RL", "DQN", "PPO", "policy gradient", "actor-critic",
        "Q-learning", "deep Q", "reward function", "Markov decision", "MDP",
        "multi-agent", "agent-based", "OOM-RL", "trade execution"
    ],
    "Sentiment Analysis & NLP": [
        "sentiment", "NLP", "natural language", "twitter", "tweet", "text classification",
        "lexicon", "word embedding", "BERT", "transformer", "LLM", "large language model",
        "news", "event-driven", "information extraction"
    ],
    "Portfolio Optimization": [
        "portfolio optim", "asset allocation", "mean-variance", "Markowitz", "efficient frontier",
        "Sharpe ratio", "risk parity", "Black-Litterman", "Kelly criterion", "capital allocation",
        "position sizing", "diversification", "rebalancing"
    ],
    "Machine Learning Methods": [
        "gradient boosting", "XGBoost", "LightGBM", "CatBoost", "random forest",
        "neural network", "deep learning", "LSTM", "CNN", "transformer", "ensemble",
        "feature engineering", "feature selection", "hyperparameter", "AutoML",
        "evolutionary algorithm", "genetic algorithm", "alpha factor", "factor mining"
    ],
    "Backtesting & Validation": [
        "backtest", "back-test", "walk-forward", "walk forward", "out-of-sample testing",
        "defensive", "training history", "circuit", "adversarial", "synthetic",
        "simulation", "paper trading", "forward performance"
    ],
    "Risk Management": [
        "risk management", "VaR", "CVaR", "expected shortfall", "drawdown",
        "stop-loss", "stop loss", "circuit breaker", "tail risk", "volatility",
        "crash", "behavioral finance", "no-arbitrage", "martingale"
    ],
    "Event-Driven Trading": [
        "event-driven", "event driven", "event study", "calendar", "financial spikes",
        "corporate event", "earnings", "macroeconomic", "economic indicator",
        "news sentiment", "information flow"
    ],
    "Factor Models & Alpha": [
        "factor model", "alpha factor", "multi-factor", "statistical arbitrage",
        "pair trading", "quantitative factor", "risk factor", "return factor",
        "financial factor", "diffusion process", "alpha decay"
    ],
    "Time Series Forecasting": [
        "time series", "forecasting", "prediction", "seasonal", "stationarity",
        "ARIMA", "GARCH", "volatility forecasting", "fractional differentiation",
        "financial time series", "FinCast"
    ],
}

MODULE_KEYWORDS = {
    "ml": ["overfitting", "cross-validation", "gradient boosting", "ensemble", "LSTM", "deep learning",
           "AutoML", "feature engineering", "hyperparameter", "evolutionary algorithm", "factor mining",
           "reinforcement learning", "RL", "meta-labeling", "triple barrier", "walk-forward",
           "purged", "embargo", "SHAP", "interpretability", "calibration", "confidence scoring"],
    "risk": ["risk management", "VaR", "CVaR", "drawdown", "stop-loss", "circuit breaker",
             "tail risk", "volatility", "position sizing", "Monte Carlo", "crash"],
    "portfolio": ["portfolio optim", "asset allocation", "Black-Litterman", "Kelly criterion",
                  "diversification", "rebalancing", "Sharpe ratio", "risk parity", "mean-variance"],
    "backtest": ["backtest", "walk-forward", "out-of-sample", "event-driven", "paper trading",
                 "simulation", "defensive"],
    "signals": ["signal", "confluence", "aggregation", "confidence", "scoring"],
    "strategies": ["trading strategy", "strategy", "execution", "entry", "exit"],
    "patterns": ["chart pattern", "technical analysis", "pattern detection", "breakout", "support resistance"],
}


def extract_paper_info(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    title = ""
    for line in lines[:20]:
        line = line.strip()
        if not line or line.startswith("<!--") or line.startswith("#"):
            continue
        if len(line) > 10:
            title = line[:150]
            break
    body_start = 0
    for i, line in enumerate(lines):
        if "abstract" in line.strip().lower() or "introduction" in line.strip().lower():
            body_start = i
            break
    abstract = " ".join(l.strip() for l in lines[body_start:body_start + 20] if l.strip())[:800]
    matched_topics = []
    full_text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(full_text_lower.count(kw) for kw in keywords)
        if score > 2:
            matched_topics.append((topic, score))
    matched_topics.sort(key=lambda x: -x[1])
    return {
        "filename": md_path.name,
        "title": title,
        "abstract": abstract,
        "topics": matched_topics[:5],
    }


def find_cross_references(papers: list) -> list:
    connections = []
    for paper in papers:
        paper_text = paper.get("abstract", "").lower()
        for module, keywords in MODULE_KEYWORDS.items():
            matches = [kw for kw in keywords if kw.lower() in paper_text]
            if matches:
                connections.append({
                    "paper": paper["filename"],
                    "paper_title": paper["title"],
                    "module": module,
                    "matches": matches,
                })
    return connections


def write_insights_report(papers: list, connections: list, output_path: Path):
    topic_counts = Counter()
    for p in papers:
        for t, s in p["topics"]:
            topic_counts[t] += 1

    topic_papers = defaultdict(list)
    for p in papers:
        for t, s in p["topics"]:
            topic_papers[t].append(p["filename"])

    report = []
    report.append("# Knowledge Graph Analysis: Research Papers → Project Modules")
    report.append("")
    report.append(f"*Generated from {len(papers)} papers across {len(TOPIC_KEYWORDS)} topic categories*")
    report.append("")

    report.append("## Paper Topic Distribution")
    report.append("")
    for topic, count in topic_counts.most_common():
        report.append(f"- **{topic}**: {count} papers")
    report.append("")

    report.append("## Paper → Module Connections")
    report.append("")
    report.append("| Paper | Module | Key Match Topics |")
    report.append("|-------|--------|-----------------|")
    for c in sorted(connections, key=lambda x: (x["module"], x["paper"])):
        paper_short = c["paper"][:50]
        matches_short = ", ".join(c["matches"][:4])
        report.append(f"| {paper_short} | **{c['module']}** | {matches_short} |")
    report.append("")

    report.append("## Cross-Paper Theme Analysis")
    report.append("")
    for topic in sorted(topic_papers.keys(), key=lambda t: -len(topic_papers[t])):
        pp = topic_papers[topic]
        report.append(f"### {topic} ({len(pp)} papers)")
        for pname in sorted(pp):
            report.append(f"- {pname}")
        report.append("")

    report.append("## Connection Strengths")
    report.append("")
    module_conn_counts = Counter(c["module"] for c in connections)
    for mod, count in module_conn_counts.most_common():
        report.append(f"- **{mod}**: {count} paper connections")
    report.append("")

    report.append("## Specific Implementation Recommendations")
    report.append("")
    recommendations = [
        ("HIGH", "ml", "Add training-history-based overfitting detection (5520 paper)"),
        ("HIGH", "ml", "Implement synthetic OOS comparison framework (Backtest Overfitting paper)"),
        ("HIGH", "signals", "Integrate sentiment scores as signal weight modifier (4+ papers)"),
        ("HIGH", "patterns", "Add event-driven pattern category (Building Calendar + Event-Based Trading)"),
        ("MEDIUM", "rl", "Create src/rl/ module with trade execution environment (2 RL papers)"),
        ("MEDIUM", "portfolio", "Implement Kelly Criterion allocator (Investing Is Compression)"),
        ("MEDIUM", "ml", "Add AutoAlpha-style factor mining pipeline (AutoAlpha paper)"),
        ("MEDIUM", "ml", "Implement circuit-based overfitting detection (Circuit Intrinsic Methods)"),
        ("MEDIUM", "risk", "Add behavioral crash regime detection (Crash-based trading paper)"),
        ("LOW", "ml", "Add adversarial overfitting detection (advrisk_neurips2019)"),
        ("LOW", "data", "Build financial event calendar database (2 event papers)"),
        ("LOW", "backtest", "Add defensive backtesting with time-reversal checks (Universal Trading paper)"),
    ]
    report.append("| Priority | Module | Recommendation |")
    report.append("|----------|--------|---------------|")
    for pri, mod, rec in recommendations:
        report.append(f"| **{pri}** | `{mod}` | {rec} |")

    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written: {output_path}")


def scan_only():
    """Just list new/unprocessed papers."""
    md_stems = {md.stem for md in PAPERS_MD.glob("*.md")}
    new = []
    for pdf in sorted(PAPERS.glob("*.pdf")):
        if pdf.stem not in md_stems:
            new.append(pdf)
    if new:
        print(f"New PDFs (no matching .md): {len(new)}")
        for p in new:
            print(f"  {p.name}")
    else:
        print("All PDFs have matching .md files.")
    print(f"Total PDFs: {len(list(PAPERS.glob('*.pdf')))}")
    print(f"Total MDs:  {len(list(PAPERS_MD.glob('*.md')))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="Only detect new papers")
    args = parser.parse_args()

    if args.scan:
        scan_only()
        return

    print("=" * 60)
    print("KNOWLEDGE GRAPH: Papers ↔ Project Modules")
    print("=" * 60)

    papers = []
    for md_file in sorted(PAPERS_MD.glob("*.md")):
        if md_file.name in ("KNOWLEDGE_GRAPH_INSIGHTS.md", "SENTIMENT_ANALYSIS_SUMMARY.md",
                            "REPOS_ARCHITECTURE_ANALYSIS.md", "research_synthesis_report.md"):
            continue
        info = extract_paper_info(md_file)
        papers.append(info)
    print(f"\nAnalyzed {len(papers)} papers")

    connections = find_cross_references(papers)
    print(f"Found {len(connections)} paper→module connections")

    write_insights_report(papers, connections, INSIGHTS_MD)
    print("Done.")


if __name__ == "__main__":
    main()
