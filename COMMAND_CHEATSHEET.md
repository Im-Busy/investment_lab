# Investment Trading System - Command Cheatsheet

> **Auto-Update Instruction:** This document is the single source of truth for all project commands. When implementing new features that include CLI scripts, flags, or workflows, the implementing AI MUST update this file immediately. Add new commands to the appropriate section following the existing format. Commit with message: "docs: update command cheatsheet for [feature-name]".

---

## Quick Reference - Environment & Package Management

### Python Environment (uv)
```bash
# Run Python command in project environment
uv run <command>

# Install package
uv add <package>

# Remove package
uv remove <package>

# Sync environment with pyproject.toml
uv sync

# Run script
uv run scripts/<script>.py

# Run pytest
uv run pytest tests/ -v

# Run linting with ruff
uv run ruff check src/
```

---

## ML Training & Model Selection

### Train ML Models
```bash
# Default training (CatBoost)
uv run scripts/train_ml_model.py --symbol SPY --start 2015-01-01 --end 2024-12-31

# Train with specific model type
uv run scripts/train_ml_model.py --symbol SPY --model-type catboost
uv run scripts/train_ml_model.py --symbol SPY --model-type chronos
uv run scripts/train_ml_model.py --symbol SPY --model-type xlstm

# Different prediction horizon
uv run scripts/train_ml_model.py --symbol SPY --horizon 5

# Memory-safe training (reduced features)
uv run scripts/train_ml_model.py --symbol SPY --start 2019-01-01 --end 2024-12-31 --features momentum,volatility

# Add suffix to model
uv run scripts/train_ml_model.py --symbol SPY --suffix v1

# Walk-forward validation
uv run scripts/train_ml_model.py --symbol SPY --walk-forward --n-splits 5
```

### ML Selector (Auto-select best model)
```bash
# Get recommendation only
uv run scripts/run_ml.py --recommend

# Auto-select and train
uv run scripts/run_ml.py --auto

# Compare all models
uv run scripts/run_ml.py --compare

# With priority mode
uv run scripts/run_ml.py --auto --priority fast
uv run scripts/run_ml.py --auto --priority accurate

# Export results
uv run scripts/run_ml.py --auto --export json

# Web UI - Streamlit
uv run streamlit run scripts/ml_selector_app.py --server.port=8501

# Web UI - Gradio (recommended)
uv run python scripts/ml_selector_gradio.py
```

### ML Validation
```bash
# ML validation enhanced
uv run scripts/ml_validation_enhanced.py
uv run scripts/ml_validation_exec.py
uv run scripts/ml_validation_ext.py

# Grid search features
uv run scripts/grid_search_features.py
```

---

## Backtesting

### ML-Enhanced Backtesting
```bash
# Default ML-enhanced backtest
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap

# With date range
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --start 2018-01-01 --end 2023-12-31

# Baseline (no ML)
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --no-ml

# Custom capital and risk
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --capital 50000 --risk-pct 0.02

# Export results
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap --export-results --output-dir reports

# Phase B7 ML backtest
uv run scripts/phase_b7_ml_backtest.py --start 2015-01-01 --end 2024-12-31
uv run scripts/phase_b7_custom_backtest.py --start 2015-01-01 --end 2024-12-31
```

### Strategy-Specific Backtests
```bash
# VWAP bounce strategy
uv run scripts/backtest_vwap_bounce.py

# EMA ribbon strategy
uv run scripts/backtest_ema_ribbon.py

# SMA crossover strategy
uv run scripts/backtest_sma_crossover.py

# Keltner channel strategy
uv run scripts/backtest_keltner_channel.py
```

### Multi-Strategy & Portfolio Backtests
```bash
# Backtest all strategies on SPY
uv run scripts/backtest_all_strategies_spy.py

# Backtest all strategies on multiple symbols
uv run scripts/backtest_all_strategies.py

# Portfolio backtest
uv run scripts/portfolio_backtest.py

# Multi-asset backtest
uv run scripts/test_multi_asset.py

# Pair trading backtests
uv run scripts/run_pair_trading_backtests.py
```

### Benchmarking
```bash
# Benchmark vectorized engine
uv run scripts/benchmark_vectorized_engine.py

# Benchmark phase 2
uv run scripts/benchmark_phase2.py

# Benchmark vectorbt
uv run scripts/benchmark_vectorbt.py

# Benchmark ranking
uv run scripts/benchmark_ranking.py

# Validate vectorized engine
uv run scripts/validate_vectorized_engine.py

# Validate all strategies and signals
uv run scripts/validate_all_strategies_signals.py
```

---

## Pattern Detection

### Pattern Detectors
```bash
# Run all pattern detectors
uv run -c "from src.patterns import scan_patterns; scan_patterns('SPY', '2020-01-01', '2024-12-31')"

# Specific pattern detection (via Python API)
# Scripts for individual patterns being developed
```

### Pattern Testing
```bash
# Test pattern selector
uv run scripts/test_pattern_selector.py

# Optimize passing patterns
uv run scripts/optimize_passing_patterns.py

# Test 23 untested patterns
uv run scripts/backtest_23_untested_patterns.py
```

---

## Feature Extraction
```bash
# Extract all features
uv run -c "from src.features import extract_all; extract_all('SPY', output='data/features')"

# Technical indicators
uv run -c "from src.features.technical_indicators import extract; extract('SPY')"

# Feature extraction scripts under development
```

---

## Risk Analysis & Optimization

### Risk Metrics
```bash
# Portfolio risk analysis
# Scripts under development in src/risk/

# Friction scoring test
uv run scripts/test_friction_scoring.py

# Diversity score test
uv run scripts/test_diversity_score.py

# Event weighting test
uv run scripts/test_event_weighting.py
```

### Optimization
```bash
# Optimize MACD parameters
uv run scripts/optimize_macd.py

# Optimize RSI parameters
uv run scripts/optimize_rsi.py

# 5+ parameter optimization
uv run scripts/test_5_plus_optimize.py
```

---

## Testing & Debugging

### Unit Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Test specific module
uv run pytest tests/test_backtest.py -v
uv run pytest tests/test_patterns.py -v
uv run pytest tests/test_ml.py -v

# Test with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run tests with detailed output
uv run pytest tests/ -v --tb=long
```

### Integration Tests & Debugging
```bash
# Test ensemble strategies
uv run scripts/test_ensemble_strategies.py

# Test phase 6 tier 1 features
uv run scripts/test_phase6_tier1.py
uv run scripts/test_phase6_tier1_r2r4r5.py

# Reverse signals testing
uv run scripts/test_reverse_signals.py
uv run scripts/test_reverse_signals_fast.py

# Setup test
uv run scripts/test_setup.py

# Debug scripts
uv run scripts/debug_imports.py
uv run scripts/debug_trades.py
uv run scripts/debug_trades_simple.py
uv run scripts/debug_get_trades.py
```

---

## Phase Implementation Scripts

### Phase B7 (ML Enhancement)
```bash
# ML enhancement
uv run scripts/phase_b_ml_enhancement.py

# ML backtest
uv run scripts/phase_b7_ml_backtest.py --start 2015-01-01 --end 2024-12-31

# Custom backtest
uv run scripts/phase_b7_custom_backtest.py --start 2015-01-01 --end 2024-12-31

# Phase 6 tier 1 integration
uv run scripts/phase6_tier1_integration.py
```

---

## Notebooks

### Execute Notebooks
```bash
# Execute and convert notebook
uv run jupyter nbconvert --to notebook --execute notebooks/01_multi_pattern_backtest.ipynb --output 01_multi_pattern_backtest_executed.ipynb

# Run specific notebook
uv run scripts/run_nb_02.py
```

### Key Notebooks
- `01_multi_pattern_backtest.ipynb` - Multi-pattern backtest
- `02_smc_backtest.ipynb` - SMC backtest
- `03_strategy_comparison.ipynb` - Strategy comparison
- `04_pattern_visualization.ipynb` - Pattern visualization
- `05_spy_longterm_backtest.ipynb` - Long-term SPY backtest
- `06_pattern_contribution.ipynb` - Pattern contribution analysis
- `07_pattern_selection_framework.ipynb` - Pattern selection framework
- `08_benchmark_ranking.ipynb` - Benchmark ranking
- `09_multi_timeframe_backtest.ipynb` - Multi-timeframe backtest
- `12_regime_aware_backtest.ipynb` - Regime-aware backtest
- `13_ml_validation.ipynb` - ML validation
- `14_regime_comparison.ipynb` - Regime detector comparison
- `15_phase2_validation.ipynb` - Phase 2 validation
- `ML_Training_Colab.ipynb` - Google Colab training notebook

---

## Model & Pattern Registry

### Model Selector
```bash
# ML selector app (Streamlit)
uv run streamlit run scripts/ml_selector_app.py

# ML selector app (Gradio)
uv run python scripts/ml_selector_gradio.py

# Export to Hugging Face Spaces
# See .useful_commands/ml_selector_commands.txt for deployment instructions
```

---

## Data Ingestion
```bash
# Data fetching (via Python API)
uv run -c "from src.data_ingestion.fetch_data import fetch_yahoo; fetch_yahoo('SPY', '2020-01-01', '2024-12-31')"
```

---

## Visualization & Analysis

### Contribution Analysis
```bash
# Contribution analysis
uv run scripts/contribution_analysis.py
uv run scripts/contribution_analysis_spy.py
```

### Validation
```bash
# Test SPY backtest
uv run scripts/test_spy_backtest.py

# Simple test
uv run scripts/simple_test.py
```

---

## Paper & Research Tools

### Paper Summarization (paper2md)
```bash
cd useful_resources/useful_repos/paper2md

# Summarize papers
uv run python summarize_md_papers.py --papers-dir ..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md

# Clear cache and re-summarize
uv run python summarize_md_papers.py --papers-dir ..\..\papers_md --out output\SENTIMENT_ANALYSIS_SUMMARY.md --clear-cache

# API Configuration
# Edit: paper2md\.env
# OPENAI_BASE_URL: https://openrouter.ai/api/v1
# OPENAI_MODEL: deepseek/deepseek-v4-flash
```

### AI Research Systems (Reference Only — Linux/Docker Required)
```bash
# RD-Agent — Quant factor/model evolution with Qlib
cd useful_resources/useful_repos/RD-Agent
rdagent fin_factor   # Iterative factor evolution
rdagent fin_model    # Iterative model evolution
rdagent fin_quant    # Factor + model joint evolution
rdagent health_check # Validate setup

# DeepScientist — Local-first research OS
cd useful_resources/useful_repos/DeepScientist
npm install -g @researai/deepscientist
ds --here            # Start research workspace

# Idea2Paper — KG-based paper generation
cd useful_resources/useful_repos/Idea2Paper
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "your idea"

# Architecture analysis reference
# See: useful_resources/papers_md/REPOS_ARCHITECTURE_ANALYSIS.md
```

---

## CLI Tools (Installed via Scoop)

### File Operations
```bash
# Find files
fd pattern

# Search content
rg "pattern"

# Search in archives/PDFs
rga "pattern"

# View file with syntax
bat file.py

# View plain file (for piping)
bat -p --paging-never file.py
```

### Config Processing
```bash
# Parse JSON
jq '.path.to.value' file.json

# Parse YAML/XML
yq '.path.to.value' file.yaml

# Convert JSON to YAML
cat file.json | yq -o yaml '.'

# Convert YAML to JSON
cat file.yaml | yq -o json '.'
```

### Document Conversion
```bash
# Markdown to HTML
pandoc input.md -o output.html

# Markdown to PDF
pandoc input.md -o output.pdf

# Any format conversion
pandoc input.docx -o output.md
```

### Git & Diffs
```bash
# Pretty diff with delta
git diff | delta
git config --global core.pager delta
```

### Disk Usage
```bash
# Directory sizes
dust

# Show all files
dust -n 999
```

### Repository Packing (for AI analysis)
```bash
# Pack current repo
npx repomix

# Pack remote repo
npx repomix --remote owner/repo

# Pack with specific format
npx repomix --style markdown
```

---

## Streamlit & Gradio Apps

### Streamlit
```bash
# ML Selector App
uv run streamlit run scripts/ml_selector_app.py --server.port=8501

# Paper extraction app (from marker)
uv run streamlit run marker/marker/scripts/streamlit_app.py
```

### Gradio
```bash
# ML Selector Gradio App
uv run python scripts/ml_selector_gradio.py
```

---

## Multi-Agent & Seach Tools (MCP)

### Web Search & Research
- **Tavily**: Web search for real-time information
- **Exa**: Semantic search for finding relevant documents
- **Microsoft Learn**: Search Microsoft/Azure documentation
- **Google Maps Platform**: Location-based services

---

## pylint & Code Quality

### Linting (ruff)
```bash
# Check code
uv run ruff check src/

# Check tests
uv run ruff check tests/

# Fix auto-fixable issues
uv run ruff check src/ --fix
```

### Type Checking (mypy)
```bash
# Check types
uv run mypy src/

# Check with strict mode
uv run mypy --strict src/
```

---

## Git Workflow
```bash
# Status, diff, log (used by AI agents)
git status
git diff
git log

# Create commit
git add .
git commit -m "feat: description"

# Push changes
git push
```

---

## Quick Command Reference by Task

### Train a model
```bash
uv run scripts/train_ml_model.py --symbol SPY
```

### Run backtest
```bash
uv run scripts/backtest_ml_enhanced.py --symbol SPY --strategy vwap
```

### Test patterns
```bash
uv run scripts/test_pattern_selector.py
```

### Optimize parameters
```bash
uv run scripts/optimize_macd.py
```

### Validate implementation
```bash
uv run scripts/validate_all_strategies_signals.py
```

### Run ML selection
```bash
uv run scripts/run_ml.py --auto
```

### Analyze contributions
```bash
uv run scripts/contribution_analysis.py
```

---

## File Structure Quick Reference

```
investment_trying/
├── src/                      # Source code
│   ├── backtest/             # Backtesting engine
│   ├── patterns/             # Pattern detectors (34+)
│   ├── strategies/           # Strategy wrappers
│   ├── ml/                   # ML models
│   ├── features/             # Feature extraction
│   ├── signals/              # Signal aggregation
│   ├── risk/                 # Risk management
│   ├── indicators/           # Technical indicators
│   └── data_ingestion/       # Data fetching
├── scripts/                  # CLI scripts (this document)
├── tests/                    # Unit/integration tests
├── notebooks/                # Jupyter notebooks
├── data/                     # Data storage
├── reports/                  # Backtest reports
├── .useful_commands/         # Command documentation
└── docs/                     # Documentation
```

---

## Maintenance

### Updating This Document

**ATTENTION ALL AI ASSISTANTS:** When implementing new functionality:

1. Add CLI commands to appropriate section above
2. Include brief description of what the command does
3. Group related commands together
4. Update file structure if new directories are created
5. Commit with: "docs: update command cheatsheet for [feature]"

### Related Documentation

- `.useful_commands/` - Detailed command files by category
- `src/*/AI_COMMANDS.txt` - Module-specific commands
- `AGENTS.md` - Project-wide AI instructions
- `.kilo/global-rules.md` - Global coding standards
- `.kilo/project-rules.md` - Project-specific rules

---

*Last Updated: 2026-04-25*
*Total Scripts: 60+*
*Categories: 15+*

