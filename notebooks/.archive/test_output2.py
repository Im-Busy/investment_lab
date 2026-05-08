# %% [markdown]
# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [28]</a>'.</span>

# %% [markdown]
# # Pattern Contribution Analysis
#
# This notebook analyzes the contribution of each pattern detector to the multi-pattern trading strategy.
#
# ## Analysis Layers
#
# 1. **Signal Event Log** - Record all pattern detection events
# 2. **Trade Attribution** - Match trades to contributing patterns
# 3. **Ablation Study** - Leave-one-out contribution analysis
# 4. **Synergy Analysis** - Pairwise pattern interactions
#
# ## Sections
#
# 1. Setup & Data
# 2. Run Baseline Backtest
# 3. Signal Event Log Analysis
# 4. Trade Attribution Analysis
# 5. Solo Backtests
# 6. Ablation Study
# 7. Pairwise Synergy
# 8. Visualizations
# 9. Summary & Recommendations

# %% [markdown]
# ## Configuration
#
# Centralized configuration for data, backtest, analysis, and output settings.

# %%
CONFIG = {
    # Data settings
    'data': {
        'filename': 'SPY_daily.csv',
        'directory': 'data/raw',
    },

    # Backtest settings
    'backtest': {
        'initial_capital': 100000,
        'commission': 0.001,
        'exclusive_orders': True,
    },

    # Strategy parameters
    'strategy': {
        'min_confidence': 0.60,
        'min_confluence_count': 2,
        'risk_per_trade': 0.02,
        'max_open_positions': 5,
    },

    # Trade attribution settings
    'attribution': {
        'tolerance_bars': 1,  # Bars tolerance for matching signals to trades
    },

    # Ablation settings
    'ablation': {
        'min_delta_sharpe': 0.05,  # Threshold for removal candidates
    },

    # Synergy analysis settings
    'synergy': {
        'top_n_patterns': 10,  # Top N patterns for pairwise analysis
        'min_synergy': 0.01,   # Threshold for complementary pairs
        'max_synergy': -0.01,  # Threshold for conflicting pairs
    },

    # Visualization settings
    'visualization': {
        'style': 'seaborn-v0_8-darkgrid',
        'palette': 'husl',
        'figure_size': (14, 6),
        'top_n_display': 20,  # Top N patterns to display in charts
    },

    # Output settings
    'output': {
        'ablation_dir': 'reports/ablation',
        'synergy_dir': 'reports/synergy',
        'charts_dir': 'reports/charts',
        'report_dir': 'reports/contribution',
    },
}

# %% [markdown]
# ## Section 1: Setup & Data

# %%
import sys
from pathlib import Path

# Setup project root
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style from config
plt.style.use(CONFIG['visualization']['style'])
sns.set_palette(CONFIG['visualization']['palette'])

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

# Import helpers
from src.utils.notebook_helpers import (
    get_data_path,
    get_output_path,
    print_config,
)

# Print configuration
print_config(CONFIG, "Pattern Contribution Analysis Configuration")

# %%
# Import analysis modules
from src.analysis import (
    SignalEventLog,
    TradeAttributor,
    AblationEngine,
    SynergyAnalyzer,
    ContributionReport,
    create_contribution_charts
)

# Import strategy and runner
from src.strategies.backtest_py.multi_pattern_strategy_optimized import MultiPatternStrategyOptimized
from src.strategies.backtest_py.runner import BacktestPyRunner

# %%
# Load data
data_path = get_data_path(CONFIG, project_root)
df = pd.read_csv(data_path, index_col=0, parse_dates=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nFirst few rows:")
df.head()

# %% [markdown]
# ## Section 2: Run Baseline Backtest
#
# Run the full multi-pattern strategy with signal logging enabled.

# %%
# Extract config values
INITIAL_CAPITAL = CONFIG['backtest']['initial_capital']
COMMISSION = CONFIG['backtest']['commission']
strategy_params = CONFIG['strategy'].copy()

# Run baseline backtest with signal logging enabled
runner = BacktestPyRunner(
    data=df,
    cash=INITIAL_CAPITAL,
    commission=COMMISSION,
    exclusive_orders=CONFIG['backtest']['exclusive_orders'],
)

# Enable signal logging on the strategy class before running
MultiPatternStrategyOptimized.enable_signal_log = True

# Run with signal logging enabled
results = runner.run(
    strategy_class=MultiPatternStrategyOptimized,
    **strategy_params
)

print(f"Baseline Results:")
print(f"  Total Return: {results['stats']['Return [%]']:.2%}")
print(f"  Sharpe Ratio: {results['stats']['Sharpe Ratio']:.3f}")
print(f"  Max Drawdown: {results['stats']['Max. Drawdown [%]']:.2%}")
print(f"  Win Rate: {results['stats']['Win Rate [%]']:.2%}")
print(f"  Total Trades: {results['stats']['# Trades']}")

# %%
# Get signal event log from strategy
# runner.results = backtesting Results object (has _strategy instance)
# results = dict from get_results() - NOT what we need here
assert runner.results is not None, "Backtest Results not in runner"
bt_results = runner.results
strategy_instance = getattr(bt_results, '_strategy', None)
assert strategy_instance is not None, "Strategy instance not found in Results object"
signal_log = getattr(strategy_instance, '_signal_event_log', None)
assert signal_log is not None, \
    "Signal event log is None. Ensure enable_signal_log=True was passed to runner.run()."

print(f"Signal Event Log:")
print(f"  Total Events: {len(signal_log)}")
print(f"  Unique Patterns: {signal_log.get_summary_stats()['unique_patterns']}")
print(f"  Unique Bars: {signal_log.get_summary_stats()['unique_bars']}")
print(f"  Avg Events per Bar: {signal_log.get_summary_stats()['avg_events_per_bar']:.2f}")

# %%
# Get trades DataFrame
trades_df = runner.get_trades()

print(f"Trades DataFrame:")
print(f"  Shape: {trades_df.shape}")
print(f"\nFirst few trades:")
trades_df.head()

# %% [markdown]
# ## Section 3: Signal Event Log Analysis
#
# Analyze pattern detection frequency and co-occurrence.

# %%
# Get detection frequency
assert signal_log is not None, "signal_log not available"
freq_df = signal_log.get_detection_frequency()

print("Pattern Detection Frequency:")
print(freq_df.to_string(index=False))

# %%
# Plot detection frequency
fig, ax = plt.subplots(figsize=(12, 6))

top_n = CONFIG['visualization']['top_n_display']
top_patterns = freq_df.head(top_n)

ax.barh(range(len(top_patterns)), top_patterns['detection_frequency_pct'], color='steelblue')
ax.set_yticks(range(len(top_patterns)))
ax.set_yticklabels(top_patterns['pattern_name'])
ax.set_xlabel('Detection Frequency (% of bars)')
ax.set_title(f'Top {top_n} Patterns by Detection Frequency')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# %%
# Get co-occurrence matrix
assert signal_log is not None, "signal_log not available"
co_occurrence = signal_log.get_co_occurrence_matrix()

print(f"Co-occurrence Matrix Shape: {co_occurrence.shape}")
print(f"\nTop 10 patterns by self co-occurrence:")
diag = np.diag(co_occurrence.values)
top_indices = np.argsort(diag)[-10:][::-1]
for idx in top_indices:
    print(f"  {co_occurrence.index[idx]}: {diag[idx]}")

# %%
# Plot co-occurrence heatmap
from src.analysis.contribution_charts import create_co_occurrence_heatmap

assert signal_log is not None, "signal_log not available"
fig = create_co_occurrence_heatmap(signal_log)
plt.show()

# %%
# Convert signal log to DataFrame for further analysis
assert signal_log is not None, "signal_log not available"
signal_df = signal_log.to_dataframe()

print(f"Signal Events DataFrame:")
print(f"  Shape: {signal_df.shape}")
print(f"\nSample events:")
signal_df.head(10)

# %%
# Analyze patterns by category
category_counts = signal_df['pattern_category'].value_counts()

print("Pattern Events by Category:")
print(category_counts)

# Plot category distribution
fig, ax = plt.subplots(figsize=(10, 6))
category_counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Pattern Category')
ax.set_ylabel('Number of Events')
ax.set_title('Pattern Events by Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 4: Trade Attribution Analysis
#
# Match trades to contributing patterns and analyze performance.

# %%
# Create trade attributor
assert signal_log is not None, "signal_log not available"
trade_attributor = TradeAttributor(
    signal_log=signal_log,
    trades_df=trades_df,
    tolerance_bars=CONFIG['attribution']['tolerance_bars']
)

# Attribute trades
attributed_trades = trade_attributor.attribute_trades()

print(f"Trade Attribution:")
print(f"  Total Trades: {len(trade_attributor)}")
print(f"  Attributed Trades: {trade_attributor.get_summary_stats()['attributed_trades']}")
print(f"  Attribution Rate: {trade_attributor.get_summary_stats()['attribution_rate']:.1%}")

# %%
# Get pattern trade statistics
pattern_trade_stats = trade_attributor.get_pattern_trade_stats()

print("Pattern Trade Statistics:")
print(pattern_trade_stats.to_string(index=False))

# %%
# Plot pattern participation
fig, axes = plt.subplots(1, 2, figsize=CONFIG['visualization']['figure_size'])

# Trade count
top_n = 15
top_patterns = pattern_trade_stats.head(top_n)

axes[0].barh(range(len(top_patterns)), top_patterns['trade_count'], color='steelblue')
axes[0].set_yticks(range(len(top_patterns)))
axes[0].set_yticklabels(top_patterns['pattern_name'])
axes[0].set_xlabel('Trade Count')
axes[0].set_title(f'Top {top_n} Patterns by Trade Count')
axes[0].invert_yaxis()

# Win rate
axes[1].barh(range(len(top_patterns)), top_patterns['win_rate'], color='darkorange')
axes[1].set_yticks(range(len(top_patterns)))
axes[1].set_yticklabels(top_patterns['pattern_name'])
axes[1].set_xlabel('Win Rate')
axes[1].set_title(f'Top {top_n} Patterns by Win Rate')
axes[1].invert_yaxis()
axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Get pattern combination statistics
combo_stats = trade_attributor.get_pattern_combination_stats()

print(f"Pattern Combinations:")
print(f"  Total Unique Combinations: {len(combo_stats)}")
print(f"\nTop 10 Combinations by Trade Count:")
print(combo_stats.head(10).to_string(index=False))

# %%
# Get best trade recipes
best_trades = trade_attributor.get_best_trade_recipes(n=10)

print("Best Trade Recipes:")
print(best_trades.to_string(index=False))

# %%
# Get worst trade recipes
worst_trades = trade_attributor.get_worst_trade_recipes(n=10)

print("Worst Trade Recipes:")
print(worst_trades.to_string(index=False))

# %%
# Get confluence vs performance
confluence_perf = trade_attributor.get_confluence_vs_performance()

print("Confluence vs Performance:")
print(confluence_perf.to_string(index=False))

# %%
# Plot confluence vs performance
from src.analysis.contribution_charts import create_confluence_performance_chart

fig = create_confluence_performance_chart(trade_attributor)
plt.show()

# %%
# Get pattern participation rate
participation = trade_attributor.get_pattern_participation_rate()

print("Pattern Participation Rate:")
print(participation.head(15).to_string(index=False))

# %%
# Plot frequency vs quality scatter
from src.analysis.contribution_charts import create_frequency_quality_scatter

assert signal_log is not None, "signal_log not available"
fig = create_frequency_quality_scatter(signal_log, trade_attributor)
plt.show()

# %%
# Get unattributed trades
unattributed = trade_attributor.get_unattributed_trades()

print(f"Unattributed Trades: {len(unattributed)}")
if not unattributed.empty:
    print(unattributed.head())

# %%
# Convert attributed trades to DataFrame
attributed_df = trade_attributor.to_dataframe()

print(f"Attributed Trades DataFrame:")
print(f"  Shape: {attributed_df.shape}")
print(f"\nSample trades:")
attributed_df.head(10)

# %% [markdown]
# ## Section 5: Solo Backtests
#
# Run backtests with each pattern individually to measure solo edge.

# %%
# Create ablation engine
ablation_engine = AblationEngine(
    data=df,
    strategy_class=MultiPatternStrategyOptimized,
    cash=INITIAL_CAPITAL,
    commission=COMMISSION,
    base_params=strategy_params,
    output_dir=CONFIG['output']['ablation_dir']
)

print(f"Ablation Engine: {ablation_engine}")

# %% [markdown]
# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

# %%
# Run solo backtests for all patterns
print("Running solo backtests...")
solo_results = ablation_engine.run_all_solo_backtests()

print(f"\nSolo Backtest Results:")
print(solo_results.to_string(index=False))

# %%
# Plot solo backtest results
fig, axes = plt.subplots(1, 2, figsize=CONFIG['visualization']['figure_size'])

# Return
solo_sorted = solo_results.sort_values('total_return_pct', ascending=False)
axes[0].barh(range(len(solo_sorted)), solo_sorted['total_return_pct'], color='steelblue')
axes[0].set_yticks(range(len(solo_sorted)))
axes[0].set_yticklabels(solo_sorted['pattern_name'])
axes[0].set_xlabel('Total Return (%)')
axes[0].set_title('Solo Backtest Returns')
axes[0].invert_yaxis()

# Sharpe ratio
solo_sorted_sharpe = solo_results.sort_values('sharpe_ratio', ascending=False)
axes[1].barh(range(len(solo_sorted_sharpe)), solo_sorted_sharpe['sharpe_ratio'], color='darkorange')
axes[1].set_yticks(range(len(solo_sorted_sharpe)))
axes[1].set_yticklabels(solo_sorted_sharpe['pattern_name'])
axes[1].set_xlabel('Sharpe Ratio')
axes[1].set_title('Solo Backtest Sharpe Ratios')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 6: Ablation Study
#
# Run leave-one-out ablation to measure marginal contribution of each pattern.

# %%
# Run baseline
print("Running baseline backtest...")
baseline = ablation_engine.run_baseline()

print(f"\nBaseline Results:")
print(f"  Return: {baseline['total_return_pct']:.2%}")
print(f"  Sharpe: {baseline['sharpe_ratio']:.3f}")
print(f"  Trades: {baseline['total_trades']}")

# %%
# Run full ablation
print("Running full ablation study...")
ablation_results = ablation_engine.run_full_ablation()

print(f"\nAblation Results:")
print(ablation_results.to_string(index=False))

# %%
# Plot ablation results
from src.analysis.contribution_charts import create_pattern_leaderboard, create_contribution_waterfall

# Pattern leaderboard
fig1 = create_pattern_leaderboard(ablation_results)
plt.show()

# Contribution waterfall
fig2 = create_contribution_waterfall(ablation_results)
plt.show()

# %%
# Get top and bottom contributors
print("Top 10 Contributors:")
print(ablation_engine.get_top_contributors(10).to_string(index=False))

print("\nBottom 10 Contributors (Potential Removal Candidates):")
print(ablation_engine.get_bottom_contributors(10).to_string(index=False))

# %%
# Get removal candidates
removal_candidates = ablation_engine.get_removal_candidates(
    min_delta_sharpe=CONFIG['ablation']['min_delta_sharpe']
)

print(f"Removal Candidates (delta_sharpe > {CONFIG['ablation']['min_delta_sharpe']}):")
if not removal_candidates.empty:
    print(removal_candidates.to_string(index=False))
else:
    print("  No removal candidates found")

# %% [markdown]
# ## Section 7: Pairwise Synergy
#
# Analyze pairwise pattern interactions to identify complementary and conflicting pairs.

# %%
# Create synergy analyzer
synergy_analyzer = SynergyAnalyzer(
    data=df,
    strategy_class=MultiPatternStrategyOptimized,
    cash=INITIAL_CAPITAL,
    commission=COMMISSION,
    base_params=strategy_params,
    output_dir=CONFIG['output']['synergy_dir']
)

print(f"Synergy Analyzer: {synergy_analyzer}")

# %%
# Run pairwise synergy analysis (top N patterns for speed)
top_n = CONFIG['synergy']['top_n_patterns']
print(f"Running pairwise synergy analysis (top {top_n} patterns)...")
synergy_results = synergy_analyzer.run_full_pairwise(top_n=top_n)

print(f"\nSynergy Results:")
print(synergy_results.to_string())

# %%
# Get best and worst synergy pairs
print("Best Synergy Pairs:")
print(synergy_analyzer.get_best_synergy_pairs(5).to_string(index=False))

print("\nWorst Synergy Pairs:")
print(synergy_analyzer.get_worst_synergy_pairs(5).to_string(index=False))

# %%
# Get complementary and conflicting pairs
complementary = synergy_analyzer.get_complementary_pairs(
    min_synergy=CONFIG['synergy']['min_synergy']
)
conflicting = synergy_analyzer.get_conflicting_pairs(
    max_synergy=CONFIG['synergy']['max_synergy']
)

print(f"Complementary Pairs: {len(complementary)}")
for a, b in complementary[:5]:
    print(f"  {a} + {b}")

print(f"\nConflicting Pairs: {len(conflicting)}")
for a, b in conflicting[:5]:
    print(f"  {a} + {b}")

# %% [markdown]
# ## Section 8: Visualizations
#
# Generate comprehensive visualizations of the contribution analysis.

# %%
# Create contribution report
assert signal_log is not None, "signal_log not available"
report = ContributionReport(
    signal_log=signal_log,
    attributor=trade_attributor,
    ablation_results=ablation_results,
    synergy_results=synergy_results
)

print(f"Contribution Report: {report}")

# %%
# Generate all charts
assert signal_log is not None, "signal_log not available"
print("Generating contribution charts...")
figures = create_contribution_charts(
    signal_log=signal_log,
    trade_attributor=trade_attributor,
    ablation_results=ablation_results,
    synergy_results=synergy_results,
    output_dir=CONFIG['output']['charts_dir']
)

print(f"Generated {len(figures)} charts")
for name, fig in figures.items():
    print(f"  - {name}")

# %%
# Display all charts
for name, fig in figures.items():
    print(f"\n{name}:")
    plt.figure(fig.number)
    plt.show()

# %% [markdown]
# ## Section 9: Summary & Recommendations
#
# Generate executive summary and actionable recommendations.

# %%
# Generate summary
import json
summary = report.generate_summary()

print("Executive Summary:")
print(json.dumps(summary, indent=2, default=str))

# %%
# Get pattern leaderboard
leaderboard = report.get_pattern_leaderboard()

print("Pattern Leaderboard:")
print(leaderboard.to_string(index=False))

# %%
# Get pattern roles
roles = report.get_pattern_roles()

print("Pattern Roles:")
for pattern, role in sorted(roles.items()):
    print(f"  {pattern}: {role}")

# %%
# Get recommendations
recommendations = report.get_recommendations()

print("Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# %%
# Generate markdown report
markdown_report = report.to_markdown()

print("Markdown Report:")
print(markdown_report[:2000])  # Print first 2000 characters
print("\n... (truncated)")

# %%
# Save all results
print("Saving results...")
saved_files = report.save(CONFIG['output']['report_dir'])

print(f"\nSaved {len(saved_files)} files:")
for name, path in saved_files.items():
    print(f"  {name}: {path}")
