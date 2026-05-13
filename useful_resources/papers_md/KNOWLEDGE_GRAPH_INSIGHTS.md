# Knowledge Graph Analysis: Research Papers → Project Modules

*Generated from 48 papers across 10 topic categories*

## Paper Topic Distribution

- **Time Series Forecasting**: 32 papers
- **Machine Learning Methods**: 30 papers
- **Overfitting & Data Leakage**: 27 papers
- **Sentiment Analysis & NLP**: 22 papers
- **Backtesting & Validation**: 22 papers
- **Risk Management**: 18 papers
- **Event-Driven Trading**: 10 papers
- **Factor Models & Alpha**: 7 papers
- **Reinforcement Learning**: 6 papers
- **Portfolio Optimization**: 6 papers

## Project Module Capabilities

### `src/ai/`
Files:

### `src/analysis/`
Files: ablation_engine.py, ablation_study_runner.py, contribution_analyzer.py, contribution_charts.py, contribution_report.py, correlation_analyzer.py, deflated_sharpe.py, pattern_performance_tracker.py, pattern_selector.py, pattern_selector_viz.py
Classes: AblationResult, AblationEngine, AblationStudyConfig, SubsetResult, AblationStudyRunner, AblationResult, ContributionAnalyzer, ContributionReport, CorrelationAnalyzerConfig, CorrelationGroup, CorrelationAnalyzer, PSRResult, FDRResult, RollingMetrics, PerformanceTrackerConfig

### `src/backtest/`
Files: adapter.py, benchmark.py, engine.py, epistemic_autopsy.py, friction_scoring.py, metrics.py, risk_metrics.py, vectorbt_adapter.py, vectorbt_alternative.py
Classes: BacktestEngineType, UnifiedBacktestConfig, UnifiedBacktestResult, BacktestAdapter, BenchmarkComparison, BacktestConfig, BacktestResult, BacktestEngine, AutopsyResult, EpistemicAutopsy, AssetClass, FrictionConfig, FrictionCosts, FrictionSummary, FrictionScorer

### `src/data_ingestion/`
Files: data_utils.py, duckdb_helpers.py, fetch_data.py

### `src/features/`
Files: alpha_factors.py, smoothing.py, technical_indicators.py

### `src/gpu/`
Files: backtest.py, features.py, pattern_detector.py, signal_aggregator.py, smooth.py, utils.py
Classes: GPUTrade, GPUBacktestResult, GPUEquitySimulator, GPUBacktestEngine, GPUFeatureEngineer, GPUSignalBatch, GPUPatternDetector, GPUAggregatedSignals, GPUSignalAggregator

### `src/indicators/`
Files: asian_range.py, fibonacci.py, ifvg.py, indicator_cache.py, kalman_hedge.py, liquidity_sweep.py, mss.py, pivots.py, pivots_numba.py, regime.py
Classes: AsianRange, IFVG, IFVGProximity, IndicatorCache, KalmanHedgeResult, KalmanHedgeRatio, SweepInfo, MSSInfo, PivotPoint, TrendDirection, VolatilityRegime, MarketPhase, RegimeState, MarketRegimeDetector, RegimeState

### `src/ml/`
Files: automl.py, backtest_bridge.py, breakout_classifier.py, change_point_regime.py, cnn_regime.py, combinatorial_purged_cv.py, cross_asset_features.py, cross_symbol_cluster.py, drawdown_target.py, dream_team_ensemble.py
Classes: AutoMLResult, AutoMLBaseline, MLBacktestResult, BacktestBridge, BreakoutResult, BreakoutClassifier, ChangePointRegimeDetector, EarlyStopping, Regime1DCNN, CNNRegimeDetector, CombinatorialPurgedCV, CrossAssetFeatures, CrossAssetFeatureExtractor, CrossSymbolCluster, DrawdownTargetGenerator

### `src/optimizer/`
Files: portfolio_optimizer.py

### `src/patterns/`
Files: base.py
Classes: RegimeState, PatternType, SignalDirection, TradeSignal, PatternResult, BasePattern, PatternDetector

### `src/portfolio/`
Files: black_litterman.py, multi_strategy_engine.py, portfolio_risk.py, signal_aggregator.py, strategy_ranker.py
Classes: BLView, BLConfig, BLResult, BlackLittermanOptimizer, StrategyBacktestResult, PortfolioBacktestResult, MultiStrategyEngine, WeightScheme, EqualWeightScheme, SharpeWeightScheme, InverseVolatilityWeightScheme, KellyWeightScheme, RiskLimitType, RiskLimit, PortfolioRiskManager

### `src/risk/`
Files: circuit_breakers.py, crash_factor.py, daily_limits.py, diversity_calculator.py, diversity_score.py, dynamic_rebalancing.py, failure_set_analyzer.py, mc_var.py, position_probability.py, position_sizing.py
Classes: CircuitBreakerState, CircuitBreakerConfig, CircuitBreaker, CrashFactorConfig, CrashFactorResult, CrashFactorModel, CrashFactorFilter, TradingState, LimitType, DailyLossState, DailyLossLimiter, CircuitBreaker, RiskMonitor, DiversityReport, DiversityMethod

### `src/signals/`
Files: event_weighting.py, position_manager.py, signal_generator.py
Classes: EventType, EventTypeInfo, PatternEventMapping, EventWeightedSignal, EventWeightedAggregator, PositionStatus, Position, PositionSizer, PositionManager, AggregatedSignal, SignalGenerator

### `src/strategies/`
Files: adaptive_router.py, adx_trend_strength.py, awesome_oscillator.py, cci_strategy.py, chaikin_oscillator.py, chandelier_exit.py, confluence.py, connors_rsi.py, donchian_breakout.py, ema_ribbon.py
Classes: StrategyCategory, AdaptiveRouter, ADXTrendStrengthStrategy, AwesomeOscillatorStrategy, CCIStrategy, ChaikinOscillatorStrategy, ChandelierExitStrategy, ConfluenceScore, PatternCompatibility, ConfluenceScorer, ConnorsRSIMeanReversion, DonchianChannelStrategy, EMARibbonStrategy, IchimokuCloudStrategy, ConfluenceLevel

### `src/utils/`
Files: helpers.py, notebook_helpers.py, validators.py
Classes: DataConfig, BacktestConfig, StrategyConfig, OutputConfig, PatternSelectionConfig, SMCConfig

### `src/visualization/`
Files: charts.py, pattern_markers.py, report.py, tearsheet.py
Classes: MarkerStyle, ChartGenerator, MarkerType, PatternMarker, PatternMarkerGenerator, ReportGenerator, TearsheetGenerator

## Paper → Module Connections

| Paper | Module | Key Match Topics |
|-------|--------|-----------------|
| Backtest Overfitting in the Machine Learning Era-  | **backtest** | backtest, out-of-sample |
| Chapter4.MontesinosLpez2022_Chapter_OverfittingMod | **backtest** | simulation |
| Circuit-Based Intrinsic Methods to Detect Overfitt | **backtest** | simulation |
| REPOS_ARCHITECTURE_ANALYSIS.md | **backtest** | backtest |
| RIsk Management and Event Driven Funds with State- | **backtest** | event-driven |
| research_synthesis_report.md | **backtest** | backtest, simulation |
| 5520_using_the_training_history_to_.md | **ml** | overfitting, RL |
| A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WIT | **ml** | deep learning, reinforcement learning, RL |
| Accounting for Spatial Autocorrelation in Algorith | **ml** | overfitting, cross-validation |
| Against a Universal Trading Strategy- No-Arbitrage | **ml** | RL |
| AutoAlpha an Efficient Hierarchical Evolutionary A | **ml** | RL |
| Backtest Overfitting in the Machine Learning Era-  | **ml** | overfitting, cross-validation, RL, purged |
| Chapter4.MontesinosLpez2022_Chapter_OverfittingMod | **ml** | cross-validation, RL |
| Circuit-Based Intrinsic Methods to Detect Overfitt | **ml** | overfitting, RL |
| Deep Learning for Portfolio Optimization.md | **ml** | deep learning |
| Defining the Determinants of Corporate Financial P | **ml** | overfitting, cross-validation, ensemble |
| Detecting Overfitting of Deep Generative Networks  | **ml** | overfitting |
| Detecting Overfitting via Adversarial Examples.md | **ml** | overfitting |
| Emergence of Statistical Financial Factors by a Di | **ml** | RL |
| Event-Based Trading- Building Superior Trading Str | **ml** | RL |
| FinCast- A Foundation Model for Financial Time-Ser | **ml** | RL |
| How is Machine Learning Useful for Macroeconomic F | **ml** | cross-validation, RL |
| I tried a bunch of things - The dangers of unexpec | **ml** | overfitting, cross-validation, hyperparameter |
| Keeping Deep Learning Models in Check - A History- | **ml** | overfitting, deep learning, RL |
| Keeping Deep Learning Models in Check - A History- | **ml** | overfitting, deep learning, RL |
| Machine Learning-Based Financial Big Data Analysis | **ml** | deep learning, feature engineering |
| OOM-RL- Out-of-Money Reinforcement Learning Market | **ml** | reinforcement learning, RL |
| Overview of leakage scenarios in supervised machin | **ml** | RL |
| Preventing Data Leakage in Classification via Inte | **ml** | overfitting |
| Time Travel is Cheating - Going Live with DeepFund | **ml** | RL |
| research_synthesis_report.md | **ml** | calibration |
| Deep Learning for Portfolio Optimization.md | **portfolio** | Sharpe ratio |
| Emergence of Statistical Financial Factors by a Di | **portfolio** | diversification |
| Investing Is Compression.md | **portfolio** | Kelly criterion |
| 1.AMultimodalEvent-drivenLSTMModelforStockPredicti | **risk** | VaR |
| 2024_SentimentAnalysisofTwitterDataUsingMachineLea | **risk** | VaR |
| 6attribution_analysis_of_bullbear_alphas_betas_cai | **risk** | risk management, VaR |
| A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WIT | **risk** | volatility |
| Algorithmic Trading and AI A Review of Strategies  | **risk** | VaR |
| Crash-based quantitative trading strategies- Persp | **risk** | crash |
| Detecting Overfitting via Adversarial Examples.md | **risk** | VaR |
| FinCast- A Foundation Model for Financial Time-Ser | **risk** | VaR |
| How is Machine Learning Useful for Macroeconomic F | **risk** | VaR |
| Machine Learning-Based Financial Big Data Analysis | **risk** | VaR |
| NLP - Powered Sentiment Analysis on the Twitter.md | **risk** | VaR |
| Overview of leakage scenarios in supervised machin | **risk** | VaR |
| Profit Mirage - Revisiting Information Leakage in  | **risk** | risk management |
| research_synthesis_report.md | **risk** | VaR |
| Event-Based Trading- Building Superior Trading Str | **signals** | signal |
| Overfitting, Underfitting and General Model Overco | **signals** | confidence |
| REPOS_ARCHITECTURE_ANALYSIS.md | **signals** | signal |
| research_synthesis_report.md | **signals** | signal |
| 1.AMultimodalEvent-drivenLSTMModelforStockPredicti | **strategies** | strategy |
| A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WIT | **strategies** | strategy |
| Against a Universal Trading Strategy- No-Arbitrage | **strategies** | trading strategy, strategy |
| An adaptive dual-level reinforcement learning appr | **strategies** | strategy, execution |
| AutoAlpha an Efficient Hierarchical Evolutionary A | **strategies** | strategy |
| Circuit-Based Intrinsic Methods to Detect Overfitt | **strategies** | exit |
| OOM-RL- Out-of-Money Reinforcement Learning Market | **strategies** | execution |
| Overfitting, Underfitting and General Model Overco | **strategies** | exit |
| REPOS_ARCHITECTURE_ANALYSIS.md | **strategies** | strategy |
| research_synthesis_report.md | **strategies** | execution |

## Cross-Paper Theme Analysis

### Time Series Forecasting (32 papers)
- 1.AMultimodalEvent-drivenLSTMModelforStockPredictionUsingOnlineNews.md
- 2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md
- 5520_using_the_training_history_to_.md
- 6attribution_analysis_of_bullbear_alphas_betas_caia_aiar_q2_2012.md
- A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WITH MACHINE LEARNING, DEEP LEARNING, AND REINFORCEMENT LEARNING METHODS.md
- A tweet sentiment classification approach using an ensemble classifier.md
- Accounting for Spatial Autocorrelation in Algorithm-Driven Hedonic Models - A Spatial Cross-Validation Approach.md
- Algorithmic Trading and AI A Review of Strategies and Market Impact.md
- Algorithmic Trading and Cookbook.md
- An adaptive dual-level reinforcement learning approach for optimal trade execution.md
- AutoAlpha an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment.md
- Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
- Building a Calendar of Events Database by Analyzing Financial Spikes.md
- Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.md
- Deep Learning for Portfolio Optimization.md
- Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.md
- Detecting Overfitting via Adversarial Examples.md
- Emergence of Statistical Financial Factors by a Diffusion Process.md
- FinCast- A Foundation Model for Financial Time-Series Forecasting.md
- How is Machine Learning Useful for Macroeconomic Forecasting.md
- I tried a bunch of things - The dangers of unexpected overfitting in classification of brain data.md
- Investing Is Compression.md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting (arXiv).md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting.md
- Leveraging hybrid model for accurate sentiment analysis of Twitter data.md
- Machine Learning-Based Financial Big Data Analysis and Forecasting.md
- Machine learning algorithm validation with a limited sample size.md
- NLP - Powered Sentiment Analysis on the Twitter.md
- Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI.md
- Overview of leakage scenarios in supervised machine learning.md
- Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.md
- Sentiment Analysis of Twitter Texts Using Machine Learning Algorithms.md

### Machine Learning Methods (30 papers)
- 1.AMultimodalEvent-drivenLSTMModelforStockPredictionUsingOnlineNews.md
- 2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md
- 5520_using_the_training_history_to_.md
- A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WITH MACHINE LEARNING, DEEP LEARNING, AND REINFORCEMENT LEARNING METHODS.md
- A tweet sentiment classification approach using an ensemble classifier.md
- Accounting for Spatial Autocorrelation in Algorithm-Driven Hedonic Models - A Spatial Cross-Validation Approach.md
- Algorithmic Trading and AI A Review of Strategies and Market Impact.md
- An adaptive dual-level reinforcement learning approach for optimal trade execution.md
- AutoAlpha an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment.md
- Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.md
- Circuit-Based Intrinsic Methods to Detect Overfitting.md
- Deep Learning for Portfolio Optimization.md
- Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.md
- Detecting Overfitting of Machine Learning Techniques for Automatic Vulnerability Detection.md
- Event-Based Trading- Building Superior Trading Strategies.md
- FinCast- A Foundation Model for Financial Time-Series Forecasting.md
- How is Machine Learning Useful for Macroeconomic Forecasting.md
- I tried a bunch of things - The dangers of unexpected overfitting in classification of brain data.md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting (arXiv).md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting.md
- Leveraging hybrid model for accurate sentiment analysis of Twitter data.md
- Machine Learning-Based Financial Big Data Analysis and Forecasting.md
- Machine learning algorithm validation with a limited sample size.md
- NLP - Powered Sentiment Analysis on the Twitter.md
- Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI.md
- Overview of leakage scenarios in supervised machine learning.md
- Preventing Data Leakage in Classification via Integrated Machine Learning Pipelines.md
- SENTIMENT_ANALYSIS_SUMMARY.md
- Sentiment Analysis of Twitter Texts Using Machine Learning Algorithms.md
- Twitter_Sentiment_Analysis_Using_Machine_Learning_.md

### Overfitting & Data Leakage (27 papers)
- 2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md
- 5520_using_the_training_history_to_.md
- A tweet sentiment classification approach using an ensemble classifier.md
- Accounting for Spatial Autocorrelation in Algorithm-Driven Hedonic Models - A Spatial Cross-Validation Approach.md
- Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
- Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.md
- Circuit-Based Intrinsic Methods to Detect Overfitting.md
- Deep Learning for Portfolio Optimization.md
- Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.md
- Detecting Overfitting of Deep Generative Networks via Latent Recovery.md
- Detecting Overfitting of Machine Learning Techniques for Automatic Vulnerability Detection.md
- Detecting Overfitting via Adversarial Examples.md
- FinCast- A Foundation Model for Financial Time-Series Forecasting.md
- How is Machine Learning Useful for Macroeconomic Forecasting.md
- I tried a bunch of things - The dangers of unexpected overfitting in classification of brain data.md
- Investing Is Compression.md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting (arXiv).md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting.md
- Leveraging hybrid model for accurate sentiment analysis of Twitter data.md
- Machine Learning-Based Financial Big Data Analysis and Forecasting.md
- Machine learning algorithm validation with a limited sample size.md
- OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md
- Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI.md
- Overview of leakage scenarios in supervised machine learning.md
- Preventing Data Leakage in Classification via Integrated Machine Learning Pipelines.md
- Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.md
- Twitter_Sentiment_Analysis_Using_Machine_Learning_.md

### Sentiment Analysis & NLP (22 papers)
- 1.AMultimodalEvent-drivenLSTMModelforStockPredictionUsingOnlineNews.md
- 2024_SentimentAnalysisofTwitterDataUsingMachineLearningTechniques.md
- A tweet sentiment classification approach using an ensemble classifier.md
- Algorithmic Trading and AI A Review of Strategies and Market Impact.md
- Algorithmic Trading and Cookbook.md
- An adaptive dual-level reinforcement learning approach for optimal trade execution.md
- Building a Calendar of Events Database by Analyzing Financial Spikes.md
- Detecting Overfitting of Machine Learning Techniques for Automatic Vulnerability Detection.md
- Emergence of Statistical Financial Factors by a Diffusion Process.md
- Event-Based Trading- Building Superior Trading Strategies.md
- FinCast- A Foundation Model for Financial Time-Series Forecasting.md
- Leveraging hybrid model for accurate sentiment analysis of Twitter data.md
- NLP - Powered Sentiment Analysis on the Twitter.md
- Preventing Data Leakage in Classification via Integrated Machine Learning Pipelines.md
- Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.md
- RIsk Management and Event Driven Funds with State-of-the-Art Information Extraction Tools.md
- SENTIMENT_ANALYSIS_SUMMARY.md
- Sentiment Analysis of Twitter Texts Using Machine Learning Algorithms.md
- Time Travel is Cheating - Going Live with DeepFund for Real-Time Fund Investment Benchmarking.md
- TwitterDistantSupervision09.md
- Twitter_Sentiment_Analysis_Using_Machine_Learning_.md
- research_synthesis_report.md

### Backtesting & Validation (22 papers)
- 5520_using_the_training_history_to_.md
- Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization.md
- Algorithmic Trading and Cookbook.md
- An adaptive dual-level reinforcement learning approach for optimal trade execution.md
- AutoAlpha an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment.md
- Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
- Chapter4.MontesinosLpez2022_Chapter_OverfittingModelTuningAndEvalu.md
- Circuit-Based Intrinsic Methods to Detect Overfitting.md
- Detecting Overfitting of Deep Generative Networks via Latent Recovery.md
- Detecting Overfitting of Machine Learning Techniques for Automatic Vulnerability Detection.md
- Detecting Overfitting via Adversarial Examples.md
- Event-Based Trading- Building Superior Trading Strategies.md
- I tried a bunch of things - The dangers of unexpected overfitting in classification of brain data.md
- Investing Is Compression.md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting (arXiv).md
- Keeping Deep Learning Models in Check - A History-Based Approach to Mitigate Overfitting.md
- Machine learning algorithm validation with a limited sample size.md
- OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md
- Overfitting, Underfitting and General Model Overconfidence and Under-Performance Pitfalls and Best Practices in Machine Learning and AI.md
- REPOS_ARCHITECTURE_ANALYSIS.md
- Time Travel is Cheating - Going Live with DeepFund for Real-Time Fund Investment Benchmarking.md
- research_synthesis_report.md

### Risk Management (18 papers)
- 1.AMultimodalEvent-drivenLSTMModelforStockPredictionUsingOnlineNews.md
- 6attribution_analysis_of_bullbear_alphas_betas_caia_aiar_q2_2012.md
- A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WITH MACHINE LEARNING, DEEP LEARNING, AND REINFORCEMENT LEARNING METHODS.md
- Against a Universal Trading Strategy- No-Arbitrage, No-Free-Lunch, and Adversarial Cantor Diagonalization.md
- Algorithmic Trading and AI A Review of Strategies and Market Impact.md
- Algorithmic Trading and Cookbook.md
- Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
- Building a Calendar of Events Database by Analyzing Financial Spikes.md
- Crash-based quantitative trading strategies- Perspective of behavioral finance.md
- Deep Learning for Portfolio Optimization.md
- Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.md
- FinCast- A Foundation Model for Financial Time-Series Forecasting.md
- JPM Why Not 100 Equities.md
- Machine Learning-Based Financial Big Data Analysis and Forecasting.md
- OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md
- RIsk Management and Event Driven Funds with State-of-the-Art Information Extraction Tools.md
- Time Travel is Cheating - Going Live with DeepFund for Real-Time Fund Investment Benchmarking.md
- research_synthesis_report.md

### Event-Driven Trading (10 papers)
- 1.AMultimodalEvent-drivenLSTMModelforStockPredictionUsingOnlineNews.md
- Algorithmic Trading and Cookbook.md
- Backtest Overfitting in the Machine Learning Era- A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment.md
- Building a Calendar of Events Database by Analyzing Financial Spikes.md
- Defining the Determinants of Corporate Financial Performance - A Machine Learning Approach.md
- How is Machine Learning Useful for Macroeconomic Forecasting.md
- Machine Learning-Based Financial Big Data Analysis and Forecasting.md
- Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.md
- RIsk Management and Event Driven Funds with State-of-the-Art Information Extraction Tools.md
- research_synthesis_report.md

### Factor Models & Alpha (7 papers)
- 6attribution_analysis_of_bullbear_alphas_betas_caia_aiar_q2_2012.md
- A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WITH MACHINE LEARNING, DEEP LEARNING, AND REINFORCEMENT LEARNING METHODS.md
- Algorithmic Trading and AI A Review of Strategies and Market Impact.md
- AutoAlpha an Efficient Hierarchical Evolutionary Algorithm for Mining Alpha Factors in Quantitative Investment.md
- Emergence of Statistical Financial Factors by a Diffusion Process.md
- How is Machine Learning Useful for Macroeconomic Forecasting.md
- research_synthesis_report.md

### Reinforcement Learning (6 papers)
- A SURVEY OF STATISTICAL ARBITRAGE PAIR TRADING WITH MACHINE LEARNING, DEEP LEARNING, AND REINFORCEMENT LEARNING METHODS.md
- An adaptive dual-level reinforcement learning approach for optimal trade execution.md
- OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md
- Overview of leakage scenarios in supervised machine learning.md
- Profit Mirage - Revisiting Information Leakage in LLM-based Financial Agents.md
- Time Travel is Cheating - Going Live with DeepFund for Real-Time Fund Investment Benchmarking.md

### Portfolio Optimization (6 papers)
- Deep Learning for Portfolio Optimization.md
- Investing Is Compression.md
- JPM Why Not 100 Equities.md
- OOM-RL- Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems.md
- RIsk Management and Event Driven Funds with State-of-the-Art Information Extraction Tools.md
- Time Travel is Cheating - Going Live with DeepFund for Real-Time Fund Investment Benchmarking.md

## Gap Analysis: What Papers Cover vs. What Project Has

### Crossover Strengths (papers + modules aligned)

- **ml**: 25 paper connections — module EXISTS and is well-aligned
- **risk**: 14 paper connections — module EXISTS and is well-aligned
- **strategies**: 10 paper connections — module EXISTS and is well-aligned
- **backtest**: 6 paper connections — module EXISTS and is well-aligned
- **signals**: 4 paper connections — module EXISTS and is well-aligned
- **portfolio**: 3 paper connections — module EXISTS and is well-aligned

### Potential Gaps (paper topics without matching project implementation)

- **Backtesting & Validation** (22 papers) — consider building dedicated module
- **Event-Driven Trading** (10 papers) — consider building dedicated module
- **Factor Models & Alpha** (7 papers) — consider building dedicated module
- **Machine Learning Methods** (30 papers) — consider building dedicated module
- **Overfitting & Data Leakage** (27 papers) — consider building dedicated module
- **Portfolio Optimization** (6 papers) — consider building dedicated module
- **Reinforcement Learning** (6 papers) — consider building dedicated module
- **Risk Management** (18 papers) — consider building dedicated module
- **Sentiment Analysis & NLP** (22 papers) — consider building dedicated module
- **Time Series Forecasting** (32 papers) — consider building dedicated module

## Key Cross-Referenced Insights & Recommendations

### 1. Overfitting Prevention is the #1 Cross-Cutting Theme

27 papers address overfitting/data leakage. The project has PurgedKFold in `src/ml/`, but papers suggest additional techniques:
- **Circuit-Based Detection** (Circuit-Based Intrinsic Methods): could complement existing PurgedKFold
- **Adversarial Examples** (advrisk paper): novel detection method not yet in project
- **Training History** (5520 paper): use loss curves to detect overfit — not implemented
- **Synthetic validation** (Backtest Overfitting paper): comprehensive OOS comparison framework

### 2. Reinforcement Learning for Trading is Under-Developed

6 papers on RL for trading. Project has no `src/rl/` module.
- **OOM-RL**: Out-of-Money alignment — novel RL paradigm for agent alignment
- **Trade Execution RL**: Adaptive dual-level RL for optimal execution (VWAP tracking)
- **Deep Portfolio RL**: Direct Sharpe ratio optimization via deep learning

### 3. Sentiment/NLP Integration Gap

22 papers on sentiment/NLP. No `src/nlp/` or `src/sentiment/` module.
- Multiple Twitter sentiment papers suggest direct integration with signal pipeline
- Ensemble classifiers for tweet sentiment could feed into `src/signals/`
- Event-driven NLP (Profit Mirage, Time Travel papers) covers LLM leakage risks

### 4. Portfolio Optimization Theory Outpaces Implementation

6 papers. Project has `src/portfolio/` with Black-Litterman, but papers suggest:
- **Kelly Criterion** (Investing Is Compression): novel entropy/divergence decomposition
- **Factor-based allocation** (Emergence paper): statistical factor emergence via diffusion
- **100% Equities vs Diversified** (JPM): empirical case for diversification over concentration

### 5. Event-Driven Trading Needs More Infrastructure

10 papers. Limited event infrastructure in project.
- **Calendar of Events** (Building paper): systematic event database approach
- **Risk for Event Funds** (RIsk Management paper): specific risk framework for event-driven
- **Financial Spikes** analysis could integrate with existing pattern detection

### 6. Alpha Factor Mining — Paper Insights to Apply

7 papers. AutoAlpha evolutionary algorithm already discussed but not implemented.
- **AutoAlpha**: hierarchical evolutionary algorithm for formulaic alpha generation
- **Pair Trading Survey**: comprehensive ML/DL/RL methods for statistical arbitrage
- **Statistical Financial Factors**: diffusion process emergence — theoretical foundation

### 7. Risk Management — Strong Foundation, Can Expand

18 papers. Project has robust `src/risk/` (VaR, CVaR, circuit breakers).
- **Crash-based strategies** (behavioral finance paper): behavioral angle not yet in risk module
- **No-Arbitrage proof** (Against Universal Trading): theoretical limits for strategy evaluation

### Specific Implementation Recommendations

| Priority | Module | Recommendation |
|----------|--------|---------------|
| **HIGH** | `ml` | Add training-history-based overfitting detection (from 5520 paper) to ML pipeline |
| **HIGH** | `ml` | Implement synthetic OOS comparison framework (Backtest Overfitting paper) |
| **HIGH** | `signals` | Integrate sentiment scores as signal weight modifier (4+ sentiment papers) |
| **HIGH** | `patterns` | Add event-driven pattern category (Building Calendar paper + Event-Based Trading) |
| **MEDIUM** | `rl` | Create `src/rl/` module with trade execution environment (2 RL papers) |
| **MEDIUM** | `portfolio` | Implement Kelly Criterion allocator (Investing Is Compression paper) |
| **MEDIUM** | `ml` | Add AutoAlpha-style factor mining pipeline (AutoAlpha paper) |
| **MEDIUM** | `ml` | Implement circuit-based overfitting detection (Circuit Intrinsic Methods paper) |
| **MEDIUM** | `risk` | Add behavioral crash regime detection (Crash-based trading paper) |
| **LOW** | `ml` | Add adversarial overfitting detection (advrisk_neurips2019 paper) |
| **LOW** | `data` | Build financial event calendar database (2 event papers) |
| **LOW** | `backtest` | Add defensive backtesting with time-reversal checks (Against Universal Trading paper) |
