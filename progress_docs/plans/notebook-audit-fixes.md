---
type: enhancement
name: "Notebook Audit Fixes"
status: in_progress
started: 2026-05-09
summary: |
  Comprehensive notebook execution audit revealed critical data extraction bugs
  masking true strategy performance. The pattern detection system is not broken —
  backtest results are simply being read with wrong dictionary keys.
root_causes:
  - "AblationEngine._run_backtest() reads flat keys from nested dict (ablation_engine.py:149-163)"
  - "SynergyAnalyzer._run_backtest() has identical bug (synergy_analyzer.py:151-165)"
  - "PatternSelector.__init__ never populates all_patterns (pattern_selector.py:223-224)"
  - "Runner.get_results() returns {'stats': {...}} but callers read top-level keys"
  - "Signal scorer generates only 1 trade in 10 years of SPY data"
  - "Regime classifier has 29% train-test accuracy gap"
priorities:
  - tier: 1
    tasks:
      - id: F1
        desc: "Fix metric extraction in AblationEngine._run_backtest() and SynergyAnalyzer._run_backtest()"
        files: ["src/analysis/ablation_engine.py", "src/analysis/synergy_analyzer.py"]
        impact: "Unblocks 34 solo pattern backtests — reveals which patterns actually generate edge"
      - id: F2
        desc: "Add pattern discovery to PatternSelector.__init__ so all_patterns is populated"
        files: ["src/analysis/pattern_selector.py"]
        impact: "Unblocks pattern selection framework (notebook 07)"
  - tier: 2
    tasks:
      - id: F3
        desc: "Re-run notebooks 06, 07 after F1/F2 fixes to get real results"
        impact: "First valid ablation study and pattern ranking"
      - id: ML1
        desc: "Fix signal scorer to generate >=10 trades for ML training"
        files: ["src/ml/pattern_scorer.py"]
        impact: "Enables ML signal quality scoring"
      - id: ML2
        desc: "Reduce regime classifier overfit (86.65% train → 57.62% test)"
        files: ["src/ml/ebm_classifier.py"]
        impact: "More reliable regime detection for backtest filtering"
  - tier: 3
    tasks:
      - id: S1
        desc: "Focus on SMA_Cross as primary strategy backbone"
      - id: ML3
        desc: "Increase feature count for LightGBM model beyond 25"
      - id: ML4
        desc: "Run meta-labeling on SPY/QQQ (not just JOE)"
      - id: S2
        desc: "Implement Triple Barrier Labeling (FS4 already done, integrate)"
      - id: S3
        desc: "Ensemble regime consensus instead of picking one detector"
      - id: S4
        desc: "Test on QQQ/BTC where patterns show better results"
---
