---
description: Run the full model diagnostic suite — calibration audit, regime shift investigation, and WFO comparison. Produces a health report with root cause analysis and actionable recommendations.
---

# Model Diagnose — Full Health Check

Load the **model-doctor** agent and run the complete diagnostic suite on the current pattern classifier model.

The agent will:
1. Run calibration audit (reliability diagram with triple-barrier labels)
2. Run regime shift investigation (KS tests on 57 features IS vs OOS)
3. Run WFO comparison (single-split vs walk-forward optimization)
4. Interpret all results against known baselines
5. Produce a health report: healthy, degraded, or broken
6. Recommend specific remediation from the playbook

## Usage

```
/model-diagnose
```

No arguments needed. Auto-detects the default model at `models/pattern_classifier_v3_SPY_20260511_224704.pkl`.
