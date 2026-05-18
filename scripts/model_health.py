"""
Production Model Health Dashboard (B14.4).

Monitors deployed ML models for:
(a) Feature drift — KS test vs training distribution
(b) Prediction distribution stability — KL divergence
(c) Recent Sharpe vs expected — alert on degradation
(d) Training history overfitting — divergence score (C1 paper)
(e) Synthetic OOS comparison — generalization score (C1 paper)
(f) Retrain triggers — auto-flag when thresholds exceeded

Usage:
    # Full health check
    uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl --train-ref 2015-01-01 --train-ref-end 2024-12-31

    # Check specific monitoring window
    uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl --monitor-start 2025-01-01

    # JSON output for automation
    uv run scripts/model_health.py SPY --model models/pattern_classifier_v3_SPY_20260511_224704.pkl --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineering import FeatureExtractor
from src.ml.overfitting_detectors import TrainingHistoryOverfitDetector, SyntheticOOSComparator
from src.ml.adversarial_overfit import AdversarialOverfitDetector
from src.ml.pattern_classifier import PatternClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    path = Path(f"data/raw/{symbol}_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.loc[start:end]
    return df.dropna()


class ModelHealthMonitor:
    """Production health monitor for deployed ML models.

    Attributes:
        model: Loaded PatternClassifier model.
        symbol: Ticker symbol.
        train_start: Training period start.
        train_end: Training period end.
        reference_features: Feature DataFrame from training period.
        reference_probs: Probability scores from training period.
    """

    FEATURE_DRIFT_WARN = 0.3  # KS statistic warning threshold
    FEATURE_DRIFT_FAIL = 0.5  # KS statistic failure threshold
    KL_DIVERGENCE_WARN = 0.1  # KL divergence warning threshold
    KL_DIVERGENCE_FAIL = 0.3  # KL divergence failure threshold
    SHARPE_DEGRADE_WARN = 0.5  # Fraction of reference Sharpe below which to warn
    OVERFIT_DIVERGENCE_THRESHOLD = 0.7  # Training history divergence threshold (C1)
    OVERFIT_GENERALIZATION_THRESHOLD = 0.3  # Synthetic OOS generalization minimum (C1)
    ADVERSARIAL_OVERFIT_THRESHOLD = 0.30  # Adversarial overfit score threshold (C6)

    def __init__(
        self,
        model_path: str,
        symbol: str,
        train_start: str = "2015-01-01",
        train_end: str = "2024-12-31",
    ) -> None:
        self.symbol = symbol
        self.model_path = model_path
        self.train_start = train_start
        self.train_end = train_end

        logger.info("Loading model: %s", model_path)
        self.model = PatternClassifier()
        self.model.load(model_path)
        self.feature_names = list(self.model.feature_names_)

        self.reference_features = None
        self.reference_probs = None
        self._build_reference()

    def _build_reference(self) -> None:
        """Build reference distribution from training period data."""
        df = load_data(self.symbol, self.train_start, self.train_end)
        logger.info(
            "Training reference: %d bars (%s → %s)", len(df), self.train_start, self.train_end
        )

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df, include_forward_returns=False)
        features = features.ffill().bfill().fillna(0)

        missing = [c for c in self.feature_names if c not in features.columns]
        for c in missing:
            features[c] = 0.0

        self.reference_features = features[self.feature_names].dropna()
        predictions = self.model.predict(self.reference_features)
        self.reference_probs = predictions["probability_profitable"]

    def check_feature_drift(
        self, monitor_start: str, monitor_end: str | None = None
    ) -> dict[str, Any]:
        """Check feature distribution drift via two-sample KS test.

        Args:
            monitor_start: Start date for monitoring window.
            monitor_end: End date for monitoring window (default: today).

        Returns:
            Dict with drift results per feature.
        """
        end = monitor_end or datetime.now().strftime("%Y-%m-%d")
        df = load_data(self.symbol, monitor_start, end)
        logger.info("Monitoring window: %d bars (%s → %s)", len(df), monitor_start, end)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df, include_forward_returns=False)
        features = features.ffill().bfill().fillna(0)

        missing = [c for c in self.feature_names if c not in features.columns]
        for c in missing:
            features[c] = 0.0
        monitor_features = features[self.feature_names].dropna()

        from scipy.stats import ks_2samp

        drift_results = {}
        n_shifted = 0
        n_flipped = 0

        for col in self.feature_names:
            ref = self.reference_features[col].dropna().values
            mon = monitor_features[col].dropna().values

            if len(ref) < 10 or len(mon) < 10:
                drift_results[col] = {
                    "ks_stat": float("nan"),
                    "p_value": float("nan"),
                    "status": "insufficient_data",
                }
                continue

            ks_stat, p_value = ks_2samp(ref, mon)
            status = "ok"
            if ks_stat > self.FEATURE_DRIFT_FAIL:
                status = "fail"
                n_shifted += 1
            elif ks_stat > self.FEATURE_DRIFT_WARN:
                status = "warn"
                n_shifted += 1

            drift_results[col] = {
                "ks_stat": float(round(ks_stat, 4)),
                "p_value": float(round(p_value, 6)),
                "status": status,
            }

        # Check for correlation flips
        ref_probs_aligned = self.reference_probs.reindex(self.reference_features.index).dropna()
        if len(ref_probs_aligned) >= 30:
            for col in self.feature_names:
                ref_corr = (
                    self.reference_features[col]
                    .loc[ref_probs_aligned.index]
                    .corr(ref_probs_aligned)
                )
                mon_probs = self.model.predict(monitor_features)["probability_profitable"]
                mon_corr = monitor_features[col].corr(mon_probs)
                if pd.notna(ref_corr) and pd.notna(mon_corr) and ref_corr * mon_corr < 0:
                    drift_results[col]["corr_flipped"] = True
                    drift_results[col]["ref_corr"] = float(round(ref_corr, 4))
                    drift_results[col]["mon_corr"] = float(round(mon_corr, 4))
                    n_flipped += 1

        logger.info(
            "Feature drift: %d/%d shifted, %d flipped",
            n_shifted,
            len(self.feature_names),
            n_flipped,
        )
        return {
            "summary": {
                "n_features": len(self.feature_names),
                "n_shifted": n_shifted,
                "n_flipped": n_flipped,
                "n_ok": len(self.feature_names) - n_shifted,
            },
            "drift_by_feature": drift_results,
            "shifted_features": [
                k for k, v in drift_results.items() if v["status"] in ("warn", "fail")
            ],
            "flipped_features": [k for k, v in drift_results.items() if v.get("corr_flipped")],
        }

    def check_prediction_stability(
        self, monitor_start: str, monitor_end: str | None = None
    ) -> dict[str, Any]:
        """Check prediction distribution stability via KL divergence.

        Args:
            monitor_start: Start date for monitoring window.
            monitor_end: End date for monitoring window.

        Returns:
            Dict with KL divergence and stability assessment.
        """
        end = monitor_end or datetime.now().strftime("%Y-%m-%d")
        df = load_data(self.symbol, monitor_start, end)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df, include_forward_returns=False)
        features = features.ffill().bfill().fillna(0)

        missing = [c for c in self.feature_names if c not in features.columns]
        for c in missing:
            features[c] = 0.0
        monitor_features = features[self.feature_names].dropna()

        monitor_preds = self.model.predict(monitor_features)
        monitor_probs = monitor_preds["probability_profitable"].dropna()

        ref_probs = self.reference_probs.dropna()

        # Compute KL divergence between binned distributions
        bins = np.linspace(0, 1, 21)
        ref_hist, _ = np.histogram(ref_probs, bins=bins, density=True)
        mon_hist, _ = np.histogram(monitor_probs, bins=bins, density=True)

        ref_hist = ref_hist + 1e-10
        mon_hist = mon_hist + 1e-10
        kl_div = float(np.sum(ref_hist * np.log(ref_hist / mon_hist)))

        status = "ok"
        if kl_div > self.KL_DIVERGENCE_FAIL:
            status = "fail"
        elif kl_div > self.KL_DIVERGENCE_WARN:
            status = "warn"

        logger.info(
            "Prediction stability: KL divergence=%.4f, ref_mean=%.6f, mon_mean=%.6f [%s]",
            kl_div,
            ref_probs.mean(),
            monitor_probs.mean(),
            status,
        )

        return {
            "kl_divergence": round(kl_div, 4),
            "ref_mean": float(round(ref_probs.mean(), 6)),
            "mon_mean": float(round(monitor_probs.mean(), 6)),
            "ref_std": float(round(ref_probs.std(), 6)),
            "mon_std": float(round(monitor_probs.std(), 6)),
            "status": status,
        }

    def check_sharpe_vs_expected(
        self,
        monitor_start: str,
        monitor_end: str | None = None,
        benchmark_sharpe: float = 0.69,  # Baseline et=0.45 trail Sharpe
    ) -> dict[str, Any]:
        """Check recent Sharpe ratio vs expected baseline.

        Args:
            monitor_start: Start date for monitoring window.
            monitor_end: End date for monitoring window.
            benchmark_sharpe: Expected Sharpe ratio from training/backtest.

        Returns:
            Dict with recent Sharpe and alert status.
        """
        end = monitor_end or datetime.now().strftime("%Y-%m-%d")
        df = load_data(self.symbol, monitor_start, end)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df, include_forward_returns=False)
        features = features.ffill().bfill().fillna(0)

        missing = [c for c in self.feature_names if c not in features.columns]
        for c in missing:
            features[c] = 0.0
        monitor_features = features[self.feature_names].dropna()

        predictions = self.model.predict(monitor_features)
        probs = predictions["probability_profitable"]

        # Compute strategy returns from signals
        threshold = 0.45
        signals = (probs >= threshold).astype(int)
        close = df["Close"]

        # Simple long-only strategy: hold when prob >= threshold, cash otherwise
        daily_returns = close.pct_change().shift(-1)
        strategy_returns = daily_returns * signals.shift(1)
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) < 20:
            return {"sharpe": 0.0, "status": "insufficient_data", "n_days": len(strategy_returns)}

        # Annualize
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        status = "ok"
        if sharpe < benchmark_sharpe * self.SHARPE_DEGRADE_WARN:
            status = "warn"
        if sharpe < 0:
            status = "fail"

        logger.info(
            "Recent Sharpe: %.2f vs expected %.2f [%s]",
            sharpe,
            benchmark_sharpe,
            status,
        )

        return {
            "sharpe": round(sharpe, 4),
            "benchmark_sharpe": benchmark_sharpe,
            "mean_return": float(round(mean_ret, 6)),
            "std_return": float(round(std_ret, 6)),
            "n_days": len(strategy_returns),
            "status": status,
        }

    def _check_overfitting_training_history(self) -> dict[str, Any]:
        """Check for overfitting using training history divergence (C1).

        Loads training history from experiment logger or CatBoost evals_result_,
        computes divergence score, and flags if model is overfit.

        Returns:
            Dict with divergence_score, is_overfit, optimal_epoch, status.
        """
        from pathlib import Path as _Path
        from src.ml.experiment_logger import ExperimentLogger

        result = {
            "divergence_score": 0.0,
            "is_overfit": False,
            "optimal_epoch": None,
            "status": "no_training_history",
        }

        model_path = _Path(self.model_path)
        if not model_path.exists():
            result["status"] = "model_not_found"
            return result

        # Try loading training history from experiment logger
        # The experiment run directory is typically next to or derived from the model
        experiments_dir = _Path("experiments/runs")
        train_losses = []
        val_losses = []

        if experiments_dir.exists():
            for run_dir in sorted(experiments_dir.iterdir(), reverse=True):
                exp_logger = ExperimentLogger(run_id=run_dir.name)
                tl, vl = exp_logger.get_training_history()
                if tl and vl and len(tl) >= 10:
                    train_losses = tl
                    val_losses = vl
                    logger.info(
                        "Loaded training history from %s: %d epochs",
                        run_dir.name,
                        len(tl),
                    )
                    break

        if not train_losses:
            # Fallback: try to extract from CatBoost model evals_result_
            try:
                import pickle  # nosec B403

                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)  # nosec B301
                cat_model = model_data.get("model")
                if cat_model is not None and hasattr(cat_model, "evals_result_"):
                    evals = cat_model.evals_result_
                    if evals:
                        exp_logger = ExperimentLogger(run_id="health_check")
                        exp_logger.log_from_catboost(evals)
                        train_losses, val_losses = exp_logger.get_training_history()
                        logger.info("Extracted training history from CatBoost evals_result_")
            except Exception as e:
                logger.debug("Could not extract CatBoost training history: %s", e)

        if not train_losses or len(train_losses) < 10:
            return result

        detector = TrainingHistoryOverfitDetector()
        is_overfit, divergence_score, optimal_epoch = detector.is_overfit(train_losses, val_losses)

        status = "ok"
        if divergence_score > self.OVERFIT_DIVERGENCE_THRESHOLD:
            status = "fail"
        elif divergence_score > self.OVERFIT_DIVERGENCE_THRESHOLD * 0.7:
            status = "warn"

        logger.info(
            "Training history overfit: divergence=%.3f, is_overfit=%s, optimal_epoch=%s [%s]",
            divergence_score,
            is_overfit,
            optimal_epoch,
            status,
        )

        return {
            "divergence_score": float(round(divergence_score, 4)),
            "is_overfit": is_overfit,
            "optimal_epoch": optimal_epoch,
            "n_epochs": len(train_losses),
            "final_train_loss": float(round(train_losses[-1], 6)) if train_losses else None,
            "final_val_loss": float(round(val_losses[-1], 6)) if val_losses else None,
            "status": status,
        }

    def _check_synthetic_oos(
        self,
        benchmark_sharpe: float = 0.69,
        sharpe_check: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check generalization via synthetic OOS comparison (C1).

        Generates synthetic null distribution of returns and compares
        real OOS performance to compute generalization score and PBO.

        Args:
            benchmark_sharpe: Expected IS Sharpe from training/backtest.
            sharpe_check: Pre-computed sharpe_check result (avoids double computation).

        Returns:
            Dict with pbo, generalization_score, is_oos_ratio, status.
        """
        if sharpe_check is None:
            sharpe_check = self.check_sharpe_vs_expected(
                monitor_start="2025-01-01",
                benchmark_sharpe=benchmark_sharpe,
            )
        real_oos_sharpe = sharpe_check.get("sharpe", 0.0)
        n_oos_days = sharpe_check.get("n_days", 252)

        comparator = SyntheticOOSComparator(n_synthetic_runs=100, seed=42)
        result = comparator.compute_generalization_score(
            real_oos_sharpe=real_oos_sharpe,
            real_is_sharpe=benchmark_sharpe,
            n_oos_days=max(n_oos_days, 60),
        )

        generalization_score = result["generalization_score"]
        status = "ok"
        if generalization_score < self.OVERFIT_GENERALIZATION_THRESHOLD:
            status = "fail"
        elif generalization_score < 0.5:
            status = "warn"

        logger.info(
            "Synthetic OOS: PBO=%.3f, generalization=%.3f, IS/OOS ratio=%.1f [%s]",
            result["pbo"],
            generalization_score,
            result["is_oos_ratio"],
            status,
        )

        return {
            "pbo": round(result["pbo"], 4),
            "generalization_score": round(generalization_score, 4),
            "is_oos_ratio": round(result["is_oos_ratio"], 2)
            if result["is_oos_ratio"] != float("inf")
            else 999.0,
            "synthetic_median_sharpe": round(result["synthetic_median_sharpe"], 4),
            "o_o_s_sharpe": real_oos_sharpe,
            "is_sharpe": benchmark_sharpe,
            "is_better_than_random": result["is_better_than_random"],
            "n_synthetic_runs": result["n_synthetic_runs"],
            "status": status,
        }

    def _check_adversarial_overfit(
        self,
        monitor_start: str = "2025-01-01",
        monitor_end: str | None = None,
    ) -> dict[str, Any]:
        """Check for overfitting via adversarial perturbation (C6).

        Injects feature noise and measures prediction stability.
        Overfit models degrade more under perturbation because they
        rely on brittle decision boundaries.

        Returns:
            Dict with overfit_score, gap_score, perturbation_stability, status.
        """
        end = monitor_end or datetime.now().strftime("%Y-%m-%d")
        df = load_data(self.symbol, monitor_start, end)

        extractor = FeatureExtractor()
        features = extractor.extract_all_features(df, include_forward_returns=False)
        features = features.ffill().bfill().fillna(0)

        missing = [c for c in self.feature_names if c not in features.columns]
        for c in missing:
            features[c] = 0.0
        monitor_features = features[self.feature_names].dropna()
        predictions = self.model.predict(monitor_features)["probability_profitable"]

        detector = AdversarialOverfitDetector(
            noise_scale=0.05,
            perturb_fraction=0.30,
            gap_threshold=self.ADVERSARIAL_OVERFIT_THRESHOLD,
            min_samples=100,
            seed=42,
        )
        result = detector.analyze(
            features=monitor_features,
            predictions=predictions,
        )

        status = "ok"
        if result.is_overfit:
            status = "fail"
        elif result.gap_score > self.ADVERSARIAL_OVERFIT_THRESHOLD * 0.5:
            status = "warn"

        logger.info(
            "Adversarial overfit: score=%.3f, gap=%.3f, pert_stability=%.3f [%s]",
            result.overfit_score,
            result.gap_score,
            result.perturbation_stability,
            status,
        )

        return {
            "overfit_score": result.overfit_score,
            "gap_score": result.gap_score,
            "perturbation_stability": result.perturbation_stability,
            "label_stability": result.label_stability,
            "bernstein_pvalue": result.bernstein_pvalue,
            "risk_level": result.risk_level,
            "is_overfit": result.is_overfit,
            "interpretation": result.interpretation,
            "status": status,
        }

    def full_check(
        self,
        monitor_start: str = "2025-01-01",
        monitor_end: str | None = None,
        benchmark_sharpe: float = 0.69,
    ) -> dict[str, Any]:
        """Run full model health check.

        Returns:
            Dict with all health metrics and retrain recommendation.
        """
        end = monitor_end or datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 60)
        logger.info(f"Model Health Check: {self.symbol} ({monitor_start} → {end})")
        logger.info("=" * 60)

        feature_drift = self.check_feature_drift(monitor_start, monitor_end)
        prediction_stability = self.check_prediction_stability(monitor_start, monitor_end)
        sharpe_check = self.check_sharpe_vs_expected(monitor_start, monitor_end, benchmark_sharpe)
        training_overfit = self._check_overfitting_training_history()
        synthetic_oos = self._check_synthetic_oos(benchmark_sharpe, sharpe_check)
        adversarial_overfit = self._check_adversarial_overfit(monitor_start, monitor_end)

        # Retrain recommendation
        retrain_triggers = []
        if feature_drift["summary"]["n_shifted"] >= len(self.feature_names) * 0.25:
            retrain_triggers.append("feature_drift")
        if feature_drift["summary"]["n_flipped"] > 0:
            retrain_triggers.append("correlation_flip")
        if prediction_stability["status"] == "fail":
            retrain_triggers.append("prediction_instability")
        if sharpe_check["status"] == "fail":
            retrain_triggers.append("sharpe_degradation")
        if training_overfit["status"] == "fail":
            retrain_triggers.append("training_history_overfit")
        if synthetic_oos["status"] == "fail":
            retrain_triggers.append("synthetic_o_o_s_failure")
        if adversarial_overfit["status"] == "fail":
            retrain_triggers.append("adversarial_overfit")

        retrain = len(retrain_triggers) > 0

        report = {
            "symbol": self.symbol,
            "monitor_window": {"start": monitor_start, "end": end},
            "timestamp": datetime.now().isoformat(),
            "retrain_recommended": retrain,
            "retrain_triggers": retrain_triggers,
            "feature_drift": feature_drift,
            "prediction_stability": prediction_stability,
            "sharpe_check": sharpe_check,
            "training_history_overfit": training_overfit,
            "synthetic_o_o_s": synthetic_oos,
            "adversarial_overfit": adversarial_overfit,
        }

        logger.info("=" * 60)
        logger.info("Health Check Summary")
        logger.info("=" * 60)
        logger.info(
            "Feature drift: %s",
            "OK"
            if feature_drift["summary"]["n_shifted"] == 0
            else f"{feature_drift['summary']['n_shifted']} shifted",
        )
        logger.info("Prediction stability: %s", prediction_stability["status"].upper())
        logger.info("Sharpe: %.2f [%s]", sharpe_check["sharpe"], sharpe_check["status"].upper())
        logger.info(
            "Training overfit: %.3f [%s]",
            training_overfit.get("divergence_score", 0),
            training_overfit["status"].upper(),
        )
        logger.info(
            "Synthetic OOS: generalization=%.3f [%s]",
            synthetic_oos.get("generalization_score", 1.0),
            synthetic_oos["status"].upper(),
        )
        logger.info(
            "Adversarial overfit: score=%.3f [%s]",
            adversarial_overfit.get("overfit_score", 0.0),
            adversarial_overfit["status"].upper(),
        )
        logger.info("Retrain: %s", "RECOMMENDED" if retrain else "not needed")
        if retrain_triggers:
            logger.info("Triggers: %s", ", ".join(retrain_triggers))

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Production Model Health Dashboard (B14.4)")
    parser.add_argument("symbol", type=str, help="Ticker symbol")
    parser.add_argument(
        "--model", type=str, default="models/pattern_classifier_v3_SPY_20260511_224704.pkl"
    )
    parser.add_argument(
        "--train-ref",
        type=str,
        default="2015-01-01",
        help="Training start for reference distribution",
    )
    parser.add_argument(
        "--train-ref-end",
        type=str,
        default="2024-12-31",
        help="Training end for reference distribution",
    )
    parser.add_argument(
        "--monitor-start", type=str, default="2025-01-01", help="Monitoring window start"
    )
    parser.add_argument(
        "--monitor-end", type=str, default=None, help="Monitoring window end (default: today)"
    )
    parser.add_argument(
        "--benchmark-sharpe", type=float, default=0.69, help="Expected Sharpe ratio"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON file")

    args = parser.parse_args()

    monitor = ModelHealthMonitor(
        model_path=args.model,
        symbol=args.symbol,
        train_start=args.train_ref,
        train_end=args.train_ref_end,
    )

    report = monitor.full_check(
        monitor_start=args.monitor_start,
        monitor_end=args.monitor_end,
        benchmark_sharpe=args.benchmark_sharpe,
    )

    if args.json or args.output:
        json_str = json.dumps(report, indent=2, default=str)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(json_str)
            logger.info("Report saved to %s", args.output)
        if args.json:
            print(json_str)

    # Exit code for automation
    if report["retrain_recommended"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
