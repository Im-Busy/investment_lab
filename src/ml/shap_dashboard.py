"""
SHAP Explainability Dashboard.

Provides model-agnostic interpretability using SHAP (SHapley Additive
exPlanations). Supports waterfall, beeswarm, force, and bar plots for
any sklearn-compatible tree-based, linear, or gradient boosting model.

Designed for CLI/scripting environments: generates PNG figures, HTML
interactive reports, and console-friendly text summaries.

Usage:
    from src.ml.shap_dashboard import SHAPDashboard

    dashboard = SHAPDashboard(model, X_train)
    dashboard.waterfall(X_test.iloc[0])       # explain single prediction
    dashboard.beeswarm(X_test[:500])          # global feature importance
    dashboard.force_plot(X_test.iloc[0])      # interactive force plot
    dashboard.generate_report(X_test[:200])   # full HTML report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureExplanation:
    """SHAP explanation for a single prediction.

    Attributes:
        feature: Feature name.
        value: Feature value for this sample.
        shap_value: SHAP contribution (positive = pushes prediction up).
        contribution_pct: Percentage contribution to total prediction change.
    """

    feature: str
    value: float
    shap_value: float
    contribution_pct: float


@dataclass
class PredictionExplanation:
    """Complete SHAP explanation for a single prediction.

    Attributes:
        base_value: Expected model output (baseline).
        prediction: Final model prediction.
        features: List of FeatureExplanation sorted by abs(shap_value) descending.
        top_positive: Features pushing prediction up.
        top_negative: Features pushing prediction down.
    """

    base_value: float
    prediction: float
    features: List[FeatureExplanation]
    top_positive: List[FeatureExplanation] = field(default_factory=list)
    top_negative: List[FeatureExplanation] = field(default_factory=list)

    def summary(self, top_n: int = 10) -> str:
        """Text summary of the explanation."""
        lines = [
            f"Base value: {self.base_value:.6f}",
            f"Prediction: {self.prediction:.6f}",
            f"--- Top {top_n} features ---",
        ]
        for f in self.features[:top_n]:
            direction = "+" if f.shap_value >= 0 else ""
            lines.append(f"  {direction}{f.shap_value:.6f}  {f.feature} = {f.value:.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class GlobalImportance:
    """Global SHAP feature importance summary.

    Attributes:
        features: List of (feature_name, mean_abs_shap) sorted descending.
        top_features: Top N feature names.
        mean_abs_values: Mean absolute SHAP values.
    """

    features: List[Tuple[str, float]]
    top_features: List[str]
    mean_abs_values: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.features, columns=["feature", "mean_abs_shap"]).set_index(
            "feature"
        )

    def summary(self, top_n: int = 15) -> str:
        """Text summary of global importance."""
        lines = ["--- Global Feature Importance (mean |SHAP|) ---"]
        for feat, imp in self.features[:top_n]:
            lines.append(f"  {imp:.6f}  {feat}")
        return "\n".join(lines)


class SHAPDashboard:
    """SHAP-based model explainability dashboard.

    Works with any sklearn-compatible model. Auto-detects the best explainer:
      - TreeExplainer for CatBoost, LightGBM, XGBoost, Random Forest
      - LinearExplainer for linear/logistic regression
      - KernelExplainer for black-box models (slower)

    All plots save to the specified output directory. Supports PNG figures
    and interactive HTML where applicable.

    Example:
        >>> dashboard = SHAPDashboard(model, X_train)
        >>> dashboard.beeswarm(X_test[:500], save_path="beeswarm.png")
        >>> explanation = dashboard.explain(X_test.iloc[0])
        >>> print(explanation.summary())
        >>> dashboard.generate_report(X_test[:200], "report/")
    """

    def __init__(
        self,
        model: Any,
        X_background: pd.DataFrame | np.ndarray,
        feature_names: Optional[List[str]] = None,
        output_dir: str | Path = "outputs/shap",
        is_classifier: Optional[bool] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize SHAP dashboard.

        Args:
            model: Trained sklearn-compatible model (must have .predict or .predict_proba).
            X_background: Background/reference data for SHAP (50-500 samples from training).
            feature_names: Feature column names.
            output_dir: Directory for saving plots.
            is_classifier: Auto-detected if None. Set True/False to override.
            class_names: Class names for classification labels in plots.
        """

        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(X_background, pd.DataFrame):
            self.feature_names = list(X_background.columns)
            X_bg = X_background.values.astype(np.float64)
        else:
            self.feature_names = feature_names or [
                f"feature_{i}" for i in range(X_background.shape[1])
            ]
            X_bg = np.asarray(X_background, dtype=np.float64)

        if is_classifier is None:
            is_classifier = hasattr(model, "predict_proba")
        self.is_classifier = is_classifier
        self.class_names = class_names

        self._explainer = self._create_explainer(model, X_bg)

    def _create_explainer(self, model: Any, X_bg: np.ndarray) -> Any:
        """Auto-select and create the appropriate SHAP explainer."""
        import shap

        model_type = type(model).__name__.lower()

        is_tree = any(
            t in model_type
            for t in (
                "catboost",
                "lgbm",
                "lightgbm",
                "xgb",
                "xgboost",
                "randomforest",
                "gradientboosting",
                "extratrees",
                "tree",
                "decisiontree",
            )
        )

        is_linear = any(
            t in model_type for t in ("linear", "logistic", "ridge", "lasso", "elasticnet")
        )

        try:
            if is_tree:
                return shap.TreeExplainer(model, feature_perturbation="interventional")
            elif is_linear:
                return shap.LinearExplainer(model, X_bg)
            else:
                # Auto-detect
                return shap.Explainer(model, X_bg)
        except Exception:
            # Fallback to auto-explainer
            return shap.Explainer(model, X_bg)

    def _preprocess_X(self, X: pd.DataFrame | np.ndarray | pd.Series) -> np.ndarray:
        """Convert X to numpy array."""
        if isinstance(X, pd.Series):
            return X.values.astype(np.float64).reshape(1, -1)
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, -1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def compute_shap(self, X: pd.DataFrame | np.ndarray) -> Any:
        """Compute SHAP values for dataset.

        Args:
            X: Feature matrix.

        Returns:
            SHAP Explanation object.
        """
        X_arr = self._preprocess_X(X)
        if hasattr(self._explainer, "shap_values"):
            return self._explainer.shap_values(X_arr)
        else:
            return self._explainer(X_arr)

    def explain(
        self,
        x: pd.Series | np.ndarray,
        top_n: int = 20,
    ) -> PredictionExplanation:
        """Explain a single prediction.

        Args:
            x: Single sample features.
            top_n: Number of top features to include.

        Returns:
            PredictionExplanation with feature-level breakdown.
        """
        X_arr = self._preprocess_X(x)

        shap_vals = self._compute_values(X_arr)
        base_val = self._get_base_value()

        if self.is_classifier:
            if shap_vals.ndim == 3:
                # Multi-class: use predicted class
                pred = self.model.predict(X_arr)[0]
                vals = shap_vals[0, :, pred] if pred < shap_vals.shape[2] else shap_vals[0, :, 0]
            elif shap_vals.ndim == 2 and shap_vals.shape[1] > 1:
                pred = self.model.predict(X_arr)[0]
                vals = shap_vals[0] if shap_vals.shape[0] > 1 else shap_vals[0, :]
            else:
                vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
        else:
            vals = shap_vals[0] if shap_vals.ndim >= 2 else shap_vals

        feature_vals = X_arr[0]
        n_features = len(feature_vals)

        total_abs = np.abs(vals).sum()
        explanations = []
        for i in range(min(n_features, len(vals))):
            explanations.append(
                FeatureExplanation(
                    feature=(self.feature_names[i] if i < len(self.feature_names) else f"f{i}"),
                    value=float(feature_vals[i]),
                    shap_value=float(vals[i]),
                    contribution_pct=(float(abs(vals[i]) / max(total_abs, 1e-8) * 100)),
                )
            )

        explanations.sort(key=lambda e: abs(e.shap_value), reverse=True)
        top_positive = [e for e in explanations if e.shap_value > 0][:5]
        top_negative = [e for e in explanations if e.shap_value < 0][:5]

        return PredictionExplanation(
            base_value=float(base_val),
            prediction=float(self.model.predict(X_arr)[0]),
            features=explanations[:top_n],
            top_positive=top_positive,
            top_negative=top_negative,
        )

    def global_importance(
        self,
        X: pd.DataFrame | np.ndarray,
        top_n: int = 20,
    ) -> GlobalImportance:
        """Compute global feature importance from SHAP values.

        Args:
            X: Feature matrix.
            top_n: Number of top features.

        Returns:
            GlobalImportance with ranked features.
        """
        shap_vals = self._compute_values(self._preprocess_X(X))

        if shap_vals.ndim == 3:
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
        elif shap_vals.ndim == 2:
            mean_abs = np.abs(shap_vals).mean(axis=0)
        else:
            mean_abs = np.abs(shap_vals)

        features = sorted(
            [
                (
                    self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    float(mean_abs[i]),
                )
                for i in range(len(mean_abs))
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return GlobalImportance(
            features=features,
            top_features=[f[0] for f in features[:top_n]],
            mean_abs_values=mean_abs,
        )

    def waterfall(
        self,
        x: pd.Series | np.ndarray,
        save_path: Optional[str | Path] = None,
        max_display: int = 20,
        show: bool = False,
    ) -> Optional[Path]:
        """Generate waterfall plot for a single prediction.

        Shows how each feature contributes from the base value to the final
        prediction.

        Args:
            x: Single sample features.
            save_path: File path to save plot (PNG). Auto-generated if None.
            max_display: Maximum features to display.
            show: Whether to display the plot interactively.

        Returns:
            Path to saved file, or None if show=True and not saved.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        X_arr = self._preprocess_X(x)
        sv = self._explainer(X_arr)

        if hasattr(sv, "__getitem__"):
            sv_sample = sv[0]
        else:
            sv_sample = sv

        plt.figure(figsize=(10, max(6, max_display * 0.3)))
        shap.plots.waterfall(sv_sample, max_display=max_display, show=False)
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "waterfall.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if show:
            print(f"Waterfall plot saved to: {save_path}")
            return save_path

        return save_path

    def beeswarm(
        self,
        X: pd.DataFrame | np.ndarray,
        save_path: Optional[str | Path] = None,
        max_display: int = 20,
        show: bool = False,
    ) -> Optional[Path]:
        """Generate beeswarm summary plot.

        Shows feature importance and the distribution of SHAP values across
        all samples. Each point is one sample; color represents feature value.

        Args:
            X: Feature matrix (limit to ~500 samples for readability).
            save_path: File path. Auto-generated if None.
            max_display: Maximum features to display.
            show: Whether to display the plot interactively.

        Returns:
            Path to saved file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        X_arr = self._preprocess_X(X)
        sv = self._explainer(X_arr)

        plt.figure(figsize=(10, max(6, max_display * 0.3)))
        shap.plots.beeswarm(sv, max_display=max_display, show=False)
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "beeswarm.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if show:
            print(f"Beeswarm plot saved to: {save_path}")
            return save_path

        return save_path

    def bar_plot(
        self,
        X: pd.DataFrame | np.ndarray,
        save_path: Optional[str | Path] = None,
        max_display: int = 20,
        show: bool = False,
    ) -> Optional[Path]:
        """Generate bar plot of global feature importance.

        Args:
            X: Feature matrix.
            save_path: File path. Auto-generated if None.
            max_display: Maximum features to display.
            show: Whether to display the plot interactively.

        Returns:
            Path to saved file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        X_arr = self._preprocess_X(X)
        sv = self._explainer(X_arr)

        plt.figure(figsize=(8, max(5, max_display * 0.25)))
        shap.plots.bar(sv, max_display=max_display, show=False)
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "bar_importance.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if show:
            print(f"Bar plot saved to: {save_path}")
            return save_path

        return save_path

    def scatter_plot(
        self,
        X: pd.DataFrame | np.ndarray,
        feature: str,
        color_feature: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Generate scatter plot of SHAP values vs feature value.

        Useful for detecting nonlinear relationships and interactions.

        Args:
            X: Feature matrix.
            feature: Feature name to plot on x-axis.
            color_feature: Feature to use for coloring points (detects interactions).
            save_path: File path. Auto-generated if None.
            show: Whether to display the plot interactively.

        Returns:
            Path to saved file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        X_arr = self._preprocess_X(X)
        sv = self._explainer(X_arr)

        plt.figure(figsize=(8, 6))
        if color_feature:
            shap.plots.scatter(sv[:, feature], color=sv[:, color_feature], show=False)
        else:
            shap.plots.scatter(sv[:, feature], show=False)
        plt.tight_layout()

        if save_path is None:
            safe_name = feature.replace("/", "_").replace(" ", "_")
            save_path = self.output_dir / f"scatter_{safe_name}.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if show:
            print(f"Scatter plot saved to: {save_path}")
            return save_path

        return save_path

    def force_plot(
        self,
        x: pd.Series | np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> Optional[Path]:
        """Generate interactive force plot as HTML.

        Args:
            x: Single sample features.
            save_path: HTML file path. Auto-generated if None.

        Returns:
            Path to saved HTML file.
        """
        import shap

        X_arr = self._preprocess_X(x)
        sv = self._explainer(X_arr)

        if save_path is None:
            save_path = self.output_dir / "force_plot.html"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(sv, "__getitem__"):
            fig = shap.plots.force(sv[0], matplotlib=False, show=False)
        else:
            fig = shap.plots.force(sv, matplotlib=False, show=False)

        shap.save_html(str(save_path), fig)
        return save_path

    def heatmap(
        self,
        X: pd.DataFrame | np.ndarray,
        save_path: Optional[str | Path] = None,
        max_display: int = 20,
        show: bool = False,
    ) -> Optional[Path]:
        """Generate heatmap of SHAP values across samples.

        Shows how each feature contributes to each sample's prediction.

        Args:
            X: Feature matrix.
            save_path: File path. Auto-generated if None.
            max_display: Maximum features to display.
            show: Whether to display the plot interactively.

        Returns:
            Path to saved file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        X_arr = self._preprocess_X(X)
        sv = self._explainer(X_arr)

        plt.figure(figsize=(12, max(6, max_display * 0.3)))
        shap.plots.heatmap(sv, max_display=max_display, show=False)
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "heatmap.png"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if show:
            print(f"Heatmap saved to: {save_path}")
            return save_path

        return save_path

    def generate_report(
        self,
        X: pd.DataFrame | np.ndarray,
        output_dir: Optional[str | Path] = None,
        n_samples_to_explain: int = 3,
        max_display: int = 20,
    ) -> Path:
        """Generate a comprehensive SHAP report.

        Creates:
          1. Beeswarm summary plot
          2. Bar importance plot
          3. Heatmap
          4. Individual waterfall plots for top predictions
          5. Text summary of global importance
          6. Text summary of individual explanations

        Args:
            X: Feature matrix for analysis.
            output_dir: Directory for report files. Defaults to output_dir/shaq_report/.
            n_samples_to_explain: Number of individual predictions to explain.
            max_display: Maximum features in plots.

        Returns:
            Path to report directory.
        """
        report_dir = Path(output_dir or self.output_dir / "report")
        report_dir.mkdir(parents=True, exist_ok=True)

        X_arr = self._preprocess_X(X)
        n_samples = min(len(X_arr), 500)

        print(f"Generating SHAP report to: {report_dir}")
        print(f"  Samples analyzed: {n_samples}")

        # 1. Beeswarm
        print("  [1/5] Generating beeswarm plot...")
        self.beeswarm(
            X_arr[:n_samples],
            save_path=report_dir / "01_beeswarm.png",
            max_display=max_display,
            show=True,
        )

        # 2. Bar importance
        print("  [2/5] Generating bar importance plot...")
        self.bar_plot(
            X_arr[:n_samples],
            save_path=report_dir / "02_bar_importance.png",
            max_display=max_display,
            show=True,
        )

        # 3. Heatmap
        print("  [3/5] Generating heatmap...")
        self.heatmap(
            X_arr[: min(n_samples, 100)],
            save_path=report_dir / "03_heatmap.png",
            max_display=max_display,
            show=True,
        )

        # 4. Global importance text
        print("  [4/5] Computing global importance...")
        importance = self.global_importance(X_arr[:n_samples])
        imp_text = importance.summary(top_n=max_display)
        (report_dir / "04_global_importance.txt").write_text(imp_text)
        importance.to_dataframe().to_csv(report_dir / "04_feature_importance.csv")

        # 5. Individual explanations
        print(f"  [5/5] Generating {n_samples_to_explain} individual explanations...")
        for i in range(min(n_samples_to_explain, n_samples)):
            x = pd.Series(X_arr[i], index=self.feature_names) if self.feature_names else X_arr[i]
            explanation = self.explain(x)

            # Save waterfall plot
            self.waterfall(
                x,
                save_path=report_dir / f"05_waterfall_{i}.png",
                max_display=max_display,
            )

            # Save text explanation
            (report_dir / f"05_explanation_{i}.txt").write_text(
                explanation.summary(top_n=max_display)
            )

            # Force plot HTML for first sample
            if i == 0:
                self.force_plot(x, save_path=report_dir / "05_force_plot.html")

        # Summary markdown
        md_lines = [
            "# SHAP Model Explanation Report",
            "",
            f"**Model type:** {type(self.model).__name__}",
            f"**Classification:** {self.is_classifier}",
            f"**Samples analyzed:** {n_samples}",
            f"**Features:** {len(self.feature_names)}",
            "",
            "## Global Feature Importance (Top 10)",
            "",
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|-------------|",
        ]
        for rank, (feat, imp) in enumerate(importance.features[:10], 1):
            md_lines.append(f"| {rank} | {feat} | {imp:.6f} |")

        md_lines.extend(
            [
                "",
                "## Plots",
                "",
                "- `01_beeswarm.png` — Feature importance with value distributions",
                "- `02_bar_importance.png` — Clean feature importance summary",
                "- `03_heatmap.png` — SHAP values across all samples",
                "- `04_global_importance.txt` — Text summary of importance",
                "- `04_feature_importance.csv` — CSV of feature importance",
                "- `05_waterfall_*.png` — Individual prediction breakdowns",
                "- `05_explanation_*.txt` — Text explanations per prediction",
            ]
        )

        (report_dir / "README.md").write_text("\n".join(md_lines))

        print(f"\nReport complete: {report_dir}")
        return report_dir

    def text_summary(
        self,
        X: pd.DataFrame | np.ndarray,
        top_n: int = 15,
    ) -> str:
        """Generate a console-friendly text summary.

        Args:
            X: Feature matrix.
            top_n: Number of top features to display.

        Returns:
            Multi-line string with importance summary.
        """
        import time

        t0 = time.perf_counter()
        importance = self.global_importance(X, top_n=top_n)
        elapsed = time.perf_counter() - t0

        lines = [
            f"SHAP Analysis Summary ({len(X)} samples, {elapsed:.1f}s)",
            f"Model: {type(self.model).__name__}",
            f"Classifier: {self.is_classifier}",
            "",
            importance.summary(top_n=top_n),
        ]
        return "\n".join(lines)

    def explain_errors(
        self,
        X: pd.DataFrame | np.ndarray,
        y_true: np.ndarray,
        n_errors: int = 5,
        output_dir: Optional[str | Path] = None,
    ) -> List[PredictionExplanation]:
        """Explain misclassified/mis-predicted samples.

        Helps debug model errors by showing what features contributed to
        incorrect predictions.

        Args:
            X: Feature matrix.
            y_true: True labels.
            n_errors: Number of error samples to explain.
            output_dir: Directory for waterfall plots. Auto if None.

        Returns:
            List of PredictionExplanation for error samples.
        """
        X_arr = self._preprocess_X(X)
        y_true_arr = np.asarray(y_true)
        y_pred = self.model.predict(X_arr)

        errors = y_pred != y_true_arr
        error_indices = np.where(errors)[0]

        if len(error_indices) == 0:
            print("No errors found!")
            return []

        n_show = min(n_errors, len(error_indices))
        out_dir = Path(output_dir or self.output_dir / "errors")
        out_dir.mkdir(parents=True, exist_ok=True)

        explanations = []
        for i in range(n_show):
            idx = error_indices[i]
            x = X_arr[idx]
            explanation = self.explain(x)

            print(f"\n--- Error {i + 1}: idx={idx} ---")
            print(f"  True: {y_true_arr[idx]}, Predicted: {y_pred[idx]}")
            print(explanation.summary(top_n=10))

            self.waterfall(
                x,
                save_path=out_dir / f"error_{i}_idx{idx}.png",
            )
            explanations.append(explanation)

        return explanations

    def compare_features(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_pairs: List[Tuple[str, str]],
        output_dir: Optional[str | Path] = None,
    ) -> None:
        """Generate interaction scatter plots for feature pairs.

        Args:
            X: Feature matrix.
            feature_pairs: List of (feature_name, color_feature_name) tuples.
            output_dir: Directory for plots.
        """
        out_dir = Path(output_dir or self.output_dir / "interactions")
        out_dir.mkdir(parents=True, exist_ok=True)

        for f1, f2 in feature_pairs:
            print(f"  Plotting {f1} colored by {f2}...")
            self.scatter_plot(
                X,
                feature=f1,
                color_feature=f2,
                save_path=out_dir / f"scatter_{f1}_by_{f2}.png",
                show=True,
            )

    def _compute_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values as numpy array."""
        sv = self._explainer(X)
        if hasattr(sv, "values"):
            return np.asarray(sv.values)
        return np.asarray(sv)

    def _get_base_value(self) -> float:
        """Get expected/base value from explainer."""
        if hasattr(self._explainer, "expected_value"):
            ev = self._explainer.expected_value
            if isinstance(ev, np.ndarray):
                return float(ev[0]) if ev.size > 0 else float(ev)
            return float(ev)
        return 0.0

    def __repr__(self) -> str:
        return (
            f"SHAPDashboard(model={type(self.model).__name__}, "
            f"features={len(self.feature_names)}, "
            f"explainer={type(self._explainer).__name__})"
        )


def quick_shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    output_dir: str | Path = "outputs/shap",
    max_samples: int = 500,
) -> SHAPDashboard:
    """Quick one-liner: generate full SHAP analysis.

    Args:
        model: Trained sklearn-compatible model.
        X_test: Test features.
        output_dir: Output directory for plots.
        max_samples: Maximum samples for analysis.

    Returns:
        Configured SHAPDashboard instance.
    """
    n_samples = min(len(X_test), max_samples)
    dashboard = SHAPDashboard(
        model=model,
        X_background=X_test.iloc[: min(100, n_samples)],
        output_dir=output_dir,
    )
    dashboard.generate_report(X_test.iloc[:n_samples])
    return dashboard
