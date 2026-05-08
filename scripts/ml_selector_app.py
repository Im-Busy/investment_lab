"""
ML Model Selector Streamlit App

Interactive web UI for model selection, training, and comparison.

Run: uv run streamlit run scripts/ml_selector_app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Model Selector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_selector import (
    ModelSelector,
    Recommendation,
    ModelResult,
)

st.title("🤖 ML Model Selector")
st.markdown("Automated model selection and evaluation for trading strategies")


def load_sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample SPY-like data for demonstration."""
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples),
            "rsi_14": np.random.uniform(0, 100, n_samples),
            "macd_signal": np.random.randn(n_samples),
            "bb_width": np.random.uniform(0, 5, n_samples),
            "volume_ratio": np.random.exponential(1, n_samples),
            "price_momentum": np.random.randn(n_samples),
            "volatility": np.random.exponential(0.5, n_samples),
        },
        index=pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
    )

    y = pd.Series(
        np.random.randint(0, 2, n_samples),
        index=X.index,
        name="target",
    )

    return X, y


def get_recommendation(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    priority: str,
    model_type: str | None,
    validation_type: str | None,
) -> Recommendation:
    """Get model and validation recommendation."""
    selector = ModelSelector()

    with st.spinner("Analyzing dataset characteristics..."):
        recommendation = selector.recommend(
            X=X,
            y=y,
            task=task,
            priority=priority,
            model_type=model_type,
            validation_type=validation_type,
            is_time_series=X.index.is_monotonic_increasing,
        )

    return recommendation


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    recommendation: Recommendation,
) -> ModelResult:
    """Train model with progress tracking."""
    selector = ModelSelector()

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Training model...")
    progress_bar.progress(20)

    result = selector.train_and_evaluate(
        X=X,
        y=y,
        model_config=recommendation.model,
        validation_config=recommendation.validation,
    )

    progress_bar.progress(100)
    status_text.text("Training complete!")

    return result


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    validation_type: str | None,
) -> list[ModelResult]:
    """Compare all suitable models."""
    selector = ModelSelector()

    with st.spinner("Training and comparing all models..."):
        validation_config = (
            selector.VALIDATION_CONFIGS[validation_type] if validation_type else None
        )

        results = selector.compare_models(
            X=X,
            y=y,
            task=task,
            validation_config=validation_config,
        )

    return results


def display_recommendation(recommendation: Recommendation) -> None:
    """Display recommendation card."""
    st.subheader("📋 Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Recommended Model",
            recommendation.model.display_name,
        )
        st.caption(f"Type: {recommendation.model.model_type}")

    with col2:
        st.metric(
            "Validation Method",
            recommendation.validation.display_name,
        )

    with col3:
        st.metric(
            "Est. Training Time",
            f"{recommendation.estimated_time_seconds:.1f}s",
        )

    st.info(f"**Reasoning:** {recommendation.reasoning}")

    st.divider()


def display_result(result: ModelResult) -> None:
    """Display training results."""
    st.subheader("📊 Training Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Train Score", f"{result.train_score:.4f}")

    with col2:
        st.metric("Test Score", f"{result.test_score:.4f}")

    with col3:
        gap_color = "normal" if result.overfit_gap < 0.1 else "inverse"
        st.metric("Overfit Gap", f"{result.overfit_gap:.4f}", delta_color=gap_color)

    st.success(f"✅ Model trained in {result.training_time:.2f}s")

    if result.metrics:
        with st.expander("📈 Additional Metrics"):
            metrics_df = pd.DataFrame(
                list(result.metrics.items()),
                columns=["Metric", "Value"],
            )
            st.table(metrics_df)

    st.divider()


def display_comparison(results: list[ModelResult]) -> None:
    """Display model comparison."""
    st.subheader("🔬 Model Comparison")

    comparison_data = [
        {
            "Model": r.model_name,
            "Validation": r.validation_method,
            "Train Score": f"{r.train_score:.4f}",
            "Test Score": f"{r.test_score:.4f}",
            "Overfit Gap": f"{r.overfit_gap:.4f}",
            "Time (s)": f"{r.training_time:.2f}",
        }
        for r in results
    ]

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    best_model = results[0] if results else None
    if best_model:
        st.success(
            f"🏆 Best Model: **{best_model.model_name}** (Test Score: {best_model.test_score:.4f})"
        )

    st.divider()


def plot_feature_importance(result: ModelResult) -> None:
    """Plot feature importance chart."""
    if not result.feature_importance:
        st.warning("No feature importance available")
        return

    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame(
        list(result.feature_importance.items()),
        columns=["feature", "importance"],
    ).sort_values("importance", ascending=True)

    fig = px.bar(
        importance_df.head(20),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Feature Importance",
        color="importance",
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        height=600,
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()


def plot_comparison_scores(results: list[ModelResult]) -> None:
    """Plot comparison scores."""
    if not results:
        return

    st.subheader("📉 Score Comparison")

    model_names = [r.model_name for r in results]
    train_scores = [r.train_score for r in results]
    test_scores = [r.test_score for r in results]
    overfit_gaps = [r.overfit_gap for r in results]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="Train Score", x=model_names, y=train_scores, marker_color="lightblue")
    )
    fig.add_trace(
        go.Bar(name="Test Score", x=model_names, y=test_scores, marker_color="lightcoral")
    )

    fig.update_layout(
        barmode="group",
        title="Train vs Test Scores by Model",
        xaxis_title="Model",
        yaxis_title="Score",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_gap = px.bar(
        x=model_names,
        y=overfit_gaps,
        title="Overfitting Gap by Model",
        labels={"x": "Model", "y": "Overfit Gap"},
        color=overfit_gaps,
        color_continuous_scale="RdYlGn",
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    st.divider()


def export_results(
    result: ModelResult,
    format: str,
) -> None:
    """Export results for download."""
    if format == "json":
        data = {
            "model_name": result.model_name,
            "model_type": result.model_type,
            "validation_method": result.validation_method,
            "train_score": result.train_score,
            "test_score": result.test_score,
            "overfit_gap": result.overfit_gap,
            "training_time": result.training_time,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance,
            "exported_at": datetime.now().isoformat(),
        }

        json_str = json.dumps(data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"ml_result_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )

    elif format == "csv" and result.feature_importance:
        importance_df = pd.DataFrame(
            list(result.feature_importance.items()),
            columns=["feature", "importance"],
        )

        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"feature_importance_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )


def main() -> None:
    """Main Streamlit app."""

    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.X = None
        st.session_state.y = None

    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📁 Data Source")

        if st.button("Clear Data"):
            st.session_state.data_loaded = False
            st.session_state.X = None
            st.session_state.y = None
            st.session_state.recommendation = None
            st.session_state.result = None
            st.session_state.comparison_results = None
            st.rerun()

        data_source = st.selectbox(
            "Data Source",
            ["Sample SPY Data", "Upload CSV", "Upload Parquet"],
            index=0,
            help="Choose data source for training",
        )

        X = st.session_state.X
        y = st.session_state.y

        if not st.session_state.data_loaded or X is None:
            if data_source == "Sample SPY Data":
                if st.button("Load Sample Data"):
                    with st.spinner("Loading sample data..."):
                        X, y = load_sample_data()
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.data_loaded = True
                    st.success(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
                    st.rerun()

            elif data_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV", type=["csv"], help="CSV file with features and target"
                )
                if uploaded_file:
                    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                    if "target" in df.columns:
                        y = df["target"]
                    else:
                        y = df.iloc[:, -1]
                        df = df.iloc[:, :-1]
                    X = df

                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
                    st.rerun()

            elif data_source == "Upload Parquet":
                uploaded_file = st.file_uploader(
                    "Upload Parquet", type=["parquet"], help="Parquet file with features and target"
                )
                if uploaded_file:
                    df = pd.read_parquet(uploaded_file)
                    if "target" in df.columns:
                        y = df["target"]
                    else:
                        y = df.iloc[:, -1]
                        df = df.iloc[:, :-1]
                    X = df

                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
                    st.rerun()

            if not st.session_state.data_loaded:
                st.info("Please load data to continue")
                st.stop()
        else:
            st.success(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

        st.divider()

        st.subheader("🎯 Task Configuration")
        task = st.selectbox(
            "Task Type",
            ["classification", "regression"],
            help="Type of ML task",
        )

        priority = st.selectbox(
            "Priority Mode",
            ["balanced", "fast", "accurate", "interpretable"],
            help="Optimization priority",
        )

        with st.expander("🔧 Advanced Options"):
            model_type = st.selectbox(
                "Force Model Type",
                [
                    None,
                    "lightgbm",
                    "xgboost",
                    "random_forest",
                    "gradient_boosting",
                    "logistic_regression",
                ],
                index=0,
                help="Override automatic model selection",
            )

            validation_type = st.selectbox(
                "Force Validation Type",
                [None, "train_test", "walk_forward", "purged_kfold"],
                index=0,
                help="Override automatic validation selection",
            )

    st.header("🤖 Model Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Get Recommendation", use_container_width=True):
            st.session_state.recommendation = get_recommendation(
                X, y, task, priority, model_type, validation_type
            )
            st.session_state.result = None
            st.session_state.comparison_results = None

    with col2:
        if st.button("🚀 Auto-Select Best Model", use_container_width=True):
            recommendation = get_recommendation(X, y, task, priority, model_type, validation_type)
            st.session_state.recommendation = recommendation

            result = train_model(X, y, recommendation)
            st.session_state.result = result
            st.session_state.comparison_results = None

    with col3:
        if st.button("🔬 Compare All Models", use_container_width=True):
            results = compare_models(X, y, task, validation_type)
            st.session_state.comparison_results = results
            st.session_state.recommendation = None
            st.session_state.result = None

    st.divider()

    if st.session_state.recommendation:
        display_recommendation(st.session_state.recommendation)

    if st.session_state.result:
        display_result(st.session_state.result)
        plot_feature_importance(st.session_state.result)

        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)

        with col1:
            export_results(st.session_state.result, "json")

        with col2:
            export_results(st.session_state.result, "csv")

    if st.session_state.comparison_results:
        display_comparison(st.session_state.comparison_results)
        plot_comparison_scores(st.session_state.comparison_results)

        if st.button("💾 Download Comparison JSON"):
            comparison_data = [
                {
                    "model_name": r.model_name,
                    "model_type": r.model_type,
                    "validation_method": r.validation_method,
                    "train_score": r.train_score,
                    "test_score": r.test_score,
                    "overfit_gap": r.overfit_gap,
                    "training_time": r.training_time,
                }
                for r in st.session_state.comparison_results
            ]

            json_str = json.dumps(comparison_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"ml_comparison_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
