"""
Evaluation Module for MindPredict AI
Handles model evaluation visualization and metrics display.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          model_name: str = "") -> go.Figure:
    """Plot an interactive confusion matrix."""
    cm_text = [[str(val) for val in row] for row in cm]

    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=class_names, y=class_names,
        title=f"Confusion Matrix: {model_name}",
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white", height=450, width=500,
    )
    return fig


def plot_roc_curves(results: Dict, class_names: List[str]) -> go.Figure:
    """Plot ROC curves for all models that support probability prediction."""
    fig = go.Figure()

    for name, res in results.items():
        y_prob = res.get("y_prob")
        if y_prob is None:
            continue

        n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 1

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(res.get("y_test_placeholder", np.arange(len(y_prob))), y_prob[:, 1])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['roc_auc']:.3f})" if res.get("roc_auc") else name,
            ))
        elif n_classes > 2:
            try:
                from sklearn.preprocessing import label_binarize
                y_true = res.get("y_pred")
                if y_true is not None:
                    y_bin = label_binarize(np.arange(n_classes), classes=range(n_classes))
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                        if i == 0:
                            fig.add_trace(go.Scatter(
                                x=fpr, y=tpr, mode="lines",
                                name=f"{name}" + (f" (AUC={res['roc_auc']:.3f})" if res.get("roc_auc") else ""),
                            ))
                            break
            except Exception:
                pass

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random Classifier",
    ))
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white", height=500,
    )
    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot grouped bar chart comparing model metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    available = [m for m in metrics if m in comparison_df.columns]

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    for i, metric in enumerate(available):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df["Model"],
            y=comparison_df[metric],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group", title="Model Performance Comparison",
        yaxis_title="Score", template="plotly_white", height=450,
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def plot_cv_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot cross-validation scores comparison with error bars."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["CV Mean"],
        error_y=dict(type="data", array=comparison_df["CV Std"], visible=True),
        marker_color="#636EFA",
        name="CV Accuracy",
    ))

    fig.update_layout(
        title="Cross-Validation Accuracy (5-Fold Stratified)",
        yaxis_title="Accuracy", template="plotly_white", height=400,
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str], model_name: str = "") -> go.Figure:
    """Plot actual vs predicted distribution."""
    df_plot = pd.DataFrame({
        "Actual": [class_names[i] for i in y_test],
        "Predicted": [class_names[i] for i in y_pred],
    })

    actual_counts = df_plot["Actual"].value_counts().reset_index()
    actual_counts.columns = ["Category", "Count"]
    actual_counts["Type"] = "Actual"

    pred_counts = df_plot["Predicted"].value_counts().reset_index()
    pred_counts.columns = ["Category", "Count"]
    pred_counts["Type"] = "Predicted"

    combined = pd.concat([actual_counts, pred_counts])

    fig = px.bar(
        combined, x="Category", y="Count", color="Type",
        barmode="group",
        title=f"Actual vs Predicted Distribution: {model_name}",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plot_feature_importance_bar(feature_importance: Dict, top_n: int = 15,
                                 model_name: str = "") -> go.Figure:
    """Plot feature importance as a horizontal bar chart."""
    if not feature_importance:
        fig = go.Figure()
        fig.update_layout(title=f"No feature importance available for {model_name}")
        return fig

    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])

    fig = px.bar(
        x=list(sorted_features.values()),
        y=list(sorted_features.keys()),
        orientation="h",
        title=f"Feature Importance: {model_name}",
        labels={"x": "Importance", "y": "Feature"},
        color=list(sorted_features.values()),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_white", height=450, showlegend=False)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def generate_evaluation_summary(results: Dict, best_name: str) -> str:
    """Generate a text summary of model evaluation."""
    best = results[best_name]
    summary = f"""
## Model Evaluation Summary

**Best Model: {best_name}**

| Metric | Score |
|--------|-------|
| Accuracy | {best['accuracy']:.4f} |
| Precision | {best['precision']:.4f} |
| Recall | {best['recall']:.4f} |
| F1-Score | {best['f1']:.4f} |
| ROC-AUC | {best['roc_auc']:.4f if best['roc_auc'] is not None else 'N/A'} |
| CV Mean | {best['cv_mean']:.4f} |
| CV Std | {best['cv_std']:.4f} |

### Healthcare Context
- **Recall** is critical: Missing a high-risk case (false negative) is more harmful than a false alarm.
- **Precision** indicates: Among predicted high-risk cases, how many are truly high-risk.
- **F1-Score** balances: Precision and recall into a single metric, suitable for imbalanced classes.
"""
    return summary
