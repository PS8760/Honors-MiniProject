"""
Explainability Module for MindPredict AI
Uses SHAP for model interpretation.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray,
                        feature_names: List[str] = None) -> Dict:
    """
    Compute SHAP values for a trained model.
    Returns dictionary with SHAP-related artifacts.
    """
    result = {"success": False, "message": "", "shap_values": None}

    try:
        explainer = None
        shap_values = None

        # Preferred: TreeExplainer (handles RandomForest, decision-tree based models)
        funded_tree = False
        try:
            if hasattr(model, "feature_importances_") and model.__class__.__name__ != "GradientBoostingClassifier":
                explainer = shap.TreeExplainer(model)
                test_data = X_test[min(200, X_test.shape[0]):X_test.shape[0]] if X_test.shape[0] > 0 else X_test
                f_test = X_test[:min(200, X_test.shape[0])]
                sv = explainer.shap_values(f_test)
                shap_values = sv
                funded_tree = True
        except Exception as e:
            funded_tree = False

        # Fallback: KernelExplainer
        if not funded_tree:
            background = shap.sample(X_train, min(50, X_train.shape[0]))
            n_classes = len(np.unique(list(y if 'y' not in locals() else model.classes_)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            f_test = X_test[:min(100, X_test.shape[0])]
            shap_values = explainer.shap_values(f_test)

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

        result.update({
            "success": True,
            "message": "SHAP values computed successfully",
            "explainer": explainer,
            "shap_values": shap_values,
            "X_test_limited": f_test,
            "feature_names": feature_names,
        })
    except Exception as e:
        result["message"] = f"Error computing SHAP values: {str(e)}"

    return result


def plot_shap_summary(shap_values, feature_names: List[str],
                      class_names: List[str] = None) -> go.Figure:
    """
    Create a SHAP summary plot as a Plotly figure.
    For multi-class, aggregates across classes.
    """
    try:
        if isinstance(shap_values, list):
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        if len(sv.shape) > 2:
            sv = np.mean(sv, axis=-1)

        mean_abs = np.mean(sv, axis=0)

        n_features = min(20, len(feature_names), len(mean_abs))
        indices = np.argsort(mean_abs)[-n_features:]

        fig = go.Figure(go.Bar(
            x=mean_abs[indices],
            y=[feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in indices],
            orientation="h",
            marker_color="#636EFA",
        ))
        fig.update_layout(
            title="SHAP Feature Importance (Mean |SHAP value|)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            template="plotly_white", height=max(400, n_features * 25),
            yaxis=dict(autorange="reversed"),
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error creating SHAP summary: {str(e)}")
        return fig


def plot_shap_beeswarm(shap_values, feature_names: List[str],
                        X_test: np.ndarray) -> go.Figure:
    """
    Create a beeswarm-style plot using Plotly.
    Shows distribution of SHAP values for each feature.
    """
    try:
        if isinstance(shap_values, list):
            sv = shap_values[0] if len(shap_values) == 1 else np.mean(shap_values, axis=0)
        else:
            sv = shap_values

        if len(sv.shape) > 2:
            sv = sv[:, :, 0] if sv.shape[2] == 1 else np.mean(sv, axis=2)

        mean_abs = np.mean(np.abs(sv), axis=0)
        n_features = min(15, len(feature_names), len(mean_abs))
        indices = np.argsort(mean_abs)[-n_features:]

        traces = []
        for idx in reversed(indices):
            fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            feature_vals = X_test[:, idx] if idx < X_test.shape[1] else np.zeros(sv.shape[0])
            shap_vals = sv[:, idx]

            trace = go.Scatter(
                x=shap_vals,
                y=[fname] * len(shap_vals),
                mode="markers",
                marker=dict(
                    size=5,
                    color=feature_vals,
                    colorscale="RdBu_r",
                    showscale=(idx == indices[0]),
                    colorbar=dict(title="Feature Value") if idx == indices[0] else None,
                    opacity=0.6,
                ),
                name=fname,
                showlegend=False,
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title="SHAP Beeswarm Plot",
            xaxis_title="SHAP Value",
            template="plotly_white", height=max(400, n_features * 30),
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error creating beeswarm: {str(e)}")
        return fig


def explain_individual_prediction(model, X_test: np.ndarray, index: int,
                                   feature_names: List[str],
                                   class_names: List[str] = None,
                                   shap_values=None) -> Dict:
    """
    Explain a single prediction using SHAP and model internals.
    Returns a structured explanation.
    """
    sample = X_test[index:index+1]

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

    pred_label = class_names[prediction] if class_names and prediction < len(class_names) else str(prediction)
    confidence = float(probabilities[prediction]) if probabilities is not None else None

    contributing_factors = []
    if shap_values is not None:
        try:
            # Normalize shap_values to a per-sample, per-feature vector for the
            # predicted class. shap returns either a list of class arrays
            # (older API) or an ndarray of shape (samples, features, classes).
            sv = shap_values
            if isinstance(sv, list):
                sv = sv[prediction] if prediction < len(sv) else sv[0]
            if sv.ndim == 3:
                sv = sv[:, :, prediction] if prediction < sv.shape[2] else sv[:, :, 0]

            if sv.ndim > 1:
                sample_sv = sv[index] if index < sv.shape[0] else sv[0]
            else:
                sample_sv = sv

            n_features = min(len(sample_sv), len(feature_names))
            factor_impacts = []
            for i in range(n_features):
                val = float(sample_sv[i])
                if abs(val) > 0.001:
                    factor_impacts.append({
                        "feature": feature_names[i],
                        "shap_value": val,
                        "feature_value": float(X_test[index, i]) if i < X_test.shape[1] else 0,
                        "direction": "increases" if val > 0 else "decreases",
                        "magnitude": abs(val),
                    })

            factor_impacts.sort(key=lambda x: x["magnitude"], reverse=True)
            contributing_factors = factor_impacts[:10]
        except Exception:
            pass

    explanation = {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(prob) for i, prob in enumerate(probabilities)
        } if probabilities is not None and class_names else None,
        "contributing_factors": contributing_factors,
        "disclaimer": (
            "This explanation reflects the model's statistical pattern recognition. "
            "It should NOT be interpreted as a medical diagnosis or causal statement. "
            "The model associated certain features with higher/lower predicted risk; "
            "this does not mean those features caused the outcome."
        ),
    }

    return explanation


def plot_individual_contribution(contributing_factors: List[Dict]) -> go.Figure:
    """Plot feature contributions for an individual prediction."""
    if not contributing_factors:
        fig = go.Figure()
        fig.update_layout(title="No significant contributing factors found")
        return fig

    features = [f["feature"] for f in contributing_factors]
    shap_vals = [f["shap_value"] for f in contributing_factors]
    colors = ["#EF553B" if v > 0 else "#636EFA" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker_color=colors,
        text=[f"{'+'if v>0 else ''}{v:.4f}" for v in shap_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Contributions to This Prediction",
        xaxis_title="SHAP Value (positive = pushes toward predicted class)",
        template="plotly_white", height=max(300, len(features) * 35),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def generate_explainability_report(shap_result: Dict, model_name: str = "") -> str:
    """Generate a human-readable explainability report."""
    if not shap_result.get("success"):
        return f"SHAP analysis could not be completed: {shap_result.get('message', 'Unknown error')}"

    feature_names = shap_result.get("feature_names", [])
    shap_values = shap_result.get("shap_values")

    if shap_values is None:
        return "No SHAP values available."

    try:
        if isinstance(shap_values, list):
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        if len(sv.shape) > 2:
            sv = np.mean(sv, axis=-1)

        mean_abs = np.mean(sv, axis=0)
        top_indices = np.argsort(mean_abs)[-5:][::-1]

        report = f"## Explainable AI Report ({model_name})\n\n"
        report += "### Top 5 Most Important Features\n\n"
        report += "| Rank | Feature | Importance |\n|------|---------|------------|\n"

        for rank, idx in enumerate(top_indices, 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            report += f"| {rank} | {fname} | {mean_abs[idx]:.4f} |\n"

        report += "\n### Interpretation\n\n"
        report += "The SHAP analysis reveals that the model primarily relies on "
        report += f"**{feature_names[top_indices[0]]}** and "
        report += f"**{feature_names[top_indices[1]]}** "
        report += "to make predictions. These features have the highest average impact on model output.\n\n"
        report += "**Important**: This reflects statistical associations in the data, "
        report += "not causal relationships. The model identified patterns that correlate with "
        report += "the target variable in the training data.\n"

        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"
