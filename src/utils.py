"""
Utility Module for MindPredict AI
Helper functions and constants.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
APP_DIR = os.path.join(PROJECT_ROOT, "app")


DISCLAIMER = """
> **Disclaimer:** This is an academic project. Predictions are statistical estimates based on 
> population-level data and must NOT be considered a medical diagnosis.
"""

ETHICAL_STATEMENT = """
**Ethical Considerations**

1. **No Clinical Claims**: This system provides AI-assisted statistical risk estimation, not diagnosis.
2. **Data Privacy**: The dataset contains only aggregated, publicly available data with no personal identifiers.
3. **Bias Awareness**: The model may reflect biases present in the training data, including geographic and demographic biases.
4. **Professional Advice**: Users experiencing mental health concerns should consult qualified healthcare professionals.
5. **Transparency**: All model decisions are explained through Explainable AI (SHAP) techniques.
6. **Limitations**: Model performance is bounded by data quality and quantity available.
"""

APP_NAME = "MindPredict AI"
APP_SUBTITLE = "Explainable Mental Health Prediction & Analytics"

PREVALENCE_FEATURE_NAMES = {
    "schizophrenia_prev": "Schizophrenia Prevalence",
    "depressive_prev": "Depressive Disorders Prevalence",
    "anxiety_prev": "Anxiety Disorders Prevalence",
    "bipolar_prev": "Bipolar Disorders Prevalence",
    "eating_prev": "Eating Disorders Prevalence",
}

DALY_FEATURE_NAMES = {
    "schizophrenia_daly": "Schizophrenia DALY Rate",
    "depressive_daly": "Depressive Disorders DALY Rate",
    "anxiety_daly": "Anxiety Disorders DALY Rate",
    "bipolar_daly": "Bipolar Disorders DALY Rate",
    "eating_daly": "Eating Disorders DALY Rate",
}


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, os.path.join(REPORTS_DIR, "figures")]:
        os.makedirs(d, exist_ok=True)


def format_percentage(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.2f}%"


def format_metric(value: float, name: str = "") -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with fallback."""
    return a / b if b != 0 else default


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "Low Risk": "#00CC96",
        "Moderate Risk": "#FFA15A",
        "High Risk": "#EF553B",
    }
    return colors.get(risk_level, "#636EFA")


def get_risk_icon(risk_level: str) -> str:
    """Get icon emoji for risk level."""
    icons = {
        "Low Risk": "🟢",
        "Moderate Risk": "🟡",
        "High Risk": "🔴",
    }
    return icons.get(risk_level, "⚪")
