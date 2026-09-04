"""
Training Module for MindPredict AI
Handles model training, evaluation, and comparison.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def get_models() -> Dict:
    """Return dictionary of models to evaluate."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       feature_names: List[str] = None,
                       class_names: List[str] = None) -> Dict:
    """
    Train all models and evaluate them.
    Returns comprehensive results dictionary.
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        roc = None
        if y_prob is not None and len(np.unique(y_test)) > 2:
            try:
                roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                try:
                    roc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")
                except Exception:
                    roc = None
        elif y_prob is not None and len(np.unique(y_test)) == 2:
            try:
                roc = roc_auc_score(y_test, y_prob[:, 1])
            except Exception:
                roc = None

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        fpr, tpr, _ = (None, None, None)
        if y_prob is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                else:
                    fpr_tprs = {}
                    for i in range(len(class_names)):
                        fpr_i, tpr_i, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                        fpr_tprs[i] = (fpr_i, tpr_i)
                    fpr, tpr = fpr_tprs, None
            except Exception:
                pass

        feature_importance = None
        if hasattr(model, "feature_importances_") and feature_names is not None:
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names[:len(importance)], importance.tolist()))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        elif name == "Logistic Regression" and hasattr(model, "coef_") and feature_names is not None:
            importance = np.abs(model.coef_).mean(axis=0)
            feature_importance = dict(zip(feature_names[:len(importance)], importance.tolist()))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        results[name] = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc, 4) if roc is not None else None,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "fpr": fpr,
            "tpr": tpr,
            "feature_importance": feature_importance,
        }

        print(f"  {name}: Accuracy={acc:.4f}, F1={f1:.4f}, CV={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def select_best_model(results: Dict) -> Tuple[str, Dict]:
    """
    Select the best model based on F1-score (more balanced than accuracy for healthcare).
    Falls back to accuracy if F1 is unavailable.
    """
    best_name = None
    best_score = -1

    for name, res in results.items():
        score = res["f1"] if res["f1"] is not None else res["accuracy"]
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, results[best_name]


def get_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a model comparison DataFrame."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1"],
            "ROC-AUC": res["roc_auc"] if res["roc_auc"] is not None else "N/A",
            "CV Mean": res["cv_mean"],
            "CV Std": res["cv_std"],
        })
    return pd.DataFrame(rows)


def save_model(model, filename: str = "best_model.joblib"):
    """Save a trained model to disk."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename: str = "best_model.joblib"):
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def save_results(results: Dict, filename: str = "model_results.json"):
    """Save model results (without model objects) to JSON."""
    path = os.path.join(MODELS_DIR, filename)

    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
            "confusion_matrix": res["confusion_matrix"],
            "cv_mean": res["cv_mean"],
            "cv_std": res["cv_std"],
            "feature_importance": res.get("feature_importance"),
        }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path
