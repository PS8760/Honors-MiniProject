"""
Preprocessing Module for MindPredict AI
Handles data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import json
import os

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def detect_duplicates(df: pd.DataFrame) -> Dict:
    """Detect duplicate records and return summary."""
    dup_count = int(df.duplicated().sum())
    dup_rows = df[df.duplicated(keep=False)]
    return {
        "total_duplicates": dup_count,
        "duplicate_percentage": round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0,
        "duplicate_samples": dup_rows.head(5).to_dict("records") if dup_count > 0 else [],
    }


def detect_invalid_values(df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
    """Detect potentially invalid values in numerical columns."""
    invalid_report = {}
    for col in numeric_cols:
        if col in df.columns:
            series = df[col].dropna()
            issues = []
            if (series < 0).any() and "share" in col.lower():
                issues.append(f"Negative values found: {(series < 0).sum()} rows")
            if (series > 100).any() and "share" in col.lower():
                issues.append(f"Values > 100% found: {(series > 100).sum()} rows")
            if series.std() == 0:
                issues.append("Constant column (zero variance)")
            invalid_report[col] = issues if issues else ["No issues detected"]
    return invalid_report


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
    """Detect outliers using IQR method."""
    outlier_report = {}
    for col in numeric_cols:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            outlier_report[col] = {
                "count": outlier_count,
                "percentage": round(outlier_count / len(series) * 100, 2),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
    return outlier_report


def create_risk_category(df: pd.DataFrame, target_col: str = "depressive_prev",
                         n_categories: int = 3) -> pd.DataFrame:
    """
    Create risk categories from continuous prevalence values.
    Low Risk / Moderate Risk / High Risk
    """
    df = df.copy()
    quantiles = df[target_col].quantile([0.33, 0.67]).values
    conditions = [
        df[target_col] <= quantiles[0],
        df[target_col] <= quantiles[1],
        df[target_col] > quantiles[1],
    ]
    choices = ["Low Risk", "Moderate Risk", "High Risk"]
    df["depression_risk"] = np.select(conditions, choices, default="Moderate Risk")
    return df


def prepare_features(df: pd.DataFrame, target_col: str = "depression_risk",
                     exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.
    Removes leakage-prone columns.
    """
    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    leakage_candidates = [c for c in df.columns if "depressive" in c.lower() and c != target_col]
    exclude_cols = list(set(exclude_cols + leakage_candidates))

    drop_cols = set()
    for c in ["entity", "code", target_col] + [x for x in exclude_cols if x != target_col]:
        c_lower = c.lower()
        for col in df.columns:
            if col.lower() == c_lower:
                drop_cols.add(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None

    return X, y


def get_feature_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify features into numerical and categorical."""
    numerical = list(X.select_dtypes(include=[np.number]).columns)
    categorical = list(X.select_dtypes(include=["object", "category"]).columns)
    return {"numerical": numerical, "categorical": categorical}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for preprocessing.
    - Numerical: Impute with median, then scale.
    - Categorical: Impute with most frequent, then one-hot encode.
    """
    feature_types = get_feature_types(X)
    numerical = feature_types["numerical"]
    categorical = feature_types["categorical"]

    transformers = []

    if numerical:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical))

    if categorical:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def leakage_check(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Check for potential data leakage by examining correlations
    between features and the target.
    """
    report = {"target": target_col, "warnings": [], "leakage_features": []}

    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        correlations = numeric_df.corr(numeric_only=True)[target_col].drop(target_col, errors="ignore")

        for feat, corr_val in correlations.items():
            if abs(corr_val) > 0.85:
                report["warnings"].append(
                    f"HIGH CORRELATION ({corr_val:.3f}): '{feat}' with target '{target_col}'"
                )
                report["leakage_features"].append(feat)

        direct_leakage = [c for c in df.columns if c == target_col or
                         (target_col.split("_")[0] in c.lower() and c != target_col)]
        for feat in direct_leakage:
            if feat not in report["leakage_features"]:
                report["leakage_features"].append(feat)
                report["warnings"].append(f"POTENTIAL LEAKAGE: '{feat}' may directly encode the target")

    if not report["warnings"]:
        report["status"] = "CLEAN - No obvious leakage detected"
    else:
        report["status"] = f"WARNING - {len(report['warnings'])} potential leakage issues found"

    return report


def preprocess_for_modeling(df: pd.DataFrame, target_col: str = "depression_risk",
                           test_size: float = 0.2, random_state: int = 42
                           ) -> Dict:
    """
    Full preprocessing pipeline:
    1. Create target categories
    2. Handle missing values
    3. Feature-target split
    4. Train/test split (stratified)
    5. Build preprocessing transformer
    6. Fit and transform
    """
    df = create_risk_category(df, target_col="depressive_prev")

    df_clean = df.dropna(subset=["depressive_prev"]).copy()

    # Exclude leakage-prone features: depressive_daly is 99% correlated with
    # the target depressive_prev; the target col itself is excluded in prepare_features.
    leakage_exclude = ["depressive_daly"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(df_clean["depression_risk"])
    class_names = list(le.classes_)

    X, _ = prepare_features(df_clean, target_col="depression_risk",
                            exclude_cols=leakage_exclude)

    feature_types = get_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_features = []
    if feature_types["categorical"]:
        ohe = preprocessor.named_transformers_.get("cat")
        if ohe and hasattr(ohe, "named_steps") and "encoder" in ohe.named_steps:
            cat_features = list(ohe.named_steps["encoder"].get_feature_names_out(feature_types["categorical"]))

    all_feature_names = feature_types["numerical"] + cat_features
    if X_train_processed.shape[1] > len(all_feature_names):
        all_feature_names += [f"remainder_{i}" for i in range(
            X_train_processed.shape[1] - len(all_feature_names))]

    result = {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_df": X_train,
        "X_test_df": X_test,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "class_names": class_names,
        "feature_names": all_feature_names,
        "feature_types": feature_types,
        "target_col": "depression_risk",
        "original_target": "depressive_prev",
    }

    return result


def save_preprocessing_artifacts(result: Dict, path: str = None, best_model: str = None):
    """Save preprocessing artifacts for later use."""
    if path is None:
        path = os.path.join(MODELS_DIR, "preprocessing_info.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    info = {
        "class_names": result["class_names"],
        "feature_names": result["feature_names"],
        "feature_types": result["feature_types"],
        "target_col": result["target_col"],
        "original_target": result["original_target"],
        "n_train": int(result["X_train"].shape[0]),
        "n_test": int(result["X_test"].shape[0]),
        "n_features": int(result["X_train"].shape[1]),
    }

    if best_model:
        info["best_model"] = best_model

    with open(path, "w") as f:
        json.dump(info, f, indent=2)
