#!/usr/bin/env python3
"""
Main training script for MindPredict AI.
Run this script once to train all models and save artifacts.

Usage: python src/run_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import json
import numpy as np
import pandas as pd

from src.data_loader import merge_prevalence_daly, full_dataset_audit, get_data_dictionary
from src.preprocessing import (
    preprocess_for_modeling, leakage_check, detect_duplicates,
    detect_outliers_iqr, save_preprocessing_artifacts, create_risk_category,
)
from src.train import (
    train_and_evaluate, select_best_model, get_comparison_table,
    save_model, save_results,
)
from src.explainability import compute_shap_values
from src.utils import MODELS_DIR, ensure_dirs


def main():
    ensure_dirs()
    print("=" * 60)
    print("MindPredict AI - Model Training Pipeline")
    print("=" * 60)

    # Phase 1: Load and audit data
    print("\n[Phase 1] Loading and auditing datasets...")
    merged_df = merge_prevalence_daly()
    print(f"  Merged dataset shape: {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")

    # Data dictionary
    data_dict = get_data_dictionary(merged_df)
    data_dict.to_csv(os.path.join(MODELS_DIR, "data_dictionary.csv"), index=False)
    print(f"  Data dictionary saved.")

    # Leakage check
    print("\n[Phase 1b] Checking for data leakage...")
    leakage_report = leakage_check(merged_df, "depressive_prev")
    print(f"  Leakage status: {leakage_report['status']}")
    if leakage_report["warnings"]:
        for w in leakage_report["warnings"]:
            print(f"    WARNING: {w}")

    # Duplicate check
    dup_info = detect_duplicates(merged_df)
    print(f"  Duplicates: {dup_info['total_duplicates']} ({dup_info['duplicate_percentage']}%)")

    # Outlier detection
    print("\n[Phase 1c] Detecting outliers...")
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = detect_outliers_iqr(merged_df, numeric_cols)
    for col, info in outlier_report.items():
        if info["count"] > 0:
            print(f"  {col}: {info['count']} outliers ({info['percentage']}%)")

    # Phase 2: Preprocessing
    print("\n[Phase 2] Preprocessing...")
    preproc_result = preprocess_for_modeling(merged_df, target_col="depression_risk")
    print(f"  Training set: {preproc_result['X_train'].shape}")
    print(f"  Test set: {preproc_result['X_test'].shape}")
    print(f"  Classes: {preproc_result['class_names']}")
    print(f"  Features: {len(preproc_result['feature_names'])}")

    print("\n[Phase 3] Training and evaluating models...")
    results = train_and_evaluate(
        X_train=preproc_result["X_train"],
        y_train=preproc_result["y_train"],
        X_test=preproc_result["X_test"],
        y_test=preproc_result["y_test"],
        feature_names=preproc_result["feature_names"],
        class_names=preproc_result["class_names"],
    )

    # Phase 4: Model Selection
    print("\n[Phase 4] Model Selection...")
    best_name, best_result = select_best_model(results)
    print(f"  Best Model: {best_name}")
    print(f"  F1-Score: {best_result['f1']}")
    print(f"  Accuracy: {best_result['accuracy']}")
    print(f"  CV Mean: {best_result['cv_mean']} (+/- {best_result['cv_std']})")

    # Save preprocessing info with best model name
    save_preprocessing_artifacts(preproc_result, best_model=best_name)

    # Comparison table
    comparison = get_comparison_table(results)
    comparison.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print("\n  Model Comparison Table:")
    print(comparison.to_string(index=False))

    # Save best model
    save_model(best_result["model"], "best_model.joblib")
    save_results(results)

    # Save comparison data for Streamlit
    comparison.to_csv(os.path.join(MODELS_DIR, "comparison_table.csv"), index=False)

    # Save per-model test predictions for the Model Comparison page
    y_pred_all = {name: res["y_pred"] for name, res in results.items()}
    joblib.dump(y_pred_all, os.path.join(MODELS_DIR, "y_pred_all.joblib"))

    # Phase 5: Explainable AI
    print("\n[Phase 5] Computing SHAP values...")
    shap_result = compute_shap_values(
        model=best_result["model"],
        X_train=preproc_result["X_train"],
        X_test=preproc_result["X_test"],
        feature_names=preproc_result["feature_names"],
    )

    # Save SHAP artifacts
    if shap_result["success"]:
        joblib.dump(shap_result, os.path.join(MODELS_DIR, "shap_result.joblib"))
        print("  SHAP values computed and saved.")
    else:
        print(f"  SHAP computation failed: {shap_result['message']}")

    # Save the full merged dataset for EDA in Streamlit
    merged_df.to_csv(os.path.join(MODELS_DIR, "merged_dataset.csv"), index=False)
    print(f"\n  Merged dataset saved for Streamlit use.")

    # Save preprocessed arrays
    joblib.dump(preproc_result["X_train"], os.path.join(MODELS_DIR, "X_train.joblib"))
    joblib.dump(preproc_result["X_test"], os.path.join(MODELS_DIR, "X_test.joblib"))
    joblib.dump(preproc_result["y_train"], os.path.join(MODELS_DIR, "y_train.joblib"))
    joblib.dump(preproc_result["y_test"], os.path.join(MODELS_DIR, "y_test.joblib"))
    joblib.dump(preproc_result["label_encoder"], os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(preproc_result["feature_names"], os.path.join(MODELS_DIR, "feature_names.joblib"))
    joblib.dump(preproc_result["class_names"], os.path.join(MODELS_DIR, "class_names.joblib"))

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print(f"All artifacts saved to: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
