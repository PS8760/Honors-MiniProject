#!/usr/bin/env python3
"""
MindPredict AI - Streamlit Application
Explainable Mental Health Prediction & Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import (
    load_all_datasets, merge_prevalence_daly,
    get_treatment_data, get_us_symptoms_data, get_country_list,
    get_year_range, full_dataset_audit,
)
from src.preprocessing import (
    create_risk_category, detect_duplicates, detect_outliers_iqr,
    detect_invalid_values, leakage_check,
)
from src.eda import (
    plot_global_trend, plot_country_comparison, plot_disorder_distribution,
    plot_correlation_heatmap, plot_daly_vs_prevalence, plot_regional_heatmap_trend,
    plot_missing_values, plot_treatment_gap, plot_us_symptoms,
    plot_feature_target_relationship, plot_country_timeseries,
    generate_insights, PREVALENCE_COLS, DALY_COLS,
)
from src.evaluate import (
    plot_confusion_matrix, plot_roc_curves, plot_model_comparison_bar,
    plot_cv_comparison, plot_actual_vs_predicted, plot_feature_importance_bar,
)
from src.explainability import (
    compute_shap_values, plot_shap_summary, plot_shap_beeswarm,
    explain_individual_prediction, plot_individual_contribution,
    generate_explainability_report,
)
from src.utils import (
    APP_NAME, DISCLAIMER,
    MODELS_DIR, get_risk_color, get_risk_icon,
)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .disclaimer-box {
        background-color: #FFF3CD;
        border: 1px solid #FFECB5;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .risk-low { color: #00CC96; font-weight: bold; }
    .risk-moderate { color: #FFA15A; font-weight: bold; }
    .risk-high { color: #EF553B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_merged_data():
    return merge_prevalence_daly()


@st.cache_data
def load_all_raw():
    return load_all_datasets()


@st.cache_resource
def load_trained_model():
    path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_shap_data():
    path = os.path.join(MODELS_DIR, "shap_result.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_model_results():
    path = os.path.join(MODELS_DIR, "model_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_all_predictions():
    path = os.path.join(MODELS_DIR, "y_pred_all.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return {}


@st.cache_data
def load_comparison():
    path = os.path.join(MODELS_DIR, "comparison_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_preproc_info():
    path = os.path.join(MODELS_DIR, "preprocessing_info.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dataset Overview",
            "🔧 Data Preprocessing",
            "📈 Exploratory Analysis",
            "🎯 Prediction",
            "⚖️ Model Comparison",
            "🔍 Explainable AI",
            "💡 Insights",
        ],
        label_visibility="collapsed",
    )


# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.header(APP_NAME)

    merged_df = load_merged_data()
    preproc_info = load_preproc_info()
    comp = load_comparison()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Records", f"{len(merged_df):,}")
    with col2:
        n_features = preproc_info["n_features"] if preproc_info else "N/A"
        st.metric("Features", n_features)
    with col3:
        n_countries = merged_df["Entity"].nunique() if "Entity" in merged_df.columns else 0
        st.metric("Countries", n_countries)
    with col4:
        yr = get_year_range()
        st.metric("Year Range", f"{yr[0]}-{yr[1]}")

    col5, col6 = st.columns(2)
    with col5:
        best_model = "N/A"
        best_score = "N/A"
        if comp is not None and not comp.empty:
            best_idx = comp["F1-Score"].idxmax()
            best_model = comp.loc[best_idx, "Model"]
            best_score = f"{comp.loc[best_idx, 'F1-Score']:.4f}"
        st.metric("Best Model", best_model)
    with col6:
        st.metric("Best F1-Score", best_score)

    st.markdown(DISCLAIMER, unsafe_allow_html=True)


# ============================================================
# DATASET OVERVIEW
# ============================================================
elif page == "📊 Dataset Overview":
    st.header("Dataset Overview")

    merged_df = load_merged_data()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{merged_df.shape[0]:,}")
    with col2:
        st.metric("Columns", merged_df.shape[1])

    tab1, tab2, tab3 = st.tabs(["Preview", "Missing Values", "Statistics"])

    with tab1:
        st.dataframe(merged_df.head(20), use_container_width=True)

    with tab2:
        missing = merged_df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.warning(f"{len(missing)} columns have missing values")
            st.bar_chart(missing.sort_values(ascending=False))
        else:
            st.success("No missing values.")
        dup_info = detect_duplicates(merged_df)
        st.info(f"Duplicate rows: {dup_info['total_duplicates']} ({dup_info['duplicate_percentage']}%)")

    with tab3:
        st.dataframe(merged_df.describe(), use_container_width=True)


# ============================================================
# DATA PREPROCESSING
# ============================================================
elif page == "🔧 Data Preprocessing":
    st.header("Data Preprocessing")

    merged_df = load_merged_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(merged_df):,}")
    with col2:
        dup = merged_df.duplicated().sum()
        st.metric("Duplicates", int(dup))
    with col3:
        missing_total = merged_df.isnull().sum().sum()
        st.metric("Missing Values", int(missing_total))

    st.subheader("Outliers (IQR)")
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = detect_outliers_iqr(merged_df, numeric_cols)
    outlier_data = []
    for col, info in outlier_report.items():
        if info["count"] > 0:
            outlier_data.append({
                "Feature": col,
                "Outliers": info["count"],
                "Percentage": f"{info['percentage']}%",
            })
    if outlier_data:
        st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
    else:
        st.success("No significant outliers detected.")

    st.subheader("Leakage Check")
    leakage = leakage_check(merged_df, "depressive_prev")
    if leakage["warnings"]:
        for w in leakage["warnings"]:
            st.warning(w)
    else:
        st.success(leakage["status"])

    st.subheader("Risk Category Distribution")
    df_cat = create_risk_category(merged_df, target_col="depressive_prev")
    risk_counts = df_cat["depression_risk"].value_counts()
    fig = px.pie(
        names=risk_counts.index, values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map={"Low Risk": "#00CC96", "Moderate Risk": "#FFA15A", "High Risk": "#EF553B"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# EDA PAGE
# ============================================================
elif page == "📈 Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    merged_df = load_merged_data()
    all_raw = load_all_raw()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### EDA Controls")

    countries = sorted(merged_df["Entity"].unique().tolist())
    indicators = list(PREVALENCE_COLS.keys())
    indicator_labels = [PREVALENCE_COLS[k] for k in indicators]

    selected_indicator = st.sidebar.selectbox(
        "Mental Health Indicator",
        indicators,
        format_func=lambda x: PREVALENCE_COLS[x],
    )

    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        countries,
        default=["India", "United States", "China", "Brazil", "Germany"] 
        if all(c in countries for c in ["India", "United States", "China", "Brazil", "Germany"])
        else countries[:5],
    )

    year_range = st.sidebar.slider(
        "Year Range",
        int(merged_df["Year"].min()),
        int(merged_df["Year"].max()),
        (int(merged_df["Year"].min()), int(merged_df["Year"].max())),
    )

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "Trends", "Country Comparison", "Distributions", "Correlations", "Additional"
    ])

    with eda_tab1:
        st.markdown("#### Global Trend Over Time")
        filtered = merged_df[
            (merged_df["Year"] >= year_range[0]) & (merged_df["Year"] <= year_range[1])
        ]
        fig = plot_global_trend(filtered, selected_indicator)
        st.plotly_chart(fig, use_container_width=True)

        if selected_countries:
            st.markdown("#### Country-Level Trends")
            fig = plot_country_timeseries(filtered, selected_countries, selected_indicator)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Regional Heatmap")
        fig = plot_regional_heatmap_trend(filtered, selected_indicator)
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab2:
        latest_year = int(merged_df["Year"].max())
        selected_year = st.selectbox("Select Year for Comparison", 
                                      sorted(merged_df["Year"].unique(), reverse=True))
        fig = plot_country_comparison(merged_df, selected_indicator, selected_year, top_n=15)
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab3:
        latest_year = int(merged_df["Year"].max())
        fig = plot_disorder_distribution(merged_df, latest_year)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Missing Values")
        fig = plot_missing_values(merged_df)
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab4:
        fig = plot_correlation_heatmap(merged_df)
        st.plotly_chart(fig, use_container_width=True)

        other_indicators = [k for k in indicators if k != selected_indicator]
        if other_indicators:
            compare_with = st.selectbox(
                "Feature for Scatter Plot",
                other_indicators,
                format_func=lambda x: PREVALENCE_COLS[x],
            )
            fig = plot_feature_target_relationship(merged_df, compare_with, selected_indicator)
            st.plotly_chart(fig, use_container_width=True)

    with eda_tab5:
        st.markdown("#### Prevalence vs Disease Burden (DALY)")
        disorder = st.selectbox(
            "Select Disorder",
            ["depressive", "anxiety", "bipolar", "schizophrenia", "eating"],
            format_func=lambda x: PREVALENCE_COLS.get(f"{x}_prev", x),
        )
        fig = plot_daly_vs_prevalence(merged_df, disorder)
        st.plotly_chart(fig, use_container_width=True)

        if "5- anxiety-disorders-treatment-gap.csv" in all_raw:
            st.markdown("#### Anxiety Disorders Treatment Gap")
            treatment_df = get_treatment_data()
            fig = plot_treatment_gap(treatment_df)
            st.plotly_chart(fig, use_container_width=True)

        if "6- depressive-symptoms-across-us-population.csv" in all_raw:
            st.markdown("#### US Depressive Symptoms (2014)")
            us_df = get_us_symptoms_data()
            fig = plot_us_symptoms(us_df)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "🎯 Prediction":
    st.header("Depression Risk Prediction")

    model = load_trained_model()
    preproc_info = load_preproc_info()
    merged_df = load_merged_data()
    comparison = load_comparison()
    best_model_name = "Random Forest"
    if comparison is not None and not comparison.empty:
        best_model_name = comparison.loc[comparison["F1-Score"].idxmax(), "Model"]

    if model is None or preproc_info is None:
        st.error("Trained model not found. Run `python src/run_training.py` first.")
        st.stop()

    col_input, col_result = st.columns([1, 1])

    feature_names = preproc_info["feature_names"]

    with col_input:
        st.markdown("### Input Parameters")

        st.markdown("**Prevalence Rates** (share of population)")
        schiz_prev = st.number_input("Schizophrenia Prevalence", 0.0, 1.0, 0.22, step=0.01, format="%.4f")
        anx_prev = st.number_input("Anxiety Prevalence", 0.0, 10.0, 3.7, step=0.1, format="%.4f")
        bip_prev = st.number_input("Bipolar Prevalence", 0.0, 5.0, 0.63, step=0.01, format="%.4f")
        eat_prev = st.number_input("Eating Disorders Prevalence", 0.0, 1.0, 0.15, step=0.01, format="%.4f")

        st.markdown("**DALY Rates** (disease burden)")
        schiz_daly = st.number_input("Schizophrenia DALY", 0.0, 500.0, 135.0, step=5.0, format="%.1f")
        bip_daly = st.number_input("Bipolar Disorder DALY", 0.0, 500.0, 148.0, step=5.0, format="%.1f")
        eat_daly = st.number_input("Eating Disorders DALY", 0.0, 200.0, 25.0, step=1.0, format="%.1f")
        anx_daly = st.number_input("Anxiety Disorders DALY", 0.0, 1000.0, 400.0, step=5.0, format="%.1f")

        year_input = st.slider("Year", 1990, 2019, 2019)

        st.caption("Note: Depressive Disorders DALY is excluded from predictions to avoid data leakage "
                   "(it is 99% correlated with the target depression prevalence).")

    with col_result:
        st.markdown("### Prediction Result")

        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            input_map = {
                "Year": year_input,
                "schizophrenia_prev": schiz_prev,
                "anxiety_prev": anx_prev,
                "bipolar_prev": bip_prev,
                "eating_prev": eat_prev,
                "schizophrenia_daly": schiz_daly,
                "bipolar_daly": bip_daly,
                "eating_daly": eat_daly,
                "anxiety_daly": anx_daly,
            }
            input_values = [input_map.get(f, 0.0) for f in feature_names]
            input_array = np.array(input_values).reshape(1, -1)

            prediction = model.predict(input_array)[0]
            probabilities = model.predict_proba(input_array)[0]

            class_names = preproc_info["class_names"]
            pred_label = class_names[prediction] if prediction < len(class_names) else str(prediction)
            confidence = probabilities[prediction]

            risk_color = get_risk_color(pred_label)

            st.markdown(f"""
            <div style="background: {risk_color}20; border: 2px solid {risk_color}; 
                        border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <h2 style="color: {risk_color}; margin: 0;">{pred_label}</h2>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                    Confidence: <strong>{confidence*100:.1f}%</strong>
                </p>
                <p style="color: #666; font-size: 0.9rem;">
                    Model: {best_model_name}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probability Distribution")
            prob_df = pd.DataFrame({
                "Risk Level": class_names,
                "Probability": probabilities,
            })
            fig = px.bar(
                prob_df, x="Risk Level", y="Probability",
                color="Risk Level",
                color_discrete_map={"Low Risk": "#00CC96", "Moderate Risk": "#FFA15A", "High Risk": "#EF553B"},
                title="Prediction Probabilities",
            )
            fig.update_layout(yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="disclaimer-box">
                <strong>⚠️ Disclaimer:</strong> This prediction is a statistical estimate based on 
                population-level data patterns. It must NOT be considered a medical diagnosis. 
                Individual risk assessment requires clinical evaluation by a qualified professional.
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================
elif page == "⚖️ Model Comparison":
    st.header("Model Comparison")

    comparison = load_comparison()
    model_results = load_model_results()
    preproc_info = load_preproc_info()
    y_pred_all = load_all_predictions()

    if comparison is None or model_results is None:
        st.error("Model results not found. Please run `python src/run_training.py` first.")
        st.stop()

    st.markdown("### Performance Metrics Comparison")
    st.dataframe(comparison.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        color="#d4edda",
    ), use_container_width=True)

    st.markdown("### Visual Comparison")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_model_comparison_bar(comparison)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = plot_cv_comparison(comparison)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Deep Dive")

    selected_model = st.selectbox("Select Model to Inspect", comparison["Model"].tolist())

    if selected_model in model_results:
        res = model_results[selected_model]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{res['accuracy']:.4f}")
        with col2:
            st.metric("F1-Score", f"{res['f1']:.4f}")
        with col3:
            st.metric("CV Mean", f"{res['cv_mean']:.4f}")
        with col4:
            st.metric("CV Std", f"{res['cv_std']:.4f}")

        st.markdown("#### Confusion Matrix")
        cm = np.array(res["confusion_matrix"])
        class_names = preproc_info["class_names"] if preproc_info else [str(i) for i in range(cm.shape[0])]
        fig = plot_confusion_matrix(cm, class_names, selected_model)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Actual vs Predicted Distribution")
        y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.joblib"))
        model_pred = y_pred_all.get(selected_model)
        if model_pred is not None and len(model_pred) == len(y_test):
            fig = plot_actual_vs_predicted(y_test, np.array(model_pred), class_names, selected_model)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Prediction distribution not available for this model.")

        if res.get("feature_importance"):
            st.markdown("#### Feature Importance")
            fig = plot_feature_importance_bar(res["feature_importance"], top_n=15, model_name=selected_model)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Classification Report")
        st.code(res.get("classification_report", "Not available"))


# ============================================================
# EXPLAINABLE AI PAGE
# ============================================================
elif page == "🔍 Explainable AI":
    st.header("Explainable AI (SHAP)")

    shap_data = load_shap_data()
    model = load_trained_model()
    preproc_info = load_preproc_info()
    comparison = load_comparison()

    if model is None or preproc_info is None:
        st.error("Model artifacts not found. Run training first.")
        st.stop()

    if shap_data and shap_data.get("success"):
        st.markdown("### Global Feature Importance (SHAP)")

        feature_names = preproc_info.get("feature_names", [])
        shap_values = shap_data.get("shap_values")

        fig = plot_shap_summary(shap_values, feature_names)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### SHAP Beeswarm Plot")
        X_full = joblib.load(os.path.join(MODELS_DIR, "X_test.joblib"))
        X_test_limited = shap_data.get("X_test_limited", X_full)
        fig = plot_shap_beeswarm(shap_values, feature_names, X_test_limited)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Individual Prediction Explanation")

        sample_idx = st.number_input(
            "Select sample index (0 to {} )".format(X_test_limited.shape[0] - 1),
            min_value=0, max_value=X_test_limited.shape[0] - 1, value=0,
        )

        if st.button("Explain This Prediction", type="primary"):
            explanation = explain_individual_prediction(
                model, X_test_limited, sample_idx, feature_names,
                preproc_info["class_names"], shap_values,
            )

            pred = explanation["prediction"]
            conf = explanation["confidence"]
            risk_color = get_risk_color(pred)

            st.markdown(f"""
            <div style="background: {risk_color}20; border: 2px solid {risk_color}; 
                        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                <h3 style="color: {risk_color}; margin: 0;">Prediction: {pred}</h3>
                <p style="font-size: 1.2rem;">Estimated probability: <strong>{conf*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            if explanation["probabilities"]:
                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability": v}
                    for k, v in explanation["probabilities"].items()
                ])
                st.dataframe(prob_df, use_container_width=True)

            if explanation["contributing_factors"]:
                st.markdown("#### Major Contributing Factors")
                for factor in explanation["contributing_factors"]:
                    direction = factor["direction"]
                    icon = "↑" if direction == "increases" else "↓"
                    color = "#EF553B" if direction == "increases" else "#00CC96"
                    st.markdown(
                        f"<span style='color:{color}; font-weight:bold;'>{icon}</span> "
                        f"**{factor['feature']}** (SHAP: {factor['shap_value']:+.4f}) — "
                        f"The model associated this feature with **{direction}** the predicted risk.",
                        unsafe_allow_html=True,
                    )

                fig = plot_individual_contribution(explanation["contributing_factors"])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="disclaimer-box">
                <strong>Disclaimer:</strong> {explanation['disclaimer']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("SHAP analysis could not be loaded. This may happen if the model type doesn't support TreeExplainer.")


# ============================================================
# INSIGHTS PAGE
# ============================================================
elif page == "💡 Insights":
    st.header("Data-Driven Insights")

    merged_df = load_merged_data()
    insights = generate_insights(merged_df)

    st.markdown(f"Generated **{len(insights)}** insights from the dataset.\n")

    categories = list(set(i["category"] for i in insights))
    selected_cat = st.multiselect("Filter by Category", categories, default=categories)

    filtered_insights = [i for i in insights if i["category"] in selected_cat]

    for insight in filtered_insights:
        importance_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        icon = importance_color.get(insight["importance"], "⚪")

        with st.expander(f"{icon} [{insight['category']}] {insight['title']}"):
            st.markdown(insight["finding"])

    st.markdown("---")
    st.markdown("### Key Findings Summary")

    high_insights = [i for i in insights if i["importance"] == "high"]
    if high_insights:
        for insight in high_insights:
            st.markdown(f"- **{insight['title']}**: {insight['finding']}")
